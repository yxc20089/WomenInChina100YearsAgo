"""Generate auditable entity-link candidates with an explicit NIL option."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from .entity_resolution import (
    MentionResolutionContext,
    annotate_candidates_with_proposal,
    resolve_with_qwen,
)
from .evidence import (
    EntityLinkArtifact,
    EntityLinkCandidate,
    EntityType,
    NERArtifact,
    ProcessingRun,
    RunKind,
)
from .generation import OpenAICompatibleGenerator
from .model_config import load_pipeline_model_configuration
from .ner_structured import verify_ollama_model_digest


_PIPELINE_MODELS = load_pipeline_model_configuration()
_SEMANTIC_MODEL = _PIPELINE_MODELS.semantic
QWEN_RESOLVER_MODEL = _SEMANTIC_MODEL.model_name
QWEN_RESOLVER_REVISION = _SEMANTIC_MODEL.model_revision
QWEN_RESOLVER_SERVED_MODEL = _SEMANTIC_MODEL.served_model
QWEN_RESOLVER_OLLAMA_DIGEST = _SEMANTIC_MODEL.ollama_manifest_digest


def normalize_name(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFC", value).strip()
        if not character.isspace() and unicodedata.category(character)[0] not in {"P", "S"}
    )


def authority_catalog_sha256(entities: Sequence["AuthorityEntity"]) -> str:
    """Hash both matched and unmatched reviewed authority records."""
    payload = [
        {
            "entity_id": str(entity.entity_id),
            "entity_type": entity.entity_type.value,
            "canonical_name": entity.canonical_name,
            "normalized_name": entity.normalized_name,
            "authority_uri": entity.authority_uri,
            "aliases": sorted(set(entity.aliases)),
        }
        for entity in sorted(entities, key=lambda item: str(item.entity_id))
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class AuthorityEntity:
    entity_id: Any
    entity_type: EntityType
    canonical_name: str
    normalized_name: str
    authority_uri: str | None
    aliases: tuple[str, ...]
    attributes: dict[str, Any] = field(default_factory=dict)


def candidate_links(
    mention: Any,
    entities: list[AuthorityEntity],
    run_id: Any,
    top_k: int = 5,
    fuzzy_threshold: float = 0.72,
) -> list[EntityLinkCandidate]:
    mention_name = normalize_name(mention.normalized_text or mention.text)
    candidates = []
    for entity in entities:
        if entity.entity_type != mention.entity_type:
            continue
        names = {entity.normalized_name, *(normalize_name(alias) for alias in entity.aliases)}
        exact = mention_name in names
        similarity = max(
            (SequenceMatcher(None, mention_name, name).ratio() for name in names if name),
            default=0.0,
        )
        if not exact and similarity < fuzzy_threshold:
            continue
        score = 1.0 if exact else similarity
        candidates.append(
            EntityLinkCandidate(
                mention_id=mention.mention_id,
                entity_id=entity.entity_id,
                authority_uri=entity.authority_uri,
                canonical_name=entity.canonical_name,
                entity_type=entity.entity_type,
                score=score,
                features={
                    "exact_normalized_match": exact,
                    "character_sequence_similarity": similarity,
                    "candidate_entity_status": "reviewed",
                },
                nil_candidate=False,
                run_id=run_id,
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    candidates = candidates[:top_k]
    best = candidates[0].score if candidates else 0.0
    candidates.append(
        EntityLinkCandidate(
            mention_id=mention.mention_id,
            canonical_name=mention.text,
            entity_type=mention.entity_type,
            score=max(0.05, 1.0 - best),
            features={"reason": "explicit_nil_or_new_entity_option", "best_catalog_score": best},
            nil_candidate=True,
            run_id=run_id,
        )
    )
    return candidates


def load_reviewed_entities(database_url: str) -> list[AuthorityEntity]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            """
            SELECT entity_id, entity_type, canonical_name, normalized_name,
                   authority_uri, attributes
            FROM evidence.entity
            WHERE entity_status = 'reviewed'
            ORDER BY entity_id
            """
        ).fetchall()
    entities = []
    for row in rows:
        try:
            entity_type = EntityType(row["entity_type"])
        except ValueError:
            continue
        aliases = tuple(row["attributes"].get("aliases", []))
        entities.append(
            AuthorityEntity(
                row["entity_id"],
                entity_type,
                row["canonical_name"],
                normalize_name(row["normalized_name"] or row["canonical_name"]),
                row["authority_uri"],
                aliases,
                row["attributes"],
            )
        )
    return entities


def load_mention_contexts(
    database_url: str, ner: NERArtifact
) -> dict[UUID, MentionResolutionContext]:
    """Load exact immutable OCR context for already-ingested NER mentions."""
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    mention_ids = [mention.mention_id for mention in ner.mentions]
    if not mention_ids:
        return {}
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            """
            SELECT mention.mention_id, mention.entity_type, mention.mention_text,
                   mention.normalized_text, mention.region_id, region.raw_text,
                   mention.text_start, mention.text_end, mention.run_id
            FROM evidence.entity_mention mention
            JOIN evidence.ocr_region region USING (region_id)
            WHERE mention.mention_id = ANY(%s)
            ORDER BY mention.mention_id
            """,
            (mention_ids,),
        ).fetchall()
    if {row["mention_id"] for row in rows} != set(mention_ids):
        raise ValueError("entity resolution context is missing ingested mentions")
    contexts = {}
    for row in rows:
        if row["run_id"] != ner.run.run_id:
            raise ValueError("entity resolution mention belongs to another NER run")
        contexts[row["mention_id"]] = MentionResolutionContext(
            mention_id=row["mention_id"],
            entity_type=EntityType(row["entity_type"]),
            mention_text=row["mention_text"],
            normalized_text=row["normalized_text"],
            region_id=row["region_id"],
            source_text=row["raw_text"],
            text_start=row["text_start"],
            text_end=row["text_end"],
        )
    return contexts


def create_link_artifact(
    ner: NERArtifact,
    entities: list[AuthorityEntity],
    top_k: int = 5,
    fuzzy_threshold: float = 0.72,
    *,
    mention_contexts: dict[UUID, MentionResolutionContext] | None = None,
    resolver: OpenAICompatibleGenerator | None = None,
    resolver_identity: dict[str, Any] | None = None,
) -> EntityLinkArtifact:
    started_at = datetime.now(timezone.utc)
    run = ProcessingRun(
        kind=RunKind.ENTITY_LINK,
        engine=(
            "exact-alias+character-similarity+qwen-candidate-bound"
            if resolver is not None
            else "exact-alias+character-similarity"
        ),
        model_name=(
            f"reviewed-authority-candidate-generator+{resolver.model}"
            if resolver is not None
            else "reviewed-authority-candidate-generator"
        ),
        model_revision=(
            f"1+{resolver.model_revision}"
            if resolver is not None
            else "1"
        ),
        configuration={
            "reviewed_entities_only": True,
            "top_k": top_k,
            "fuzzy_threshold": fuzzy_threshold,
            "normalization": "NFC+strip-space-punctuation-symbols",
            "authority_catalog_sha256": authority_catalog_sha256(entities),
            "resolver": resolver_identity,
            "identity_mutation": False,
            "model_selection_domain": "supplied_link_candidate_ids_only",
        },
        started_at=started_at,
    )
    links = []
    aliases_by_entity_id = {
        UUID(str(entity.entity_id)): entity.aliases
        for entity in entities
        if entity.entity_id is not None
    }
    proposal_counts = {"LINK": 0, "NIL": 0, "ABSTAIN": 0, "not_run": 0}
    if resolver is not None:
        if mention_contexts is None:
            raise ValueError("Qwen resolution requires exact mention contexts")
        if set(mention_contexts) != {mention.mention_id for mention in ner.mentions}:
            raise ValueError("mention contexts must cover exactly the NER artifact mentions")
    for mention in ner.mentions:
        roster = candidate_links(
            mention, entities, run.run_id, top_k, fuzzy_threshold
        )
        if resolver is not None and any(not item.nil_candidate for item in roster):
            proposal = resolve_with_qwen(
                mention_contexts[mention.mention_id],
                roster,
                resolver,
                aliases_by_entity_id,
            )
            roster = annotate_candidates_with_proposal(roster, proposal)
            proposal_counts[proposal.decision.value] += 1
        else:
            proposal_counts["not_run"] += 1
        links.extend(roster)
    run = run.model_copy(
        update={
            "configuration": {
                **run.configuration,
                "proposal_counts": proposal_counts,
            },
            "completed_at": datetime.now(timezone.utc),
        }
    )
    warnings = [
        "Entity links are candidates only. NIL is always retained and no mention is force-linked."
    ]
    if not entities:
        warnings.append("The reviewed entity catalog is empty; all generated candidates are NIL/new-entity options.")
    if resolver is not None:
        warnings.append(
            "Qwen proposals select only immutable candidate IDs; review is required before mention.entity_id changes."
        )
    return EntityLinkArtifact(
        source_ner_run_id=ner.run.run_id,
        run=run,
        links=links,
        warnings=warnings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--ner-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.72)
    parser.add_argument("--resolver", choices=("none", "qwen"), default="none")
    parser.add_argument(
        "--model-config",
        help="Complete model configuration; individual model overrides are not accepted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    if args.top_k < 1 or not 0 <= args.fuzzy_threshold <= 1:
        raise SystemExit("--top-k must be positive and --fuzzy-threshold must be between 0 and 1")
    ner = NERArtifact.model_validate_json(args.ner_artifact.read_text(encoding="utf-8"))
    entities = load_reviewed_entities(args.database_url)
    resolver = None
    resolver_identity = None
    contexts = None
    if args.resolver == "qwen":
        pipeline_configuration = load_pipeline_model_configuration(args.model_config)
        semantic = pipeline_configuration.semantic
        local_artifact_sha256 = semantic.ollama_manifest_digest.removeprefix("sha256:")
        resolver = OpenAICompatibleGenerator(
            semantic.base_url,
            semantic.served_model,
            api_key=os.environ.get("ENTITY_LLM_API_KEY"),
            model_revision=local_artifact_sha256,
            timeout_seconds=semantic.timeout_seconds,
            max_output_tokens=semantic.max_output_tokens,
            seed=semantic.seed,
            allow_remote=False,
        )
        verification = verify_ollama_model_digest(
            resolver,
            semantic.ollama_manifest_digest,
            semantic.runtime_version,
        )
        contexts = load_mention_contexts(args.database_url, ner)
        resolver_identity = {
            "family": "qwen_candidate_bound",
            "model": semantic.model_name,
            "model_revision": semantic.model_revision,
            "served_model": semantic.served_model,
            "local_artifact_sha256": local_artifact_sha256,
            "runtime": semantic.runtime_name,
            "runtime_version": semantic.runtime_version,
            "runtime_verification": verification.model_dump(mode="json"),
            "quantization": semantic.quantization,
            "temperature": semantic.temperature,
            "top_p": 1,
            "reasoning_effort": "none",
            "seed": semantic.seed,
            "max_output_tokens": semantic.max_output_tokens,
            "timeout_seconds": semantic.timeout_seconds,
            "acceleration": semantic.acceleration,
            "pipeline_model_configuration_sha256": pipeline_configuration.sha256,
        }
    artifact = create_link_artifact(
        ner,
        entities,
        args.top_k,
        args.fuzzy_threshold,
        mention_contexts=contexts,
        resolver=resolver,
        resolver_identity=resolver_identity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "links": len(artifact.links),
                "nil_links": sum(link.nil_candidate for link in artifact.links),
                "authority_catalog_sha256": artifact.run.configuration[
                    "authority_catalog_sha256"
                ],
                "warnings": artifact.warnings,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
