"""Generate auditable entity-link candidates with an explicit NIL option."""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

from .evidence import (
    EntityLinkArtifact,
    EntityLinkCandidate,
    EntityType,
    NERArtifact,
    ProcessingRun,
    RunKind,
)


def normalize_name(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFC", value).strip()
        if not character.isspace() and unicodedata.category(character)[0] not in {"P", "S"}
    )


@dataclass(frozen=True, slots=True)
class AuthorityEntity:
    entity_id: Any
    entity_type: EntityType
    canonical_name: str
    normalized_name: str
    authority_uri: str | None
    aliases: tuple[str, ...]


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
            )
        )
    return entities


def create_link_artifact(
    ner: NERArtifact,
    entities: list[AuthorityEntity],
    top_k: int = 5,
    fuzzy_threshold: float = 0.72,
) -> EntityLinkArtifact:
    started_at = datetime.now(timezone.utc)
    run = ProcessingRun(
        kind=RunKind.ENTITY_LINK,
        engine="exact-alias+character-similarity",
        model_name="reviewed-authority-candidate-generator",
        model_revision="1",
        configuration={
            "reviewed_entities_only": True,
            "top_k": top_k,
            "fuzzy_threshold": fuzzy_threshold,
            "normalization": "NFC+strip-space-punctuation-symbols",
        },
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
    )
    links = [
        link
        for mention in ner.mentions
        for link in candidate_links(mention, entities, run.run_id, top_k, fuzzy_threshold)
    ]
    warnings = [
        "Entity links are candidates only. NIL is always retained and no mention is force-linked."
    ]
    if not entities:
        warnings.append("The reviewed entity catalog is empty; all generated candidates are NIL/new-entity options.")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    if args.top_k < 1 or not 0 <= args.fuzzy_threshold <= 1:
        raise SystemExit("--top-k must be positive and --fuzzy-threshold must be between 0 and 1")
    ner = NERArtifact.model_validate_json(args.ner_artifact.read_text(encoding="utf-8"))
    artifact = create_link_artifact(
        ner,
        load_reviewed_entities(args.database_url),
        args.top_k,
        args.fuzzy_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "links": len(artifact.links),
                "nil_links": sum(link.nil_candidate for link in artifact.links),
                "warnings": artifact.warnings,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
