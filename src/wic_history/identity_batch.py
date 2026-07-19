"""Frozen-cohort identity resolution after ingestion, never continuous merging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Sequence
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from .link_pipeline import normalize_name
from .model_config import load_pipeline_model_configuration
from .semantic_tasks import (
    IdentityPairResponse,
    SemanticAbstention,
    SemanticTaskResult,
    build_verified_semantic_client,
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _clients() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    return psycopg, dict_row


def normalize_identity_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def character_jaccard(left: str, right: str) -> float:
    left_set = set(normalize_identity_surface(left))
    right_set = set(normalize_identity_surface(right))
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


@dataclass(frozen=True, slots=True)
class FrozenIdentityCohort:
    cohort_id: str
    profiles: int
    snapshot_sha256: str
    configuration_sha256: str
    reused: bool


@dataclass(frozen=True, slots=True)
class PairGenerationResult:
    cohort_id: str
    pairs: int
    deterministic_blocked: int
    embedding_scored: int
    reranker_scored: int


@dataclass(frozen=True, slots=True)
class IdentityDecisionResult:
    identity_pair_decision_id: str
    run_id: str
    decision: str
    reused: bool


@dataclass(frozen=True, slots=True)
class IdentityPairReviewResult:
    identity_pair_decision_id: str
    review_id: str
    review_status: str
    model_decision: str
    canonical_entity_id: str | None
    mention_resolutions: int
    entity_redirect_id: str | None
    reused: bool


def freeze_identity_cohort(
    database_url: str,
    *,
    created_by: str,
    entity_type: str | None = None,
    model_config_path: str | None = None,
) -> FrozenIdentityCohort:
    """Freeze unresolved reviewed mentions and reviewed entities into immutable profiles."""
    if not created_by.strip():
        raise ValueError("created_by must not be blank")
    configuration = load_pipeline_model_configuration(model_config_path)
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        mention_rows = connection.execute(
            """
            SELECT mention.mention_id, mention.entity_type, mention.mention_text,
                   mention.mention_form, span.evidence_span_id,
                   span.surface_text, version.text_content,
                   mention.coherent_unit_revision_id
            FROM evidence.entity_mention mention
            JOIN evidence.evidence_span span USING (evidence_span_id)
            JOIN evidence.text_version version USING (text_version_id)
            LEFT JOIN evidence.mention_resolution resolution
              ON resolution.mention_id = mention.mention_id
             AND resolution.review_status = 'reviewed'
             AND resolution.superseded_at IS NULL
            WHERE mention.mention_status = 'reviewed'
              AND resolution.mention_resolution_id IS NULL
              AND (%s::text IS NULL OR mention.entity_type = %s)
            ORDER BY mention.mention_id
            """,
            (entity_type, entity_type),
        ).fetchall()
        entity_rows = connection.execute(
            """
            SELECT entity.entity_id, entity.entity_type, entity.canonical_name,
                   COALESCE(
                       array_agg(DISTINCT assertion.name_surface)
                           FILTER (WHERE assertion.review_status = 'reviewed'),
                       '{}'::text[]
                   ) AS asserted_names,
                   COALESCE(
                       array_agg(DISTINCT resolution_evidence.evidence_span_id)
                           FILTER (WHERE resolution_evidence.evidence_span_id IS NOT NULL),
                       '{}'::uuid[]
                   ) AS evidence_span_ids
            FROM evidence.entity entity
            LEFT JOIN evidence.entity_name_assertion assertion USING (entity_id)
            LEFT JOIN evidence.mention_resolution resolution
              ON resolution.proposed_entity_id = entity.entity_id
             AND resolution.review_status = 'reviewed'
             AND resolution.superseded_at IS NULL
            LEFT JOIN evidence.entity_mention resolution_mention
              ON resolution_mention.mention_id = resolution.mention_id
            LEFT JOIN evidence.evidence_span resolution_evidence
              ON resolution_evidence.evidence_span_id = resolution_mention.evidence_span_id
            WHERE entity.entity_status = 'reviewed'
              AND (%s::text IS NULL OR entity.entity_type = %s)
            GROUP BY entity.entity_id
            ORDER BY entity.entity_id
            """,
            (entity_type, entity_type),
        ).fetchall()
    profiles: list[dict[str, Any]] = []
    for row in mention_rows:
        names = [row["mention_text"]]
        profile = {
            "profile_kind": "mention",
            "entity_type": row["entity_type"],
            "entity_id": None,
            "mention_ids": [str(row["mention_id"])],
            "evidence_span_ids": [str(row["evidence_span_id"])],
            "name_surfaces": names,
            "attributes": {
                "mention_form": row["mention_form"],
                "coherent_unit_revision_id": (
                    str(row["coherent_unit_revision_id"])
                    if row["coherent_unit_revision_id"]
                    else None
                ),
                "context": row["text_content"][:1000],
            },
        }
        profile["profile_sha256"] = _canonical_sha256(profile)
        profiles.append(profile)
    for row in entity_rows:
        names = list(
            dict.fromkeys([row["canonical_name"], *list(row["asserted_names"] or [])])
        )
        profile = {
            "profile_kind": "entity",
            "entity_type": row["entity_type"],
            "entity_id": str(row["entity_id"]),
            "mention_ids": [],
            "evidence_span_ids": [
                str(value) for value in row["evidence_span_ids"] or []
            ],
            "name_surfaces": names,
            "attributes": {},
        }
        profile["profile_sha256"] = _canonical_sha256(profile)
        profiles.append(profile)
    snapshot = {
        "scope": {
            "entity_type": entity_type,
            "source_policy": "reviewed_entities_and_unresolved_reviewed_mentions",
        },
        "profiles": sorted(profiles, key=lambda item: item["profile_sha256"]),
    }
    snapshot_sha256 = _canonical_sha256(snapshot)
    cohort_id = uuid5(
        NAMESPACE_URL,
        f"wic-identity-cohort:{snapshot_sha256}:{configuration.sha256}",
    )
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        existing = connection.execute(
            "SELECT cohort_id FROM evidence.identity_resolution_cohort WHERE cohort_id = %s",
            (cohort_id,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO evidence.identity_resolution_cohort (
                    cohort_id, scope, snapshot_sha256, configuration_sha256,
                    created_by, status
                ) VALUES (%s, %s::jsonb, %s, %s, %s, 'frozen')
                """,
                (
                    cohort_id,
                    json.dumps(snapshot["scope"], ensure_ascii=False),
                    snapshot_sha256,
                    configuration.sha256,
                    created_by,
                ),
            )
            for profile in profiles:
                profile_id = uuid5(cohort_id, profile["profile_sha256"])
                connection.execute(
                    """
                    INSERT INTO evidence.identity_profile (
                        identity_profile_id, cohort_id, profile_kind,
                        entity_type, entity_id, mention_ids, evidence_span_ids,
                        name_surfaces, profile_sha256, attributes
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        profile_id,
                        cohort_id,
                        profile["profile_kind"],
                        profile["entity_type"],
                        profile["entity_id"],
                        profile["mention_ids"],
                        profile["evidence_span_ids"],
                        profile["name_surfaces"],
                        profile["profile_sha256"],
                        json.dumps(profile["attributes"], ensure_ascii=False),
                    ),
                )
    return FrozenIdentityCohort(
        str(cohort_id),
        len(profiles),
        snapshot_sha256,
        configuration.sha256,
        existing is not None,
    )


def generate_identity_pairs(
    database_url: str,
    cohort_id: UUID,
    *,
    embedding_candidates: dict[tuple[UUID, UUID], float] | None = None,
    embedding_scorer: Callable[[dict[str, Any], dict[str, Any]], float] | None = None,
    reranker_scorer: Any | None = None,
    embedding_run_id: UUID | None = None,
    reranker_run_id: UUID | None = None,
    minimum_character_jaccard: float = 0.35,
    maximum_pairs_per_profile: int = 20,
    require_model_scores: bool = True,
) -> PairGenerationResult:
    """Union deterministic and embedding top-k blocks, then rerank bounded pairs."""
    if not 0 <= minimum_character_jaccard <= 1 or maximum_pairs_per_profile < 1:
        raise ValueError("invalid blocking thresholds")
    if embedding_candidates is not None and embedding_scorer is not None:
        raise ValueError("supply embedding candidates or a legacy scorer, not both")
    if require_model_scores and (
        (embedding_candidates is None and embedding_scorer is None)
        or reranker_scorer is None
        or embedding_run_id is None
        or reranker_run_id is None
    ):
        raise ValueError(
            "production identity pairing requires configured embedding/reranker "
            "scores and their processing-run IDs"
        )
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM evidence.identity_profile
                WHERE cohort_id = %s ORDER BY identity_profile_id
                """,
                (cohort_id,),
            )
        ]
        if not rows:
            raise ValueError("identity cohort is absent or empty")
        profiles = {row["identity_profile_id"]: row for row in rows}
        deterministic: dict[tuple[UUID, UUID], tuple[list[str], float]] = {}
        same_type_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for left_index, left in enumerate(rows):
            left_normalized = {
                normalize_identity_surface(value) for value in left["name_surfaces"]
            }
            for right in rows[left_index + 1 :]:
                if left["entity_type"] != right["entity_type"]:
                    continue
                same_type_pairs.append((left, right))
                right_normalized = {
                    normalize_identity_surface(value) for value in right["name_surfaces"]
                }
                methods: list[str] = []
                if left_normalized & right_normalized:
                    methods.append("exact_normalized_name")
                best_jaccard = max(
                    character_jaccard(one, two)
                    for one in left["name_surfaces"]
                    for two in right["name_surfaces"]
                )
                if best_jaccard >= minimum_character_jaccard:
                    methods.append("character_jaccard")
                if methods:
                    pair_key = tuple(
                        sorted(
                            (left["identity_profile_id"], right["identity_profile_id"])
                        )
                    )
                    deterministic[pair_key] = (methods, best_jaccard)

        candidate_scores: dict[tuple[UUID, UUID], float] = {}
        for supplied_key, supplied_score in (embedding_candidates or {}).items():
            if len(supplied_key) != 2 or supplied_key[0] == supplied_key[1]:
                raise ValueError("embedding candidate keys require two distinct profiles")
            pair_key = tuple(sorted(supplied_key))
            candidate_scores[pair_key] = max(
                candidate_scores.get(pair_key, -1.0), float(supplied_score)
            )
        if embedding_scorer is not None:
            for left, right in same_type_pairs:
                pair_key = tuple(
                    sorted((left["identity_profile_id"], right["identity_profile_id"]))
                )
                candidate_scores[pair_key] = float(embedding_scorer(left, right))
        for pair_key, score in candidate_scores.items():
            if len(pair_key) != 2 or pair_key[0] not in profiles or pair_key[1] not in profiles:
                raise ValueError("embedding candidates must refer only to cohort profiles")
            if profiles[pair_key[0]]["entity_type"] != profiles[pair_key[1]]["entity_type"]:
                raise ValueError("embedding candidate crossed entity types")
            if not -1 <= score <= 1:
                raise ValueError("embedding scores must be in [-1, 1]")

        selected_keys = set(deterministic) | set(candidate_scores)
        selected_pairs = [(profiles[left], profiles[right]) for left, right in selected_keys]
        reranker_values: list[float | None]
        if reranker_scorer is None:
            reranker_values = [None] * len(selected_pairs)
        elif hasattr(reranker_scorer, "score_pairs"):
            reranker_values = [
                float(value) for value in reranker_scorer.score_pairs(selected_pairs)
            ]
        else:
            reranker_values = [
                float(reranker_scorer(left, right)) for left, right in selected_pairs
            ]
        if len(reranker_values) != len(selected_pairs) or any(
            value is not None and not 0 <= value <= 1 for value in reranker_values
        ):
            raise ValueError("reranker must return one [0, 1] score per pair")

        scored: list[
            tuple[dict[str, Any], dict[str, Any], list[str], float | None, float | None, float]
        ] = []
        for (left, right), reranker_score in zip(
            selected_pairs, reranker_values, strict=True
        ):
            pair_key = (left["identity_profile_id"], right["identity_profile_id"])
            methods, best_jaccard = deterministic.get(pair_key, ([], 0.0))
            methods = list(methods)
            embedding_score = candidate_scores.get(pair_key)
            if embedding_score is not None:
                methods.append("embedding_top_k")
            scored.append(
                (
                    left,
                    right,
                    methods,
                    embedding_score,
                    reranker_score,
                    best_jaccard,
                )
            )
        scored.sort(
            key=lambda item: (
                bool(set(item[2]) & {"exact_normalized_name", "character_jaccard"}),
                item[4] if item[4] is not None else -1,
                item[3] if item[3] is not None else -1,
                item[5],
                str(item[0]["identity_profile_id"]),
                str(item[1]["identity_profile_id"]),
            ),
            reverse=True,
        )
        degrees: dict[UUID, int] = {}
        proposals: list[
            tuple[dict[str, Any], dict[str, Any], list[str], float | None, float | None]
        ] = []
        for left, right, methods, embedding_score, reranker_score, _ in scored:
            left_id = left["identity_profile_id"]
            right_id = right["identity_profile_id"]
            if (
                degrees.get(left_id, 0) >= maximum_pairs_per_profile
                or degrees.get(right_id, 0) >= maximum_pairs_per_profile
            ):
                continue
            proposals.append((left, right, methods, embedding_score, reranker_score))
            degrees[left_id] = degrees.get(left_id, 0) + 1
            degrees[right_id] = degrees.get(right_id, 0) + 1

        proposals.sort(
            key=lambda item: (
                str(item[0]["identity_profile_id"]),
                -(item[4] if item[4] is not None else -1),
                -(item[3] if item[3] is not None else -1),
                str(item[1]["identity_profile_id"]),
            )
        )
        ranks: dict[UUID, int] = {}
        for left, right, methods, embedding_score, reranker_score in proposals:
            left_id = left["identity_profile_id"]
            ranks[left_id] = ranks.get(left_id, 0) + 1
            pair_id = uuid5(
                cohort_id,
                f"pair:{left_id}:{right['identity_profile_id']}",
            )
            connection.execute(
                """
                INSERT INTO evidence.identity_pair_candidate (
                    identity_pair_candidate_id, cohort_id, left_profile_id,
                    right_profile_id, blocking_methods, embedding_score,
                    reranker_score, candidate_rank, embedding_run_id,
                    reranker_run_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cohort_id, left_profile_id, right_profile_id) DO NOTHING
                """,
                (
                    pair_id,
                    cohort_id,
                    left_id,
                    right["identity_profile_id"],
                    methods,
                    embedding_score,
                    reranker_score,
                    ranks[left_id],
                    embedding_run_id,
                    reranker_run_id,
                ),
            )
        connection.execute(
            """
            UPDATE evidence.identity_resolution_cohort
            SET status = 'deciding' WHERE cohort_id = %s AND status = 'frozen'
            """,
            (cohort_id,),
        )
    return PairGenerationResult(
        str(cohort_id),
        len(proposals),
        sum(
            bool(set(item[2]) & {"exact_normalized_name", "character_jaccard"})
            for item in proposals
        ),
        sum(item[3] is not None for item in proposals),
        sum(item[4] is not None for item in proposals),
    )


def load_identity_pair(
    database_url: str, identity_pair_candidate_id: UUID
) -> tuple[dict[str, Any], dict[str, Any], list[UUID]]:
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        row = connection.execute(
            """
            SELECT pair.identity_pair_candidate_id,
                   to_jsonb(left_profile) AS left_profile,
                   to_jsonb(right_profile) AS right_profile
            FROM evidence.identity_pair_candidate pair
            JOIN evidence.identity_profile left_profile
              ON left_profile.identity_profile_id = pair.left_profile_id
            JOIN evidence.identity_profile right_profile
              ON right_profile.identity_profile_id = pair.right_profile_id
            WHERE pair.identity_pair_candidate_id = %s
            """,
            (identity_pair_candidate_id,),
        ).fetchone()
    if row is None:
        raise ValueError("unknown identity pair candidate")
    evidence_ids = sorted(
        {
            *(UUID(str(value)) for value in row["left_profile"]["evidence_span_ids"]),
            *(UUID(str(value)) for value in row["right_profile"]["evidence_span_ids"]),
        }
    )
    return row["left_profile"], row["right_profile"], evidence_ids


def persist_identity_pair_decision(
    database_url: str,
    identity_pair_candidate_id: UUID,
    result: SemanticTaskResult[IdentityPairResponse],
    *,
    model_config_path: str | None = None,
) -> IdentityDecisionResult:
    configuration = load_pipeline_model_configuration(model_config_path)
    model = configuration.semantic
    model_identity = model.provenance_identity()
    run_id = uuid5(
        identity_pair_candidate_id,
        f"qwen-pair:{configuration.sha256}:{result.prompt_schema_sha256}:{result.prompt_sha256}",
    )
    decision_id = uuid5(run_id, "identity-pair-decision")
    now = datetime.now(timezone.utc)
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        existing = connection.execute(
            """
            SELECT identity_pair_decision_id, decision
            FROM evidence.identity_pair_decision
            WHERE identity_pair_decision_id = %s
            """,
            (decision_id,),
        ).fetchone()
        if existing is not None:
            return IdentityDecisionResult(
                str(decision_id), str(run_id), existing["decision"], True
            )
        connection.execute(
            """
            INSERT INTO evidence.processing_run (
                run_id, kind, engine, model_name, model_revision,
                software_version, configuration, status, started_at, completed_at
            ) VALUES (
                %s, 'entity_link', %s, %s, %s, %s, %s::jsonb,
                'completed', %s, %s
            )
            """,
            (
                run_id,
                f"structured-semantic:{model.provider}",
                model_identity["model_name"],
                model_identity["model_revision"],
                model_identity["runtime_version"],
                json.dumps(
                    {
                        "task": "identity_pair",
                        "pipeline_model_configuration_sha256": configuration.sha256,
                        **model_identity,
                        "prompt_sha256": result.prompt_sha256,
                        "prompt_schema_sha256": result.prompt_schema_sha256,
                        "raw_output_sha256": result.raw_output_sha256,
                    }
                ),
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO evidence.identity_pair_decision (
                identity_pair_decision_id, identity_pair_candidate_id, run_id,
                decision, supporting_evidence_ids,
                contradiction_evidence_ids, prompt_sha256, raw_output_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                decision_id,
                identity_pair_candidate_id,
                run_id,
                result.response.decision,
                result.response.supporting_evidence_ids,
                result.response.contradiction_evidence_ids,
                result.prompt_sha256,
                result.raw_output_sha256,
            ),
        )
    return IdentityDecisionResult(
        str(decision_id), str(run_id), result.response.decision, False
    )


def review_identity_pair_decision(
    database_url: str,
    identity_pair_decision_id: UUID,
    *,
    action: Literal["accept", "reject", "needs_review"],
    reviewer: str,
    note: str | None = None,
    canonical_entity_id: UUID | None = None,
    new_canonical_name: str | None = None,
    review_id: UUID | None = None,
) -> IdentityPairReviewResult:
    """Promote a reviewed pair without letting the model choose the canonical entity."""
    if not reviewer.strip():
        raise ValueError("reviewer must not be blank")
    review_id = review_id or uuid4()
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        existing_review = connection.execute(
            """
            SELECT target_kind, target_id, decision, reviewer, note, new_value
            FROM evidence.review_decision WHERE review_id = %s
            """,
            (review_id,),
        ).fetchone()
        if existing_review is not None:
            if (
                existing_review["target_kind"] != "identity_pair_decision"
                or existing_review["target_id"] != identity_pair_decision_id
                or existing_review["decision"] != action
                or existing_review["reviewer"] != reviewer
                or existing_review["note"] != note
            ):
                raise ValueError("review_id retry differs from the stored review")
            stored = existing_review["new_value"] or {}
            if (
                stored.get("requested_canonical_entity_id")
                != (str(canonical_entity_id) if canonical_entity_id else None)
                or stored.get("new_canonical_name") != new_canonical_name
            ):
                raise ValueError("review_id retry canonical selection has changed")
            return IdentityPairReviewResult(
                str(identity_pair_decision_id),
                str(review_id),
                stored.get("review_status", "candidate"),
                stored.get("model_decision", "INSUFFICIENT"),
                stored.get("canonical_entity_id"),
                int(stored.get("mention_resolutions", 0)),
                stored.get("entity_redirect_id"),
                True,
            )

        row = connection.execute(
            """
            SELECT decision.*, pair.cohort_id,
                   to_jsonb(left_profile) AS left_profile,
                   to_jsonb(right_profile) AS right_profile
            FROM evidence.identity_pair_decision decision
            JOIN evidence.identity_pair_candidate pair
              ON pair.identity_pair_candidate_id = decision.identity_pair_candidate_id
            JOIN evidence.identity_profile left_profile
              ON left_profile.identity_profile_id = pair.left_profile_id
            JOIN evidence.identity_profile right_profile
              ON right_profile.identity_profile_id = pair.right_profile_id
            WHERE decision.identity_pair_decision_id = %s
            FOR UPDATE OF decision
            """,
            (identity_pair_decision_id,),
        ).fetchone()
        if row is None:
            raise ValueError("unknown identity pair decision")
        if row["review_status"] != "candidate":
            raise ValueError("only candidate identity decisions can enter review")
        model_decision = row["decision"]
        left = row["left_profile"]
        right = row["right_profile"]
        if left["entity_type"] != right["entity_type"]:
            raise ValueError("identity pair crossed entity types")
        if action != "accept" or model_decision != "SAME":
            if canonical_entity_id is not None or new_canonical_name is not None:
                raise ValueError("canonical selection applies only to accepted SAME pairs")

        resolved_entity_id: UUID | None = None
        redirect_id: UUID | None = None
        resolution_count = 0
        if action == "accept" and model_decision == "SAME":
            entity_ids = [
                UUID(str(value))
                for value in (left.get("entity_id"), right.get("entity_id"))
                if value is not None
            ]
            mention_ids = [
                UUID(str(value))
                for value in [
                    *(left.get("mention_ids") or []),
                    *(right.get("mention_ids") or []),
                ]
            ]
            if len(set(entity_ids)) != len(entity_ids):
                raise ValueError("identity pair repeats an entity profile")
            entity_rows = connection.execute(
                """
                SELECT entity_id, entity_type, canonical_name, entity_status
                FROM evidence.entity
                WHERE entity_id = ANY(%s::uuid[])
                ORDER BY entity_id FOR UPDATE
                """,
                (entity_ids,),
            ).fetchall() if entity_ids else []
            if len(entity_rows) != len(entity_ids) or any(
                entity["entity_type"] != left["entity_type"]
                or entity["entity_status"] != "reviewed"
                for entity in entity_rows
            ):
                raise ValueError("pair entities must still be reviewed and type-compatible")

            if len(entity_ids) == 2:
                if canonical_entity_id not in set(entity_ids):
                    raise ValueError(
                        "entity-to-entity SAME review requires an explicit canonical pair member"
                    )
                if new_canonical_name is not None:
                    raise ValueError("existing entity merge cannot create a new canonical name")
                resolved_entity_id = canonical_entity_id
                superseded_entity_id = next(
                    value for value in entity_ids if value != resolved_entity_id
                )
                terminal = connection.execute(
                    """
                    SELECT 1 FROM evidence.entity_redirect
                    WHERE superseded_entity_id = %s AND reversed_at IS NULL
                    """,
                    (resolved_entity_id,),
                ).fetchone()
                if terminal is not None:
                    raise ValueError("canonical entity must be the active terminal entity")
                redirect_id = uuid4()
                redirect_review_id = uuid4()
                connection.execute(
                    """
                    INSERT INTO evidence.review_decision (
                        review_id, target_kind, target_id, decision, reviewer, note,
                        previous_value, new_value
                    ) VALUES (
                        %s, 'entity_redirect', %s, 'accept', %s, %s,
                        %s::jsonb, %s::jsonb
                    )
                    """,
                    (
                        redirect_review_id,
                        redirect_id,
                        reviewer,
                        note or f"Promoted identity decision {identity_pair_decision_id}",
                        json.dumps(
                            {
                                "superseded_entity_id": str(superseded_entity_id),
                                "canonical_entity_id": str(resolved_entity_id),
                            }
                        ),
                        json.dumps(
                            {
                                "active": True,
                                "identity_pair_decision_id": str(identity_pair_decision_id),
                            }
                        ),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO evidence.entity_redirect (
                        entity_redirect_id, superseded_entity_id,
                        canonical_entity_id, review_id, reason
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        redirect_id,
                        superseded_entity_id,
                        resolved_entity_id,
                        redirect_review_id,
                        note or f"Reviewed identity decision {identity_pair_decision_id}",
                    ),
                )
                connection.execute(
                    """
                    UPDATE evidence.entity
                    SET entity_status = 'merged', updated_at = now()
                    WHERE entity_id = %s
                    """,
                    (superseded_entity_id,),
                )
            elif len(entity_ids) == 1:
                resolved_entity_id = entity_ids[0]
                if canonical_entity_id not in {None, resolved_entity_id}:
                    raise ValueError("canonical entity is not the entity in this pair")
                if new_canonical_name is not None:
                    raise ValueError("mention-to-entity review reuses the reviewed entity")
            else:
                if canonical_entity_id is not None:
                    raise ValueError("mention-to-mention review cannot select an unrelated entity")
                if not (new_canonical_name or "").strip():
                    raise ValueError(
                        "mention-to-mention SAME review requires a historian canonical name"
                    )
                resolved_entity_id = uuid4()
                connection.execute(
                    """
                    INSERT INTO evidence.entity (
                        entity_id, entity_type, canonical_name, normalized_name,
                        entity_status, attributes
                    ) VALUES (%s, %s, %s, %s, 'reviewed', %s::jsonb)
                    """,
                    (
                        resolved_entity_id,
                        left["entity_type"],
                        new_canonical_name.strip(),
                        normalize_name(new_canonical_name.strip()),
                        json.dumps(
                            {
                                "created_from_identity_pair_decision_id": str(
                                    identity_pair_decision_id
                                ),
                                "created_by": reviewer,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )

            mention_rows = connection.execute(
                """
                SELECT mention.mention_id, mention.entity_type,
                       mention.mention_status, mention.evidence_span_id,
                       resolution.mention_resolution_id AS active_resolution_id
                FROM evidence.entity_mention mention
                LEFT JOIN evidence.mention_resolution resolution
                  ON resolution.mention_id = mention.mention_id
                 AND resolution.review_status = 'reviewed'
                 AND resolution.superseded_at IS NULL
                WHERE mention.mention_id = ANY(%s::uuid[])
                ORDER BY mention.mention_id FOR UPDATE OF mention
                """,
                (mention_ids,),
            ).fetchall() if mention_ids else []
            if len(mention_rows) != len(set(mention_ids)) or any(
                mention["mention_status"] != "reviewed"
                or mention["entity_type"] != left["entity_type"]
                or mention["active_resolution_id"] is not None
                for mention in mention_rows
            ):
                raise ValueError(
                    "pair mentions must be reviewed, type-compatible, and unresolved"
                )
            for mention in mention_rows:
                resolution_id = uuid4()
                resolution_review_id = uuid4()
                connection.execute(
                    """
                    INSERT INTO evidence.review_decision (
                        review_id, target_kind, target_id, decision, reviewer, note,
                        previous_value, new_value
                    ) VALUES (
                        %s, 'mention_resolution', %s, 'accept', %s, %s,
                        %s::jsonb, %s::jsonb
                    )
                    """,
                    (
                        resolution_review_id,
                        resolution_id,
                        reviewer,
                        note or f"Promoted identity decision {identity_pair_decision_id}",
                        json.dumps({"active_resolution": None}),
                        json.dumps(
                            {
                                "mention_id": str(mention["mention_id"]),
                                "proposed_entity_id": str(resolved_entity_id),
                                "identity_pair_decision_id": str(
                                    identity_pair_decision_id
                                ),
                            }
                        ),
                    ),
                )
                supporting = sorted(
                    {
                        *row["supporting_evidence_ids"],
                        mention["evidence_span_id"],
                    }
                )
                connection.execute(
                    """
                    INSERT INTO evidence.mention_resolution (
                        mention_resolution_id, mention_id, proposed_entity_id,
                        is_nil, resolution_scope, run_id, proposal,
                        supporting_evidence_ids, contradiction_evidence_ids,
                        review_status, review_id
                    ) VALUES (
                        %s, %s, %s, false, 'corpus', %s, 'SAME',
                        %s, %s, 'reviewed', %s
                    )
                    """,
                    (
                        resolution_id,
                        mention["mention_id"],
                        resolved_entity_id,
                        row["run_id"],
                        supporting,
                        row["contradiction_evidence_ids"],
                        resolution_review_id,
                    ),
                )
                resolution_count += 1

        desired_status = {
            "accept": "reviewed",
            "reject": "rejected",
            "needs_review": "candidate",
        }[action]
        new_value = {
            "review_status": desired_status,
            "model_decision": model_decision,
            "canonical_entity_id": (
                str(resolved_entity_id) if resolved_entity_id is not None else None
            ),
            "mention_resolutions": resolution_count,
            "entity_redirect_id": str(redirect_id) if redirect_id else None,
            "requested_canonical_entity_id": (
                str(canonical_entity_id) if canonical_entity_id else None
            ),
            "new_canonical_name": new_canonical_name,
        }
        connection.execute(
            """
            INSERT INTO evidence.review_decision (
                review_id, target_kind, target_id, decision, reviewer, note,
                previous_value, new_value
            ) VALUES (
                %s, 'identity_pair_decision', %s, %s, %s, %s,
                %s::jsonb, %s::jsonb
            )
            """,
            (
                review_id,
                identity_pair_decision_id,
                action,
                reviewer,
                note,
                json.dumps(
                    {
                        "review_status": row["review_status"],
                        "model_decision": model_decision,
                    }
                ),
                json.dumps(new_value),
            ),
        )
        if action != "needs_review":
            connection.execute(
                """
                UPDATE evidence.identity_pair_decision
                SET review_status = %s, review_id = %s
                WHERE identity_pair_decision_id = %s
                """,
                (desired_status, review_id, identity_pair_decision_id),
            )
        outstanding = connection.execute(
            """
            SELECT count(*) AS count
            FROM evidence.identity_pair_candidate pair
            LEFT JOIN evidence.identity_pair_decision decision
              ON decision.identity_pair_candidate_id = pair.identity_pair_candidate_id
             AND decision.review_status IN ('reviewed', 'rejected')
            WHERE pair.cohort_id = %s
              AND decision.identity_pair_decision_id IS NULL
            """,
            (row["cohort_id"],),
        ).fetchone()["count"]
        if outstanding == 0:
            connection.execute(
                """
                UPDATE evidence.identity_resolution_cohort
                SET status = 'completed', completed_at = now()
                WHERE cohort_id = %s
                """,
                (row["cohort_id"],),
            )
    return IdentityPairReviewResult(
        str(identity_pair_decision_id),
        str(review_id),
        desired_status,
        model_decision,
        str(resolved_entity_id) if resolved_entity_id else None,
        resolution_count,
        str(redirect_id) if redirect_id else None,
        False,
    )


def prepare_identity_batch(
    database_url: str,
    *,
    created_by: str,
    entity_type: str | None = None,
    model_config_path: str | None = None,
    embedding_batch_size: int = 16,
    reranker_batch_size: int = 8,
    neighbors_per_profile: int = 40,
    maximum_pairs_per_profile: int = 20,
) -> dict[str, Any]:
    """Freeze, embed, retrieve, rerank, and persist one candidate cohort."""
    from .identity_models import (
        QwenIdentityEmbedder,
        QwenIdentityReranker,
        complete_identity_reranker_run,
        load_identity_embedding_neighbors,
        persist_identity_profile_embeddings,
        start_identity_reranker_run,
    )

    configuration = load_pipeline_model_configuration(model_config_path)
    cohort = freeze_identity_cohort(
        database_url,
        created_by=created_by,
        entity_type=entity_type,
        model_config_path=model_config_path,
    )
    embedder = QwenIdentityEmbedder(configuration)
    embedded = persist_identity_profile_embeddings(
        database_url,
        UUID(cohort.cohort_id),
        model_config_path=model_config_path,
        batch_size=embedding_batch_size,
        embedder=embedder,
    )
    neighbors = load_identity_embedding_neighbors(
        database_url,
        UUID(cohort.cohort_id),
        model_config_path=model_config_path,
        neighbors_per_profile=neighbors_per_profile,
    )
    reranker = QwenIdentityReranker(configuration)
    reranker_run = start_identity_reranker_run(
        database_url,
        UUID(cohort.cohort_id),
        model_config_path=model_config_path,
    )

    class _BatchedReranker:
        def score_pairs(self, pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]]):
            return reranker.score_pairs(pairs, batch_size=reranker_batch_size)

    pairs = generate_identity_pairs(
        database_url,
        UUID(cohort.cohort_id),
        embedding_candidates=neighbors,
        reranker_scorer=_BatchedReranker(),
        embedding_run_id=UUID(embedded.run_id),
        reranker_run_id=UUID(reranker_run.run_id),
        maximum_pairs_per_profile=maximum_pairs_per_profile,
    )
    complete_identity_reranker_run(
        database_url,
        UUID(reranker_run.run_id),
        candidate_pairs=pairs.pairs,
    )
    return {
        "schema_version": "1.0",
        "cohort": asdict(cohort),
        "embedding": asdict(embedded),
        "reranker_run": asdict(reranker_run),
        "pairs": asdict(pairs),
        "identity_models": {
            "embedding": configuration.identity.embedding.model_dump(mode="json"),
            "reranker": configuration.identity.reranker.model_dump(mode="json"),
        },
        "pipeline_model_configuration": configuration.source_path.as_posix(),
        "pipeline_model_configuration_sha256": configuration.sha256,
        "publication_state": "candidate_only",
        "next_required_step": "Run bounded Qwen pair decisions, then historian review.",
    }


def _write_semantic_result(path: Path, result: SemanticTaskResult[Any]) -> None:
    payload = {
        "schema_version": "1.0",
        "task": result.task,
        "prompt_sha256": result.prompt_sha256,
        "prompt_schema_sha256": result.prompt_schema_sha256,
        "response_format_sha256": result.response_format_sha256,
        "raw_output_sha256": result.raw_output_sha256,
        "finish_reason": result.finish_reason,
        "response": result.response.model_dump(mode="json"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_identity_decisions(
    database_url: str,
    cohort_id: UUID,
    output_dir: Path,
    *,
    model_config_path: str | None = None,
) -> dict[str, Any]:
    """Ask the configured Qwen model for bounded, non-promoting pair decisions."""
    configuration = load_pipeline_model_configuration(model_config_path)
    client = build_verified_semantic_client(model_config_path)
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        candidate_ids = [
            row["identity_pair_candidate_id"]
            for row in connection.execute(
                """
                SELECT pair.identity_pair_candidate_id
                FROM evidence.identity_pair_candidate pair
                LEFT JOIN evidence.identity_pair_decision decision
                  ON decision.identity_pair_candidate_id = pair.identity_pair_candidate_id
                WHERE pair.cohort_id = %s
                  AND decision.identity_pair_decision_id IS NULL
                ORDER BY pair.candidate_rank, pair.identity_pair_candidate_id
                """,
                (cohort_id,),
            )
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions: list[dict[str, Any]] = []
    abstentions: list[dict[str, str]] = []
    for pair_id in candidate_ids:
        left, right, evidence_ids = load_identity_pair(database_url, pair_id)
        try:
            result = client.identity_pair(
                left_profile=left,
                right_profile=right,
                evidence_ids=evidence_ids,
            )
        except SemanticAbstention as exc:
            abstentions.append(
                {"identity_pair_candidate_id": str(pair_id), "reason": str(exc)}
            )
            continue
        artifact = output_dir / f"{pair_id}.json"
        _write_semantic_result(artifact, result)
        stored = persist_identity_pair_decision(
            database_url,
            pair_id,
            result,
            model_config_path=model_config_path,
        )
        decisions.append(asdict(stored))
    with psycopg.connect(database_url) as connection:
        connection.execute(
            """
            UPDATE evidence.identity_resolution_cohort
            SET status = 'reviewing'
            WHERE cohort_id = %s AND status IN ('frozen', 'deciding')
            """,
            (cohort_id,),
        )
    receipt = {
        "schema_version": "1.0",
        "cohort_id": str(cohort_id),
        "pipeline_model_configuration": configuration.source_path.as_posix(),
        "pipeline_model_configuration_sha256": configuration.sha256,
        "semantic_model": configuration.semantic.model_dump(mode="json"),
        "decisions": decisions,
        "abstentions": abstentions,
        "publication_state": "awaiting_historian_review",
    }
    receipt_path = output_dir / "receipt.json"
    temporary = receipt_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--model-config",
        help="Complete model configuration; individual model overrides are forbidden",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze and rank a new identity cohort")
    prepare.add_argument("--created-by", required=True)
    prepare.add_argument("--entity-type")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--embedding-batch-size", type=int, default=16)
    prepare.add_argument("--reranker-batch-size", type=int, default=8)
    prepare.add_argument("--neighbors-per-profile", type=int, default=40)
    prepare.add_argument("--maximum-pairs-per-profile", type=int, default=20)
    decide = commands.add_parser("decide", help="Run bounded Qwen pair decisions")
    decide.add_argument("--cohort-id", type=UUID, required=True)
    decide.add_argument("--output-dir", type=Path, required=True)
    review = commands.add_parser("review", help="Review and optionally promote one pair")
    review.add_argument("--identity-pair-decision-id", type=UUID, required=True)
    review.add_argument(
        "--action", choices=("accept", "reject", "needs_review"), required=True
    )
    review.add_argument("--reviewer", required=True)
    review.add_argument("--note")
    review.add_argument("--canonical-entity-id", type=UUID)
    review.add_argument("--new-canonical-name")
    review.add_argument("--review-id", type=UUID)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("--database-url or DATABASE_URL is required")
    if args.command == "prepare":
        receipt = prepare_identity_batch(
            args.database_url,
            created_by=args.created_by,
            entity_type=args.entity_type,
            model_config_path=args.model_config,
            embedding_batch_size=args.embedding_batch_size,
            reranker_batch_size=args.reranker_batch_size,
            neighbors_per_profile=args.neighbors_per_profile,
            maximum_pairs_per_profile=args.maximum_pairs_per_profile,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    elif args.command == "decide":
        receipt = run_identity_decisions(
            args.database_url,
            args.cohort_id,
            args.output_dir,
            model_config_path=args.model_config,
        )
    else:
        receipt = review_identity_pair_decision(
            args.database_url,
            args.identity_pair_decision_id,
            action=args.action,
            reviewer=args.reviewer,
            note=args.note,
            canonical_entity_id=args.canonical_entity_id,
            new_canonical_name=args.new_canonical_name,
            review_id=args.review_id,
        )
        receipt = asdict(receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
