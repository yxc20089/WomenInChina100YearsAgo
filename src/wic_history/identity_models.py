"""Pinned embedding and reranking adapters for frozen identity cohorts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from .model_config import PipelineModelConfiguration, load_pipeline_model_configuration


IDENTITY_RETRIEVAL_INSTRUCTION = (
    "Retrieve another historical identity profile that may refer to the same "
    "real-world entity. Treat names, titles, dates, places, relationships, and "
    "source context as evidence; name similarity alone is not identity."
)
IDENTITY_RERANK_INSTRUCTION = (
    "Rank whether these two historical identity profiles deserve close human "
    "comparison for possible coreference. This is candidate relevance, not a "
    "same-person decision."
)


def _clients() -> tuple[Any, Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    return psycopg, dict_row, Jsonb


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"


def identity_profile_text(profile: dict[str, Any]) -> str:
    """Serialize only frozen profile evidence into stable model input text."""
    names = [str(value) for value in profile.get("name_surfaces") or []]
    attributes = profile.get("attributes") or {}
    evidence = [str(value) for value in profile.get("evidence_span_ids") or []]
    payload = {
        "entity_type": profile.get("entity_type"),
        "name_surfaces": names,
        "context": attributes.get("context"),
        "attributes": {
            key: value
            for key, value in sorted(attributes.items())
            if key != "context"
        },
        "evidence_span_ids": evidence,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class QwenIdentityEmbedder:
    """Official Sentence Transformers path, pinned by the central config."""

    def __init__(self, configuration: PipelineModelConfiguration):
        selected = configuration.identity.embedding
        if selected.engine not in {"sentence-transformers", "transformers"}:
            raise ValueError("identity embedding engine must use Sentence Transformers")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - minimal installations
            raise RuntimeError("Install the NER extra: uv sync --extra ner") from exc
        self.configuration = configuration
        self.model_name = selected.model_name
        self.model_revision = selected.model_revision
        self.dimension = selected.dimension
        self.normalize = selected.normalize
        self.model = SentenceTransformer(
            selected.model_name,
            revision=selected.model_revision,
        )

    def encode_documents(
        self, profiles: Sequence[dict[str, Any]], *, batch_size: int = 16
    ) -> list[list[float]]:
        values = self.model.encode(
            [identity_profile_text(profile) for profile in profiles],
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(profiles) > batch_size,
        )
        vectors = values.tolist() if hasattr(values, "tolist") else list(values)
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError(
                f"Expected {self.dimension}-dimensional identity embeddings"
            )
        return [[float(value) for value in vector] for vector in vectors]


class QwenIdentityReranker:
    """Official CrossEncoder path; scores candidate relevance, never identity."""

    def __init__(self, configuration: PipelineModelConfiguration):
        selected = configuration.identity.reranker
        if selected.engine not in {"sentence-transformers", "transformers"}:
            raise ValueError("identity reranker engine must use Sentence Transformers")
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - minimal installations
            raise RuntimeError("Install the NER extra: uv sync --extra ner") from exc
        self.configuration = configuration
        self.model_name = selected.model_name
        self.model_revision = selected.model_revision
        self._activation = torch.nn.Sigmoid()
        self.model = CrossEncoder(
            selected.model_name,
            revision=selected.model_revision,
            prompts={"identity": IDENTITY_RERANK_INSTRUCTION},
            default_prompt_name="identity",
        )

    def score_pairs(
        self,
        pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
        *,
        batch_size: int = 8,
    ) -> list[float]:
        if not pairs:
            return []
        model_pairs = [
            (identity_profile_text(left), identity_profile_text(right))
            for left, right in pairs
        ]
        values = self.model.predict(
            model_pairs,
            batch_size=batch_size,
            activation_fn=self._activation,
            show_progress_bar=len(pairs) > batch_size,
        )
        scores = values.tolist() if hasattr(values, "tolist") else list(values)
        result = [float(value) for value in scores]
        if any(value < 0 or value > 1 for value in result):
            raise ValueError("identity reranker scores must be in [0, 1]")
        return result

    def __call__(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        return self.score_pairs([(left, right)], batch_size=1)[0]


@dataclass(frozen=True, slots=True)
class IdentityEmbeddingResult:
    cohort_id: str
    run_id: str
    profiles: int
    inserted: int
    model_name: str
    model_revision: str
    reused: bool


@dataclass(frozen=True, slots=True)
class IdentityRerankerRunResult:
    cohort_id: str
    run_id: str
    model_name: str
    model_revision: str
    reused: bool


def start_identity_reranker_run(
    database_url: str,
    cohort_id: UUID,
    *,
    model_config_path: str | None = None,
) -> IdentityRerankerRunResult:
    """Open or reuse the pinned reranker run for one immutable cohort."""
    configuration = load_pipeline_model_configuration(model_config_path)
    selected = configuration.identity.reranker
    run_id = uuid5(
        NAMESPACE_URL,
        (
            f"wic-identity-reranker:{cohort_id}:{configuration.sha256}:"
            f"{selected.model_name}:{selected.model_revision}"
        ),
    )
    psycopg, dict_row, Jsonb = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        cohort = connection.execute(
            """
            SELECT configuration_sha256 FROM evidence.identity_resolution_cohort
            WHERE cohort_id = %s
            """,
            (cohort_id,),
        ).fetchone()
        if cohort is None or cohort["configuration_sha256"] != configuration.sha256:
            raise ValueError("identity cohort is absent or uses a different model config")
        existing = connection.execute(
            "SELECT status FROM evidence.processing_run WHERE run_id = %s",
            (run_id,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO evidence.processing_run (
                    run_id, kind, engine, model_name, model_revision,
                    software_version, configuration, status, started_at
                ) VALUES (
                    %s, 'entity_link', %s, %s, %s,
                    'sentence-transformers-5.x', %s, 'running', %s
                )
                """,
                (
                    run_id,
                    selected.engine,
                    selected.model_name,
                    selected.model_revision,
                    Jsonb(
                        {
                            "task": "identity_pair_candidate_reranking",
                            "cohort_id": str(cohort_id),
                            "pipeline_model_configuration_sha256": configuration.sha256,
                            "instruction": IDENTITY_RERANK_INSTRUCTION,
                            "score_semantics": "candidate_relevance_not_identity_probability",
                        }
                    ),
                    datetime.now(timezone.utc),
                ),
            )
        elif existing["status"] not in {"running", "completed"}:
            raise ValueError("identity reranker run is not reusable")
    return IdentityRerankerRunResult(
        str(cohort_id),
        str(run_id),
        selected.model_name,
        selected.model_revision,
        existing is not None and existing["status"] == "completed",
    )


def complete_identity_reranker_run(
    database_url: str,
    run_id: UUID,
    *,
    candidate_pairs: int,
) -> None:
    if candidate_pairs < 0:
        raise ValueError("candidate_pairs cannot be negative")
    psycopg, dict_row, Jsonb = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        row = connection.execute(
            "SELECT status, configuration FROM evidence.processing_run WHERE run_id = %s FOR UPDATE",
            (run_id,),
        ).fetchone()
        if row is None or row["status"] not in {"running", "completed"}:
            raise ValueError("unknown or unusable identity reranker run")
        prior_count = (row["configuration"] or {}).get("candidate_pairs")
        if row["status"] == "completed" and prior_count != candidate_pairs:
            raise ValueError("completed identity reranker run count has drifted")
        connection.execute(
            """
            UPDATE evidence.processing_run
            SET configuration = configuration || %s,
                status = 'completed', completed_at = COALESCE(completed_at, now())
            WHERE run_id = %s
            """,
            (Jsonb({"candidate_pairs": candidate_pairs}), run_id),
        )


def persist_identity_profile_embeddings(
    database_url: str,
    cohort_id: UUID,
    *,
    model_config_path: str | None = None,
    batch_size: int = 16,
    embedder: Any | None = None,
) -> IdentityEmbeddingResult:
    """Embed a frozen cohort once and persist its vectors in authoritative pgvector."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    configuration = load_pipeline_model_configuration(model_config_path)
    selected = configuration.identity.embedding
    psycopg, dict_row, Jsonb = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        cohort = connection.execute(
            """
            SELECT cohort_id, snapshot_sha256, configuration_sha256
            FROM evidence.identity_resolution_cohort WHERE cohort_id = %s
            """,
            (cohort_id,),
        ).fetchone()
        if cohort is None:
            raise ValueError("unknown identity cohort")
        if cohort["configuration_sha256"] != configuration.sha256:
            raise ValueError("identity cohort was frozen under a different model config")
        profiles = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM evidence.identity_profile
                WHERE cohort_id = %s ORDER BY identity_profile_id
                """,
                (cohort_id,),
            )
        ]
        if not profiles:
            raise ValueError("identity cohort is empty")
        run_id = uuid5(
            NAMESPACE_URL,
            (
                f"wic-identity-embedding:{cohort_id}:{cohort['snapshot_sha256']}:"
                f"{configuration.sha256}:{selected.model_name}:{selected.model_revision}"
            ),
        )
        existing = connection.execute(
            """
            SELECT run_id, status FROM evidence.processing_run WHERE run_id = %s
            """,
            (run_id,),
        ).fetchone()
        if existing is not None:
            count = connection.execute(
                """
                SELECT count(*) AS count FROM retrieval.embedding
                WHERE run_id = %s AND target_kind = 'identity_profile'
                """,
                (run_id,),
            ).fetchone()["count"]
            if existing["status"] != "completed" or count != len(profiles):
                raise ValueError("identity embedding run is partial; repair before retry")
            return IdentityEmbeddingResult(
                str(cohort_id), str(run_id), len(profiles), 0,
                selected.model_name, selected.model_revision, True,
            )

    adapter = embedder or QwenIdentityEmbedder(configuration)
    vectors = adapter.encode_documents(profiles, batch_size=batch_size)
    if len(vectors) != len(profiles):
        raise ValueError("identity embedder did not return one vector per profile")
    now = datetime.now(timezone.utc)
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        connection.execute(
            """
            INSERT INTO evidence.processing_run (
                run_id, kind, engine, model_name, model_revision,
                software_version, configuration, status, started_at, completed_at
            ) VALUES (
                %s, 'embedding', %s, %s, %s, 'sentence-transformers-5.x',
                %s, 'completed', %s, %s
            )
            """,
            (
                run_id,
                selected.engine,
                selected.model_name,
                selected.model_revision,
                Jsonb(
                    {
                        "task": "identity_profile_candidate_retrieval",
                        "cohort_id": str(cohort_id),
                        "pipeline_model_configuration_sha256": configuration.sha256,
                        "dimension": selected.dimension,
                        "normalized": selected.normalize,
                        "instruction": IDENTITY_RETRIEVAL_INSTRUCTION,
                        "batch_size": batch_size,
                    }
                ),
                now,
                now,
            ),
        )
        inserted = 0
        for profile, vector in zip(profiles, vectors, strict=True):
            cursor = connection.execute(
                """
                INSERT INTO retrieval.embedding (
                    target_kind, target_id, run_id, model_name,
                    model_revision, embedding
                ) VALUES ('identity_profile', %s, %s, %s, %s, %s::vector)
                ON CONFLICT (
                    target_kind, target_id, model_name, model_revision
                ) WHERE input_sha256 IS NULL
                    AND content_sha256 IS NULL
                    AND configuration_sha256 IS NULL
                DO NOTHING
                """,
                (
                    profile["identity_profile_id"],
                    run_id,
                    selected.model_name,
                    selected.model_revision,
                    _vector_literal(vector),
                ),
            )
            inserted += cursor.rowcount
    return IdentityEmbeddingResult(
        str(cohort_id), str(run_id), len(profiles), inserted,
        selected.model_name, selected.model_revision, False,
    )


def load_identity_embedding_neighbors(
    database_url: str,
    cohort_id: UUID,
    *,
    model_config_path: str | None = None,
    neighbors_per_profile: int = 40,
) -> dict[tuple[UUID, UUID], float]:
    """Return the union of directed top-k neighbors as undirected pair scores."""
    if neighbors_per_profile < 1:
        raise ValueError("neighbors_per_profile must be positive")
    configuration = load_pipeline_model_configuration(model_config_path)
    selected = configuration.identity.embedding
    psycopg, dict_row, _ = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            """
            SELECT source.identity_profile_id AS source_id,
                   neighbor.identity_profile_id AS neighbor_id,
                   1 - (source_embedding.embedding <=> neighbor.embedding)
                       AS cosine_similarity
            FROM evidence.identity_profile source
            JOIN retrieval.embedding source_embedding
              ON source_embedding.target_kind = 'identity_profile'
             AND source_embedding.target_id = source.identity_profile_id
             AND source_embedding.model_name = %s
             AND source_embedding.model_revision = %s
            CROSS JOIN LATERAL (
                SELECT candidate.identity_profile_id, candidate_embedding.embedding
                FROM evidence.identity_profile candidate
                JOIN retrieval.embedding candidate_embedding
                  ON candidate_embedding.target_kind = 'identity_profile'
                 AND candidate_embedding.target_id = candidate.identity_profile_id
                 AND candidate_embedding.model_name = %s
                 AND candidate_embedding.model_revision = %s
                WHERE candidate.cohort_id = source.cohort_id
                  AND candidate.entity_type = source.entity_type
                  AND candidate.identity_profile_id <> source.identity_profile_id
                ORDER BY source_embedding.embedding <=> candidate_embedding.embedding,
                         candidate.identity_profile_id
                LIMIT %s
            ) neighbor
            WHERE source.cohort_id = %s
            ORDER BY source.identity_profile_id, cosine_similarity DESC,
                     neighbor.identity_profile_id
            """,
            (
                selected.model_name,
                selected.model_revision,
                selected.model_name,
                selected.model_revision,
                neighbors_per_profile,
                cohort_id,
            ),
        ).fetchall()
    result: dict[tuple[UUID, UUID], float] = {}
    for row in rows:
        pair = tuple(sorted((row["source_id"], row["neighbor_id"])))
        score = max(-1.0, min(1.0, float(row["cosine_similarity"])))
        result[pair] = max(result.get(pair, -1.0), score)
    return result


def build_identity_model_adapters(
    model_config_path: str | None = None,
) -> tuple[PipelineModelConfiguration, QwenIdentityEmbedder, QwenIdentityReranker]:
    """Build both candidate models from the single authoritative config."""
    configuration = load_pipeline_model_configuration(model_config_path)
    return (
        configuration,
        QwenIdentityEmbedder(configuration),
        QwenIdentityReranker(configuration),
    )
