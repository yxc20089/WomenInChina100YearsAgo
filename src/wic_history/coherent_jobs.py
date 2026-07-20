from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final
from uuid import UUID, uuid4

from .article_embedding import ArticleEmbeddingRequest, ReviewedArticleUnavailableError, embed_reviewed_articles
from .article_embedding_vectors import DIMENSION, POLICY
from .coherent_search import COHERENT_ALIAS, COHERENT_INDEX_PREFIX, coherent_index_body
from .model_config import load_pipeline_model_configuration
from .rag_experiment import REVIEWED_UNIT_EXPORT_SQL, build_coherent_unit_documents


_RETRIEVAL_MODEL: Final = load_pipeline_model_configuration().retrieval.passage_embedding
_MAPPING_SHA256: Final = hashlib.sha256(
    json.dumps(coherent_index_body(), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
COHERENT_CONFIGURATION: Final[dict[str, str | int]] = {
    "engine": "sentence-transformers",
    "model": _RETRIEVAL_MODEL.model_name,
    "revision": _RETRIEVAL_MODEL.model_revision,
    "dimension": DIMENSION,
    "normalize_embeddings": "true",
    "window_policy": POLICY,
    "tokenizer_limit": 8190,
    "model_limit": 8190,
    "effective_limit": 8190,
    "overlap_tokens": 1023,
    "embedding_configuration_sha256": "3ca671bf5e5d01b8e9016c28c9e39cbc0c194099c7fd109506896647967b8903",
    "alias": COHERENT_ALIAS,
    "index_prefix": COHERENT_INDEX_PREFIX,
    "projection_kind": "opensearch_coherent_unit",
    "mapping_sha256": _MAPPING_SHA256,
}
COHERENT_STAGES: Final = ("coherent_unit_embedding", "coherent_unit_search_projection")
_SNAPSHOT_LOCK: Final = "wic-coherent-active-snapshot-v1"


@dataclass(frozen=True, slots=True)
class ActiveRevision:
    revision_id: UUID
    input_sha256: str
    content_sha256: str = ""


class CoherentJobError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class CoherentPlanResult:
    batch_id: str | None
    plan_key: str | None
    revisions: int
    jobs: int
    created: bool


@dataclass(frozen=True, slots=True)
class CoherentJobContext:
    job_id: str
    batch_id: str
    stage: str
    input_fingerprint: str
    configuration: Mapping[str, Any]
    revision_id: UUID | None


@dataclass(frozen=True, slots=True)
class CoherentExecution:
    artifact_uri: str
    output_sha256: str
    result: dict[str, Any]
    adopted: bool


def coherent_plan_key(revisions: tuple[ActiveRevision, ...]) -> str:
    payload = {
        "contract": "wic-coherent-article-jobs-v1",
        "active_revisions": [asdict(revision) for revision in sorted(revisions, key=lambda item: str(item.revision_id))],
        "configuration": COHERENT_CONFIGURATION,
    }
    return hashlib.sha256(
        json.dumps(payload, default=str, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def lock_coherent_mutation(connection: Any) -> None:
    connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (_SNAPSHOT_LOCK,))


def _psycopg() -> tuple[Any, Any]:
    import psycopg
    from psycopg.rows import dict_row

    return psycopg, dict_row


def _active_documents(connection: Any) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    rows = connection.execute(
        REVIEWED_UNIT_EXPORT_SQL, {"volume_number": None, "page_number": None}
    ).fetchall()
    return build_coherent_unit_documents(rows)


def _active_revisions(
    documents: Sequence[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[ActiveRevision, ...]:
    return tuple(
        ActiveRevision(
            UUID(document["id"]),
            document["metadata"]["input_sha256"],
            document["metadata"]["content_sha256"],
        )
        for document, _ in documents
    )


def enqueue_coherent_jobs(
    connection: Any, *, created_by: str, max_revisions: int = 1000
) -> CoherentPlanResult:
    if not created_by.strip() or max_revisions < 1:
        raise CoherentJobError("created_by and a positive max_revisions are required")
    lock_coherent_mutation(connection)
    documents = _active_documents(connection)
    revisions = _active_revisions(documents)
    if len(revisions) > max_revisions:
        raise CoherentJobError(f"Coherent backfill found {len(revisions)} revisions above the {max_revisions} revision guard")
    if not revisions:
        return CoherentPlanResult(None, None, 0, 0, False)
    plan_key = coherent_plan_key(revisions)
    batch_id = uuid4()
    inserted = connection.execute(
        """INSERT INTO pipeline.ingestion_batch (
               batch_id, plan_key, name, scope, configuration, created_by
           ) VALUES (%s, %s, 'active reviewed article search', %s::jsonb, %s::jsonb, %s)
           ON CONFLICT (plan_key) DO NOTHING RETURNING batch_id""",
        (
            batch_id,
            plan_key,
            json.dumps({"active_revisions": [asdict(item) for item in revisions]}, default=str),
            json.dumps(COHERENT_CONFIGURATION),
            created_by.strip(),
        ),
    ).fetchone()
    if inserted is None:
        existing = connection.execute(
            "SELECT batch_id FROM pipeline.ingestion_batch WHERE plan_key = %s",
            (plan_key,),
        ).fetchone()
        if existing is None:
            raise CoherentJobError("Coherent plan conflict could not be reloaded")
        return CoherentPlanResult(str(existing["batch_id"]), plan_key, len(revisions), len(revisions) + 1, False)
    parent_ids: list[UUID] = []
    for revision in revisions:
        job_id = uuid4()
        parent_ids.append(job_id)
        fingerprint = _sha256({"revision": asdict(revision), "configuration": COHERENT_CONFIGURATION})
        job_configuration = {
            **COHERENT_CONFIGURATION,
            "planned_revision_id": str(revision.revision_id),
            "planned_input_sha256": revision.input_sha256,
            "planned_content_sha256": revision.content_sha256,
            "planned_embedding_fingerprint": fingerprint,
        }
        connection.execute(
            """INSERT INTO pipeline.ingestion_job (
                   job_id, batch_id, job_key, stage, scope_kind,
                   coherent_unit_revision_id, input_fingerprint, configuration
               ) VALUES (%s, %s, %s, 'coherent_unit_embedding',
                         'coherent_unit_revision', %s, %s, %s::jsonb)""",
            (job_id, batch_id, fingerprint, revision.revision_id, fingerprint, json.dumps(job_configuration)),
        )
    projection_id = uuid4()
    connection.execute(
        """INSERT INTO pipeline.ingestion_job (
               job_id, batch_id, job_key, stage, scope_kind,
               input_fingerprint, configuration
           ) VALUES (%s, %s, %s, 'coherent_unit_search_projection',
                     'batch', %s, %s::jsonb)""",
        (
            projection_id,
            batch_id,
            _sha256({"projection": plan_key}),
            plan_key,
            json.dumps(
                {
                    **COHERENT_CONFIGURATION,
                    "planned_snapshot_sha256": plan_key,
                    "planned_revision_count": len(revisions),
                }
            ),
        ),
    )
    with connection.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO pipeline.ingestion_job_dependency (job_id, depends_on_job_id) VALUES (%s, %s)",
            [(projection_id, parent_id) for parent_id in parent_ids],
        )
        cursor.executemany(
            "INSERT INTO pipeline.ingestion_job_event (job_id, event_type, worker_id) VALUES (%s, 'planned', %s)",
            [(job_id, created_by.strip()) for job_id in (*parent_ids, projection_id)],
        )
    return CoherentPlanResult(str(batch_id), plan_key, len(revisions), len(revisions) + 1, True)


def backfill_coherent_jobs(
    database_url: str, *, created_by: str, max_revisions: int
) -> CoherentPlanResult:
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        return enqueue_coherent_jobs(
            connection, created_by=created_by, max_revisions=max_revisions
        )


def load_coherent_job_context(database_url: str, job_id: UUID | str) -> CoherentJobContext:
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        row = connection.execute(
            """SELECT job_id, batch_id, stage, input_fingerprint,
                      configuration, coherent_unit_revision_id
               FROM pipeline.ingestion_job WHERE job_id = %s
                 AND scope_kind IN ('batch', 'coherent_unit_revision')""",
            (job_id,),
        ).fetchone()
    if row is None:
        raise CoherentJobError("Coherent ingestion job does not exist")
    return CoherentJobContext(str(row["job_id"]), str(row["batch_id"]), row["stage"], row["input_fingerprint"], row["configuration"], row["coherent_unit_revision_id"])


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, default=str, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _revision_superseded(database_url: str, revision_id: UUID) -> bool:
    import psycopg

    with psycopg.connect(database_url) as connection:
        row = connection.execute(
            """SELECT superseded_at IS NOT NULL
               FROM evidence.coherent_unit_revision WHERE revision_id = %s""",
            (revision_id,),
        ).fetchone()
    return bool(row and row[0])


def execute_coherent_embedding(
    database_url: str, context: CoherentJobContext
) -> CoherentExecution:
    revision_id = context.revision_id
    if revision_id is None:
        raise CoherentJobError("Coherent embedding job lacks its exact revision")
    planned = ActiveRevision(
        revision_id,
        str(context.configuration.get("planned_input_sha256", "")),
        str(context.configuration.get("planned_content_sha256", "")),
    )
    expected_fingerprint = _sha256(
        {"revision": asdict(planned), "configuration": COHERENT_CONFIGURATION}
    )
    if (
        any(context.configuration.get(key) != value for key, value in COHERENT_CONFIGURATION.items())
        or context.configuration.get("planned_revision_id") != str(revision_id)
        or context.configuration.get("planned_embedding_fingerprint") != expected_fingerprint
        or context.input_fingerprint != expected_fingerprint
    ):
        raise CoherentJobError("Coherent embedding plan fingerprint is invalid")
    current = next(
        (
            item
            for item in _active_revisions_from_database(database_url)
            if item.revision_id == revision_id
        ),
        None,
    )
    if current != planned:
        result = {
            "revision_id": str(revision_id),
            "planned_input_sha256": planned.input_sha256,
            "planned_content_sha256": planned.content_sha256,
            "input_fingerprint": expected_fingerprint,
            "embeddings_inserted": 0,
            "embeddings_reused": 0,
            "stale_noop": True,
            "active": False,
        }
        reconciliation = _schedule_current_reconcile(database_url, context.job_id)
        result["reconciliation_plan_key"] = reconciliation.plan_key
        result["reconciliation_scheduled"] = True
        return CoherentExecution(f"coherent-unit://embedding/{revision_id}", _sha256(result), result, False)
    request = ArticleEmbeddingRequest(
        database_url,
        str(context.configuration["model"]),
        str(context.configuration["revision"]),
        16,
        revision_id,
        str(context.configuration["embedding_configuration_sha256"]),
    )
    try:
        summary = embed_reviewed_articles(request)
    except ReviewedArticleUnavailableError:
        if not _revision_superseded(database_url, revision_id):
            raise
        reconciliation = _schedule_current_reconcile(database_url, context.job_id)
        result = {
            "revision_id": str(revision_id),
            "planned_input_sha256": planned.input_sha256,
            "planned_content_sha256": planned.content_sha256,
            "input_fingerprint": expected_fingerprint,
            "embeddings_inserted": 0,
            "embeddings_reused": 0,
            "stale_noop": True,
            "active": False,
            "reconciliation_plan_key": reconciliation.plan_key,
            "reconciliation_scheduled": True,
        }
    else:
        result = {
            "revision_id": str(revision_id),
            "planned_input_sha256": planned.input_sha256,
            "planned_content_sha256": planned.content_sha256,
            "input_fingerprint": expected_fingerprint,
            "embeddings_inserted": summary.embeddings_inserted,
            "embeddings_reused": summary.embeddings_reused,
            "stale_noop": False,
            "active": True,
        }
    return CoherentExecution(f"coherent-unit://embedding/{revision_id}", _sha256(result), result, bool(result["embeddings_reused"]))


def _active_revisions_from_database(database_url: str) -> tuple[ActiveRevision, ...]:
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        return _active_revisions(_active_documents(connection))


def _schedule_current_reconcile(database_url: str, worker_id: str) -> CoherentPlanResult:
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        return enqueue_coherent_jobs(
            connection, created_by=f"stale-job:{worker_id}", max_revisions=100_000
        )
