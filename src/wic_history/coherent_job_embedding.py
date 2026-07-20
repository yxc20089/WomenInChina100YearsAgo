from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from uuid import UUID

from .article_embedding import (
    ArticleEmbeddingRequest,
    ReviewedArticleUnavailableError,
    embed_reviewed_articles,
)
from .coherent_jobs import (
    COHERENT_CONFIGURATION,
    ActiveRevision,
    CoherentExecution,
    CoherentJobContext,
    CoherentJobError,
    CoherentPlanResult,
    active_revisions,
    enqueue_coherent_jobs,
    load_active_documents,
)
from .coherent_job_database import database_clients
from .coherent_job_hashing import coherent_sha256


def _active_revisions_from_database(database_url: str) -> tuple[ActiveRevision, ...]:
    psycopg, dict_row = database_clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        return active_revisions(load_active_documents(connection))


def _schedule_current_reconcile(
    database_url: str, worker_id: str
) -> CoherentPlanResult:
    psycopg, dict_row = database_clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        return enqueue_coherent_jobs(
            connection, created_by=f"stale-job:{worker_id}", max_revisions=100_000
        )


def _revision_superseded(database_url: str, revision_id: UUID) -> bool:
    psycopg, dict_row = database_clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        row = connection.execute(
            "SELECT superseded_at IS NOT NULL AS superseded FROM evidence.coherent_unit_revision WHERE revision_id = %s",
            (revision_id,),
        ).fetchone()
    return bool(row and row["superseded"])


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
    fingerprint = coherent_sha256(
        {"revision": asdict(planned), "configuration": COHERENT_CONFIGURATION}
    )
    if (
        any(
            context.configuration.get(key) != value
            for key, value in COHERENT_CONFIGURATION.items()
        )
        or context.configuration.get("planned_revision_id") != str(revision_id)
        or context.configuration.get("planned_embedding_fingerprint") != fingerprint
        or context.input_fingerprint != fingerprint
    ):
        raise CoherentJobError("Coherent embedding plan fingerprint is invalid")
    try:
        engine_version = version("sentence-transformers")
    except PackageNotFoundError as exc:
        raise CoherentJobError("Planned embedding engine is unavailable") from exc
    if engine_version != context.configuration["engine_version"]:
        raise CoherentJobError("Worker embedding engine version differs from its plan")
    current = next(
        (
            item
            for item in _active_revisions_from_database(database_url)
            if item.revision_id == revision_id
        ),
        None,
    )
    if current != planned:
        reconciliation = _schedule_current_reconcile(database_url, context.job_id)
        result = {
            "revision_id": str(revision_id),
            "planned_input_sha256": planned.input_sha256,
            "planned_content_sha256": planned.content_sha256,
            "input_fingerprint": fingerprint,
            "embedding_configuration_sha256": context.configuration[
                "embedding_configuration_sha256"
            ],
            "embeddings_inserted": 0,
            "embeddings_reused": 0,
            "stale_noop": True,
            "active": False,
            "reconciliation_plan_key": reconciliation.plan_key,
            "reconciliation_scheduled": True,
        }
        return CoherentExecution(
            f"coherent-unit://embedding/{revision_id}",
            coherent_sha256(result),
            result,
            False,
        )
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
            "input_fingerprint": fingerprint,
            "embedding_configuration_sha256": context.configuration[
                "embedding_configuration_sha256"
            ],
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
            "input_fingerprint": fingerprint,
            "embedding_configuration_sha256": context.configuration[
                "embedding_configuration_sha256"
            ],
            "embeddings_inserted": summary.embeddings_inserted,
            "embeddings_reused": summary.embeddings_reused,
            "stale_noop": False,
            "active": True,
        }
    return CoherentExecution(
        f"coherent-unit://embedding/{revision_id}",
        coherent_sha256(result),
        result,
        bool(result["embeddings_reused"]),
    )
