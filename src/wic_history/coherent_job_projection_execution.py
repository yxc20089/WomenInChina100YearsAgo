from __future__ import annotations

from uuid import UUID

from .coherent_job_projection import build_coherent_manifest
from .coherent_jobs import (
    COHERENT_CONFIGURATION,
    CoherentExecution,
    CoherentJobContext,
    CoherentJobError,
    active_revisions,
    coherent_plan_key,
    load_active_documents,
)
from .coherent_job_database import database_clients
from .coherent_job_hashing import coherent_sha256
from .coherent_search import project_coherent_units, restore_coherent_alias
from .coherent_search_contracts import JsonValue


def execute_coherent_projection(
    database_url: str, context: CoherentJobContext, *, opensearch_url: str
) -> CoherentExecution:
    psycopg, dict_row = database_clients()
    from psycopg.types.json import Jsonb

    projected = None
    revision_count = 0
    try:
        with psycopg.connect(database_url, row_factory=dict_row) as connection:
            if any(
                context.configuration.get(key) != value
                for key, value in COHERENT_CONFIGURATION.items()
            ):
                raise CoherentJobError(
                    "Coherent projection configuration is not pinned"
                )
            if (
                context.configuration.get("planned_snapshot_sha256")
                != context.input_fingerprint
            ):
                raise CoherentJobError("Coherent projection snapshot plan is invalid")
            _ = connection.execute(
                "SELECT pg_advisory_xact_lock_shared(hashtextextended(%s, 0))",
                ("wic-coherent-active-snapshot-v1",),
            )
            documents = load_active_documents(connection)
            revisions = active_revisions(documents)
            revision_count = len(revisions)
            if (
                not revisions
                or coherent_plan_key(revisions) != context.input_fingerprint
                or context.configuration.get("planned_revision_count") != revision_count
            ):
                raise CoherentJobError(
                    "Coherent projection active snapshot is empty or stale"
                )
            manifest = build_coherent_manifest(connection, documents)
            if (
                coherent_plan_key(active_revisions(load_active_documents(connection)))
                != context.input_fingerprint
            ):
                raise CoherentJobError("Coherent projection active snapshot changed")
            projected = project_coherent_units(opensearch_url, manifest)
            if (
                projected.documents_indexed != revision_count
                or projected.source_snapshot_sha256 != manifest.snapshot_sha256
            ):
                raise CoherentJobError(
                    "Coherent projection core returned a misleading receipt"
                )
            _ = connection.execute(
                """INSERT INTO retrieval.projection_build (
                       build_id, projection_kind, source_schema_version, configuration,
                       status, completed_at, artifact_uri, source_snapshot_sha256,
                       document_count, published_at
                   ) VALUES (%s, 'opensearch_coherent_unit', '1.0', %s, 'completed',
                             now(), %s, %s, %s, now())""",
                (
                    UUID(projected.build_id),
                    Jsonb(dict(COHERENT_CONFIGURATION)),
                    f"opensearch://{projected.index_name}",
                    context.input_fingerprint,
                    projected.documents_indexed,
                ),
            )
    except (CoherentJobError, psycopg.Error):
        if projected is not None:
            restore_coherent_alias(opensearch_url, projected)
        raise
    result: dict[str, JsonValue] = {
        "projection_build_id": projected.build_id,
        "index_name": projected.index_name,
        "documents_indexed": projected.documents_indexed,
        "source_snapshot_sha256": context.input_fingerprint,
        "projection_manifest_sha256": projected.source_snapshot_sha256,
        "planned_snapshot_sha256": context.input_fingerprint,
        "planned_revision_count": revision_count,
        "published": True,
    }
    return CoherentExecution(
        f"opensearch://{projected.index_name}", coherent_sha256(result), result, False
    )
