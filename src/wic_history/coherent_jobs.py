from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final, final
from uuid import UUID, uuid4

from importlib.metadata import version as _package_version

from .article_embedding_vectors import (
    DIMENSION,
    POLICY,
    pinned_window_configuration,
    window_configuration_sha256,
)
from .coherent_search import COHERENT_ALIAS, COHERENT_INDEX_PREFIX, coherent_index_body
from .coherent_search_contracts import JsonValue
from .coherent_job_database import (
    DatabaseConnection,
    database_clients,
    lock_coherent_mutation,
)
from .coherent_job_hashing import coherent_sha256
from .model_config import load_pipeline_model_configuration
from .rag_experiment import (
    REVIEWED_UNIT_EXPORT_SQL,
    build_coherent_unit_documents_isolated,
)


_RETRIEVAL_MODEL: Final = (
    load_pipeline_model_configuration().retrieval.passage_embedding
)
_MAPPING_SHA256: Final = coherent_sha256(coherent_index_body())
# plan-time window configuration and hash come from the same code path the
# worker verifies against at run time; the engine version binds whatever is
# installed when the plan is created (a stale plan under a bumped engine
# fails closed at the worker and is re-planned, instead of every job failing
# against a hardcoded version string)
_PINNED_WINDOW: Final = pinned_window_configuration()
COHERENT_CONFIGURATION: Final[dict[str, str | int | bool]] = {
    "engine": "sentence-transformers",
    "engine_version": _package_version("sentence-transformers"),
    "runtime": "transformers-cpu",
    "model": _RETRIEVAL_MODEL.model_name,
    "revision": _RETRIEVAL_MODEL.model_revision,
    "dimension": DIMENSION,
    "normalize_embeddings": True,
    "window_policy": POLICY,
    "tokenizer_limit": _PINNED_WINDOW.tokenizer_limit,
    "model_limit": _PINNED_WINDOW.model_limit,
    "effective_limit": _PINNED_WINDOW.effective_limit,
    "overlap_tokens": _PINNED_WINDOW.overlap_tokens,
    "embedding_configuration_sha256": window_configuration_sha256(_PINNED_WINDOW),
    "alias": COHERENT_ALIAS,
    "index_prefix": COHERENT_INDEX_PREFIX,
    "projection_kind": "opensearch_coherent_unit",
    "mapping_sha256": _MAPPING_SHA256,
}
COHERENT_STAGES: Final = ("coherent_unit_embedding", "coherent_unit_search_projection")


@dataclass(frozen=True, slots=True)
class ActiveRevision:
    revision_id: UUID
    input_sha256: str
    content_sha256: str = ""


@final
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
    skipped: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class CoherentJobContext:
    job_id: str
    batch_id: str
    stage: str
    input_fingerprint: str
    configuration: Mapping[str, JsonValue]
    revision_id: UUID | None


@dataclass(frozen=True, slots=True)
class CoherentExecution:
    artifact_uri: str
    output_sha256: str
    result: dict[str, JsonValue]
    adopted: bool


def coherent_plan_key(revisions: tuple[ActiveRevision, ...]) -> str:
    payload = {
        "contract": "wic-coherent-article-jobs-v1",
        "active_revisions": [
            asdict(revision)
            for revision in sorted(revisions, key=lambda item: str(item.revision_id))
        ],
        "configuration": COHERENT_CONFIGURATION,
    }
    return coherent_sha256(payload)


def load_active_documents_with_skipped(
    connection: DatabaseConnection,
) -> tuple[
    list[tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]],
    list[tuple[str, str]],
]:
    """Materialize the active corpus, isolating unmaterializable articles.

    A damaged article (ambiguous alignment, hash mismatch) is excluded from
    the snapshot and reported in the skipped list — it abstains into review
    instead of failing corpus-wide planning and projection.
    """
    rows = connection.execute(
        REVIEWED_UNIT_EXPORT_SQL, {"volume_number": None, "page_number": None}
    ).fetchall()
    return build_coherent_unit_documents_isolated(rows)


def load_active_documents(
    connection: DatabaseConnection,
) -> list[tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]]:
    documents, _skipped = load_active_documents_with_skipped(connection)
    return documents


def active_revisions(
    documents: Sequence[tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]],
) -> tuple[ActiveRevision, ...]:
    revisions: list[ActiveRevision] = []
    for document, _ in documents:
        metadata = document["metadata"]
        if not isinstance(metadata, Mapping):
            raise CoherentJobError("Coherent article metadata is invalid")
        revisions.append(
            ActiveRevision(
                UUID(str(document["id"])),
                str(metadata["input_sha256"]),
                str(metadata["content_sha256"]),
            )
        )
    return tuple(revisions)


def enqueue_coherent_jobs(
    connection: DatabaseConnection, *, created_by: str, max_revisions: int = 1000
) -> CoherentPlanResult:
    if not created_by.strip() or max_revisions < 1:
        raise CoherentJobError("created_by and a positive max_revisions are required")
    lock_coherent_mutation(connection)
    documents, skipped = load_active_documents_with_skipped(connection)
    revisions = active_revisions(documents)
    if len(revisions) > max_revisions:
        raise CoherentJobError(
            f"Coherent backfill found {len(revisions)} revisions above the {max_revisions} revision guard"
        )
    if not revisions:
        return CoherentPlanResult(None, None, 0, 0, False, tuple(skipped))
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
            json.dumps(
                {"active_revisions": [asdict(item) for item in revisions]}, default=str
            ),
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
        return CoherentPlanResult(
            str(existing["batch_id"]),
            plan_key,
            len(revisions),
            len(revisions) + 1,
            False,
            tuple(skipped),
        )
    parent_ids: list[UUID] = []
    for revision in revisions:
        job_id = uuid4()
        parent_ids.append(job_id)
        fingerprint = coherent_sha256(
            {"revision": asdict(revision), "configuration": COHERENT_CONFIGURATION}
        )
        job_configuration = {
            **COHERENT_CONFIGURATION,
            "planned_revision_id": str(revision.revision_id),
            "planned_input_sha256": revision.input_sha256,
            "planned_content_sha256": revision.content_sha256,
            "planned_embedding_fingerprint": fingerprint,
        }
        _ = connection.execute(
            """INSERT INTO pipeline.ingestion_job (
                   job_id, batch_id, job_key, stage, scope_kind,
                   coherent_unit_revision_id, input_fingerprint, configuration
               ) VALUES (%s, %s, %s, 'coherent_unit_embedding',
                         'coherent_unit_revision', %s, %s, %s::jsonb)""",
            (
                job_id,
                batch_id,
                fingerprint,
                revision.revision_id,
                fingerprint,
                json.dumps(job_configuration),
            ),
        )
    projection_id = uuid4()
    _ = connection.execute(
        """INSERT INTO pipeline.ingestion_job (
               job_id, batch_id, job_key, stage, scope_kind,
               input_fingerprint, configuration
           ) VALUES (%s, %s, %s, 'coherent_unit_search_projection',
                     'batch', %s, %s::jsonb)""",
        (
            projection_id,
            batch_id,
            coherent_sha256({"projection": plan_key}),
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
        _ = cursor.executemany(
            "INSERT INTO pipeline.ingestion_job_dependency (job_id, depends_on_job_id) VALUES (%s, %s)",
            [(projection_id, parent_id) for parent_id in parent_ids],
        )
        _ = cursor.executemany(
            "INSERT INTO pipeline.ingestion_job_event (job_id, event_type, worker_id) VALUES (%s, 'planned', %s)",
            [(job_id, created_by.strip()) for job_id in (*parent_ids, projection_id)],
        )
    return CoherentPlanResult(
        str(batch_id), plan_key, len(revisions), len(revisions) + 1, True, tuple(skipped)
    )


def backfill_coherent_jobs(
    database_url: str, *, created_by: str, max_revisions: int
) -> CoherentPlanResult:
    psycopg, dict_row = database_clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        result = enqueue_coherent_jobs(
            connection, created_by=created_by, max_revisions=max_revisions
        )
    return result


def load_coherent_job_context(
    database_url: str, job_id: UUID | str
) -> CoherentJobContext:
    psycopg, dict_row = database_clients()
    row: Mapping[str, JsonValue] | None = None
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
    configuration = row["configuration"]
    if not isinstance(configuration, Mapping):
        raise CoherentJobError("Coherent ingestion configuration is invalid")
    raw_revision = row["coherent_unit_revision_id"]
    return CoherentJobContext(
        str(row["job_id"]),
        str(row["batch_id"]),
        str(row["stage"]),
        str(row["input_fingerprint"]),
        configuration,
        None if raw_revision is None else UUID(str(raw_revision)),
    )
