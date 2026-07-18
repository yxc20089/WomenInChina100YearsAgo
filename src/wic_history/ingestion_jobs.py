"""Durable, dependency-aware orchestration for full-corpus ingestion jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any, Iterable, Sequence
from uuid import UUID, uuid4


PAGE_STAGES = ("render_lossless", "ocr", "embedding", "ner")
AGGREGATE_STAGES = ("search_projection", "rag_export", "graph_projection")
ALL_STAGES = (*PAGE_STAGES, *AGGREGATE_STAGES)
STAGE_DEPENDENCY = {
    "ocr": "render_lossless",
    "embedding": "ocr",
    "ner": "ocr",
}
AGGREGATE_DEPENDENCY_STAGE = {
    "search_projection": "embedding",
    "rag_export": "ocr",
    "graph_projection": "ner",
}
DEFAULT_CONFIGURATION: dict[str, dict[str, Any]] = {
    "render_lossless": {
        "output_root": "artifacts/ingestion-pages",
        "evidence_tier": "unreviewed_input",
        "geometric_transform": "none",
    },
    "ocr": {
        "engine": "PaddleOCR",
        "model": "PP-OCRv6_medium_det+PP-OCRv6_medium_rec",
        "revision": "paddleocr-3.7.0-official",
        "language": "ch",
        "tile_size": 1200,
        "overlap": 120,
        "worker_batch_size": 5,
        "worker_mode": "platform_default",
        "output_root": "artifacts/ingestion-ocr",
    },
    "embedding": {
        "model": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "dimension": 1024,
        "batch_size": 16,
        "output_root": "artifacts/ingestion-embedding",
    },
    "ner": {
        "adapter": "rules+gliner",
        "model": "knowledgator/gliner-x-large",
        "revision": "4a4437f439a78d67c87781b42e8c45373d2adcb0",
        "ontology_version": "women-history-zh-v1",
        "input_variant": "raw_ocr",
        "max_regions": None,
        "threshold": 0.45,
        "batch_size": 2,
        "word_splitter_language": "zh-hant",
        "flat_ner": False,
        "multi_label": True,
        "dataset_id": None,
        "split_id": None,
        "output_root": "artifacts/ingestion-ner",
        "status": "candidate_only",
    },
    "search_projection": {
        "alias": "wic-regions-current",
        "index_prefix": "wic-regions-batch",
        "batch_size": 500,
        "source_policy": "active_page_selection_only",
        "output_root": "artifacts/ingestion-search",
    },
    "rag_export": {
        "source_policy": "active_page_selection_only",
        "input_unit": "ocr_page",
        "output_root": "artifacts/ingestion-rag",
    },
    "graph_projection": {
        "reviewed_only": True,
        "output_root": "artifacts/ingestion-graph",
    },
}


@dataclass(frozen=True, slots=True)
class PageTarget:
    source_object_id: UUID
    volume_id: UUID
    volume_number: int
    page_number: int
    publication_year: int
    source_uri: str
    source_sha256: str | None
    etag: str | None
    size_bytes: int
    integrity_status: str


@dataclass(frozen=True, slots=True)
class PlanResult:
    batch_id: str
    plan_key: str
    pages: int
    jobs: int
    dependencies: int
    created: bool


@dataclass(frozen=True, slots=True)
class JobLease:
    job_id: str
    batch_id: str
    stage: str
    scope_kind: str
    volume_number: int | None
    page_number: int | None
    input_fingerprint: str
    configuration: dict[str, Any]
    attempt_count: int
    max_attempts: int
    lease_owner: str
    lease_expires_at: str


@dataclass(frozen=True, slots=True)
class JobTransition:
    job_id: str
    status: str
    attempt_count: int
    max_attempts: int


@dataclass(frozen=True, slots=True)
class BatchStatus:
    batch_id: str
    name: str
    status: str
    total_jobs: int
    ready_jobs: int
    blocked_jobs: int
    dead_letter_jobs: int
    by_status: dict[str, int]
    by_stage: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class FailedJob:
    job_id: str
    stage: str
    volume_number: int | None
    page_number: int | None
    attempt_count: int
    max_attempts: int
    error_details: dict[str, Any] | None
    completed_at: str | None


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_argument(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256")
    return value


def normalize_stages(stages: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(stage.strip() for stage in stages if stage.strip()))
    unknown = sorted(set(requested) - set(PAGE_STAGES))
    if unknown:
        raise ValueError(f"Unsupported page stages: {', '.join(unknown)}")
    if not requested:
        raise ValueError("At least one page stage is required")
    for stage in requested:
        dependency = STAGE_DEPENDENCY.get(stage)
        if dependency and dependency not in requested:
            raise ValueError(f"Stage {stage} requires {dependency} in the same plan")
    return tuple(stage for stage in PAGE_STAGES if stage in requested)


def normalize_aggregate_stages(
    stages: Iterable[str], page_stages: Sequence[str]
) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(stage.strip() for stage in stages if stage.strip()))
    unknown = sorted(set(requested) - set(AGGREGATE_STAGES))
    if unknown:
        raise ValueError(f"Unsupported aggregate stages: {', '.join(unknown)}")
    for stage in requested:
        dependency = AGGREGATE_DEPENDENCY_STAGE[stage]
        if dependency not in page_stages:
            raise ValueError(
                f"Aggregate stage {stage} requires page stage {dependency}"
            )
    return tuple(stage for stage in AGGREGATE_STAGES if stage in requested)


def _psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    return psycopg, dict_row


def _load_targets(
    connection: Any,
    *,
    volume_number: int | None,
    page_number: int | None,
    include_suspect: bool,
) -> list[PageTarget]:
    if page_number is not None and volume_number is None:
        raise ValueError("A page filter requires a volume filter")
    rows = connection.execute(
        """
        SELECT s.source_object_id, v.volume_id, v.volume_number,
               v.publication_year, v.page_count, s.source_uri, s.sha256,
               s.etag, s.size_bytes, s.integrity_status
        FROM archive.volume v
        JOIN archive.source_object s USING (source_object_id)
        WHERE v.page_count IS NOT NULL
          AND (%(volume_number)s::integer IS NULL
               OR v.volume_number = %(volume_number)s::integer)
          AND (%(include_suspect)s OR s.integrity_status = 'ok_fast_checks')
        ORDER BY v.volume_number
        """,
        {
            "volume_number": volume_number,
            "include_suspect": include_suspect,
        },
    ).fetchall()
    targets = []
    for row in rows:
        pages = [page_number] if page_number is not None else range(1, row[4] + 1)
        for page in pages:
            if not 1 <= page <= row[4]:
                raise ValueError(
                    f"Page {page} is outside volume {row[2]}'s 1–{row[4]} range"
                )
            targets.append(
                PageTarget(
                    source_object_id=row[0],
                    volume_id=row[1],
                    volume_number=row[2],
                    page_number=page,
                    publication_year=row[3],
                    source_uri=row[5],
                    source_sha256=row[6],
                    etag=row[7],
                    size_bytes=row[8],
                    integrity_status=row[9],
                )
            )
    if not targets:
        raise ValueError("No manifest-validated pages match the requested scope")
    return targets


def create_plan(
    database_url: str,
    *,
    name: str,
    created_by: str,
    volume_number: int | None = None,
    page_number: int | None = None,
    stages: Iterable[str] = PAGE_STAGES,
    aggregate_stages: Iterable[str] = (),
    configuration: dict[str, dict[str, Any]] | None = None,
    include_suspect: bool = False,
    max_pages: int = 1000,
    allow_large_plan: bool = False,
) -> PlanResult:
    """Create an immutable, idempotent page DAG from authoritative volume rows."""
    psycopg, _ = _psycopg()
    normalized_stages = normalize_stages(stages)
    normalized_aggregate_stages = normalize_aggregate_stages(
        aggregate_stages, normalized_stages
    )
    all_requested_stages = (*normalized_stages, *normalized_aggregate_stages)
    stage_configuration = {
        stage: {
            **DEFAULT_CONFIGURATION[stage],
            **(configuration or {}).get(stage, {}),
        }
        for stage in all_requested_stages
    }
    if not name.strip() or not created_by.strip():
        raise ValueError("Batch name and created_by must not be blank")
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    with psycopg.connect(database_url) as connection:
        targets = _load_targets(
            connection,
            volume_number=volume_number,
            page_number=page_number,
            include_suspect=include_suspect,
        )
        if len(targets) > max_pages and not allow_large_plan:
            raise ValueError(
                f"Plan contains {len(targets):,} pages, above the {max_pages:,}-page guard; "
                "use --allow-large-plan only after cost/capacity review"
            )
        scope = {
            "volume_number": volume_number,
            "page_number": page_number,
            "include_suspect": include_suspect,
            "page_count": len(targets),
        }
        target_snapshots = [
            {
                "source_object_id": str(target.source_object_id),
                "volume_id": str(target.volume_id),
                "volume_number": target.volume_number,
                "page_number": target.page_number,
                "source_sha256": target.source_sha256,
                "etag": target.etag,
                "size_bytes": target.size_bytes,
                "integrity_status": target.integrity_status,
            }
            for target in targets
        ]
        plan_payload = {
            "contract": "wic-ingestion-dag-v1",
            "scope": scope,
            "stages": normalized_stages,
            "configuration": stage_configuration,
            "targets": target_snapshots,
        }
        if normalized_aggregate_stages:
            plan_payload["aggregate_stages"] = normalized_aggregate_stages
        plan_key = canonical_sha256(plan_payload)
        existing = connection.execute(
            """
            SELECT batch_id FROM pipeline.ingestion_batch WHERE plan_key = %s
            """,
            (plan_key,),
        ).fetchone()
        if existing:
            counts = connection.execute(
                """
                SELECT count(*),
                       (SELECT count(*)
                        FROM pipeline.ingestion_job_dependency dependency
                        JOIN pipeline.ingestion_job job USING (job_id)
                        WHERE job.batch_id = %s)
                FROM pipeline.ingestion_job WHERE batch_id = %s
                """,
                (existing[0], existing[0]),
            ).fetchone()
            return PlanResult(
                str(existing[0]), plan_key, len(targets), counts[0], counts[1], False
            )

        batch_id = connection.execute(
            """
            INSERT INTO pipeline.ingestion_batch (
                plan_key, name, scope, configuration, created_by
            ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s)
            RETURNING batch_id
            """,
            (
                plan_key,
                name.strip(),
                json.dumps(scope, ensure_ascii=False),
                json.dumps(
                    {
                        "contract": "wic-ingestion-dag-v1",
                        "stages": normalized_stages,
                        "aggregate_stages": normalized_aggregate_stages,
                        "stage_configuration": stage_configuration,
                    },
                    ensure_ascii=False,
                ),
                created_by.strip(),
            ),
        ).fetchone()[0]
        jobs: list[tuple[Any, ...]] = []
        dependencies: list[tuple[UUID, UUID]] = []
        events: list[tuple[UUID, str]] = []
        page_stage_ids: dict[str, list[UUID]] = {
            stage: [] for stage in normalized_stages
        }
        page_stage_keys: dict[str, list[str]] = {
            stage: [] for stage in normalized_stages
        }
        for target, snapshot in zip(targets, target_snapshots, strict=True):
            source_fingerprint = canonical_sha256(snapshot)
            stage_ids: dict[str, UUID] = {}
            stage_keys: dict[str, str] = {}
            for stage in normalized_stages:
                dependency = STAGE_DEPENDENCY.get(stage)
                input_fingerprint = (
                    canonical_sha256(
                        {
                            "parent_job_key": stage_keys[dependency],
                            "stage": stage,
                            "configuration": stage_configuration[stage],
                        }
                    )
                    if dependency
                    else source_fingerprint
                )
                job_key = canonical_sha256(
                    {
                        "stage": stage,
                        "volume_number": target.volume_number,
                        "page_number": target.page_number,
                        "input_fingerprint": input_fingerprint,
                        "configuration": stage_configuration[stage],
                    }
                )
                job_id = uuid4()
                jobs.append(
                    (
                        job_id,
                        batch_id,
                        job_key,
                        stage,
                        "page",
                        target.source_object_id,
                        target.volume_id,
                        target.page_number,
                        input_fingerprint,
                        json.dumps(stage_configuration[stage], ensure_ascii=False),
                    )
                )
                events.append((job_id, created_by.strip()))
                stage_ids[stage] = job_id
                stage_keys[stage] = job_key
                page_stage_ids[stage].append(job_id)
                page_stage_keys[stage].append(job_key)
                if dependency:
                    dependencies.append((job_id, stage_ids[dependency]))
        for stage in normalized_aggregate_stages:
            dependency_stage = AGGREGATE_DEPENDENCY_STAGE[stage]
            parent_keys = sorted(page_stage_keys[dependency_stage])
            input_fingerprint = canonical_sha256(
                {
                    "parent_job_keys": parent_keys,
                    "stage": stage,
                    "configuration": stage_configuration[stage],
                }
            )
            job_key = canonical_sha256(
                {
                    "stage": stage,
                    "scope": scope,
                    "input_fingerprint": input_fingerprint,
                    "configuration": stage_configuration[stage],
                }
            )
            job_id = uuid4()
            jobs.append(
                (
                    job_id,
                    batch_id,
                    job_key,
                    stage,
                    "batch",
                    None,
                    None,
                    None,
                    input_fingerprint,
                    json.dumps(stage_configuration[stage], ensure_ascii=False),
                )
            )
            events.append((job_id, created_by.strip()))
            dependencies.extend(
                (job_id, parent_id)
                for parent_id in page_stage_ids[dependency_stage]
            )
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO pipeline.ingestion_job (
                    job_id, batch_id, job_key, stage, scope_kind,
                    source_object_id, volume_id, page_number,
                    input_fingerprint, configuration
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                jobs,
            )
            cursor.executemany(
                """
                INSERT INTO pipeline.ingestion_job_dependency (
                    job_id, depends_on_job_id
                ) VALUES (%s, %s)
                """,
                dependencies,
            )
            cursor.executemany(
                """
                INSERT INTO pipeline.ingestion_job_event (
                    job_id, event_type, worker_id
                ) VALUES (%s, 'planned', %s)
                """,
                events,
            )
        return PlanResult(
            str(batch_id),
            plan_key,
            len(targets),
            len(jobs),
            len(dependencies),
            True,
        )


def _cancel_failed_descendants(
    connection: Any,
    failed_job_ids: Sequence[UUID],
) -> int:
    """Cancel pending descendants that can never satisfy a failed dependency."""
    if not failed_job_ids:
        return 0
    rows = connection.execute(
        """
        WITH RECURSIVE descendants(job_id) AS (
            SELECT dependency.job_id
            FROM pipeline.ingestion_job_dependency dependency
            WHERE dependency.depends_on_job_id = ANY(%s)
            UNION
            SELECT dependency.job_id
            FROM pipeline.ingestion_job_dependency dependency
            JOIN descendants parent
              ON parent.job_id = dependency.depends_on_job_id
        )
        UPDATE pipeline.ingestion_job job
        SET status = 'cancelled', completed_at = now(),
            error_details = COALESCE(job.error_details, '{}'::jsonb)
                || jsonb_build_object(
                    'type', 'dependency_failed',
                    'failed_parent_job_ids', %s::jsonb
                )
        WHERE job.job_id IN (SELECT job_id FROM descendants)
          AND job.status = 'pending'
        RETURNING job.job_id
        """,
        (
            list(failed_job_ids),
            json.dumps([str(job_id) for job_id in failed_job_ids]),
        ),
    ).fetchall()
    if rows:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO pipeline.ingestion_job_event (
                    job_id, event_type, details
                ) VALUES (%s, 'cancelled', %s::jsonb)
                """,
                [
                    (
                        row[0],
                        json.dumps(
                            {
                                "reason": "dependency_failed",
                                "failed_parent_job_ids": [
                                    str(job_id) for job_id in failed_job_ids
                                ],
                            }
                        ),
                    )
                    for row in rows
                ],
            )
    return len(rows)


def _refresh_batch_status(connection: Any, batch_id: UUID) -> None:
    connection.execute(
        """
        UPDATE pipeline.ingestion_batch batch
        SET status = CASE
                WHEN EXISTS (
                    SELECT 1 FROM pipeline.ingestion_job job
                    WHERE job.batch_id = batch.batch_id
                      AND job.status = 'failed'
                ) THEN 'failed'
                WHEN EXISTS (
                    SELECT 1 FROM pipeline.ingestion_job job
                    WHERE job.batch_id = batch.batch_id
                      AND job.status = 'cancelled'
                ) THEN 'cancelled'
                ELSE 'completed'
            END,
            completed_at = now()
        WHERE batch.batch_id = %s AND batch.status = 'active'
          AND NOT EXISTS (
              SELECT 1 FROM pipeline.ingestion_job job
              WHERE job.batch_id = batch.batch_id
                AND job.status IN ('pending', 'leased', 'running')
          )
        """,
        (batch_id,),
    )


def _requeue_expired(connection: Any) -> None:
    expired = connection.execute(
        """
        UPDATE pipeline.ingestion_job
        SET status = CASE
                WHEN attempt_count >= max_attempts THEN 'failed'
                ELSE 'pending'
            END,
            available_at = now(),
            lease_owner = NULL,
            lease_expires_at = NULL,
            error_details = COALESCE(error_details, '{}'::jsonb)
                || jsonb_build_object('last_error', 'lease_expired')
        WHERE status IN ('leased', 'running')
          AND lease_expires_at <= now()
        RETURNING job_id, batch_id, status
        """
    ).fetchall()
    if expired:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO pipeline.ingestion_job_event (job_id, event_type)
                VALUES (%s, 'lease_expired')
                """,
                [(row[0],) for row in expired],
            )
        failed_ids = [row[0] for row in expired if row[2] == "failed"]
        _cancel_failed_descendants(connection, failed_ids)
        for batch_id in {row[1] for row in expired}:
            _refresh_batch_status(connection, batch_id)


def claim_job(
    database_url: str,
    *,
    worker_id: str,
    lease_seconds: int = 900,
    stage: str | None = None,
    batch_id: UUID | None = None,
) -> JobLease | None:
    """Atomically lease one ready job using PostgreSQL SKIP LOCKED."""
    psycopg, dict_row = _psycopg()
    if not worker_id.strip():
        raise ValueError("worker_id must not be blank")
    if not 30 <= lease_seconds <= 86400:
        raise ValueError("lease_seconds must be between 30 and 86400")
    if stage is not None and stage not in ALL_STAGES:
        raise ValueError("claim stage must be a supported ingestion stage")
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        _requeue_expired(connection)
        row = connection.execute(
            """
            SELECT job.job_id
            FROM pipeline.ingestion_job job
            JOIN pipeline.ingestion_batch batch USING (batch_id)
            WHERE job.status = 'pending'
              AND job.available_at <= now()
              AND job.attempt_count < job.max_attempts
              AND batch.status = 'active'
              AND (%(stage)s::text IS NULL OR job.stage = %(stage)s::text)
              AND (%(batch_id)s::uuid IS NULL OR job.batch_id = %(batch_id)s::uuid)
              AND NOT EXISTS (
                  SELECT 1
                  FROM pipeline.ingestion_job_dependency dependency
                  JOIN pipeline.ingestion_job parent
                    ON parent.job_id = dependency.depends_on_job_id
                  WHERE dependency.job_id = job.job_id
                    AND parent.status <> 'completed'
              )
            ORDER BY job.priority DESC, job.created_at, job.job_id
            FOR UPDATE OF job SKIP LOCKED
            LIMIT 1
            """,
            {"stage": stage, "batch_id": batch_id},
        ).fetchone()
        if row is None:
            return None
        lease = connection.execute(
            """
            UPDATE pipeline.ingestion_job job
            SET status = 'leased',
                attempt_count = attempt_count + 1,
                lease_owner = %(worker_id)s,
                lease_expires_at = now() + %(lease_duration)s,
                started_at = COALESCE(started_at, now()),
                error_details = NULL
            WHERE job.job_id = %(job_id)s
            RETURNING job.job_id, job.batch_id, job.stage, job.scope_kind,
                      (SELECT volume_number FROM archive.volume volume
                       WHERE volume.volume_id = job.volume_id) AS volume_number,
                      job.page_number,
                      job.input_fingerprint, job.configuration,
                      job.attempt_count, job.max_attempts,
                      job.lease_owner, job.lease_expires_at
            """,
            {
                "worker_id": worker_id.strip(),
                "lease_duration": timedelta(seconds=lease_seconds),
                "job_id": row["job_id"],
            },
        ).fetchone()
        connection.execute(
            """
            INSERT INTO pipeline.ingestion_job_event (
                job_id, event_type, worker_id,
                details
            ) VALUES (%s, 'leased', %s, jsonb_build_object('lease_seconds', %s))
            """,
            (row["job_id"], worker_id.strip(), lease_seconds),
        )
        return JobLease(
            job_id=str(lease["job_id"]),
            batch_id=str(lease["batch_id"]),
            stage=lease["stage"],
            scope_kind=lease["scope_kind"],
            volume_number=lease["volume_number"],
            page_number=lease["page_number"],
            input_fingerprint=lease["input_fingerprint"],
            configuration=lease["configuration"],
            attempt_count=lease["attempt_count"],
            max_attempts=lease["max_attempts"],
            lease_owner=lease["lease_owner"],
            lease_expires_at=lease["lease_expires_at"].isoformat(),
        )


def start_job(database_url: str, job_id: UUID, worker_id: str) -> JobTransition:
    return _worker_transition(database_url, job_id, worker_id, "start")


def heartbeat_job(
    database_url: str,
    job_id: UUID,
    worker_id: str,
    *,
    lease_seconds: int = 900,
) -> JobTransition:
    return _worker_transition(
        database_url, job_id, worker_id, "heartbeat", lease_seconds=lease_seconds
    )


def _worker_transition(
    database_url: str,
    job_id: UUID,
    worker_id: str,
    action: str,
    *,
    lease_seconds: int = 900,
) -> JobTransition:
    psycopg, _ = _psycopg()
    if action not in {"start", "heartbeat"}:
        raise ValueError("Unsupported worker transition")
    if not worker_id.strip():
        raise ValueError("worker_id must not be blank")
    if not 30 <= lease_seconds <= 86400:
        raise ValueError("lease_seconds must be between 30 and 86400")
    with psycopg.connect(database_url) as connection:
        if action == "start":
            row = connection.execute(
                """
                UPDATE pipeline.ingestion_job
                SET status = 'running'
                WHERE job_id = %s AND status = 'leased'
                  AND lease_owner = %s AND lease_expires_at > now()
                RETURNING job_id, status, attempt_count, max_attempts
                """,
                (job_id, worker_id.strip()),
            ).fetchone()
            event_type = "started"
        else:
            row = connection.execute(
                """
                UPDATE pipeline.ingestion_job
                SET lease_expires_at = now() + %s
                WHERE job_id = %s AND status IN ('leased', 'running')
                  AND lease_owner = %s AND lease_expires_at > now()
                RETURNING job_id, status, attempt_count, max_attempts
                """,
                (timedelta(seconds=lease_seconds), job_id, worker_id.strip()),
            ).fetchone()
            event_type = "heartbeat"
        if row is None:
            raise ValueError("Job lease is absent, expired, or owned by another worker")
        connection.execute(
            """
            INSERT INTO pipeline.ingestion_job_event (job_id, event_type, worker_id)
            VALUES (%s, %s, %s)
            """,
            (job_id, event_type, worker_id.strip()),
        )
        return JobTransition(str(row[0]), row[1], row[2], row[3])


def complete_job(
    database_url: str,
    job_id: UUID,
    worker_id: str,
    *,
    artifact_uri: str,
    output_sha256: str,
    result: dict[str, Any] | None = None,
) -> JobTransition:
    psycopg, _ = _psycopg()
    if not worker_id.strip() or not artifact_uri.strip():
        raise ValueError("worker_id and artifact_uri must not be blank")
    sha256_argument(output_sha256)
    result_data = result or {}
    with psycopg.connect(database_url) as connection:
        job = connection.execute(
            """
            SELECT stage, configuration
            FROM pipeline.ingestion_job
            WHERE job_id = %s AND status IN ('leased', 'running')
              AND lease_owner = %s AND lease_expires_at > now()
            FOR UPDATE
            """,
            (job_id, worker_id.strip()),
        ).fetchone()
        if job is None:
            raise ValueError("Job lease is absent, expired, or owned by another worker")
        validate_stage_result(job[0], job[1], result_data)
        row = connection.execute(
            """
            UPDATE pipeline.ingestion_job
            SET status = 'completed', artifact_uri = %s, output_sha256 = %s,
                result = %s::jsonb, completed_at = now(),
                lease_owner = NULL, lease_expires_at = NULL
            WHERE job_id = %s AND status IN ('leased', 'running')
              AND lease_owner = %s AND lease_expires_at > now()
            RETURNING job_id, status, attempt_count, max_attempts, batch_id
            """,
            (
                artifact_uri.strip(),
                output_sha256,
                json.dumps(result_data, ensure_ascii=False),
                job_id,
                worker_id.strip(),
            ),
        ).fetchone()
        if row is None:
            raise ValueError("Job lease is absent, expired, or owned by another worker")
        connection.execute(
            """
            INSERT INTO pipeline.ingestion_job_event (
                job_id, event_type, worker_id, details
            ) VALUES (%s, 'completed', %s, %s::jsonb)
            """,
            (job_id, worker_id.strip(), json.dumps(result_data, ensure_ascii=False)),
        )
        _refresh_batch_status(connection, row[4])
        return JobTransition(str(row[0]), row[1], row[2], row[3])


def _required_uuid(result: dict[str, Any], field: str) -> None:
    try:
        UUID(str(result[field]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Stage result requires UUID field {field}") from exc


def _required_count(result: dict[str, Any], field: str) -> None:
    value = result.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Stage result requires nonnegative integer field {field}")


def validate_stage_result(
    stage: str, configuration: dict[str, Any], result: dict[str, Any]
) -> None:
    """Reject completion metadata that contradicts the immutable job contract."""
    if stage == "render_lossless":
        if not re.fullmatch(r"[0-9a-f]{64}", str(result.get("render_sha256", ""))):
            raise ValueError(
                "Stage result requires lowercase SHA-256 field render_sha256"
            )
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(result.get("source_object_sha256", ""))
        ):
            raise ValueError(
                "Stage result requires lowercase SHA-256 field source_object_sha256"
            )
    elif stage == "ocr":
        _required_uuid(result, "ocr_run_id")
        _required_count(result, "regions")
    elif stage == "embedding":
        _required_uuid(result, "embedding_run_id")
        _required_count(result, "embeddings")
    elif stage == "ner":
        _required_uuid(result, "ner_run_id")
        _required_count(result, "mentions")
        if result.get("candidate_only") is not True:
            raise ValueError("NER stage results must remain candidate_only")
        expected_limit = configuration.get("max_regions")
        observed_limit = result.get("bounded_regions")
        if expected_limit != observed_limit:
            raise ValueError(
                "NER bounded_regions must exactly match the planned max_regions"
            )
    elif stage == "search_projection":
        _required_uuid(result, "projection_build_id")
        _required_count(result, "documents_indexed")
        if not str(result.get("index_name", "")).startswith("wic-regions-"):
            raise ValueError("Search projection requires managed index_name")
    elif stage == "rag_export":
        _required_count(result, "documents")
        _required_count(result, "exported_regions")
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(result.get("manifest_sha256", ""))
        ):
            raise ValueError("RAG export requires manifest_sha256")
    elif stage == "graph_projection":
        _required_uuid(result, "projection_build_id")
        for field in ("entities", "claims", "mentions", "claim_evidence"):
            _required_count(result, field)
        if result.get("reviewed_only") is not True:
            raise ValueError("Graph projection must remain reviewed_only")
    else:
        raise ValueError(f"No completion result contract exists for stage {stage}")


def fail_job(
    database_url: str,
    job_id: UUID,
    worker_id: str,
    *,
    error_type: str,
    message: str,
    retry_delay_seconds: int = 60,
) -> JobTransition:
    psycopg, _ = _psycopg()
    if not worker_id.strip() or not error_type.strip() or not message.strip():
        raise ValueError("worker_id, error_type and message must not be blank")
    if not 0 <= retry_delay_seconds <= 86400:
        raise ValueError("retry_delay_seconds must be between 0 and 86400")
    details = {"type": error_type.strip(), "message": message.strip()}
    with psycopg.connect(database_url) as connection:
        row = connection.execute(
            """
            UPDATE pipeline.ingestion_job
            SET status = CASE
                    WHEN attempt_count < max_attempts THEN 'pending'
                    ELSE 'failed'
                END,
                available_at = CASE
                    WHEN attempt_count < max_attempts THEN now() + %s
                    ELSE available_at
                END,
                error_details = %s::jsonb,
                completed_at = CASE
                    WHEN attempt_count >= max_attempts THEN now()
                    ELSE NULL
                END,
                lease_owner = NULL, lease_expires_at = NULL
            WHERE job_id = %s AND status IN ('leased', 'running')
              AND lease_owner = %s AND lease_expires_at > now()
            RETURNING job_id, status, attempt_count, max_attempts, batch_id
            """,
            (
                timedelta(seconds=retry_delay_seconds),
                json.dumps(details, ensure_ascii=False),
                job_id,
                worker_id.strip(),
            ),
        ).fetchone()
        if row is None:
            raise ValueError("Job lease is absent, expired, or owned by another worker")
        event_type = "retry_scheduled" if row[1] == "pending" else "failed"
        connection.execute(
            """
            INSERT INTO pipeline.ingestion_job_event (
                job_id, event_type, worker_id, details
            ) VALUES (%s, %s, %s, %s::jsonb)
            """,
            (
                job_id,
                event_type,
                worker_id.strip(),
                json.dumps(details, ensure_ascii=False),
            ),
        )
        if row[1] == "failed":
            _cancel_failed_descendants(connection, [row[0]])
        _refresh_batch_status(connection, row[4])
        return JobTransition(str(row[0]), row[1], row[2], row[3])


def batch_status(database_url: str, batch_id: UUID) -> BatchStatus:
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        batch = connection.execute(
            """
            SELECT batch_id, name, status
            FROM pipeline.ingestion_batch WHERE batch_id = %s
            """,
            (batch_id,),
        ).fetchone()
        if batch is None:
            raise ValueError("Ingestion batch does not exist")
        rows = connection.execute(
            """
            SELECT stage, status, count(*) AS count
            FROM pipeline.ingestion_job
            WHERE batch_id = %s
            GROUP BY stage, status
            ORDER BY stage, status
            """,
            (batch_id,),
        ).fetchall()
        ready = connection.execute(
            """
            SELECT count(*) AS count
            FROM pipeline.ingestion_job job
            WHERE job.batch_id = %s AND job.status = 'pending'
              AND job.available_at <= now()
              AND NOT EXISTS (
                  SELECT 1
                  FROM pipeline.ingestion_job_dependency dependency
                  JOIN pipeline.ingestion_job parent
                    ON parent.job_id = dependency.depends_on_job_id
                  WHERE dependency.job_id = job.job_id
                    AND parent.status <> 'completed'
              )
            """,
            (batch_id,),
        ).fetchone()["count"]
        blocked = connection.execute(
            """
            SELECT count(*) AS count
            FROM pipeline.ingestion_job job
            WHERE job.batch_id = %s AND job.status = 'pending'
              AND EXISTS (
                  SELECT 1
                  FROM pipeline.ingestion_job_dependency dependency
                  JOIN pipeline.ingestion_job parent
                    ON parent.job_id = dependency.depends_on_job_id
                  WHERE dependency.job_id = job.job_id
                    AND parent.status <> 'completed'
              )
            """,
            (batch_id,),
        ).fetchone()["count"]
    by_status: dict[str, int] = {}
    by_stage: dict[str, dict[str, int]] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + row["count"]
        by_stage.setdefault(row["stage"], {})[row["status"]] = row["count"]
    return BatchStatus(
        str(batch["batch_id"]),
        batch["name"],
        batch["status"],
        sum(by_status.values()),
        ready,
        blocked,
        by_status.get("failed", 0),
        dict(sorted(by_status.items())),
        {stage: dict(sorted(counts.items())) for stage, counts in by_stage.items()},
    )


def batch_failures(database_url: str, batch_id: UUID) -> list[FailedJob]:
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pipeline.ingestion_batch WHERE batch_id = %s",
            (batch_id,),
        ).fetchone()
        if exists is None:
            raise ValueError("Ingestion batch does not exist")
        rows = connection.execute(
            """
            SELECT job.job_id, job.stage, volume.volume_number,
                   job.page_number, job.attempt_count, job.max_attempts,
                   job.error_details, job.completed_at
            FROM pipeline.ingestion_job job
            LEFT JOIN archive.volume volume USING (volume_id)
            WHERE job.batch_id = %s AND job.status = 'failed'
            ORDER BY job.completed_at, job.created_at, job.job_id
            """,
            (batch_id,),
        ).fetchall()
    return [
        FailedJob(
            job_id=str(row["job_id"]),
            stage=row["stage"],
            volume_number=row["volume_number"],
            page_number=row["page_number"],
            attempt_count=row["attempt_count"],
            max_attempts=row["max_attempts"],
            error_details=row["error_details"],
            completed_at=(
                row["completed_at"].isoformat() if row["completed_at"] else None
            ),
        )
        for row in rows
    ]


def cancel_batch(
    database_url: str,
    batch_id: UUID,
    *,
    cancelled_by: str,
    reason: str,
) -> BatchStatus:
    """Explicitly cancel unfinished jobs while preserving completed artifacts."""
    psycopg, _ = _psycopg()
    if not cancelled_by.strip() or not reason.strip():
        raise ValueError("cancelled_by and reason must not be blank")
    with psycopg.connect(database_url) as connection:
        batch = connection.execute(
            """
            SELECT status FROM pipeline.ingestion_batch
            WHERE batch_id = %s FOR UPDATE
            """,
            (batch_id,),
        ).fetchone()
        if batch is None:
            raise ValueError("Ingestion batch does not exist")
        if batch[0] in {"completed", "failed"}:
            raise ValueError(f"Cannot cancel terminal {batch[0]} batch")
        if batch[0] == "active":
            rows = connection.execute(
                """
                UPDATE pipeline.ingestion_job
                SET status = 'cancelled', completed_at = now(),
                    lease_owner = NULL, lease_expires_at = NULL,
                    error_details = jsonb_build_object(
                        'type', 'batch_cancelled', 'message', %s::text
                    )
                WHERE batch_id = %s
                  AND status IN ('pending', 'leased', 'running')
                RETURNING job_id
                """,
                (reason.strip(), batch_id),
            ).fetchall()
            if rows:
                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO pipeline.ingestion_job_event (
                            job_id, event_type, worker_id, details
                        ) VALUES (%s, 'cancelled', %s, %s::jsonb)
                        """,
                        [
                            (
                                row[0],
                                cancelled_by.strip(),
                                json.dumps(
                                    {
                                        "reason": "batch_cancelled",
                                        "message": reason.strip(),
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                            for row in rows
                        ],
                    )
            connection.execute(
                """
                UPDATE pipeline.ingestion_batch
                SET status = 'cancelled', completed_at = now()
                WHERE batch_id = %s
                """,
                (batch_id,),
            )
    return batch_status(database_url, batch_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--name", required=True)
    plan.add_argument("--created-by", required=True)
    plan.add_argument("--volume", type=int)
    plan.add_argument("--page", type=int)
    plan.add_argument("--stages", default=",".join(PAGE_STAGES))
    plan.add_argument(
        "--aggregate-stages",
        default="",
        help="Comma-separated batch stages: search_projection,rag_export,graph_projection",
    )
    plan.add_argument("--configuration", help="JSON object keyed by stage")
    plan.add_argument("--include-suspect", action="store_true")
    plan.add_argument("--max-pages", type=int, default=1000)
    plan.add_argument("--allow-large-plan", action="store_true")

    claim = subparsers.add_parser("claim")
    claim.add_argument("--worker", required=True)
    claim.add_argument("--lease-seconds", type=int, default=900)
    claim.add_argument("--stage", choices=ALL_STAGES)
    claim.add_argument("--batch-id", type=UUID)

    start = subparsers.add_parser("start")
    start.add_argument("--job-id", type=UUID, required=True)
    start.add_argument("--worker", required=True)

    heartbeat = subparsers.add_parser("heartbeat")
    heartbeat.add_argument("--job-id", type=UUID, required=True)
    heartbeat.add_argument("--worker", required=True)
    heartbeat.add_argument("--lease-seconds", type=int, default=900)

    complete = subparsers.add_parser("complete")
    complete.add_argument("--job-id", type=UUID, required=True)
    complete.add_argument("--worker", required=True)
    complete.add_argument("--artifact-uri", required=True)
    complete.add_argument("--output-sha256", type=sha256_argument, required=True)
    complete.add_argument("--result", default="{}", help="JSON object")

    fail = subparsers.add_parser("fail")
    fail.add_argument("--job-id", type=UUID, required=True)
    fail.add_argument("--worker", required=True)
    fail.add_argument("--error-type", required=True)
    fail.add_argument("--message", required=True)
    fail.add_argument("--retry-delay-seconds", type=int, default=60)

    status = subparsers.add_parser("status")
    status.add_argument("--batch-id", type=UUID, required=True)

    failures = subparsers.add_parser("failures")
    failures.add_argument("--batch-id", type=UUID, required=True)

    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("--batch-id", type=UUID, required=True)
    cancel.add_argument("--cancelled-by", required=True)
    cancel.add_argument("--reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    if args.command == "plan":
        configuration = json.loads(args.configuration) if args.configuration else None
        result: Any = create_plan(
            args.database_url,
            name=args.name,
            created_by=args.created_by,
            volume_number=args.volume,
            page_number=args.page,
            stages=args.stages.split(","),
            aggregate_stages=args.aggregate_stages.split(","),
            configuration=configuration,
            include_suspect=args.include_suspect,
            max_pages=args.max_pages,
            allow_large_plan=args.allow_large_plan,
        )
    elif args.command == "claim":
        result = claim_job(
            args.database_url,
            worker_id=args.worker,
            lease_seconds=args.lease_seconds,
            stage=args.stage,
            batch_id=args.batch_id,
        )
        if result is None:
            print("null")
            return 2
    elif args.command == "start":
        result = start_job(args.database_url, args.job_id, args.worker)
    elif args.command == "heartbeat":
        result = heartbeat_job(
            args.database_url,
            args.job_id,
            args.worker,
            lease_seconds=args.lease_seconds,
        )
    elif args.command == "complete":
        result = complete_job(
            args.database_url,
            args.job_id,
            args.worker,
            artifact_uri=args.artifact_uri,
            output_sha256=args.output_sha256,
            result=json.loads(args.result),
        )
    elif args.command == "fail":
        result = fail_job(
            args.database_url,
            args.job_id,
            args.worker,
            error_type=args.error_type,
            message=args.message,
            retry_delay_seconds=args.retry_delay_seconds,
        )
    elif args.command == "status":
        result = batch_status(args.database_url, args.batch_id)
    elif args.command == "failures":
        result = batch_failures(args.database_url, args.batch_id)
    else:
        result = cancel_batch(
            args.database_url,
            args.batch_id,
            cancelled_by=args.cancelled_by,
            reason=args.reason,
        )
    payload = (
        [asdict(item) for item in result]
        if isinstance(result, list)
        else asdict(result) if result is not None else None
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
