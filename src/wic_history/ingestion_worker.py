"""Execute one leased ingestion job with provenance checks and safe retries."""

from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import UUID

from .corpus_manifest import build_s3_client
from .embedding_pipeline import embed_regions
from .evidence import NERArtifact, OCRPageArtifact
from .gold_render import (
    ingestion_candidate,
    render_lossless_plan,
    write_lossless_results,
)
from .ingestion_jobs import (
    JobLease,
    claim_job,
    complete_job,
    fail_job,
    heartbeat_job,
    start_job,
)
from .ner_pipeline import main as ner_main
from .ocr_pipeline import main as ocr_main
from .ocr_pipeline import resolve_render_provenance
from .render_samples import sha256_file
from .repository import ingest_ner_artifact, ingest_ocr_artifact


@dataclass(frozen=True, slots=True)
class PageJobContext:
    job_id: str
    stage: str
    configuration: dict[str, Any]
    source_uri: str
    bucket: str
    object_key: str
    media_type: str
    size_bytes: int
    etag: str | None
    source_sha256: str | None
    integrity_status: str
    volume_number: int
    publication_year: int
    page_number: int
    page_count: int
    parent_stage: str | None
    parent_artifact_uri: str | None
    parent_output_sha256: str | None
    parent_result: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class StageExecution:
    artifact_uri: str
    output_sha256: str
    result: dict[str, Any]
    adopted: bool


@dataclass(frozen=True, slots=True)
class WorkerRunResult:
    job_id: str | None
    stage: str | None
    status: str
    adopted: bool | None = None
    artifact_uri: str | None = None
    error_type: str | None = None
    message: str | None = None


def _psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    return psycopg, dict_row


def load_job_context(database_url: str, job_id: UUID | str) -> PageJobContext:
    """Load immutable page/source context plus the single completed parent."""
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        row = connection.execute(
            """
            SELECT job.job_id, job.stage, job.configuration,
                   source.source_uri, source.bucket, source.object_key,
                   source.media_type, source.size_bytes, source.etag,
                   source.sha256 AS source_sha256, source.integrity_status,
                   volume.volume_number, volume.publication_year,
                   job.page_number, volume.page_count,
                   parent.stage AS parent_stage,
                   parent.artifact_uri AS parent_artifact_uri,
                   parent.output_sha256 AS parent_output_sha256,
                   parent.result AS parent_result
            FROM pipeline.ingestion_job job
            JOIN archive.source_object source USING (source_object_id)
            JOIN archive.volume volume USING (volume_id)
            LEFT JOIN pipeline.ingestion_job_dependency dependency
              ON dependency.job_id = job.job_id
            LEFT JOIN pipeline.ingestion_job parent
              ON parent.job_id = dependency.depends_on_job_id
            WHERE job.job_id = %s AND job.scope_kind = 'page'
            """,
            (job_id,),
        ).fetchone()
    if row is None:
        raise ValueError("Page ingestion job does not exist")
    if not row["bucket"] or not row["object_key"]:
        raise ValueError("Source object lacks its S3 bucket/key identity")
    if row["parent_stage"] and not row["parent_artifact_uri"]:
        raise ValueError("Parent job has no completed artifact URI")
    return PageJobContext(
        job_id=str(row["job_id"]),
        stage=row["stage"],
        configuration=row["configuration"],
        source_uri=row["source_uri"],
        bucket=row["bucket"],
        object_key=row["object_key"],
        media_type=row["media_type"],
        size_bytes=row["size_bytes"],
        etag=row["etag"],
        source_sha256=row["source_sha256"],
        integrity_status=row["integrity_status"],
        volume_number=row["volume_number"],
        publication_year=row["publication_year"],
        page_number=row["page_number"],
        page_count=row["page_count"],
        parent_stage=row["parent_stage"],
        parent_artifact_uri=row["parent_artifact_uri"],
        parent_output_sha256=row["parent_output_sha256"],
        parent_result=row["parent_result"],
    )


def resolve_workspace_path(workspace_root: Path, value: str | Path) -> Path:
    """Resolve a repository artifact path and refuse workspace escape."""
    root = workspace_root.resolve()
    supplied = Path(value)
    resolved = (supplied if supplied.is_absolute() else root / supplied).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Artifact path escapes the workspace: {value}")
    return resolved


def stage_output_dir(context: PageJobContext, workspace_root: Path) -> Path:
    configured = context.configuration.get(
        "output_root", f"artifacts/ingestion-{context.stage}"
    )
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("Stage output_root must be a nonblank workspace path")
    return resolve_workspace_path(workspace_root, configured) / "jobs" / context.job_id


def workspace_artifact_uri(workspace_root: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_record_image(
    workspace_root: Path, record: dict[str, Any]
) -> Path:
    image_path = resolve_workspace_path(workspace_root, record["render_path"])
    if not image_path.is_file():
        raise ValueError(f"Rendered page image is absent: {image_path}")
    if sha256_file(image_path) != record.get("render_sha256"):
        raise ValueError("Rendered page image disagrees with its manifest SHA-256")
    return image_path


def _render_manifest_execution(
    manifest_path: Path,
    context: PageJobContext,
    workspace_root: Path,
) -> StageExecution:
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        record
        for record in records
        if record.get("status") == "rendered"
        and record.get("source_uri") == context.source_uri
        and record.get("volume_number") == context.volume_number
        and record.get("publication_year") == context.publication_year
        and record.get("page_number") == context.page_number
    ]
    if len(matches) != 1:
        raise ValueError("Render manifest must have one matching page record")
    record = matches[0]
    image_path = _resolve_record_image(workspace_root, record)
    source_sha256, _ = resolve_render_provenance(
        image_path,
        manifest_path,
        source_uri=context.source_uri,
        page_number=context.page_number,
        volume_number=context.volume_number,
        publication_year=context.publication_year,
        supplied_source_sha256=context.source_sha256,
        artifact_root=workspace_root,
    )
    return StageExecution(
        artifact_uri=workspace_artifact_uri(workspace_root, manifest_path),
        output_sha256=sha256_file(manifest_path),
        result={
            "render_sha256": record["render_sha256"],
            "source_object_sha256": source_sha256,
            "decoded_pixel_sha256": record.get("decoded_pixel_sha256"),
            "render_width": record.get("render_width"),
            "render_height": record.get("render_height"),
            "reused_verified_artifact": True,
        },
        adopted=True,
    )


def _existing_render(
    database_url: str,
    context: PageJobContext,
    workspace_root: Path,
) -> StageExecution | None:
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            """
            SELECT derivative.image_uri, derivative.image_sha256,
                   derivative.render_manifest_uri, derivative.evidence_tier
            FROM archive.page page
            JOIN archive.volume volume USING (volume_id)
            JOIN archive.page_derivative derivative USING (page_id)
            WHERE volume.volume_number = %s AND page.page_number = %s
              AND derivative.preference_rank >= 20
              AND derivative.render_manifest_uri IS NOT NULL
            ORDER BY derivative.preference_rank DESC, derivative.width DESC,
                     derivative.height DESC, derivative.created_at
            """,
            (context.volume_number, context.page_number),
        ).fetchall()
    for row in rows:
        try:
            manifest_path = resolve_workspace_path(
                workspace_root, row["render_manifest_uri"]
            )
            records = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            matches = [
                record
                for record in records
                if record.get("status") == "rendered"
                and record.get("source_uri") == context.source_uri
                and record.get("volume_number") == context.volume_number
                and record.get("page_number") == context.page_number
                and record.get("render_sha256") == row["image_sha256"]
            ]
            if len(matches) != 1:
                continue
            return _render_manifest_execution(
                manifest_path, context, workspace_root
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def execute_render(
    database_url: str,
    context: PageJobContext,
    workspace_root: Path,
    *,
    cache_dir: Path,
    offline: bool,
    profile: str | None,
    credentials_csv: Path | None,
    region: str,
) -> StageExecution:
    job_manifest = stage_output_dir(context, workspace_root) / "lossless_manifest.jsonl"
    if job_manifest.is_file():
        return _render_manifest_execution(job_manifest, context, workspace_root)
    adopted = _existing_render(database_url, context, workspace_root)
    if adopted:
        return adopted
    output_dir = stage_output_dir(context, workspace_root)
    selection = ingestion_candidate(
        source_uri=context.source_uri,
        source_key=context.object_key,
        volume_number=context.volume_number,
        publication_year=context.publication_year,
        page_number=context.page_number,
        job_id=context.job_id,
    )
    extension = Path(context.object_key).suffix.lower()
    record = {
        "bucket": context.bucket,
        "key": context.object_key,
        "source_uri": context.source_uri,
        "extension": extension,
        "media_type": context.media_type,
        "size_bytes": context.size_bytes,
        "etag": context.etag,
        "full_sha256": context.source_sha256,
        "integrity_status": context.integrity_status,
        "volume_number": context.volume_number,
        "publication_year": context.publication_year,
        "page_count": context.page_count,
    }
    client = None
    if not offline:
        client = build_s3_client(profile, credentials_csv, region)
    results = render_lossless_plan(
        [selection],
        [record],
        cache_dir,
        output_dir,
        client=client,
        bucket=context.bucket,
    )
    result = results[0]
    if result.get("status") != "rendered":
        raise RuntimeError(result.get("issue", "lossless render failed"))
    if context.source_sha256 and result["source_object_sha256"] != context.source_sha256:
        raise ValueError("Rendered source SHA-256 conflicts with the authoritative catalog")
    write_lossless_results(output_dir, results)
    manifest_path = output_dir / "lossless_manifest.jsonl"
    return StageExecution(
        artifact_uri=workspace_artifact_uri(workspace_root, manifest_path),
        output_sha256=sha256_file(manifest_path),
        result={
            "render_sha256": result["render_sha256"],
            "source_object_sha256": result["source_object_sha256"],
            "decoded_pixel_sha256": result.get("decoded_pixel_sha256"),
            "render_width": result.get("render_width"),
            "render_height": result.get("render_height"),
            "reused_verified_artifact": False,
        },
        adopted=False,
    )


def _parent_manifest(
    context: PageJobContext, workspace_root: Path
) -> tuple[Path, dict[str, Any], Path]:
    if context.parent_stage != "render_lossless" or not context.parent_artifact_uri:
        raise ValueError("OCR jobs require one completed render_lossless parent")
    manifest_path = resolve_workspace_path(workspace_root, context.parent_artifact_uri)
    if context.parent_output_sha256 != sha256_file(manifest_path):
        raise ValueError("Render parent artifact checksum no longer matches its job")
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        record
        for record in records
        if record.get("status") == "rendered"
        and record.get("source_uri") == context.source_uri
        and record.get("volume_number") == context.volume_number
        and record.get("publication_year") == context.publication_year
        and record.get("page_number") == context.page_number
    ]
    if len(matches) != 1:
        raise ValueError("Render parent manifest does not contain exactly one job page")
    image_path = _resolve_record_image(workspace_root, matches[0])
    return manifest_path, matches[0], image_path


def _ocr_artifact_matches(
    artifact: OCRPageArtifact,
    context: PageJobContext,
    image_sha256: str,
) -> bool:
    configuration = context.configuration
    return (
        artifact.source.source_uri == context.source_uri
        and artifact.source.volume_number == context.volume_number
        and artifact.source.publication_year == context.publication_year
        and artifact.source.page_number == context.page_number
        and artifact.image_sha256 == image_sha256
        and artifact.run.engine == configuration.get("engine")
        and artifact.run.model_name == configuration.get("model")
        and artifact.run.model_revision == configuration.get("revision")
        and artifact.run.configuration.get("language") == configuration.get("language")
        and artifact.run.configuration.get("tile_size") == configuration.get("tile_size")
        and artifact.run.configuration.get("overlap") == configuration.get("overlap")
    )


def _existing_ocr(
    database_url: str,
    context: PageJobContext,
    workspace_root: Path,
    image_sha256: str,
) -> StageExecution | None:
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            """
            SELECT input.artifact_uri
            FROM evidence.ocr_run_input input
            JOIN evidence.processing_run run USING (run_id)
            JOIN archive.page page USING (page_id)
            JOIN archive.volume volume USING (volume_id)
            JOIN archive.page_derivative derivative USING (derivative_id)
            WHERE volume.volume_number = %s AND page.page_number = %s
              AND derivative.image_sha256 = %s
              AND run.status = 'completed' AND run.kind = 'ocr'
            ORDER BY run.completed_at, run.run_id
            """,
            (context.volume_number, context.page_number, image_sha256),
        ).fetchall()
    for row in rows:
        try:
            artifact_path = resolve_workspace_path(workspace_root, row["artifact_uri"])
            artifact = OCRPageArtifact.model_validate_json(
                artifact_path.read_text(encoding="utf-8")
            )
            if not _ocr_artifact_matches(artifact, context, image_sha256):
                continue
            stored = ingest_ocr_artifact(database_url, artifact_path)
            return StageExecution(
                artifact_uri=workspace_artifact_uri(workspace_root, artifact_path),
                output_sha256=sha256_file(artifact_path),
                result={
                    "ocr_run_id": stored.run_id,
                    "regions": stored.regions_verified,
                    "derivative_id": stored.derivative_id,
                    "reused_verified_artifact": True,
                },
                adopted=True,
            )
        except (OSError, ValueError):
            continue
    return None


def execute_ocr(
    database_url: str,
    context: PageJobContext,
    workspace_root: Path,
) -> StageExecution:
    supported = {
        "engine": "PaddleOCR",
        "model": "PP-OCRv6_medium_det+PP-OCRv6_medium_rec",
        "revision": "paddleocr-3.7.0-official",
    }
    for field, expected in supported.items():
        if context.configuration.get(field) != expected:
            raise ValueError(f"OCR worker does not implement {field}={context.configuration.get(field)!r}")
    manifest_path, render, image_path = _parent_manifest(context, workspace_root)
    source_sha256, _ = resolve_render_provenance(
        image_path,
        manifest_path,
        source_uri=context.source_uri,
        page_number=context.page_number,
        volume_number=context.volume_number,
        publication_year=context.publication_year,
        supplied_source_sha256=context.source_sha256,
        artifact_root=workspace_root,
    )
    output_path = stage_output_dir(context, workspace_root) / "ocr.json"
    if output_path.is_file():
        artifact = OCRPageArtifact.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
        if not _ocr_artifact_matches(artifact, context, render["render_sha256"]):
            raise ValueError("Existing job OCR artifact contradicts its immutable plan")
        stored = ingest_ocr_artifact(database_url, output_path)
        return StageExecution(
            artifact_uri=workspace_artifact_uri(workspace_root, output_path),
            output_sha256=sha256_file(output_path),
            result={
                "ocr_run_id": stored.run_id,
                "regions": stored.regions_verified,
                "derivative_id": stored.derivative_id,
                "reused_verified_artifact": True,
            },
            adopted=True,
        )
    adopted = _existing_ocr(
        database_url, context, workspace_root, render["render_sha256"]
    )
    if adopted:
        return adopted
    arguments = [
        "--image", str(image_path),
        "--source-uri", context.source_uri,
        "--source-sha256", source_sha256,
        "--render-manifest", str(manifest_path),
        "--page", str(context.page_number),
        "--volume", str(context.volume_number),
        "--year", str(context.publication_year),
        "--language", str(context.configuration["language"]),
        "--tile-size", str(context.configuration["tile_size"]),
        "--overlap", str(context.configuration["overlap"]),
        "--worker-batch-size", str(context.configuration.get("worker_batch_size", 5)),
        "--output", str(output_path),
    ]
    if context.configuration.get("worker_mode") == "reuse_model":
        arguments.append("--reuse-model")
    elif context.configuration.get("worker_mode") == "isolate_tiles":
        arguments.append("--isolate-tiles")
    exit_code = ocr_main(arguments)
    if exit_code:
        raise RuntimeError(f"OCR command exited with status {exit_code}")
    stored = ingest_ocr_artifact(database_url, output_path)
    return StageExecution(
        artifact_uri=workspace_artifact_uri(workspace_root, output_path),
        output_sha256=sha256_file(output_path),
        result={
            "ocr_run_id": stored.run_id,
            "regions": stored.regions_verified,
            "derivative_id": stored.derivative_id,
            "reused_verified_artifact": False,
        },
        adopted=False,
    )


def _parent_ocr(
    context: PageJobContext, workspace_root: Path
) -> tuple[Path, OCRPageArtifact]:
    if context.parent_stage != "ocr" or not context.parent_artifact_uri:
        raise ValueError(f"{context.stage} jobs require one completed OCR parent")
    artifact_path = resolve_workspace_path(workspace_root, context.parent_artifact_uri)
    if context.parent_output_sha256 != sha256_file(artifact_path):
        raise ValueError("OCR parent artifact checksum no longer matches its job")
    artifact = OCRPageArtifact.model_validate_json(
        artifact_path.read_text(encoding="utf-8")
    )
    expected_run_id = str((context.parent_result or {}).get("ocr_run_id", ""))
    if str(artifact.run.run_id) != expected_run_id:
        raise ValueError("OCR parent result disagrees with its artifact run UUID")
    return artifact_path, artifact


def _embedding_run(
    database_url: str,
    source_ocr_run_id: str,
    model: str,
    revision: str,
) -> tuple[str | None, int, int]:
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        region_count = connection.execute(
            "SELECT count(*) AS count FROM evidence.ocr_region WHERE run_id = %s",
            (source_ocr_run_id,),
        ).fetchone()["count"]
        rows = connection.execute(
            """
            SELECT embedding.run_id, count(*) AS count
            FROM retrieval.embedding embedding
            JOIN evidence.ocr_region region
              ON embedding.target_kind = 'region'
             AND embedding.target_id = region.region_id
            JOIN evidence.processing_run run ON run.run_id = embedding.run_id
            WHERE region.run_id = %s AND embedding.model_name = %s
              AND embedding.model_revision = %s AND run.status = 'completed'
            GROUP BY embedding.run_id
            ORDER BY count(*) DESC, embedding.run_id
            """,
            (source_ocr_run_id, model, revision),
        ).fetchall()
    total = sum(row["count"] for row in rows)
    complete = next((row for row in rows if row["count"] == region_count), None)
    return (str(complete["run_id"]) if complete else None, region_count, total)


def execute_embedding(
    database_url: str,
    context: PageJobContext,
    workspace_root: Path,
) -> StageExecution:
    _, ocr = _parent_ocr(context, workspace_root)
    model = context.configuration.get("model")
    revision = context.configuration.get("revision")
    if not isinstance(model, str) or not isinstance(revision, str):
        raise ValueError("Embedding plan must pin model and revision")
    run_id, region_count, total_existing = _embedding_run(
        database_url, str(ocr.run.run_id), model, revision
    )
    adopted = run_id is not None
    inserted = 0
    if run_id is None:
        if total_existing:
            raise ValueError(
                "Partial embeddings already exist for this OCR/model revision; repair before retry"
            )
        generated = embed_regions(
            database_url,
            model_name=model,
            model_revision=revision,
            batch_size=int(context.configuration.get("batch_size", 16)),
            source_ocr_run_id=str(ocr.run.run_id),
        )
        if generated.regions_processed != generated.embeddings_inserted:
            raise ValueError("Fresh embedding run did not persist every OCR region")
        run_id = generated.run_id
        inserted = generated.embeddings_inserted
    receipt_path = stage_output_dir(context, workspace_root) / "embedding-receipt.json"
    _atomic_json(
        receipt_path,
        {
            "schema_version": "1.0",
            "job_id": context.job_id,
            "source_ocr_run_id": str(ocr.run.run_id),
            "embedding_run_id": run_id,
            "model": model,
            "revision": revision,
            "embeddings": region_count,
            "inserted_by_job": inserted,
            "reused_verified_artifact": adopted,
        },
    )
    return StageExecution(
        artifact_uri=workspace_artifact_uri(workspace_root, receipt_path),
        output_sha256=sha256_file(receipt_path),
        result={
            "embedding_run_id": run_id,
            "embeddings": region_count,
            "source_ocr_run_id": str(ocr.run.run_id),
            "reused_verified_artifact": adopted,
        },
        adopted=adopted,
    )


def _ner_artifact_matches(
    artifact: NERArtifact,
    context: PageJobContext,
    source_ocr_run_id: str,
) -> bool:
    configuration = context.configuration
    run_configuration = artifact.run.configuration
    expected_name = f"{configuration.get('model')}+historical-women-zh-rules"
    expected_revision = f"{configuration.get('revision')}+rules-v1"
    return (
        str(artifact.source_ocr_run_id) == source_ocr_run_id
        and artifact.input_variant == configuration.get("input_variant")
        and artifact.ontology_version == configuration.get("ontology_version")
        and artifact.adapter_id == configuration.get("adapter")
        and artifact.run.model_name == expected_name
        and artifact.run.model_revision == expected_revision
        and run_configuration.get("threshold") == configuration.get("threshold")
        and run_configuration.get("batch_size") == configuration.get("batch_size")
        and run_configuration.get("word_splitter_language")
        == configuration.get("word_splitter_language")
        and run_configuration.get("max_regions") == configuration.get("max_regions")
        and run_configuration.get("flat_ner") == configuration.get("flat_ner")
        and run_configuration.get("multi_label") == configuration.get("multi_label")
    )


def _existing_ner(
    database_url: str,
    context: PageJobContext,
    workspace_root: Path,
    source_ocr_run_id: str,
) -> StageExecution | None:
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            """
            SELECT input.artifact_uri
            FROM evidence.ner_run_input input
            JOIN evidence.processing_run run USING (run_id)
            WHERE input.source_ocr_run_id = %s AND run.status = 'completed'
            ORDER BY run.completed_at, run.run_id
            """,
            (source_ocr_run_id,),
        ).fetchall()
    for row in rows:
        try:
            artifact_path = resolve_workspace_path(workspace_root, row["artifact_uri"])
            artifact = NERArtifact.model_validate_json(
                artifact_path.read_text(encoding="utf-8")
            )
            if not _ner_artifact_matches(artifact, context, source_ocr_run_id):
                continue
            stored = ingest_ner_artifact(database_url, artifact_path)
            return StageExecution(
                artifact_uri=workspace_artifact_uri(workspace_root, artifact_path),
                output_sha256=sha256_file(artifact_path),
                result={
                    "ner_run_id": stored.run_id,
                    "mentions": stored.mentions_verified,
                    "candidate_only": True,
                    "bounded_regions": context.configuration.get("max_regions"),
                    "source_ocr_run_id": source_ocr_run_id,
                    "reused_verified_artifact": True,
                },
                adopted=True,
            )
        except (OSError, ValueError):
            continue
    return None


def execute_ner(
    database_url: str,
    context: PageJobContext,
    workspace_root: Path,
) -> StageExecution:
    ocr_path, ocr = _parent_ocr(context, workspace_root)
    source_ocr_run_id = str(ocr.run.run_id)
    configuration = context.configuration
    if configuration.get("adapter") != "rules+gliner":
        raise ValueError("NER worker currently implements only rules+gliner")
    output_path = stage_output_dir(context, workspace_root) / "ner.json"
    if output_path.is_file():
        artifact = NERArtifact.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
        if not _ner_artifact_matches(artifact, context, source_ocr_run_id):
            raise ValueError("Existing job NER artifact contradicts its immutable plan")
        stored = ingest_ner_artifact(database_url, output_path)
        return StageExecution(
            artifact_uri=workspace_artifact_uri(workspace_root, output_path),
            output_sha256=sha256_file(output_path),
            result={
                "ner_run_id": stored.run_id,
                "mentions": stored.mentions_verified,
                "candidate_only": True,
                "bounded_regions": configuration.get("max_regions"),
                "source_ocr_run_id": source_ocr_run_id,
                "reused_verified_artifact": True,
            },
            adopted=True,
        )
    adopted = _existing_ner(
        database_url, context, workspace_root, source_ocr_run_id
    )
    if adopted:
        return adopted
    arguments = [
        "--ocr-artifact", str(ocr_path),
        "--output", str(output_path),
        "--model", str(configuration["model"]),
        "--revision", str(configuration["revision"]),
        "--threshold", str(configuration["threshold"]),
        "--batch-size", str(configuration["batch_size"]),
        "--word-splitter-language", str(configuration["word_splitter_language"]),
    ]
    if configuration.get("dataset_id") is not None:
        arguments.extend(["--dataset-id", str(configuration["dataset_id"])])
    if configuration.get("split_id") is not None:
        arguments.extend(["--split-id", str(configuration["split_id"])])
    max_regions = configuration.get("max_regions")
    if max_regions is not None:
        arguments.extend(["--max-regions", str(max_regions)])
    if configuration.get("flat_ner"):
        arguments.append("--flat-ner")
    if not configuration.get("multi_label"):
        arguments.append("--single-label")
    exit_code = ner_main(arguments)
    if exit_code:
        raise RuntimeError(f"NER command exited with status {exit_code}")
    stored = ingest_ner_artifact(database_url, output_path)
    return StageExecution(
        artifact_uri=workspace_artifact_uri(workspace_root, output_path),
        output_sha256=sha256_file(output_path),
        result={
            "ner_run_id": stored.run_id,
            "mentions": stored.mentions_verified,
            "candidate_only": True,
            "bounded_regions": max_regions,
            "source_ocr_run_id": source_ocr_run_id,
            "reused_verified_artifact": False,
        },
        adopted=False,
    )


def execute_stage(
    database_url: str,
    context: PageJobContext,
    workspace_root: Path,
    *,
    cache_dir: Path,
    offline: bool,
    profile: str | None,
    credentials_csv: Path | None,
    region: str,
) -> StageExecution:
    if context.stage == "render_lossless":
        return execute_render(
            database_url,
            context,
            workspace_root,
            cache_dir=cache_dir,
            offline=offline,
            profile=profile,
            credentials_csv=credentials_csv,
            region=region,
        )
    if context.stage == "ocr":
        return execute_ocr(database_url, context, workspace_root)
    if context.stage == "embedding":
        return execute_embedding(database_url, context, workspace_root)
    if context.stage == "ner":
        return execute_ner(database_url, context, workspace_root)
    raise ValueError(f"No page worker implementation exists for stage {context.stage}")


class LeaseHeartbeat:
    def __init__(
        self,
        database_url: str,
        job_id: UUID,
        worker_id: str,
        lease_seconds: int,
    ):
        self.database_url = database_url
        self.job_id = job_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.interval = max(10, lease_seconds // 3)
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                heartbeat_job(
                    self.database_url,
                    self.job_id,
                    self.worker_id,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as exc:  # pragma: no cover - timing dependent
                self._error = exc
                self._stop.set()

    def __enter__(self) -> LeaseHeartbeat:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def raise_if_failed(self) -> None:
        if self._error:
            raise RuntimeError(f"Lease heartbeat failed: {self._error}") from self._error


def run_one(
    database_url: str,
    *,
    worker_id: str,
    workspace_root: Path,
    cache_dir: Path,
    offline: bool = False,
    profile: str | None = None,
    credentials_csv: Path | None = None,
    region: str = "us-east-1",
    lease_seconds: int = 900,
    retry_delay_seconds: int = 60,
    stage: str | None = None,
    batch_id: UUID | None = None,
    executor: Callable[..., StageExecution] = execute_stage,
) -> WorkerRunResult:
    lease: JobLease | None = claim_job(
        database_url,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        stage=stage,
        batch_id=batch_id,
    )
    if lease is None:
        return WorkerRunResult(None, stage, "idle")
    job_id = UUID(lease.job_id)
    start_job(database_url, job_id, worker_id)
    try:
        context = load_job_context(database_url, job_id)
        with LeaseHeartbeat(
            database_url, job_id, worker_id, lease_seconds
        ) as heartbeat:
            execution = executor(
                database_url,
                context,
                workspace_root.resolve(),
                cache_dir=cache_dir,
                offline=offline,
                profile=profile,
                credentials_csv=credentials_csv,
                region=region,
            )
            heartbeat.raise_if_failed()
        transition = complete_job(
            database_url,
            job_id,
            worker_id,
            artifact_uri=execution.artifact_uri,
            output_sha256=execution.output_sha256,
            result=execution.result,
        )
        return WorkerRunResult(
            lease.job_id,
            lease.stage,
            transition.status,
            execution.adopted,
            execution.artifact_uri,
        )
    except Exception as exc:
        transition = fail_job(
            database_url,
            job_id,
            worker_id,
            error_type=type(exc).__name__,
            message=str(exc),
            retry_delay_seconds=retry_delay_seconds,
        )
        return WorkerRunResult(
            lease.job_id,
            lease.stage,
            transition.status,
            error_type=type(exc).__name__,
            message=str(exc),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--worker", required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/wic-source-cache"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--credentials-csv", type=Path)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--lease-seconds", type=int, default=900)
    parser.add_argument("--retry-delay-seconds", type=int, default=60)
    parser.add_argument(
        "--stage", choices=("render_lossless", "ocr", "embedding", "ner")
    )
    parser.add_argument("--batch-id", type=UUID)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    result = run_one(
        args.database_url,
        worker_id=args.worker,
        workspace_root=args.workspace_root,
        cache_dir=args.cache_dir,
        offline=args.offline,
        profile=args.profile,
        credentials_csv=args.credentials_csv,
        region=args.region,
        lease_seconds=args.lease_seconds,
        retry_delay_seconds=args.retry_delay_seconds,
        stage=args.stage,
        batch_id=args.batch_id,
    )
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 1 if result.status in {"pending", "failed"} else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
