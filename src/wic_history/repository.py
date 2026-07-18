"""Transactional loaders for the authoritative PostgreSQL evidence store.

The repository stores observed archive metadata and immutable processing
artifacts. OpenSearch and Neo4j are deliberately excluded: they are rebuildable
projections, never alternate sources of historical truth.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .evidence import NERArtifact, OCRPageArtifact


@dataclass(frozen=True, slots=True)
class ManifestIngestResult:
    objects_processed: int
    volumes_processed: int


@dataclass(frozen=True, slots=True)
class OCRIngestResult:
    artifact_id: str
    page_id: str
    run_id: str
    regions_verified: int


def _psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.types.json import Jsonb
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    return psycopg, Jsonb


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc


def ingest_manifest(database_url: str, manifest_path: Path) -> ManifestIngestResult:
    """Upsert a corpus audit manifest without manufacturing page records."""
    psycopg, Jsonb = _psycopg()
    object_count = 0
    volume_count = 0
    with psycopg.connect(database_url) as connection:
        for record in read_jsonl(manifest_path):
            required = {
                "source_uri",
                "media_type",
                "size_bytes",
                "integrity_status",
                "bucket",
                "key",
            }
            missing = sorted(required - record.keys())
            if missing:
                raise ValueError(f"Manifest record missing fields: {', '.join(missing)}")
            details = {
                key: record.get(key)
                for key in (
                    "schema_version",
                    "extension",
                    "last_modified",
                    "storage_class",
                    "etag_is_simple_md5_candidate",
                    "integrity_checks",
                    "issues",
                    "full_sha256_status",
                    "page_count_status",
                    "text_layer_status",
                )
            }
            source_object_id = connection.execute(
                """
                INSERT INTO archive.source_object (
                    source_uri, bucket, object_key, media_type, size_bytes, etag,
                    sha256, integrity_status, integrity_details, observed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (source_uri) DO UPDATE SET
                    bucket = EXCLUDED.bucket,
                    object_key = EXCLUDED.object_key,
                    media_type = EXCLUDED.media_type,
                    size_bytes = EXCLUDED.size_bytes,
                    etag = EXCLUDED.etag,
                    sha256 = COALESCE(EXCLUDED.sha256, archive.source_object.sha256),
                    integrity_status = EXCLUDED.integrity_status,
                    integrity_details = EXCLUDED.integrity_details,
                    observed_at = EXCLUDED.observed_at
                RETURNING source_object_id
                """,
                (
                    record["source_uri"],
                    record["bucket"],
                    record["key"],
                    record["media_type"],
                    int(record["size_bytes"]),
                    record.get("etag"),
                    record.get("full_sha256"),
                    record["integrity_status"],
                    Jsonb(details),
                ),
            ).fetchone()[0]
            object_count += 1

            volume_number = record.get("volume_number")
            if volume_number is None:
                continue
            publication_year = record.get("publication_year")
            if publication_year is None:
                raise ValueError(f"Volume {volume_number} has no publication_year")
            volume_metadata = {
                "manifest_schema_version": record.get("schema_version"),
                "page_count_status": record.get("page_count_status"),
                "text_layer_status": record.get("text_layer_status"),
            }
            connection.execute(
                """
                INSERT INTO archive.volume (
                    source_object_id, volume_number, publication_year, page_count, metadata
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (volume_number) DO UPDATE SET
                    source_object_id = EXCLUDED.source_object_id,
                    publication_year = EXCLUDED.publication_year,
                    page_count = EXCLUDED.page_count,
                    metadata = EXCLUDED.metadata
                """,
                (
                    source_object_id,
                    int(volume_number),
                    int(publication_year),
                    record.get("page_count"),
                    Jsonb(volume_metadata),
                ),
            )
            volume_count += 1
    return ManifestIngestResult(object_count, volume_count)


def _verify_run(connection: Any, artifact: Any, Jsonb: Any) -> None:
    run = artifact.run
    status = "completed" if run.completed_at else "running"
    connection.execute(
        """
        INSERT INTO evidence.processing_run (
            run_id, kind, engine, model_name, model_revision, software_version,
            configuration, status, started_at, completed_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO NOTHING
        """,
        (
            run.run_id,
            run.kind.value,
            run.engine,
            run.model_name,
            run.model_revision,
            run.software_version,
            Jsonb(run.configuration),
            status,
            run.started_at,
            run.completed_at,
        ),
    )
    stored = connection.execute(
        """
        SELECT kind, engine, model_name, model_revision, software_version,
               configuration, status, started_at, completed_at
        FROM evidence.processing_run WHERE run_id = %s
        """,
        (run.run_id,),
    ).fetchone()
    expected = (
        run.kind.value,
        run.engine,
        run.model_name,
        run.model_revision,
        run.software_version,
        run.configuration,
        status,
        run.started_at,
        run.completed_at,
    )
    if stored != expected:
        raise ValueError(f"Processing run UUID {run.run_id} already has different provenance")


def _region_record(region: Any) -> tuple[Any, ...]:
    return (
        region.region_id,
        region.parent_region_id,
        region.kind.value,
        region.reading_order,
        region.polygon.model_dump(mode="json"),
        region.raw_text,
        region.normalized_text,
        region.confidence,
        region.language,
        region.direction,
        region.engine_payload,
    )


def ingest_ocr_artifact(database_url: str, artifact_path: Path) -> OCRIngestResult:
    """Validate and atomically store a coordinate-preserving OCR artifact."""
    psycopg, Jsonb = _psycopg()
    artifact = OCRPageArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    source = artifact.source
    if source.volume_number is None:
        raise ValueError("OCR artifacts for this corpus require volume_number")

    with psycopg.connect(database_url) as connection:
        volume = connection.execute(
            """
            SELECT v.volume_id, v.publication_year, s.source_uri
            FROM archive.volume v
            JOIN archive.source_object s USING (source_object_id)
            WHERE v.volume_number = %s
            """,
            (source.volume_number,),
        ).fetchone()
        if volume is None:
            raise ValueError(
                f"Volume {source.volume_number} is absent; ingest the corpus manifest first"
            )
        volume_id, publication_year, stored_uri = volume
        if stored_uri != source.source_uri or (
            source.publication_year is not None and publication_year != source.publication_year
        ):
            raise ValueError("OCR source pointer disagrees with the authoritative volume record")

        _verify_run(connection, artifact, Jsonb)
        page_metadata = {
            "ocr_artifact_id": str(artifact.artifact_id),
            "artifact_schema_version": artifact.schema_version,
            "warnings": artifact.warnings,
        }
        page_id = connection.execute(
            """
            INSERT INTO archive.page (
                volume_id, page_number, source_image_uri, source_image_sha256,
                width, height, dpi, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (volume_id, page_number) DO UPDATE SET
                source_image_uri = EXCLUDED.source_image_uri,
                source_image_sha256 = EXCLUDED.source_image_sha256,
                width = EXCLUDED.width,
                height = EXCLUDED.height,
                dpi = EXCLUDED.dpi,
                metadata = EXCLUDED.metadata
            RETURNING page_id
            """,
            (
                volume_id,
                source.page_number,
                artifact.image_uri,
                artifact.image_sha256,
                artifact.width,
                artifact.height,
                artifact.dpi,
                Jsonb(page_metadata),
            ),
        ).fetchone()[0]

        rows = [_region_record(region) for region in artifact.regions]
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO evidence.ocr_region (
                    region_id, page_id, parent_region_id, run_id, region_kind,
                    reading_order, polygon, raw_text, normalized_text, confidence,
                    language, direction, engine_payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (region_id) DO NOTHING
                """,
                [
                    (
                        row[0], page_id, row[1], artifact.run.run_id, row[2], row[3],
                        Jsonb(row[4]), row[5], row[6], row[7], row[8], row[9], Jsonb(row[10]),
                    )
                    for row in rows
                ],
            )

        stored_rows = connection.execute(
            """
            SELECT region_id, parent_region_id, region_kind, reading_order, polygon,
                   raw_text, normalized_text, confidence, language, direction, engine_payload
            FROM evidence.ocr_region
            WHERE page_id = %s AND run_id = %s
            ORDER BY reading_order
            """,
            (page_id, artifact.run.run_id),
        ).fetchall()
        if stored_rows != rows:
            raise ValueError(
                "Stored OCR regions differ from the artifact; evidence rows are immutable"
            )

    return OCRIngestResult(
        artifact_id=str(artifact.artifact_id),
        page_id=str(page_id),
        run_id=str(artifact.run.run_id),
        regions_verified=len(rows),
    )


@dataclass(frozen=True, slots=True)
class NERIngestResult:
    artifact_id: str
    run_id: str
    mentions_verified: int


def ingest_ner_artifact(database_url: str, artifact_path: Path) -> NERIngestResult:
    """Store exact-offset NER candidates without creating canonical entities."""
    psycopg, Jsonb = _psycopg()
    artifact = NERArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    region_ids = [mention.source.region_id for mention in artifact.mentions]
    with psycopg.connect(database_url) as connection:
        _verify_run(connection, artifact, Jsonb)
        region_rows = connection.execute(
            """
            SELECT r.region_id, r.run_id, r.raw_text, p.page_number,
                   v.volume_number, v.publication_year, s.source_uri
            FROM evidence.ocr_region r
            JOIN archive.page p USING (page_id)
            JOIN archive.volume v USING (volume_id)
            JOIN archive.source_object s USING (source_object_id)
            WHERE r.region_id = ANY(%s)
            """,
            (region_ids,),
        ).fetchall() if region_ids else []
        sources = {row[0]: row[1:] for row in region_rows}
        expected_ids = {region_id for region_id in region_ids if region_id is not None}
        if sources.keys() != expected_ids:
            missing = sorted(str(value) for value in expected_ids - sources.keys())
            raise ValueError(f"NER artifact references unknown OCR regions: {', '.join(missing)}")

        rows = []
        for mention in artifact.mentions:
            source = mention.source
            ocr_run_id, raw_text, page_number, volume_number, publication_year, source_uri = sources[
                source.region_id
            ]
            if ocr_run_id != artifact.source_ocr_run_id:
                raise ValueError("NER artifact source_ocr_run_id does not match its OCR region")
            if (
                page_number != source.page_number
                or volume_number != source.volume_number
                or publication_year != source.publication_year
                or source_uri != source.source_uri
            ):
                raise ValueError("NER source pointer disagrees with the OCR evidence record")
            if source.text_start is None or source.text_end is None:
                raise ValueError("NER mentions require exact source text offsets")
            if raw_text[source.text_start : source.text_end] != mention.text:
                raise ValueError("NER mention text does not match the cited OCR character span")
            rows.append(
                (
                    mention.mention_id,
                    source.region_id,
                    mention.run_id,
                    mention.entity_type.value,
                    mention.text,
                    mention.normalized_text,
                    source.text_start,
                    source.text_end,
                    source.polygon.model_dump(mode="json") if source.polygon else None,
                    mention.confidence,
                    mention.attributes,
                )
            )

        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO evidence.entity_mention (
                    mention_id, region_id, run_id, entity_type, mention_text,
                    normalized_text, text_start, text_end, polygon, confidence,
                    mention_status, attributes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'candidate', %s)
                ON CONFLICT (mention_id) DO NOTHING
                """,
                [
                    (*row[:8], Jsonb(row[8]) if row[8] else None, row[9], Jsonb(row[10]))
                    for row in rows
                ],
            )
        stored = connection.execute(
            """
            SELECT mention_id, region_id, run_id, entity_type, mention_text,
                   normalized_text, text_start, text_end, polygon, confidence, attributes
            FROM evidence.entity_mention WHERE run_id = %s ORDER BY mention_id
            """,
            (artifact.run.run_id,),
        ).fetchall()
        expected = sorted(rows, key=lambda row: row[0])
        if stored != expected:
            raise ValueError(
                "Stored NER candidates differ from the artifact; evidence rows are immutable"
            )
    return NERIngestResult(str(artifact.artifact_id), str(artifact.run.run_id), len(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest", help="Load a corpus manifest JSONL")
    manifest.add_argument("path", type=Path)
    ocr = subparsers.add_parser("ocr", help="Load one or more OCR artifact JSON files")
    ocr.add_argument("paths", type=Path, nargs="+")
    ner = subparsers.add_parser("ner", help="Load one or more NER candidate artifacts")
    ner.add_argument("paths", type=Path, nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    if args.command == "manifest":
        result: Any = ingest_manifest(args.database_url, args.path)
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 0
    for path in args.paths:
        result = (
            ingest_ocr_artifact(args.database_url, path)
            if args.command == "ocr"
            else ingest_ner_artifact(args.database_url, path)
        )
        print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
