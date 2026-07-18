"""Build a read-only, integrity-aware S3 corpus manifest.

The fast audit deliberately avoids full-object downloads. It validates container
signatures and, where meaningful, trailer markers using bounded S3 range reads.
Full content hashes, page counts, and text-layer inspection belong to the deep
audit stage and are therefore explicitly left unverified here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = "1.0"
DEFAULT_BUCKET = "ccaa-us-east-1-504133794192"
DEFAULT_PREFIX = "sb_raw/"
HEADER_BYTES = 64
TRAILER_BYTES = 65_536
XREF_BYTES = 2 * 1024 * 1024
OBJECT_BYTES = 32_768

# Transcribed from the archive's `申报分年一览表.jpg`. A volume maps to
# exactly one publication year. The index distinguishes editions in 1938/1939,
# but the year remains sufficient for the Phase 0 manifest.
YEAR_VOLUME_RANGES: tuple[tuple[int, int, int], ...] = (
    (1872, 1, 1), (1873, 2, 3), (1874, 4, 5), (1875, 6, 7),
    (1876, 8, 9), (1877, 10, 11), (1878, 12, 13), (1879, 14, 15),
    (1880, 16, 17), (1881, 18, 19), (1882, 20, 21), (1883, 22, 23),
    (1884, 24, 25), (1885, 26, 27), (1886, 28, 29), (1887, 30, 31),
    (1888, 32, 33), (1889, 34, 35), (1890, 36, 37), (1891, 38, 39),
    (1892, 40, 42), (1893, 43, 45), (1894, 46, 48), (1895, 49, 51),
    (1896, 52, 54), (1897, 55, 57), (1898, 58, 60), (1899, 61, 63),
    (1900, 64, 66), (1901, 67, 69), (1902, 70, 72), (1903, 73, 75),
    (1904, 76, 78), (1905, 79, 81), (1906, 82, 85), (1907, 86, 91),
    (1908, 92, 97), (1909, 98, 103), (1910, 104, 109), (1911, 110, 115),
    (1912, 116, 119), (1913, 120, 125), (1914, 126, 131),
    (1915, 132, 137), (1916, 138, 143), (1917, 144, 149),
    (1918, 150, 155), (1919, 156, 161), (1920, 162, 167),
    (1921, 168, 176), (1922, 177, 187), (1923, 188, 198),
    (1924, 199, 208), (1925, 209, 219), (1926, 220, 230),
    (1927, 231, 241), (1928, 242, 253), (1929, 254, 265),
    (1930, 266, 277), (1931, 278, 289), (1932, 290, 299),
    (1933, 300, 311), (1934, 312, 323), (1935, 324, 335),
    (1936, 336, 347), (1937, 348, 355), (1938, 356, 357),
    (1939, 358, 358), (1938, 359, 360), (1939, 361, 367),
    (1940, 368, 373), (1941, 374, 378),
    (1942, 379, 382), (1943, 383, 384), (1944, 385, 386),
    (1945, 387, 387), (1946, 388, 391), (1947, 392, 395),
    (1948, 396, 399), (1949, 400, 400),
)

VOLUME_RE = re.compile(r"申报影印本(?P<volume>\d+)\.(?:pdf|djvu)$", re.IGNORECASE)


@dataclass(slots=True)
class ManifestRecord:
    schema_version: str
    bucket: str
    key: str
    source_uri: str
    size_bytes: int
    etag: str | None
    etag_is_simple_md5_candidate: bool
    last_modified: str | None
    storage_class: str | None
    extension: str
    media_type: str
    volume_number: int | None
    publication_year: int | None
    integrity_status: str
    integrity_checks: dict[str, bool | None]
    issues: list[str] = field(default_factory=list)
    full_sha256: str | None = None
    full_sha256_status: str = "not_computed_fast_audit"
    page_count: int | None = None
    page_count_status: str = "not_computed_fast_audit"
    text_layer_status: str = "not_inspected_fast_audit"


def publication_year_for_volume(volume: int | None) -> int | None:
    """Return the indexed publication year for a numbered archive volume."""
    if volume is None:
        return None
    matches = [year for year, start, end in YEAR_VOLUME_RANGES if start <= volume <= end]
    if not matches:
        return None
    return matches[0]


def parse_volume_number(key: str) -> int | None:
    match = VOLUME_RE.search(key.rsplit("/", 1)[-1])
    return int(match.group("volume")) if match else None


def media_type_for_extension(extension: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".djvu": "image/vnd.djvu",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(extension, "application/octet-stream")


def validate_bytes(extension: str, header: bytes, trailer: bytes) -> tuple[str, dict[str, bool | None], list[str]]:
    """Validate bounded header/trailer bytes without claiming full integrity."""
    checks: dict[str, bool | None] = {
        "signature_valid": None,
        "trailer_valid": None,
    }
    issues: list[str] = []

    if extension == ".pdf":
        checks["signature_valid"] = header.startswith(b"%PDF-")
        checks["trailer_valid"] = b"startxref" in trailer and b"%%EOF" in trailer
        if not checks["signature_valid"]:
            issues.append("invalid_pdf_signature")
        if not checks["trailer_valid"]:
            issues.append("missing_pdf_trailer_or_eof")
    elif extension == ".djvu":
        checks["signature_valid"] = header.startswith(b"AT&TFORM")
        checks["trailer_valid"] = None
        if not checks["signature_valid"]:
            issues.append("invalid_djvu_signature")
    elif extension in {".jpg", ".jpeg"}:
        checks["signature_valid"] = header.startswith(b"\xff\xd8\xff")
        checks["trailer_valid"] = trailer.endswith(b"\xff\xd9")
        if not checks["signature_valid"]:
            issues.append("invalid_jpeg_signature")
        if not checks["trailer_valid"]:
            issues.append("missing_jpeg_eoi")
    else:
        issues.append("unsupported_extension")
        return "unsupported", checks, issues

    return ("ok_fast_checks" if not issues else "suspect"), checks, issues


def parse_classic_xref(data: bytes) -> tuple[dict[int, int], int]:
    """Parse a classic PDF xref table and return object offsets and root ID.

    This intentionally does not attempt to implement PDF xref streams or every
    incremental-update edge case. Unsupported structures are reported instead
    of guessed.
    """
    lines = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
    if not lines or lines[0].strip() != b"xref":
        raise ValueError("xref_stream_or_missing_classic_xref")

    offsets: dict[int, int] = {}
    index = 1
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == b"trailer":
            break
        header_match = re.fullmatch(rb"(\d+)\s+(\d+)", line)
        if not header_match:
            raise ValueError("invalid_xref_subsection")
        first, count = map(int, header_match.groups())
        index += 1
        for object_id in range(first, first + count):
            if index >= len(lines):
                raise ValueError("truncated_xref")
            entry = lines[index].strip().split()
            index += 1
            if len(entry) >= 3 and entry[2] == b"n":
                offsets[object_id] = int(entry[0])

    trailer_data = b"\n".join(lines[index:])
    root_match = re.search(rb"/Root\s+(\d+)\s+\d+\s+R", trailer_data)
    if not root_match:
        raise ValueError("missing_root_reference")
    return offsets, int(root_match.group(1))


def extract_pdf_page_count(
    client: Any,
    bucket: str,
    key: str,
    size: int,
    trailer: bytes,
    trailer_start: int,
) -> tuple[int | None, str, list[str]]:
    """Read a PDF page-tree count using small random-access S3 ranges."""
    start_match = re.search(rb"startxref\s+(\d+)\s+%%EOF", trailer)
    if not start_match:
        return None, "unavailable_suspect_container", ["page_count_missing_startxref"]
    xref_offset = int(start_match.group(1))
    if not 0 <= xref_offset < size:
        return None, "parse_error", ["page_count_invalid_xref_offset"]

    if xref_offset >= trailer_start:
        xref_data = trailer[xref_offset - trailer_start :]
    else:
        xref_data = _read_range(
            client,
            bucket,
            key,
            xref_offset,
            min(size - 1, xref_offset + XREF_BYTES - 1),
        )

    try:
        offsets, root_id = parse_classic_xref(xref_data)
        root_offset = offsets[root_id]
        root_data = _read_range(
            client, bucket, key, root_offset, min(size - 1, root_offset + OBJECT_BYTES - 1)
        )
        pages_match = re.search(rb"/Pages\s+(\d+)\s+\d+\s+R", root_data)
        if not pages_match:
            raise ValueError("missing_pages_reference")
        pages_id = int(pages_match.group(1))
        pages_offset = offsets[pages_id]
        pages_data = _read_range(
            client, bucket, key, pages_offset, min(size - 1, pages_offset + OBJECT_BYTES - 1)
        )
        count_match = re.search(rb"/Count\s+(\d+)", pages_data)
        if not count_match:
            raise ValueError("missing_page_count")
        return int(count_match.group(1)), "read_from_pdf_page_tree", []
    except (KeyError, ValueError) as exc:
        return None, "unsupported_or_parse_error", [f"page_count_{exc}"]


def extract_djvu_page_count(header: bytes) -> tuple[int | None, str, list[str]]:
    """Read a bundled multipage DjVu page count from its DIRM payload."""
    if not header.startswith(b"AT&TFORM") or header[12:16] != b"DJVM":
        return None, "unsupported_or_parse_error", ["page_count_not_bundled_djvu"]
    directory_offset = header.find(b"DIRM", 16)
    # Chunk ID (4) + length (4) + flags/version (1) + file count (2).
    count_offset = directory_offset + 9
    if directory_offset < 0 or len(header) < count_offset + 2:
        return None, "unsupported_or_parse_error", ["page_count_missing_djvu_directory"]
    return int.from_bytes(header[count_offset : count_offset + 2], "big"), "read_from_djvu_directory", []


def iter_s3_objects(client: Any, bucket: str, prefix: str) -> Iterator[dict[str, Any]]:
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1_000}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        yield from response.get("Contents", [])
        if not response.get("IsTruncated"):
            return
        token = response["NextContinuationToken"]


def _read_range(client: Any, bucket: str, key: str, start: int, end: int) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
    return response["Body"].read()


def inspect_object(
    client: Any,
    bucket: str,
    obj: dict[str, Any],
    pdf_page_counts: bool = False,
) -> ManifestRecord:
    key = obj["Key"]
    size = int(obj.get("Size", 0))
    extension = Path(key).suffix.lower()
    issues: list[str] = []

    header = b""
    trailer = b""
    trailer_start = 0
    if size <= 0:
        status = "empty"
        checks: dict[str, bool | None] = {"signature_valid": None, "trailer_valid": None}
        issues.append("empty_object")
    else:
        header = _read_range(client, bucket, key, 0, min(size - 1, HEADER_BYTES - 1))
        trailer_start = max(0, size - TRAILER_BYTES)
        trailer = _read_range(client, bucket, key, trailer_start, size - 1)
        status, checks, issues = validate_bytes(extension, header, trailer)

    page_count: int | None = None
    page_count_status = "not_computed_fast_audit"
    if pdf_page_counts and extension == ".pdf" and size > 0:
        page_count, page_count_status, page_issues = extract_pdf_page_count(
            client, bucket, key, size, trailer, trailer_start
        )
        issues.extend(page_issues)
    elif extension == ".djvu" and size > 0:
        page_count, page_count_status, page_issues = extract_djvu_page_count(header)
        issues.extend(page_issues)

    volume = parse_volume_number(key)
    etag = str(obj.get("ETag", "")).strip('"') or None
    modified = obj.get("LastModified")
    if isinstance(modified, datetime):
        modified_text = modified.astimezone(timezone.utc).isoformat()
    else:
        modified_text = str(modified) if modified is not None else None

    return ManifestRecord(
        schema_version=SCHEMA_VERSION,
        bucket=bucket,
        key=key,
        source_uri=f"s3://{bucket}/{key}",
        size_bytes=size,
        etag=etag,
        etag_is_simple_md5_candidate=bool(etag and re.fullmatch(r"[0-9a-fA-F]{32}", etag)),
        last_modified=modified_text,
        storage_class=obj.get("StorageClass"),
        extension=extension,
        media_type=media_type_for_extension(extension),
        volume_number=volume,
        publication_year=publication_year_for_volume(volume),
        integrity_status=status,
        integrity_checks=checks,
        issues=issues,
        page_count=page_count,
        page_count_status=page_count_status,
    )


def inspect_objects(
    client: Any,
    bucket: str,
    objects: Sequence[dict[str, Any]],
    workers: int,
    pdf_page_counts: bool = False,
) -> list[ManifestRecord]:
    def inspect(obj: dict[str, Any]) -> ManifestRecord:
        try:
            return inspect_object(client, bucket, obj, pdf_page_counts=pdf_page_counts)
        except Exception as exc:  # preserve the inventory even when a range read fails
            key = obj["Key"]
            volume = parse_volume_number(key)
            extension = Path(key).suffix.lower()
            etag = str(obj.get("ETag", "")).strip('"') or None
            return ManifestRecord(
                schema_version=SCHEMA_VERSION,
                bucket=bucket,
                key=key,
                source_uri=f"s3://{bucket}/{key}",
                size_bytes=int(obj.get("Size", 0)),
                etag=etag,
                etag_is_simple_md5_candidate=bool(etag and re.fullmatch(r"[0-9a-fA-F]{32}", etag)),
                last_modified=str(obj.get("LastModified")) if obj.get("LastModified") else None,
                storage_class=obj.get("StorageClass"),
                extension=extension,
                media_type=media_type_for_extension(extension),
                volume_number=volume,
                publication_year=publication_year_for_volume(volume),
                integrity_status="read_error",
                integrity_checks={"signature_valid": None, "trailer_valid": None},
                issues=[f"range_read_failed:{type(exc).__name__}"],
            )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(inspect, objects))
    return sorted(records, key=lambda record: record.key)


def potential_duplicate_groups(records: Iterable[ManifestRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.etag:
            groups[(record.size_bytes, record.etag)].append(record)
    return [
        {
            "size_bytes": size,
            "etag": etag,
            "classification": "potential_duplicate_not_content_verified",
            "keys": sorted(record.key for record in group),
        }
        for (size, etag), group in sorted(groups.items())
        if len(group) > 1
    ]


def summarize(records: Sequence[ManifestRecord], bucket: str, prefix: str) -> dict[str, Any]:
    by_extension = Counter(record.extension or "[none]" for record in records)
    bytes_by_extension = Counter()
    for record in records:
        bytes_by_extension[record.extension or "[none]"] += record.size_bytes
    statuses = Counter(record.integrity_status for record in records)
    issues = Counter(issue for record in records for issue in record.issues)
    years = Counter(str(record.publication_year) for record in records if record.publication_year)
    volumes = [record.volume_number for record in records if record.volume_number is not None]
    duplicate_groups = potential_duplicate_groups(records)
    known_page_counts = [record.page_count for record in records if record.page_count is not None]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_mode": "fast_bounded_range_reads",
        "source": {"bucket": bucket, "prefix": prefix},
        "object_count": len(records),
        "total_bytes": sum(record.size_bytes for record in records),
        "counts_by_extension": dict(sorted(by_extension.items())),
        "bytes_by_extension": dict(sorted(bytes_by_extension.items())),
        "integrity_status_counts": dict(sorted(statuses.items())),
        "issue_counts": dict(sorted(issues.items())),
        "publication_year_counts": dict(sorted(years.items())),
        "numbered_volume_count": len(volumes),
        "volume_number_min": min(volumes) if volumes else None,
        "volume_number_max": max(volumes) if volumes else None,
        "missing_volume_numbers": sorted(set(range(1, 401)) - set(volumes)),
        "duplicate_volume_numbers": sorted(
            volume for volume, count in Counter(volumes).items() if count > 1
        ),
        "potential_duplicate_group_count": len(duplicate_groups),
        "objects_with_page_count": len(known_page_counts),
        "known_page_count_total": sum(known_page_counts),
        "page_count_status_counts": dict(sorted(Counter(record.page_count_status for record in records).items())),
        "limitations": [
            "Fast checks do not prove complete container validity.",
            "ETag is recorded but is not treated as a full-content SHA-256 checksum.",
            "Unresolved page counts and embedded text layers require deep inspection.",
            "Potential duplicates require full cryptographic hashes or image comparison.",
        ],
    }


def write_outputs(
    output_dir: Path,
    records: Sequence[ManifestRecord],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "manifest.jsonl"
    csv_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"
    duplicate_path = output_dir / "potential_duplicates.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")

    csv_fields = [
        "schema_version", "bucket", "key", "source_uri", "size_bytes", "etag",
        "etag_is_simple_md5_candidate", "last_modified", "storage_class", "extension",
        "media_type", "volume_number", "publication_year", "integrity_status", "issues",
        "full_sha256_status", "page_count", "page_count_status", "text_layer_status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["issues"] = ";".join(record.issues)
            writer.writerow({field: row.get(field) for field in csv_fields})

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    duplicate_path.write_text(
        json.dumps(potential_duplicate_groups(records), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_s3_client(profile: str | None, credentials_csv: Path | None, region: str) -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise SystemExit("boto3 is required; install the project dependencies") from exc

    if profile and credentials_csv:
        raise SystemExit("Use either --profile or --credentials-csv, not both")
    if credentials_csv:
        with credentials_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        try:
            access_key = row["Access key ID"].strip()
            secret_key = row["Secret access key"].strip()
        except KeyError as exc:
            raise SystemExit("Credential CSV must contain Access key ID and Secret access key") from exc
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
    else:
        session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("s3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/corpus-audit"))
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile")
    parser.add_argument("--credentials-csv", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--pdf-page-counts",
        action="store_true",
        help="Read classic PDF xref/page-tree objects with bounded ranges to obtain page counts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = build_s3_client(args.profile, args.credentials_csv, args.region)
    objects = [obj for obj in iter_s3_objects(client, args.bucket, args.prefix) if obj.get("Size", 0) > 0]
    records = inspect_objects(
        client,
        args.bucket,
        objects,
        args.workers,
        pdf_page_counts=args.pdf_page_counts,
    )
    summary = summarize(records, args.bucket, args.prefix)
    write_outputs(args.output_dir, records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if summary["integrity_status_counts"].get("read_error", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
