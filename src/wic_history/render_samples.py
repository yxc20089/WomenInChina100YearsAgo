"""Download selected source volumes to a cache and render screening pages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .corpus_manifest import build_s3_client


RENDER_SCHEMA_VERSION = "1.0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cache_source(
    client: Any,
    bucket: str,
    record: dict[str, Any],
    cache_dir: Path,
) -> tuple[Path, str]:
    """Download atomically when absent; reuse only an exact-size cache entry."""
    volume = int(record["volume_number"])
    extension = record["extension"]
    destination = cache_dir / f"volume-{volume:03d}{extension}"
    expected_size = int(record["size_bytes"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size == expected_size:
        return destination, "cache_hit_size_verified"
    if destination.exists():
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    client.download_file(bucket, record["key"], str(partial))
    if partial.stat().st_size != expected_size:
        partial.unlink()
        raise ValueError("download_size_mismatch")
    partial.replace(destination)
    return destination, "downloaded_size_verified"


def render_pdf_pages(
    source_path: Path,
    candidates: Sequence[dict[str, str]],
    output_root: Path,
    expected_page_count: int,
    dpi: int,
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required for PDF rendering") from exc

    document = fitz.open(source_path)
    if document.page_count != expected_page_count:
        actual_page_count = document.page_count
        document.close()
        raise ValueError(
            f"page_count_mismatch:manifest={expected_page_count}:document={actual_page_count}"
        )

    volume = int(candidates[0]["volume_number"])
    volume_dir = output_root / "images" / f"v{volume:03d}"
    volume_dir.mkdir(parents=True, exist_ok=True)
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    results: list[dict[str, Any]] = []
    try:
        for candidate in sorted(candidates, key=lambda row: int(row["page_number"])):
            page_number = int(candidate["page_number"])
            if not 1 <= page_number <= document.page_count:
                raise ValueError(f"page_out_of_range:{page_number}")
            output_path = volume_dir / f"p{page_number:04d}.jpg"
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
            pixmap.save(output_path, output="jpeg", jpg_quality=jpeg_quality)
            results.append(
                {
                    "schema_version": RENDER_SCHEMA_VERSION,
                    "sample_id": candidate["sample_id"],
                    "source_uri": candidate["source_uri"],
                    "volume_number": volume,
                    "publication_year": int(candidate["publication_year"]),
                    "page_number": page_number,
                    "render_path": str(output_path),
                    "render_sha256": sha256_file(output_path),
                    "render_width": pixmap.width,
                    "render_height": pixmap.height,
                    "render_dpi": dpi,
                    "render_format": "image/jpeg",
                    "jpeg_quality": jpeg_quality,
                    "renderer": "PyMuPDF",
                    "renderer_version": fitz.version[0],
                    "status": "rendered",
                }
            )
            del pixmap
            del page
    finally:
        document.close()
        # PyMuPDF keeps a display-list/image cache that can grow across the
        # very large source volumes. Release it between volumes so a full plan
        # does not accumulate hundreds of pages of decoder state.
        fitz.TOOLS.store_shrink(100)
    return results


def render_plan(
    client: Any,
    bucket: str,
    candidates: Sequence[dict[str, str]],
    manifest: Sequence[dict[str, Any]],
    cache_dir: Path,
    output_dir: Path,
    volumes: set[int] | None,
    dpi: int,
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    manifest_by_key = {record["key"]: record for record in manifest}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        volume = int(candidate["volume_number"])
        if volumes is None or volume in volumes:
            grouped[candidate["source_key"]].append(candidate)

    results: list[dict[str, Any]] = []
    for source_key, source_candidates in sorted(
        grouped.items(), key=lambda item: int(item[1][0]["volume_number"])
    ):
        record = manifest_by_key[source_key]
        extension = record["extension"]
        if extension != ".pdf":
            for candidate in source_candidates:
                results.append(
                    {
                        "schema_version": RENDER_SCHEMA_VERSION,
                        "sample_id": candidate["sample_id"],
                        "source_uri": candidate["source_uri"],
                        "volume_number": int(candidate["volume_number"]),
                        "publication_year": int(candidate["publication_year"]),
                        "page_number": int(candidate["page_number"]),
                        "status": "unsupported_renderer",
                        "issue": f"no_renderer_for_{extension.removeprefix('.')}",
                    }
                )
            continue

        try:
            source_path, cache_status = cache_source(client, bucket, record, cache_dir)
            rendered = render_pdf_pages(
                source_path,
                source_candidates,
                output_dir,
                int(record["page_count"]),
                dpi,
                jpeg_quality,
            )
            for result in rendered:
                result["source_cache_status"] = cache_status
            results.extend(rendered)
        except Exception as exc:
            for candidate in source_candidates:
                results.append(
                    {
                        "schema_version": RENDER_SCHEMA_VERSION,
                        "sample_id": candidate["sample_id"],
                        "source_uri": candidate["source_uri"],
                        "volume_number": int(candidate["volume_number"]),
                        "publication_year": int(candidate["publication_year"]),
                        "page_number": int(candidate["page_number"]),
                        "status": "render_error",
                        "issue": f"{type(exc).__name__}:{exc}",
                    }
                )
    return sorted(results, key=lambda row: (row["volume_number"], row["page_number"]))


def write_results(output_dir: Path, results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "render_manifest.jsonl"
    existing: list[dict[str, Any]] = []
    if manifest_path.exists():
        existing = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    merged = {result["sample_id"]: result for result in existing}
    merged.update({result["sample_id"]: result for result in results})
    all_results = sorted(
        merged.values(), key=lambda row: (row["volume_number"], row["page_number"])
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        for result in all_results:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    statuses = Counter(result["status"] for result in all_results)
    summary = {
        "schema_version": RENDER_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(all_results),
        "updated_candidate_count": len(results),
        "status_counts": dict(sorted(statuses.items())),
        "rendered_bytes": sum(
            os.path.getsize(result["render_path"])
            for result in all_results
            if result["status"] == "rendered" and Path(result["render_path"]).exists()
        ),
        "note": "JPEGs are screening derivatives, not authoritative evidence images.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("artifacts/benchmark-sample/candidate_pages.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/corpus-audit/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmark-pages"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/wic-source-cache"))
    parser.add_argument("--bucket", default="ccaa-us-east-1-504133794192")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile")
    parser.add_argument("--credentials-csv", type=Path)
    parser.add_argument("--volume", action="append", type=int, dest="volumes")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--jpeg-quality", type=int, default=72)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = build_s3_client(args.profile, args.credentials_csv, args.region)
    results = render_plan(
        client=client,
        bucket=args.bucket,
        candidates=read_csv(args.candidates),
        manifest=read_jsonl(args.manifest),
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        volumes=set(args.volumes) if args.volumes else None,
        dpi=args.dpi,
        jpeg_quality=args.jpeg_quality,
    )
    summary = write_results(args.output_dir, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["status_counts"].get("render_error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
