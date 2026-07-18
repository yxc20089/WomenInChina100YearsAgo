"""Render historian-selected pages as source-resolution lossless evidence images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from .corpus_manifest import build_s3_client
from .render_samples import (
    cache_source,
    djvulibre_version,
    read_jsonl,
    sha256_file,
)


GOLD_RENDER_SCHEMA_VERSION = "1.0"
REQUIRED_SELECTION_FIELDS = (
    "page_genre",
    "layout",
    "scan_quality",
    "women_relevance",
    "reviewer",
)


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_annotations(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0" or not isinstance(
        data.get("annotations"), dict
    ):
        raise ValueError("unsupported screening annotation file")
    return data


def selected_candidates(
    candidates: list[dict[str, str]],
    annotations: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates_by_id = {candidate["sample_id"]: candidate for candidate in candidates}
    selected = []
    for sample_id, annotation in annotations["annotations"].items():
        if annotation.get("gold_status") != "include":
            continue
        candidate = candidates_by_id.get(sample_id)
        if candidate is None:
            raise ValueError(f"included sample is absent from candidate plan: {sample_id}")
        missing = [
            field
            for field in REQUIRED_SELECTION_FIELDS
            if not str(annotation.get(field, "")).strip()
            or annotation.get(field) == "unreviewed"
        ]
        if missing:
            raise ValueError(
                f"included sample {sample_id} lacks reviewed selection fields: {missing}"
            )
        selected.append(
            {
                **candidate,
                "selection": {
                    key: annotation.get(key)
                    for key in (
                        *REQUIRED_SELECTION_FIELDS,
                        "notes",
                        "reviewed_at",
                        "gold_status",
                    )
                },
            }
        )
    return sorted(
        selected,
        key=lambda item: (int(item["volume_number"]), int(item["page_number"])),
    )


def pilot_candidates(
    candidates: list[dict[str, str]], sample_ids: list[str]
) -> list[dict[str, Any]]:
    candidates_by_id = {candidate["sample_id"]: candidate for candidate in candidates}
    selected = []
    for sample_id in sample_ids:
        if sample_id not in candidates_by_id:
            raise ValueError(f"pilot sample is absent from candidate plan: {sample_id}")
        selected.append(
            {
                **candidates_by_id[sample_id],
                "selection": {
                    "gold_status": "non_gold_pilot",
                    "reviewer": None,
                    "note": "Pipeline verification only; not historian-selected gold.",
                },
            }
        )
    return selected


def _pixel_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _save_decoded_as_png(image_bytes: bytes, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(image_bytes)) as source_image:
        source_image.load()
        if source_image.mode not in {"1", "L", "LA", "RGB", "RGBA"}:
            source_image = source_image.convert("RGBA" if "A" in source_image.mode else "RGB")
        source_pixel_sha256 = _pixel_sha256(source_image)
        width, height = source_image.size
        mode = source_image.mode
        temporary = output_path.with_suffix(output_path.suffix + ".part")
        source_image.save(temporary, format="PNG", compress_level=6, optimize=False)
    with Image.open(temporary) as written_image:
        written_image.load()
        written_pixel_sha256 = _pixel_sha256(written_image)
    if source_pixel_sha256 != written_pixel_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError("decoded pixels changed while writing lossless PNG")
    temporary.replace(output_path)
    return {
        "render_path": str(output_path),
        "render_sha256": sha256_file(output_path),
        "decoded_pixel_sha256": written_pixel_sha256,
        "render_width": width,
        "render_height": height,
        "image_mode": mode,
        "render_format": "image/png",
        "png_compress_level": 6,
        "pillow_version": Image.__version__,
    }


def extract_pdf_page(
    source_path: Path,
    page_number: int,
    output_path: Path,
    expected_page_count: int,
) -> dict[str, Any]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required for PDF extraction") from exc

    with fitz.open(source_path) as document:
        if document.page_count != expected_page_count:
            raise ValueError(
                f"page_count_mismatch:manifest={expected_page_count}:document={document.page_count}"
            )
        if not 1 <= page_number <= document.page_count:
            raise ValueError(f"page_out_of_range:{page_number}")
        page = document.load_page(page_number - 1)
        if page.rotation != 0:
            raise ValueError("rotated PDF pages require an explicit pixel-orientation review")
        display_items = page.get_bboxlog()
        if len(display_items) != 1 or display_items[0][0] not in {
            "fill-image",
            "fill-imgmask",
        }:
            raise ValueError(
                "PDF page contains composited visible content; direct raster extraction is unsafe"
            )
        images = page.get_images(full=True)
        unique_xrefs = sorted({image[0] for image in images})
        if len(unique_xrefs) != 1:
            raise ValueError(
                f"expected one full-page raster, found {len(unique_xrefs)} image objects"
            )
        xref = unique_xrefs[0]
        placements = page.get_image_rects(xref, transform=True)
        if len(placements) != 1:
            raise ValueError(f"expected one image placement, found {len(placements)}")
        rectangle, matrix = placements[0]
        tolerance = 0.01
        if any(
            abs(value) > tolerance
            for value in (
                rectangle.x0 - page.rect.x0,
                rectangle.y0 - page.rect.y0,
                rectangle.x1 - page.rect.x1,
                rectangle.y1 - page.rect.y1,
                matrix.b,
                matrix.c,
            )
        ) or matrix.a <= 0 or matrix.d <= 0:
            raise ValueError("embedded raster is not one unrotated full-page placement")
        image_record = next(image for image in images if image[0] == xref)
        extracted = document.extract_image(xref)
        saved = _save_decoded_as_png(extracted["image"], output_path)
        if (
            saved["render_width"] != extracted["width"]
            or saved["render_height"] != extracted["height"]
        ):
            raise ValueError("extracted PNG dimensions disagree with PDF image metadata")
        return {
            **saved,
            "render_method": "direct_embedded_raster_decode",
            "geometric_transform": "none",
            "pixel_encoding_transform": "source_codec_decode_then_lossless_png_reencode",
            "renderer": "PyMuPDF/extract_image + Pillow/PNG",
            "renderer_version": fitz.version[0],
            "source_image_xref": xref,
            "source_image_extension": extracted["ext"],
            "source_image_width": extracted["width"],
            "source_image_height": extracted["height"],
            "source_image_bits_per_component": image_record[4],
            "source_image_filter": image_record[8],
        }


def extract_djvu_page(
    source_path: Path,
    page_number: int,
    output_path: Path,
    expected_page_count: int,
) -> dict[str, Any]:
    ddjvu_path = shutil.which("ddjvu")
    djvused_path = shutil.which("djvused")
    if not ddjvu_path or not djvused_path:
        raise RuntimeError("DjVuLibre ddjvu and djvused executables are required")
    count = subprocess.run(
        [djvused_path, str(source_path), "-e", "n"],
        capture_output=True,
        text=True,
        check=True,
    )
    actual_page_count = int(count.stdout.strip())
    if actual_page_count != expected_page_count:
        raise ValueError(
            f"page_count_mismatch:manifest={expected_page_count}:document={actual_page_count}"
        )
    if not 1 <= page_number <= actual_page_count:
        raise ValueError(f"page_out_of_range:{page_number}")
    size = subprocess.run(
        [djvused_path, str(source_path), "-e", f"select {page_number}; size"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    expected_size = {
        key: int(value)
        for key, value in (item.split("=") for item in size.split())
    }
    temporary_pnm = output_path.with_suffix(".pnm.part")
    temporary_pnm.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            ddjvu_path,
            "-format=pnm",
            f"-page={page_number}",
            str(source_path),
            str(temporary_pnm),
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        temporary_pnm.unlink(missing_ok=True)
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"ddjvu_exit_{completed.returncode}:{stderr[-500:]}")
    try:
        image_bytes = temporary_pnm.read_bytes()
        saved = _save_decoded_as_png(image_bytes, output_path)
    finally:
        temporary_pnm.unlink(missing_ok=True)
    if (
        saved["render_width"] != expected_size["width"]
        or saved["render_height"] != expected_size["height"]
    ):
        raise ValueError("native DjVu output dimensions disagree with page metadata")
    return {
        **saved,
        "render_method": "native_djvu_decode",
        "geometric_transform": "none",
        "pixel_encoding_transform": "source_codec_decode_then_lossless_png_reencode",
        "renderer": "DjVuLibre/ddjvu + Pillow/PNG",
        "renderer_version": djvulibre_version(ddjvu_path),
        "source_image_width": expected_size["width"],
        "source_image_height": expected_size["height"],
    }


def _resolve_source(
    client: Any | None,
    bucket: str,
    record: dict[str, Any],
    cache_dir: Path,
) -> tuple[Path, str]:
    destination = cache_dir / (
        f"volume-{int(record['volume_number']):03d}{record['extension']}"
    )
    expected_size = int(record["size_bytes"])
    if destination.exists() and destination.stat().st_size == expected_size:
        return destination, "cache_hit_size_verified"
    if client is None:
        raise ValueError("source is not present in the verified local cache (offline mode)")
    return cache_source(client, bucket, record, cache_dir)


def render_lossless_plan(
    selections: list[dict[str, Any]],
    corpus_manifest: list[dict[str, Any]],
    cache_dir: Path,
    output_dir: Path,
    *,
    client: Any | None = None,
    bucket: str = "ccaa-us-east-1-504133794192",
    source_hash: Callable[[Path], str] = sha256_file,
) -> list[dict[str, Any]]:
    manifest_by_key = {record["key"]: record for record in corpus_manifest}
    source_hashes: dict[Path, str] = {}
    results = []
    for selection in selections:
        sample_id = selection["sample_id"]
        source_key = selection["source_key"]
        record = manifest_by_key.get(source_key)
        base = {
            "schema_version": GOLD_RENDER_SCHEMA_VERSION,
            "sample_id": sample_id,
            "source_uri": selection["source_uri"],
            "source_key": source_key,
            "volume_number": int(selection["volume_number"]),
            "publication_year": int(selection["publication_year"]),
            "page_number": int(selection["page_number"]),
            "selection": selection["selection"],
        }
        if record is None:
            results.append(
                {**base, "status": "render_error", "issue": "source_missing_from_corpus_manifest"}
            )
            continue
        extension = record["extension"]
        if extension not in {".pdf", ".djvu"}:
            results.append(
                {**base, "status": "unsupported_renderer", "issue": f"unsupported_{extension}"}
            )
            continue
        try:
            source_path, cache_status = _resolve_source(
                client, bucket, record, cache_dir
            )
            if source_path not in source_hashes:
                source_hashes[source_path] = source_hash(source_path)
            output_path = (
                output_dir
                / "images"
                / f"v{int(selection['volume_number']):03d}"
                / f"p{int(selection['page_number']):04d}.png"
            )
            extractor = extract_pdf_page if extension == ".pdf" else extract_djvu_page
            extracted = extractor(
                source_path,
                int(selection["page_number"]),
                output_path,
                int(record["page_count"]),
            )
            results.append(
                {
                    **base,
                    **extracted,
                    "source_object_size_bytes": int(record["size_bytes"]),
                    "source_object_etag": record.get("etag"),
                    "source_object_sha256": source_hashes[source_path],
                    "source_cache_status": cache_status,
                    "status": "rendered",
                }
            )
        except Exception as exc:
            results.append(
                {
                    **base,
                    "status": "render_error",
                    "issue": f"{type(exc).__name__}:{exc}",
                }
            )
    return sorted(
        results, key=lambda item: (item["volume_number"], item["page_number"])
    )


def write_lossless_results(
    output_dir: Path, results: list[dict[str, Any]]
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "lossless_manifest.jsonl"
    temporary_manifest = manifest_path.with_suffix(".jsonl.part")
    temporary_manifest.write_text(
        "".join(
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
            for result in results
        ),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    statuses = Counter(result["status"] for result in results)
    summary = {
        "schema_version": GOLD_RENDER_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_pages": len(results),
        "status_counts": dict(sorted(statuses.items())),
        "rendered_bytes": sum(
            Path(result["render_path"]).stat().st_size
            for result in results
            if result["status"] == "rendered"
        ),
        "gold_pages": sum(
            result["selection"].get("gold_status") == "include"
            for result in results
            if result["status"] == "rendered"
        ),
        "non_gold_pilot_pages": sum(
            result["selection"].get("gold_status") == "non_gold_pilot"
            for result in results
            if result["status"] == "rendered"
        ),
        "note": (
            "Only records with selection.gold_status=include are historian-selected gold. "
            "Pilot records must never be used as quality ground truth."
        ),
    }
    summary_path = output_dir / "summary.json"
    temporary_summary = summary_path.with_suffix(".json.part")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("artifacts/benchmark-review/annotations.json"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("artifacts/benchmark-sample/candidate_pages.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/corpus-audit/manifest.jsonl"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/wic-source-cache"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bucket", default="ccaa-us-east-1-504133794192")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile")
    parser.add_argument("--credentials-csv", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--pilot-sample-id",
        action="append",
        help="Render an explicitly non-gold pipeline pilot instead of annotation selections",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = read_candidates(args.candidates)
    if args.pilot_sample_id:
        selections = pilot_candidates(candidates, args.pilot_sample_id)
        output_dir = args.output_dir or Path("artifacts/lossless-pilot")
    else:
        selections = selected_candidates(candidates, read_annotations(args.annotations))
        output_dir = args.output_dir or Path("artifacts/gold-pages")
        if not selections:
            raise SystemExit(
                "No historian-reviewed gold_status=include pages exist; use the screening UI first."
            )
    client = None
    if not args.offline:
        client = build_s3_client(args.profile, args.credentials_csv, args.region)
    results = render_lossless_plan(
        selections,
        read_jsonl(args.manifest),
        args.cache_dir,
        output_dir,
        client=client,
        bucket=args.bucket,
    )
    summary = write_lossless_results(output_dir, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["status_counts"].get("render_error", 0) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
