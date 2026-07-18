"""Coordinate-preserving PaddleOCR 3.x ingestion for newspaper page images."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

from .evidence import (
    OCRPageArtifact,
    OCRRegion,
    Point,
    Polygon,
    ProcessingRun,
    RegionKind,
    RunKind,
    SourcePointer,
)
from .render_samples import sha256_file


@dataclass(frozen=True, slots=True)
class Tile:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True, slots=True)
class DetectedLine:
    text: str
    confidence: float
    points: tuple[tuple[float, float], ...]
    engine_payload: dict[str, Any]


def normalize_ocr_text(text: str) -> str:
    """Preserve historical glyphs; apply Unicode canonical composition only."""
    return unicodedata.normalize("NFC", text).strip()


def tile_bounds(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("overlap must be non-negative and smaller than tile_size")

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        stride = tile_size - overlap
        values = list(range(0, max(1, length - tile_size + 1), stride))
        final = length - tile_size
        if values[-1] != final:
            values.append(final)
        return values

    return [
        Tile(x, y, min(width, x + tile_size), min(height, y + tile_size))
        for y in starts(height)
        for x in starts(width)
    ]


def translate_line(line: DetectedLine, tile: Tile) -> DetectedLine:
    return DetectedLine(
        text=line.text,
        confidence=line.confidence,
        points=tuple((x + tile.left, y + tile.top) for x, y in line.points),
        engine_payload={**line.engine_payload, "tile": [tile.left, tile.top, tile.right, tile.bottom]},
    )


def bounding_box(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    points_list = list(points)
    xs = [point[0] for point in points_list]
    ys = [point[1] for point in points_list]
    return min(xs), min(ys), max(xs), max(ys)


def intersection_over_union(one: DetectedLine, two: DetectedLine) -> float:
    ax1, ay1, ax2, ay2 = bounding_box(one.points)
    bx1, by1, bx2, by2 = bounding_box(two.points)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    one_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    two_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = one_area + two_area - intersection
    return intersection / union if union else 0.0


def deduplicate_lines(lines: Iterable[DetectedLine], iou_threshold: float = 0.5) -> list[DetectedLine]:
    """Remove overlap duplicates only when normalized text also agrees."""
    kept: list[DetectedLine] = []
    for candidate in sorted(lines, key=lambda line: line.confidence, reverse=True):
        normalized = normalize_ocr_text(candidate.text)
        duplicate = any(
            normalized == normalize_ocr_text(existing.text)
            and intersection_over_union(candidate, existing) >= iou_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    # This is a deterministic geometric fallback, not an assertion that it is
    # the newspaper's true reading order. PP-Structure will later replace it.
    return sorted(kept, key=lambda line: (bounding_box(line.points)[1], bounding_box(line.points)[0]))


def run_tiled_detection(
    image: Image.Image,
    predict_tile: Callable[[Image.Image], list[DetectedLine]],
    tile_size: int = 1200,
    overlap: int = 120,
) -> tuple[list[DetectedLine], list[Tile]]:
    tiles = tile_bounds(image.width, image.height, tile_size, overlap)
    all_lines: list[DetectedLine] = []
    for tile in tiles:
        crop = image.crop((tile.left, tile.top, tile.right, tile.bottom))
        all_lines.extend(translate_line(line, tile) for line in predict_tile(crop))
    return deduplicate_lines(all_lines), tiles


def run_batched_isolated_detection(
    image: Image.Image,
    language: str,
    tile_size: int,
    overlap: int,
    worker_batch_size: int = 5,
    timeout_seconds: int = 600,
) -> tuple[list[DetectedLine], list[Tile]]:
    """Process small tile batches in disposable native-runtime processes."""
    if worker_batch_size <= 0:
        raise ValueError("worker_batch_size must be positive")
    tiles = tile_bounds(image.width, image.height, tile_size, overlap)
    all_lines: list[DetectedLine] = []
    for batch_start in range(0, len(tiles), worker_batch_size):
        batch = list(enumerate(tiles[batch_start : batch_start + worker_batch_size], batch_start))
        print(
            f"OCR tile batch {batch_start // worker_batch_size + 1}/"
            f"{(len(tiles) + worker_batch_size - 1) // worker_batch_size}",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="wic-ocr-batch-") as directory:
            root = Path(directory)
            manifest_items = []
            for tile_index, tile in batch:
                tile_path = root / f"tile-{tile_index:04d}.png"
                image.crop((tile.left, tile.top, tile.right, tile.bottom)).save(tile_path, format="PNG")
                manifest_items.append({"tile_index": tile_index, "path": str(tile_path)})
            manifest_path = root / "manifest.json"
            output_path = root / "output.json"
            manifest_path.write_text(json.dumps(manifest_items), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wic_history.ocr_batch_worker",
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                    "--language",
                    language,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if completed.returncode or not output_path.exists():
                details = (completed.stderr or completed.stdout)[-2000:]
                raise RuntimeError(
                    f"batched PaddleOCR worker failed with {completed.returncode}: {details}"
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        for tile_result in payload:
            tile = tiles[int(tile_result["tile_index"])]
            for item in tile_result["lines"]:
                line = DetectedLine(
                    text=item["text"],
                    confidence=float(item["confidence"]),
                    points=tuple((float(x), float(y)) for x, y in item["points"]),
                    engine_payload=item.get("engine_payload", {}),
                )
                all_lines.append(translate_line(line, tile))
        # The macOS Paddle runtime can retain native allocations briefly after
        # process exit. Avoid overlapping that release with the next worker.
        time.sleep(2)
    return deduplicate_lines(all_lines), tiles


class PaddleOCRPredictor:
    """Lazy PP-OCRv6 medium detector/recognizer wrapper."""

    def __init__(self, language: str = "ch"):
        from paddleocr import PaddleOCR

        self.language = language
        self.engine = PaddleOCR(
            ocr_version="PP-OCRv6",
            lang=language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            # Native orientation inference terminated on a full historical page
            # on Apple Silicon. Direction is evaluated separately in benchmark.
            use_textline_orientation=False,
            text_recognition_batch_size=1,
        )

    def __call__(self, image: Image.Image) -> list[DetectedLine]:
        import numpy as np

        outputs = list(self.engine.predict(np.asarray(image.convert("RGB"))))
        lines: list[DetectedLine] = []
        for output in outputs:
            payload = output.json
            result = payload.get("res", payload)
            texts = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])
            polygons = result.get("rec_polys", [])
            for index, (text, score, polygon) in enumerate(zip(texts, scores, polygons, strict=True)):
                points = tuple((float(point[0]), float(point[1])) for point in polygon)
                lines.append(
                    DetectedLine(
                        text=str(text),
                        confidence=float(score),
                        points=points,
                        engine_payload={"result_index": index},
                    )
                )
        return lines


class IsolatedPaddleOCRPredictor:
    """Run each tile in a fresh process to release native Paddle state."""

    def __init__(self, language: str = "ch", timeout_seconds: int = 300):
        self.language = language
        self.timeout_seconds = timeout_seconds

    def __call__(self, image: Image.Image) -> list[DetectedLine]:
        with tempfile.TemporaryDirectory(prefix="wic-ocr-tile-") as directory:
            input_path = Path(directory) / "tile.png"
            output_path = Path(directory) / "result.json"
            image.save(input_path, format="PNG")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wic_history.ocr_worker",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--language",
                    self.language,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode or not output_path.exists():
                details = (completed.stderr or completed.stdout)[-2000:]
                raise RuntimeError(
                    f"isolated PaddleOCR worker failed with {completed.returncode}: {details}"
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        return [
            DetectedLine(
                text=item["text"],
                confidence=float(item["confidence"]),
                points=tuple((float(x), float(y)) for x, y in item["points"]),
                engine_payload=item.get("engine_payload", {}),
            )
            for item in payload
        ]


def create_ocr_artifact(
    image_path: Path,
    source_uri: str,
    page_number: int,
    volume_number: int | None,
    publication_year: int | None,
    predictor: Callable[[Image.Image], list[DetectedLine]],
    tile_size: int,
    overlap: int,
    language: str,
    screening_derivative: bool,
    isolated_tiles: bool = False,
    page_detector: Callable[[Image.Image, int, int], tuple[list[DetectedLine], list[Tile]]] | None = None,
    source_sha256: str | None = None,
    evidence_tier: str = "unreviewed_input",
    render_manifest_path: str | None = None,
) -> OCRPageArtifact:
    started_at = datetime.now(timezone.utc)
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        dpi_value = source_image.info.get("dpi", (None, None))[0]
    detected, tiles = (
        page_detector(image, tile_size, overlap)
        if page_detector is not None
        else run_tiled_detection(image, predictor, tile_size, overlap)
    )
    run = ProcessingRun(
        kind=RunKind.OCR,
        engine="PaddleOCR",
        model_name="PP-OCRv6_medium_det+PP-OCRv6_medium_rec",
        model_revision="paddleocr-3.7.0-official",
        software_version="3.7.0",
        configuration={
            "language": language,
            "tile_size": tile_size,
            "overlap": overlap,
            "tile_count": len(tiles),
            "orientation_model_enabled": False,
            "isolated_tile_workers": isolated_tiles,
            "evidence_tier": evidence_tier,
            "render_manifest": render_manifest_path,
        },
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
    )
    regions = [
        OCRRegion(
            kind=RegionKind.TEXT,
            polygon=Polygon(points=[Point(x=x, y=y) for x, y in line.points]),
            reading_order=index,
            raw_text=line.text,
            normalized_text=normalize_ocr_text(line.text),
            confidence=line.confidence,
            language="zh-Hant",
            direction="unknown",
            engine_payload=line.engine_payload,
        )
        for index, line in enumerate(detected)
    ]
    warnings = [
        "Reading order is a geometric fallback and has not been validated for vertical newspaper columns."
    ]
    if screening_derivative:
        warnings.append(
            "Input is a lossy screening derivative; this artifact is a technical smoke test, not gold OCR evidence."
        )
    elif evidence_tier == "non_gold_lossless_pilot":
        warnings.append(
            "Input is a source-resolution lossless pipeline pilot, not historian-selected gold."
        )
    elif evidence_tier != "historian_selected_gold":
        warnings.append(
            "Input has no historian-selected gold provenance and must not support OCR quality claims."
        )
    return OCRPageArtifact(
        source=SourcePointer(
            source_uri=source_uri,
            source_sha256=source_sha256,
            volume_number=volume_number,
            publication_year=publication_year,
            page_number=page_number,
        ),
        image_uri=str(image_path),
        image_sha256=sha256_file(image_path),
        width=image.width,
        height=image.height,
        dpi=int(round(dpi_value)) if dpi_value else None,
        run=run,
        regions=regions,
        warnings=warnings,
    )


def sha256_argument(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256")
    return value


def resolve_render_provenance(
    image_path: Path,
    manifest_path: Path,
    *,
    source_uri: str,
    page_number: int,
    volume_number: int | None,
    publication_year: int | None,
    supplied_source_sha256: str | None = None,
) -> tuple[str, str]:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    matches = [
        row
        for row in rows
        if Path(row.get("render_path", "")).resolve() == image_path.resolve()
    ]
    if len(matches) != 1:
        raise ValueError("render manifest must contain exactly one record for the OCR image")
    record = matches[0]
    if record.get("status") != "rendered":
        raise ValueError("render manifest record is not successfully rendered")
    expected = {
        "source_uri": source_uri,
        "page_number": page_number,
        "volume_number": volume_number,
        "publication_year": publication_year,
    }
    for key, value in expected.items():
        if value is not None and record.get(key) != value:
            raise ValueError(f"render manifest {key} disagrees with the OCR request")
    if sha256_file(image_path) != record.get("render_sha256"):
        raise ValueError("OCR image bytes disagree with the render manifest hash")
    source_sha256 = record.get("source_object_sha256")
    if not isinstance(source_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_sha256
    ):
        raise ValueError("render manifest lacks a valid full source-object SHA-256")
    if supplied_source_sha256 and supplied_source_sha256 != source_sha256:
        raise ValueError("supplied source SHA-256 disagrees with the render manifest")
    selection_status = (record.get("selection") or {}).get("gold_status")
    evidence_tiers = {
        "include": "historian_selected_gold",
        "non_gold_pilot": "non_gold_lossless_pilot",
    }
    if selection_status not in evidence_tiers:
        raise ValueError("render manifest does not contain an eligible gold or pilot selection")
    return source_sha256, evidence_tiers[selection_status]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-sha256", type=sha256_argument)
    parser.add_argument("--render-manifest", type=Path)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--volume", type=int)
    parser.add_argument("--year", type=int)
    parser.add_argument("--language", default="ch")
    parser.add_argument("--tile-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--screening-derivative", action="store_true")
    worker_group = parser.add_mutually_exclusive_group()
    worker_group.add_argument(
        "--isolate-tiles",
        action="store_true",
        help="Run each tile in a fresh process; recommended for the macOS Paddle runtime",
    )
    worker_group.add_argument(
        "--reuse-model",
        action="store_true",
        help="Reuse one Paddle model process; recommended for Linux/GPU throughput",
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=5,
        help="Tiles per disposable worker in the default macOS bounded-memory mode",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.screening_derivative and args.render_manifest:
        raise SystemExit("--screening-derivative and --render-manifest are mutually exclusive")
    source_sha256 = args.source_sha256
    evidence_tier = "screening_derivative" if args.screening_derivative else "unreviewed_input"
    if args.render_manifest:
        source_sha256, evidence_tier = resolve_render_provenance(
            args.image,
            args.render_manifest,
            source_uri=args.source_uri,
            page_number=args.page,
            volume_number=args.volume,
            publication_year=args.year,
            supplied_source_sha256=args.source_sha256,
        )
    isolate_tiles = args.isolate_tiles
    batch_isolate = sys.platform == "darwin" and not args.isolate_tiles and not args.reuse_model
    reuse_model = args.reuse_model or (sys.platform != "darwin" and not isolate_tiles)
    predictor = IsolatedPaddleOCRPredictor(args.language) if isolate_tiles else (
        PaddleOCRPredictor(args.language) if reuse_model else lambda image: []
    )

    def batched_detector(
        image: Image.Image, tile_size: int, overlap: int
    ) -> tuple[list[DetectedLine], list[Tile]]:
        return run_batched_isolated_detection(
            image,
            language=args.language,
            tile_size=tile_size,
            overlap=overlap,
            worker_batch_size=args.worker_batch_size,
        )

    page_detector = batched_detector if batch_isolate else None
    artifact = create_ocr_artifact(
        image_path=args.image,
        source_uri=args.source_uri,
        page_number=args.page,
        volume_number=args.volume,
        publication_year=args.year,
        predictor=predictor,
        tile_size=args.tile_size,
        overlap=args.overlap,
        language=args.language,
        screening_derivative=args.screening_derivative,
        isolated_tiles=isolate_tiles or batch_isolate,
        page_detector=page_detector,
        source_sha256=source_sha256,
        evidence_tier=evidence_tier,
        render_manifest_path=str(args.render_manifest) if args.render_manifest else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "regions": len(artifact.regions),
                "warnings": artifact.warnings,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
