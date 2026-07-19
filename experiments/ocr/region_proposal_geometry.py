"""Deterministic geometry diagnostics over detector line proposals.

Takes the text-detection polygons of one page artifact and computes, with no
learned model involved:

- missed-ink analysis: fraction of ink (text) pixels outside the union of
  detector boxes, plus the largest connected missed components, so gaps in
  detector coverage are visible instead of silently absent;
- column clustering: greedy grouping of detector boxes into vertical column
  chains by x-interval overlap and bounded vertical gaps, a *proposal* for
  crop-scale units (columns are transport geometry, not semantic articles);
- a review overlay PNG drawn on a polarity-corrected proxy: detector boxes,
  column boxes, and missed-ink components in distinct colors.

All outputs are diagnostic proposals for human review; nothing here assigns
semantic boundaries or enters reviewed retrieval.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCHEMA_VERSION = "region-proposal-geometry-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bboxes(regions: list[dict]) -> list[tuple[int, int, int, int]]:
    out = []
    for region in regions:
        xs = [p["x"] for p in region["polygon"]["points"]]
        ys = [p["y"] for p in region["polygon"]["points"]]
        out.append((min(xs), min(ys), max(xs), max(ys)))
    return out


def missed_ink(
    page: np.ndarray, boxes: list[tuple[int, int, int, int]], ink_threshold: int, scale: int
) -> tuple[float, float, list[dict]]:
    """Ink fraction outside boxes and largest missed components (at 1/scale)."""
    small = page[::scale, ::scale]
    ink = small > ink_threshold  # source is white text on black
    covered = np.zeros_like(ink)
    for x1, y1, x2, y2 in boxes:
        covered[y1 // scale : max(y1 // scale + 1, -(-y2 // scale)),
                x1 // scale : max(x1 // scale + 1, -(-x2 // scale))] = True
    missed = ink & ~covered
    total_ink = int(ink.sum())
    missed_total = int(missed.sum())

    # simple 4-connected component labelling on the missed mask
    labels = np.zeros(missed.shape, dtype=np.int32)
    current = 0
    components: dict[int, list[int]] = {}
    stack: list[tuple[int, int]] = []
    height, width = missed.shape
    for yy in range(height):
        for xx in range(width):
            if missed[yy, xx] and labels[yy, xx] == 0:
                current += 1
                labels[yy, xx] = current
                stack.append((yy, xx))
                pixels = 0
                min_x = max_x = xx
                min_y = max_y = yy
                while stack:
                    cy, cx = stack.pop()
                    pixels += 1
                    min_x, max_x = min(min_x, cx), max(max_x, cx)
                    min_y, max_y = min(min_y, cy), max(max_y, cy)
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < height and 0 <= nx < width and missed[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
                components[current] = [pixels, min_x, min_y, max_x, max_y]
    largest = sorted(components.values(), key=lambda c: c[0], reverse=True)[:20]
    largest_entries = [
        {
            "ink_pixels_at_scale": pixels,
            "bbox_source_xyxy": [min_x * scale, min_y * scale, (max_x + 1) * scale, (max_y + 1) * scale],
        }
        for pixels, min_x, min_y, max_x, max_y in largest
    ]
    fraction = missed_total / total_ink if total_ink else 0.0
    return fraction, total_ink, largest_entries


def cluster_columns(
    boxes: list[tuple[int, int, int, int]],
    min_x_overlap: float,
    max_y_gap_factor: float,
) -> list[dict]:
    """Greedy vertical chaining of boxes into column proposals."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0], boxes[i][1]))
    assigned = [-1] * len(boxes)
    columns: list[dict] = []
    for index in order:
        x1, y1, x2, y2 = boxes[index]
        width = max(1, x2 - x1)
        best = -1
        best_gap = None
        for column_id, column in enumerate(columns):
            cx1, cy1, cx2, cy2 = column["bbox"]
            overlap = min(x2, cx2) - max(x1, cx1)
            narrower = min(width, max(1, cx2 - cx1))
            if overlap / narrower < min_x_overlap:
                continue
            gap = y1 - cy2
            if gap < -0.5 * (y2 - y1):
                continue
            if gap > max_y_gap_factor * max(y2 - y1, cy2 - cy1):
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = column_id, gap
        if best < 0:
            columns.append({"bbox": [x1, y1, x2, y2], "members": [index]})
            assigned[index] = len(columns) - 1
        else:
            column = columns[best]
            cx1, cy1, cx2, cy2 = column["bbox"]
            column["bbox"] = [min(cx1, x1), min(cy1, y1), max(cx2, x2), max(cy2, y2)]
            column["members"].append(index)
            assigned[index] = best
    return columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--proxy", required=True, help="polarity-corrected proxy for the overlay")
    parser.add_argument("--ink-threshold", type=int, default=128)
    parser.add_argument("--analysis-scale", type=int, default=8)
    parser.add_argument("--min-x-overlap", type=float, default=0.5)
    parser.add_argument("--max-y-gap-factor", type=float, default=1.5)
    parser.add_argument("--overlay-out", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact = json.loads(Path(args.artifact).read_text())
    regions = artifact["regions"]
    boxes = _bboxes(regions)
    page = np.asarray(Image.open(args.image).convert("L"))
    page_sha = _sha256_file(Path(args.image))
    if page_sha != artifact["image_sha256"]:
        raise SystemExit("page hash does not match artifact image_sha256; refusing")

    fraction, total_ink, largest = missed_ink(page, boxes, args.ink_threshold, args.analysis_scale)
    columns = cluster_columns(boxes, args.min_x_overlap, args.max_y_gap_factor)

    proxy = Image.open(args.proxy).convert("RGB")
    scale_x = proxy.width / page.shape[1]
    scale_y = proxy.height / page.shape[0]
    draw = ImageDraw.Draw(proxy)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle(
            [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
            outline=(220, 40, 40),
            width=1,
        )
    for column in columns:
        x1, y1, x2, y2 = column["bbox"]
        draw.rectangle(
            [x1 * scale_x - 2, y1 * scale_y - 2, x2 * scale_x + 2, y2 * scale_y + 2],
            outline=(40, 90, 220),
            width=2,
        )
    for component in largest:
        x1, y1, x2, y2 = component["bbox_source_xyxy"]
        draw.rectangle(
            [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
            outline=(240, 160, 20),
            width=3,
        )
    overlay_path = Path(args.overlay_out)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    proxy.save(overlay_path)

    column_sizes = sorted(len(c["members"]) for c in columns)
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_not_qualified",
        "semantic_boundary_authority": "none",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_artifact": {"path": args.artifact, "run": artifact.get("run", {})},
        "page_image": {"path": args.image, "sha256": page_sha},
        "parameters": {
            "ink_threshold": args.ink_threshold,
            "analysis_scale": args.analysis_scale,
            "min_x_overlap": args.min_x_overlap,
            "max_y_gap_factor": args.max_y_gap_factor,
        },
        "missed_ink": {
            "fraction_of_ink_outside_boxes": round(fraction, 4),
            "total_ink_pixels_at_scale": total_ink,
            "largest_missed_components": largest,
        },
        "columns": {
            "count": len(columns),
            "member_count_median": column_sizes[len(column_sizes) // 2] if column_sizes else 0,
            "singleton_columns": sum(1 for c in columns if len(c["members"]) == 1),
            "proposals": [
                {"bbox_source_xyxy": c["bbox"], "member_regions": len(c["members"])}
                for c in columns
            ],
        },
        "overlay": {"path": str(overlay_path)},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=1))
    print(json.dumps({
        "missed_ink_fraction": round(fraction, 4),
        "columns": len(columns),
        "singletons": sum(1 for c in columns if len(c["members"]) == 1),
    }))
    print(f"[artifact] {out_path}")
    print(f"[overlay] {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
