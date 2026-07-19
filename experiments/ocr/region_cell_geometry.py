"""Frame-aware deterministic region proposals for one broadsheet page.

Decomposes a ruled newspaper page with no learned model:

1. ruling-line extraction: long horizontal/vertical ink runs on the binarized
   source become a separator mask (ad frames, column rules, borders);
2. cell decomposition: connected components of the non-separator area are
   layout cells; every downstream grouping is confined to one cell;
3. detector assignment: text-detection boxes from an existing artifact are
   assigned to cells by center point;
4. display-ink proposals: connected missed-ink components inside a cell that
   exceed a size threshold become `display_or_figure` region proposals —
   these capture large display glyphs and illustrations that line detectors
   under-segment;
5. column clustering within each cell (x-interval overlap, bounded y-gap);
6. a review overlay: separators, cells, detector boxes, columns, and
   display proposals in distinct colors.

Outputs are diagnostic proposals for human review and cross-validation
against crop-level OCR; nothing here asserts semantic article boundaries.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCHEMA_VERSION = "region-cell-geometry-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _erode_1d(mask: np.ndarray, length: int, axis: int) -> np.ndarray:
    """Binary erosion with a 1xL (or Lx1) structuring element via shifts."""
    result = mask.copy()
    for offset in range(1, length):
        shifted = np.zeros_like(mask)
        if axis == 1:
            shifted[:, :-offset] = mask[:, offset:]
        else:
            shifted[:-offset, :] = mask[offset:, :]
        result &= shifted
    return result


def _dilate_1d(mask: np.ndarray, length: int, axis: int) -> np.ndarray:
    result = mask.copy()
    for offset in range(1, length):
        shifted = np.zeros_like(mask)
        if axis == 1:
            shifted[:, offset:] = mask[:, :-offset]
        else:
            shifted[offset:, :] = mask[:-offset, :]
        result |= shifted
    return result


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Two-pass 4-connected labelling with union-find (numpy + small python)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    parent = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    next_label = 0
    height, width = mask.shape
    for yy in range(height):
        row = mask[yy]
        for xx in range(width):
            if not row[xx]:
                continue
            up = labels[yy - 1, xx] if yy else 0
            left = labels[yy, xx - 1] if xx else 0
            if not up and not left:
                next_label += 1
                parent.append(next_label)
                labels[yy, xx] = next_label
            elif up and left:
                ru, rl = find(up), find(left)
                labels[yy, xx] = rl
                if ru != rl:
                    parent[ru] = rl
            else:
                labels[yy, xx] = up or left
    flat = labels.ravel()
    for index in range(flat.size):
        if flat[index]:
            flat[index] = find(flat[index])
    unique = np.unique(labels)
    remap = {int(value): i for i, value in enumerate(unique)}
    for index in range(flat.size):
        flat[index] = remap[int(flat[index])]
    return labels, len(unique) - 1


def cluster_columns(boxes, indices, min_x_overlap, max_y_gap_factor):
    order = sorted(indices, key=lambda i: (boxes[i][0], boxes[i][1]))
    columns = []
    for index in order:
        x1, y1, x2, y2 = boxes[index]
        width = max(1, x2 - x1)
        best, best_gap = -1, None
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
        else:
            column = columns[best]
            cx1, cy1, cx2, cy2 = column["bbox"]
            column["bbox"] = [min(cx1, x1), min(cy1, y1), max(cx2, x2), max(cy2, y2)]
            column["members"].append(index)
    return columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--ink-threshold", type=int, default=128)
    parser.add_argument("--line-scale", type=int, default=4, help="work scale for ruling lines")
    parser.add_argument("--min-line-source-px", type=int, default=700)
    parser.add_argument("--min-hline-source-px", type=int, default=280,
                        help="horizontal rules (between stacked ads) are only column-wide")
    parser.add_argument("--cell-label-scale", type=int, default=16)
    parser.add_argument("--min-cell-area-fraction", type=float, default=0.0004)
    parser.add_argument("--display-min-ink-source-px", type=int, default=6000)
    parser.add_argument("--min-x-overlap", type=float, default=0.5)
    parser.add_argument("--max-y-gap-factor", type=float, default=1.2)
    parser.add_argument("--overlay-out", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact = json.loads(Path(args.artifact).read_text())
    regions = artifact["regions"]
    page = np.asarray(Image.open(args.image).convert("L"))
    page_sha = _sha256_file(Path(args.image))
    if page_sha != artifact["image_sha256"]:
        raise SystemExit("page hash does not match artifact image_sha256; refusing")
    height, width = page.shape

    # 1. ruling lines at line_scale, restricted to text-free ink.
    # Detector boxes cover running text; ink outside them is frames, rules,
    # display glyphs, and illustrations. Frame bridging is safe there because
    # no inter-character gaps exist to merge, and a thickness veto afterwards
    # rejects display glyphs and figures.
    ls = args.line_scale
    small = page[::ls, ::ls]
    ink = small > args.ink_threshold

    text_boxes = []
    for region in artifact["regions"]:
        xs = [p["x"] for p in region["polygon"]["points"]]
        ys = [p["y"] for p in region["polygon"]["points"]]
        text_boxes.append((min(xs), min(ys), max(xs), max(ys)))
    text_mask = np.zeros_like(ink)
    for x1, y1, x2, y2 in text_boxes:
        text_mask[max(0, y1 // ls - 1) : -(-y2 // ls) + 1,
                  max(0, x1 // ls - 1) : -(-x2 // ls) + 1] = True
    text_free = ink & ~text_mask

    run = max(3, args.min_line_source_px // ls)
    run_h = max(3, args.min_hline_source_px // ls)
    gap = max(1, 48 // ls)  # bridge frame damage; safe on text-free ink
    thick_limit = max(2, 120 // ls)  # only very fat strokes (>~120 source px) are vetoed

    bridged_h = _dilate_1d(text_free, gap, axis=1)
    horizontal = _dilate_1d(_erode_1d(bridged_h, run_h, axis=1), run_h, axis=1)
    h_thick = _dilate_1d(_erode_1d(horizontal, thick_limit, axis=0), thick_limit, axis=0)
    horizontal &= ~h_thick

    bridged_v = _dilate_1d(text_free, gap, axis=0)
    vertical = _dilate_1d(_erode_1d(bridged_v, run, axis=0), run, axis=0)
    v_thick = _dilate_1d(_erode_1d(vertical, thick_limit, axis=1), thick_limit, axis=1)
    vertical &= ~v_thick

    # extend endpoints so rules meet the frames they abut (closes cell circuits)
    extend = max(2, 64 // ls)
    horizontal = _dilate_1d(horizontal, extend, axis=1)
    vertical = _dilate_1d(vertical, extend, axis=0)
    separator = horizontal | vertical
    # thicken separators slightly so labelling does not leak through hairlines
    separator = _dilate_1d(_dilate_1d(separator, 3, axis=1), 3, axis=0)
    # page border counts as separator
    separator[0, :] = separator[-1, :] = True
    separator[:, 0] = separator[:, -1] = True

    # 2. cells at coarser label scale (block-OR keeps hairline separators)
    cs = max(1, args.cell_label_scale // ls)
    sep_h = separator.shape[0] - separator.shape[0] % cs
    sep_w = separator.shape[1] - separator.shape[1] % cs
    blocked = separator[:sep_h, :sep_w].reshape(sep_h // cs, cs, sep_w // cs, cs).any(axis=(1, 3))
    open_area = ~blocked
    labels, count = _label(open_area)
    label_scale = ls * cs
    min_cell_pixels = args.min_cell_area_fraction * labels.size
    cells = {}
    for value in range(1, count + 1):
        ys, xs = np.nonzero(labels == value)
        if ys.size < min_cell_pixels:
            continue
        cells[value] = {
            "cell_id": len(cells),
            "bbox_source_xyxy": [
                int(xs.min() * label_scale),
                int(ys.min() * label_scale),
                int((xs.max() + 1) * label_scale),
                int((ys.max() + 1) * label_scale),
            ],
            "area_fraction": round(ys.size / labels.size, 5),
        }

    def cell_at(x: int, y: int):
        value = labels[min(labels.shape[0] - 1, y // label_scale),
                       min(labels.shape[1] - 1, x // label_scale)]
        return cells.get(int(value))

    # 3. assign detector boxes to cells
    boxes = []
    for region in regions:
        xs = [p["x"] for p in region["polygon"]["points"]]
        ys = [p["y"] for p in region["polygon"]["points"]]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    box_cell = []
    for x1, y1, x2, y2 in boxes:
        cell = cell_at((x1 + x2) // 2, (y1 + y2) // 2)
        box_cell.append(cell["cell_id"] if cell else None)

    # 4. missed display ink per cell
    ds = 8
    ink8 = page[::ds, ::ds] > args.ink_threshold
    covered = np.zeros_like(ink8)
    for x1, y1, x2, y2 in boxes:
        covered[y1 // ds : max(y1 // ds + 1, -(-y2 // ds)),
                x1 // ds : max(x1 // ds + 1, -(-x2 // ds))] = True
    sep8 = np.kron(separator, np.ones((1, 1), dtype=bool))
    sep8 = separator[:: max(1, ds // ls), :: max(1, ds // ls)]
    sep8 = sep8[: ink8.shape[0], : ink8.shape[1]]
    missed = ink8 & ~covered  # separator-independent: false rules must not eat display glyphs
    miss_labels, miss_count = _label(missed[:: 2, :: 2])  # label at scale 16
    display = []
    min_display = args.display_min_ink_source_px / (16 * 16)
    for value in range(1, miss_count + 1):
        ys, xs = np.nonzero(miss_labels == value)
        if ys.size < min_display:
            continue
        bbox = [int(xs.min() * 16), int(ys.min() * 16), int((xs.max() + 1) * 16), int((ys.max() + 1) * 16)]
        cell = cell_at((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        display.append(
            {
                "kind": "display_or_figure",
                "bbox_source_xyxy": bbox,
                "ink_pixels_scale16": int(ys.size),
                "cell_id": cell["cell_id"] if cell else None,
            }
        )

    # 5. columns within each cell
    from collections import defaultdict

    per_cell = defaultdict(list)
    for index, cell_id in enumerate(box_cell):
        if cell_id is not None:
            per_cell[cell_id].append(index)
    columns = []
    for cell_id, indices in sorted(per_cell.items()):
        for column in cluster_columns(boxes, indices, args.min_x_overlap, args.max_y_gap_factor):
            columns.append(
                {
                    "cell_id": cell_id,
                    "bbox_source_xyxy": column["bbox"],
                    "member_regions": len(column["members"]),
                }
            )

    # 6. overlay
    proxy = Image.open(args.proxy).convert("RGB")
    sx = proxy.width / width
    sy = proxy.height / height
    draw = ImageDraw.Draw(proxy)
    sep_ys, sep_xs = np.nonzero(separator)
    sep_img = Image.new("L", (separator.shape[1], separator.shape[0]), 0)
    sep_pixels = sep_img.load()
    for yy, xx in zip(sep_ys.tolist(), sep_xs.tolist()):
        sep_pixels[xx, yy] = 255
    sep_img = sep_img.resize(proxy.size)
    green = Image.new("RGB", proxy.size, (30, 160, 60))
    proxy = Image.composite(green, proxy, sep_img.point(lambda v: 120 if v else 0))
    draw = ImageDraw.Draw(proxy)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=(220, 40, 40), width=1)
    for column in columns:
        x1, y1, x2, y2 = column["bbox_source_xyxy"]
        draw.rectangle([x1 * sx - 1, y1 * sy - 1, x2 * sx + 1, y2 * sy + 1], outline=(40, 90, 220), width=2)
    for proposal in display:
        x1, y1, x2, y2 = proposal["bbox_source_xyxy"]
        draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=(240, 160, 20), width=3)
    for cell in cells.values():
        x1, y1, x2, y2 = cell["bbox_source_xyxy"]
        draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=(90, 30, 140), width=2)
    overlay_path = Path(args.overlay_out)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    proxy.save(overlay_path)

    unassigned = sum(1 for cell_id in box_cell if cell_id is None)
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_not_qualified",
        "semantic_boundary_authority": "none",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_artifact": {"path": args.artifact, "run": artifact.get("run", {})},
        "page_image": {"path": args.image, "sha256": page_sha},
        "parameters": {
            key: getattr(args, key.replace("-", "_"))
            for key in (
                "ink_threshold", "line_scale", "min_line_source_px", "cell_label_scale",
                "min_cell_area_fraction", "display_min_ink_source_px",
                "min_x_overlap", "max_y_gap_factor",
            )
        },
        "cells": {"count": len(cells), "list": sorted(cells.values(), key=lambda c: c["cell_id"])},
        "detector_boxes": {"count": len(boxes), "unassigned_to_cell": unassigned},
        "display_proposals": {"count": len(display), "list": display},
        "columns": {
            "count": len(columns),
            "singletons": sum(1 for c in columns if c["member_regions"] == 1),
            "list": columns,
        },
        "overlay": {"path": str(overlay_path)},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=1))
    print(json.dumps({
        "cells": len(cells),
        "columns": len(columns),
        "column_singletons": sum(1 for c in columns if c["member_regions"] == 1),
        "display_proposals": len(display),
        "boxes_unassigned": unassigned,
    }))
    print(f"[artifact] {out_path}")
    print(f"[overlay] {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
