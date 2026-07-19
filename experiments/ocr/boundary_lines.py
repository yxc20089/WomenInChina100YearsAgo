"""Detect printed boundary rules of a broadsheet page and trace layout cells.

Implements the operator-described boundary definition: rules are LONG ink
runs, ROUGHLY horizontal or vertical (page warp allowed), THIN, and may have
short white intervals. Pipeline, all deterministic (OpenCV):

1. binarize the negative scan (ink = bright);
2. directional morphological closing bridges the white intervals;
3. directional opening keeps only long runs;
4. connected-component shape filter: keep components that are long, thin and
   axis-dominant — this passes warped/dashed rules and frame borders while
   rejecting text columns and display glyphs;
5. the kept components plus the page border form the separator mask; cells
   are traced as connected components of its complement;
6. optional: group externally supplied layout detections (e.g.
   PP-DocLayoutV2 elements) into per-cell blocks.

Outputs overlay PNG + JSON. Diagnostic proposals only; no semantic authority.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SCHEMA_VERSION = "boundary-lines-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_lines(ink: np.ndarray, axis: str, close_px: int, min_len_px: int,
                 max_thick_px: int, min_aspect: float,
                 isolation_band_px: int = 6, max_neighbor_ink: float = 0.15):
    """Line components along one axis; returns (mask, component boxes).

    A candidate must be ISOLATED: the bands immediately beside it (perpendicular
    to its direction) must be nearly ink-free. This rejects aligned character
    strokes, which always sit inside dense glyph ink, while true rules and
    frame borders have clear space alongside.
    """
    if axis == "h":
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, 1))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len_px, 1))
    else:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_px))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len_px))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, close_kernel)
    lines = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    keep = np.zeros_like(lines)
    boxes = []
    band = isolation_band_px
    page_h, page_w = ink.shape
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        major, minor = (w, h) if axis == "h" else (h, w)
        # thickness estimated from area/major: robust for warped diagonal bboxes
        est_thick = area / max(1, major)
        if not (major >= min_len_px and est_thick <= max_thick_px
                and major / max(1, minor) >= min_aspect):
            continue
        if axis == "h":
            above = ink[max(0, y - band):y, x:x + w]
            below = ink[y + h:min(page_h, y + h + band), x:x + w]
        else:
            above = ink[y:y + h, max(0, x - band):x]
            below = ink[y:y + h, x + w:min(page_w, x + w + band)]
        neighbor = np.concatenate([above.ravel(), below.ravel()])
        if neighbor.size and (neighbor > 0).mean() > max_neighbor_ink:
            continue
        keep[labels == index] = 255
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return keep, boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="lossless page (negative polarity)")
    parser.add_argument("--work-scale", type=int, default=4)
    parser.add_argument("--ink-threshold", type=int, default=128)
    parser.add_argument("--close-src-px", type=int, default=80,
                        help="max white interval bridged inside one rule")
    parser.add_argument("--min-len-src-px", type=int, default=400)
    parser.add_argument("--max-thick-src-px", type=int, default=48)
    parser.add_argument("--min-aspect", type=float, default=6.0)
    parser.add_argument("--endpoint-extend-src-px", type=int, default=72)
    parser.add_argument("--isolation-band-src-px", type=int, default=8)
    parser.add_argument("--max-neighbor-ink", type=float, default=0.2)
    parser.add_argument("--min-cell-area-frac", type=float, default=0.0006)
    parser.add_argument("--detections", default="", help="optional layout detections JSON (proxy coords)")
    parser.add_argument("--text-regions", default="",
                        help="OCR artifact with det polygons; masked out before line detection")
    parser.add_argument("--mask-shrink-src-px", type=int, default=12,
                        help="shrink layout-detection masks so frame edges survive")
    parser.add_argument("--mask-classes", default="",
                        help="comma list of detection classes to mask (empty = all)")
    parser.add_argument("--support-filter", action="store_true",
                        help="detect lines on full ink; reject candidates mostly inside text/image masks instead of erasing ink first")
    parser.add_argument("--max-text-support", type=float, default=0.5)
    parser.add_argument("--max-image-support", type=float, default=0.3)
    parser.add_argument("--detections-scale", type=float, default=4.0,
                        help="source px per detection-proxy px")
    parser.add_argument("--proxy", required=True, help="black-on-white proxy for the overlay")
    parser.add_argument("--overlay-out", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    scale = args.work_scale
    small = source[::scale, ::scale]
    ink = ((small > args.ink_threshold).astype(np.uint8)) * 255

    text_mask = np.zeros_like(ink, dtype=bool)
    image_mask = np.zeros_like(ink, dtype=bool)
    line_ink = ink.copy()
    if args.text_regions:
        artifact = json.loads(Path(args.text_regions).read_text())
        for region in artifact["regions"]:
            xs = [p["x"] for p in region["polygon"]["points"]]
            ys = [p["y"] for p in region["polygon"]["points"]]
            text_mask[max(0, min(ys) // scale):max(ys) // scale + 1,
                      max(0, min(xs) // scale):max(xs) // scale + 1] = True
    if args.detections:
        shrink = args.mask_shrink_src_px
        wanted = {c for c in args.mask_classes.split(",") if c}
        for det in json.loads(Path(args.detections).read_text()):
            if wanted and det.get("cls") not in wanted:
                continue
            sx = args.detections_scale
            x1, y1, x2, y2 = [v * sx for v in det["proxy_xyxy"]]
            if x2 - x1 > 2 * shrink and y2 - y1 > 2 * shrink:
                image_mask[int((y1 + shrink) / scale):int((y2 - shrink) / scale),
                           int((x1 + shrink) / scale):int((x2 - shrink) / scale)] = True

    close_px = max(3, args.close_src_px // scale)
    min_len = max(8, args.min_len_src_px // scale)
    max_thick = max(2, args.max_thick_src_px // scale)

    iso = max(1, args.isolation_band_src_px // scale)
    if args.support_filter:
        line_ink = ink.copy()
    else:
        line_ink = ink.copy()
        line_ink[text_mask | image_mask] = 0
    h_mask, h_boxes = detect_lines(line_ink, "h", close_px, min_len, max_thick, args.min_aspect, iso, args.max_neighbor_ink)
    v_mask, v_boxes = detect_lines(line_ink, "v", close_px, min_len, max_thick, args.min_aspect, iso, args.max_neighbor_ink)
    if args.support_filter:
        def support_ok(mask):
            keep = np.zeros_like(mask)
            n, lab = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
            kept_boxes = []
            for i in range(1, n):
                comp = lab == i
                size = comp.sum()
                if not size:
                    continue
                t = (comp & text_mask).sum() / size
                m = (comp & image_mask).sum() / size
                if t <= args.max_text_support and m <= args.max_image_support:
                    keep[comp] = 255
                    ys, xs = np.nonzero(comp)
                    kept_boxes.append([int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)])
            return keep, kept_boxes
        h_mask, h_boxes = support_ok(h_mask)
        v_mask, v_boxes = support_ok(v_mask)

    extend = max(2, args.endpoint_extend_src_px // scale)
    h_ext = cv2.dilate(h_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * extend + 1, 3)))
    v_ext = cv2.dilate(v_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2 * extend + 1)))
    separator = cv2.bitwise_or(h_ext, v_ext)
    separator[0, :] = separator[-1, :] = 255
    separator[:, 0] = separator[:, -1] = 255

    open_area = (separator == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(open_area, connectivity=4)
    total = open_area.size
    cells = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < args.min_cell_area_frac * total:
            continue
        cells.append({
            "cell_id": len(cells),
            "bbox_source_xyxy": [int(x * scale), int(y * scale),
                                 int((x + w) * scale), int((y + h) * scale)],
            "area_fraction": round(area / total, 5),
            "label": int(index),
        })

    blocks = []
    detections = []
    if args.detections:
        detections = json.loads(Path(args.detections).read_text())
        label_of = {c["label"]: c["cell_id"] for c in cells}
        per_cell = {}
        for det in detections:
            sx = args.detections_scale
            x1, y1, x2, y2 = [v * sx for v in det["proxy_xyxy"]]
            cx = min(labels.shape[1] - 1, int((x1 + x2) / 2 / scale))
            cy = min(labels.shape[0] - 1, int((y1 + y2) / 2 / scale))
            cell_id = label_of.get(int(labels[cy, cx]))
            det["cell_id"] = cell_id
            if cell_id is not None:
                per_cell.setdefault(cell_id, []).append([x1, y1, x2, y2])
        for cell_id, boxes in sorted(per_cell.items()):
            arr = np.array(boxes)
            blocks.append({
                "cell_id": cell_id,
                "bbox_source_xyxy": [int(arr[:, 0].min()), int(arr[:, 1].min()),
                                     int(arr[:, 2].max()), int(arr[:, 3].max())],
                "member_detections": len(boxes),
            })

    proxy = cv2.imread(args.proxy, cv2.IMREAD_COLOR)
    px = proxy.shape[1] / source.shape[1]
    py = proxy.shape[0] / source.shape[0]
    line_overlay = cv2.resize(cv2.bitwise_or(h_mask, v_mask),
                              (proxy.shape[1], proxy.shape[0]), interpolation=cv2.INTER_NEAREST)
    proxy[line_overlay > 0] = (60, 180, 60)
    for cell in cells:
        x1, y1, x2, y2 = cell["bbox_source_xyxy"]
        cv2.rectangle(proxy, (int(x1 * px), int(y1 * py)), (int(x2 * px), int(y2 * py)),
                      (160, 30, 120), 2)
    for block in blocks:
        x1, y1, x2, y2 = block["bbox_source_xyxy"]
        cv2.rectangle(proxy, (int(x1 * px), int(y1 * py)), (int(x2 * px), int(y2 * py)),
                      (30, 30, 220), 3)
    Path(args.overlay_out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.overlay_out, proxy)

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_not_qualified",
        "semantic_boundary_authority": "none",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "page_image": {"path": args.image, "sha256": _sha256_file(Path(args.image))},
        "parameters": {
            key: getattr(args, key)
            for key in ("work_scale", "ink_threshold", "close_src_px", "min_len_src_px",
                        "max_thick_src_px", "min_aspect", "endpoint_extend_src_px",
                        "min_cell_area_frac")
        },
        "lines": {
            "horizontal": len(h_boxes),
            "vertical": len(v_boxes),
            "horizontal_boxes_workscale": h_boxes,
            "vertical_boxes_workscale": v_boxes,
        },
        "cells": {"count": len(cells), "list": [
            {k: c[k] for k in ("cell_id", "bbox_source_xyxy", "area_fraction")} for c in cells
        ]},
        "blocks": {"count": len(blocks), "list": blocks},
        "overlay": {"path": args.overlay_out},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=1))
    print(json.dumps({"h_lines": len(h_boxes), "v_lines": len(v_boxes),
                      "cells": len(cells), "blocks": len(blocks)}))
    print(f"[artifact] {args.output}")
    print(f"[overlay] {args.overlay_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
