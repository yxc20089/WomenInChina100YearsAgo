"""Canonical morphological frame extraction: rules -> sealed mask -> cell tree.

The industry-converged lattice recipe (Camelot/img2table family), per the
2026-07-19 method research:

1. deskew once, using the median angle of the page's own long rules;
2. directional MORPH_CLOSE with perpendicular dimension 1 seals breaks in a
   rule without fusing nearby parallel rules (nested frames survive);
3. long directional opening keeps only long runs; a thinness gate
   (length:thickness) rejects display-glyph strokes;
4. the sealed union mask + page border is traced with
   findContours(RETR_TREE): HOLE contours are the cells, and the hierarchy
   records nesting (frames inside frames) for free;
5. cells become crops from the deskewed page (original polarity), with the
   deskew angle recorded for provenance.

Diagnostic proposals for review; no semantic authority.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

SCHEMA_VERSION = "frame-rectangles-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directional_lines(ink: np.ndarray, axis: str, close_px: int, open_px: int,
                      max_thick_px: int, min_aspect: float) -> np.ndarray:
    if axis == "h":
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, 1))
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_px, 1))
    else:
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_px))
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, open_px))
    sealed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, close_k)
    lines = cv2.morphologyEx(sealed, cv2.MORPH_OPEN, open_k)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    keep = np.zeros_like(lines)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        major = w if axis == "h" else h
        thickness = area / max(1, major)
        if thickness <= max_thick_px and major / max(1.0, thickness) >= min_aspect:
            keep[labels == index] = 255
    return keep


def estimate_skew(ink: np.ndarray, open_px: int) -> float:
    """Median angle (degrees) of long horizontal rules; 0 when none found."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_px, 1))
    rough = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    segments = cv2.HoughLinesP(rough, 1, math.pi / 720, 60,
                               minLineLength=open_px * 2, maxLineGap=open_px // 4)
    if segments is None:
        return 0.0
    angles = []
    for (x1, y1, x2, y2), in segments:
        if x2 != x1:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) <= 5:
                angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--work-scale", type=int, default=4)
    parser.add_argument("--ink-threshold", type=int, default=128)
    parser.add_argument("--close-src-px", type=int, default=240)
    parser.add_argument("--open-frac", type=float, default=20.0,
                        help="opening kernel = dimension/this")
    parser.add_argument("--max-thick-src-px", type=int, default=22)
    parser.add_argument("--min-aspect", type=float, default=20.0)
    parser.add_argument("--seal-src-px", type=int, default=100,
                        help="junction sealing dilation before tracing")
    parser.add_argument("--min-cell-dim-src-px", type=int, default=180)
    parser.add_argument("--min-cell-area-frac", type=float, default=0.0004)
    parser.add_argument("--text-regions", default="",
                        help="OCR artifact with det polygons for the support filter")
    parser.add_argument("--detections", default="", help="layout detections JSON (proxy coords)")
    parser.add_argument("--detections-scale", type=float, default=4.0)
    parser.add_argument("--mask-classes", default="image,doc_title")
    parser.add_argument("--max-text-support", type=float, default=0.45)
    parser.add_argument("--max-image-support", type=float, default=0.3)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--crops-dir", default="")
    parser.add_argument("--overlay-out", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    scale = args.work_scale
    work = source[::scale, ::scale]
    ink0 = ((work > args.ink_threshold).astype(np.uint8)) * 255
    height, width = ink0.shape

    open_h = max(8, int(width / args.open_frac))
    open_v = max(8, int(height / args.open_frac))

    skew = estimate_skew(ink0, open_h)
    center = (width / 2, height / 2)
    rotation = cv2.getRotationMatrix2D(center, skew, 1.0)
    ink = cv2.warpAffine(ink0, rotation, (width, height), flags=cv2.INTER_NEAREST)
    deskewed_work = cv2.warpAffine(work, rotation, (width, height), flags=cv2.INTER_LINEAR)
    # full-res deskewed page for accurate crops
    src_center = (source.shape[1] / 2, source.shape[0] / 2)
    src_rotation = cv2.getRotationMatrix2D(src_center, skew, 1.0)
    deskewed_source = cv2.warpAffine(source, src_rotation,
                                     (source.shape[1], source.shape[0]),
                                     flags=cv2.INTER_LINEAR)

    close_px = max(3, args.close_src_px // scale)
    thick_px = max(2, args.max_thick_src_px // scale)
    h_lines = directional_lines(ink, "h", close_px, open_h, thick_px, args.min_aspect)
    v_lines = directional_lines(ink, "v", close_px, open_v, thick_px, args.min_aspect)

    # vectorize components into rules, merge parallel duplicates, complete
    # each rule along raw ink (masks/filters decide EXISTENCE, raw ink decides
    # EXTENT), snap endpoints to crossing rules, then draw the sealed mask.
    def to_rules(mask, axis):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        rules = []
        for i in range(1, count):
            x, y, w, h, _ = stats[i]
            if axis == "h":
                rules.append([x, x + w, y + h / 2, 1])
            else:
                rules.append([y, y + h, x + w / 2, 1])
        return rules

    def merge_parallel(rules, perp_tol):
        rules = sorted(rules, key=lambda r: r[2])
        merged = []
        for rule in rules:
            if merged:
                last = merged[-1]
                overlap = min(rule[1], last[1]) - max(rule[0], last[0])
                shorter = max(1, min(rule[1] - rule[0], last[1] - last[0]))
                if abs(rule[2] - last[2]) <= perp_tol and overlap / shorter >= 0.3:
                    last[0] = min(last[0], rule[0]); last[1] = max(last[1], rule[1])
                    last[2] = (last[2] + rule[2]) / 2; last[3] += rule[3]
                    continue
            merged.append(list(rule))
        return merged

    ink_bool = ink > 0
    gap_tol = max(2, args.seal_src_px * 2 // scale)
    def complete(rules, axis):
        for rule in rules:
            perp = int(round(rule[2]))
            if axis == "h":
                strip = ink_bool[max(0, perp - 2):perp + 3, :]
            else:
                strip = ink_bool[:, max(0, perp - 2):perp + 3].T
            profile = strip.any(axis=0)
            n = profile.size
            lo, hi = int(rule[0]), int(rule[1])
            pos, gap = lo, 0
            while pos - 1 >= 0 and gap <= gap_tol:
                pos -= 1
                gap = 0 if profile[pos] else gap + 1
            rule[0] = min(lo, pos + gap)
            pos, gap = hi, 0
            while pos + 1 < n and gap <= gap_tol:
                pos += 1
                gap = 0 if profile[pos] else gap + 1
            rule[1] = max(hi, pos - gap)
        return rules

    snap = max(4, 360 // scale)
    def snap_rules(primary, cross, tol):
        for rule in primary:
            for c_lo, c_hi, c_perp, _ in cross:
                if c_lo - tol <= rule[2] <= c_hi + tol:
                    if 0 < rule[0] - c_perp <= snap:
                        rule[0] = c_perp
                    if 0 < c_perp - rule[1] <= snap:
                        rule[1] = c_perp
        return primary

    text_mask = np.zeros(ink.shape, dtype=bool)
    if args.text_regions:
        for region in json.loads(Path(args.text_regions).read_text())["regions"]:
            xs = [p["x"] for p in region["polygon"]["points"]]
            ys = [p["y"] for p in region["polygon"]["points"]]
            text_mask[max(0, min(ys) // scale):max(ys) // scale + 1,
                      max(0, min(xs) // scale):max(xs) // scale + 1] = True
    image_mask = np.zeros(ink.shape, dtype=bool)
    if args.detections:
        wanted = {c for c in args.mask_classes.split(",") if c}
        for det in json.loads(Path(args.detections).read_text()):
            if wanted and det.get("cls") not in wanted:
                continue
            sx = args.detections_scale
            x1, y1, x2, y2 = [v * sx / scale for v in det["proxy_xyxy"]]
            image_mask[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = True
    if skew:
        text_mask = cv2.warpAffine(text_mask.astype(np.uint8), rotation,
                                   (width, height), flags=cv2.INTER_NEAREST).astype(bool)
        image_mask = cv2.warpAffine(image_mask.astype(np.uint8), rotation,
                                    (width, height), flags=cv2.INTER_NEAREST).astype(bool)

    def support_filter(rules, axis):
        kept = []
        for rule in rules:
            perp = int(round(rule[2]))
            lo, hi = int(rule[0]), int(rule[1])
            if axis == "h":
                t = text_mask[perp, lo:hi]
                m = image_mask[perp, lo:hi]
            else:
                t = text_mask[lo:hi, perp]
                m = image_mask[lo:hi, perp]
            if t.size and (t.mean() > args.max_text_support or m.mean() > args.max_image_support):
                continue
            kept.append(rule)
        return kept

    perp_tol = max(2, 44 // scale)
    h_rules = support_filter(merge_parallel(to_rules(h_lines, "h"), perp_tol), "h")
    v_rules = support_filter(merge_parallel(to_rules(v_lines, "v"), perp_tol), "v")
    h_rules = complete(h_rules, "h")
    v_rules = complete(v_rules, "v")
    tol = max(2, 40 // scale)
    for _ in range(2):
        h_rules = snap_rules(h_rules, v_rules, tol)
        v_rules = snap_rules(v_rules, h_rules, tol)

    separator = np.zeros_like(ink)
    ext = max(2, 24 // scale)
    for lo, hi, perp, _ in h_rules:
        cv2.line(separator, (int(lo - ext), int(perp)), (int(hi + ext), int(perp)), 255, 3)
    for lo, hi, perp, _ in v_rules:
        cv2.line(separator, (int(perp), int(lo - ext)), (int(perp), int(hi + ext)), 255, 3)
    separator[0:2, :] = separator[-2:, :] = 255
    separator[:, 0:2] = separator[:, -2:] = 255
    h_sealed = separator  # for overlay naming compatibility

    contours, hierarchy = cv2.findContours(separator, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []

    def depth_of(index: int) -> int:
        depth = 0
        parent = hierarchy[index][3]
        while parent != -1:
            depth += 1
            parent = hierarchy[parent][3]
        return depth

    cells = []
    for index, contour in enumerate(contours):
        depth = depth_of(index)
        if depth % 2 == 0:
            continue  # even depth = ink contours (rule rings); odd = holes = cells
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) * scale < args.min_cell_dim_src_px:
            continue
        if w * h < args.min_cell_area_frac * separator.size:
            continue
        cells.append({
            "cell_id": len(cells),
            "contour_index": index,
            "parent_contour": int(hierarchy[index][3]),
            "nesting_depth": (depth - 1) // 2,
            "bbox_deskewed_source_xyxy": [x * scale, y * scale,
                                          (x + w) * scale, (y + h) * scale],
            "area_fraction": round(w * h / separator.size, 5),
        })

    crops_dir = Path(args.crops_dir) if args.crops_dir else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
        for cell in cells:
            x1, y1, x2, y2 = cell["bbox_deskewed_source_xyxy"]
            path = crops_dir / (
                f"cell{cell['cell_id']:02d}_d{cell['nesting_depth']}_{x2-x1}x{y2-y1}.png")
            cv2.imwrite(str(path), deskewed_source[y1:y2, x1:x2])
            cell["crop_path"] = str(path)

    overlay = cv2.cvtColor(255 - deskewed_work, cv2.COLOR_GRAY2BGR)
    lines_view = cv2.bitwise_or(h_lines, v_lines)
    overlay[lines_view > 0] = (60, 180, 60)
    palette = [(30, 30, 220), (30, 140, 220), (160, 30, 200)]
    for cell in cells:
        x1, y1, x2, y2 = [v // scale for v in cell["bbox_deskewed_source_xyxy"]]
        color = palette[min(cell["nesting_depth"], len(palette) - 1)]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, str(cell["cell_id"]), (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    Path(args.overlay_out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.overlay_out, overlay)

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_not_qualified",
        "semantic_boundary_authority": "none",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "page_image": {"path": args.image, "sha256": _sha256_file(Path(args.image))},
        "deskew_degrees": round(skew, 4),
        "parameters": {
            key: getattr(args, key.replace("-", "_"))
            for key in ("work_scale", "ink_threshold", "close_src_px", "open_frac",
                        "max_thick_src_px", "min_aspect", "seal_src_px",
                        "min_cell_dim_src_px", "min_cell_area_frac")
        },
        "cells": {"count": len(cells),
                  "by_depth": {},
                  "list": cells},
        "overlay": {"path": args.overlay_out},
    }
    from collections import Counter
    output["cells"]["by_depth"] = dict(
        Counter(c["nesting_depth"] for c in cells))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=1))
    print(json.dumps({"skew_deg": round(skew, 3), "cells": len(cells),
                      "by_depth": output["cells"]["by_depth"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
