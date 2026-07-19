"""Boundary-line cells via probabilistic Hough segments + support filtering.

Rules per the operator definition: long, near-horizontal/vertical dark lines,
white intervals allowed, slight warp allowed. HoughLinesP supplies exactly
that (maxLineGap = intervals, per-segment angles = warp). Each segment is
kept only if it is NOT supported mainly by known text or image ink (OCR det
boxes; layout-model image/doc_title boxes), then near-collinear segments are
chained into rules, rules + page border become the separator mask, and cells
are traced from its complement. NO whitespace cutting: cells follow printed
lines only. Crops are original polarity.

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

SCHEMA_VERSION = "boundary-hough-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chain_segments(segments, axis, join_perp_px, join_gap_px, min_total_px):
    """Chain near-collinear segments (axis 'h' or 'v') into rules."""
    items = []
    for x1, y1, x2, y2 in segments:
        if axis == "h":
            lo, hi, perp = min(x1, x2), max(x1, x2), (y1 + y2) / 2
        else:
            lo, hi, perp = min(y1, y2), max(y1, y2), (x1 + x2) / 2
        items.append([lo, hi, perp])
    items.sort(key=lambda s: (round(s[2] / join_perp_px), s[0]))
    rules = []
    for lo, hi, perp in items:
        merged = False
        for rule in rules:
            if abs(rule[2] - perp) <= join_perp_px and lo - rule[1] <= join_gap_px and hi >= rule[0] - join_gap_px:
                rule[0] = min(rule[0], lo)
                rule[1] = max(rule[1], hi)
                rule[2] = (rule[2] * rule[3] + perp) / (rule[3] + 1)
                rule[3] += 1
                merged = True
                break
        if not merged:
            rules.append([lo, hi, perp, 1])
    return [r for r in rules if r[1] - r[0] >= min_total_px]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--work-scale", type=int, default=4)
    parser.add_argument("--ink-threshold", type=int, default=128)
    parser.add_argument("--text-regions", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--detections-scale", type=float, default=4.0)
    parser.add_argument("--mask-classes", default="image,doc_title")
    parser.add_argument("--hough-threshold", type=int, default=90)
    parser.add_argument("--min-seg-src-px", type=int, default=280)
    parser.add_argument("--max-gap-src-px", type=int, default=60)
    parser.add_argument("--max-angle-deg", type=float, default=4.0)
    parser.add_argument("--max-text-support", type=float, default=0.45)
    parser.add_argument("--max-image-support", type=float, default=0.3)
    parser.add_argument("--join-perp-src-px", type=int, default=28)
    parser.add_argument("--join-gap-src-px", type=int, default=220)
    parser.add_argument("--min-rule-src-px", type=int, default=560)
    parser.add_argument("--endpoint-extend-src-px", type=int, default=80)
    parser.add_argument("--min-cell-area-frac", type=float, default=0.0006)
    parser.add_argument("--min-cell-dim-src-px", type=int, default=180)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--crops-dir", default="")
    parser.add_argument("--overlay-out", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    scale = args.work_scale
    small = source[::scale, ::scale]
    ink = (small > args.ink_threshold).astype(np.uint8) * 255
    height, width = ink.shape

    text_mask = np.zeros(ink.shape, dtype=bool)
    for region in json.loads(Path(args.text_regions).read_text())["regions"]:
        xs = [p["x"] for p in region["polygon"]["points"]]
        ys = [p["y"] for p in region["polygon"]["points"]]
        text_mask[max(0, min(ys) // scale):max(ys) // scale + 1,
                  max(0, min(xs) // scale):max(xs) // scale + 1] = True
    image_mask = np.zeros(ink.shape, dtype=bool)
    wanted = {c for c in args.mask_classes.split(",") if c}
    for det in json.loads(Path(args.detections).read_text()):
        if wanted and det.get("cls") not in wanted:
            continue
        sx = args.detections_scale
        x1, y1, x2, y2 = [v * sx / scale for v in det["proxy_xyxy"]]
        image_mask[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = True

    segments = cv2.HoughLinesP(
        ink, 1, math.pi / 180, args.hough_threshold,
        minLineLength=args.min_seg_src_px // scale,
        maxLineGap=args.max_gap_src_px // scale,
    )
    segments = [] if segments is None else [s[0].tolist() for s in segments]

    h_segments, v_segments = [], []
    rejected = 0
    for x1, y1, x2, y2 in segments:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        if angle <= args.max_angle_deg or angle >= 180 - args.max_angle_deg:
            axis = "h"
        elif abs(angle - 90) <= args.max_angle_deg:
            axis = "v"
        else:
            continue
        length = max(abs(x2 - x1), abs(y2 - y1))
        steps = max(8, int(length))
        ts = np.linspace(0, 1, steps)
        px = np.clip((x1 + ts * (x2 - x1)).astype(int), 0, width - 1)
        py = np.clip((y1 + ts * (y2 - y1)).astype(int), 0, height - 1)
        t_support = text_mask[py, px].mean()
        i_support = image_mask[py, px].mean()
        if t_support > args.max_text_support or i_support > args.max_image_support:
            rejected += 1
            continue
        (h_segments if axis == "h" else v_segments).append([x1, y1, x2, y2])

    h_rules = chain_segments(h_segments, "h", args.join_perp_src_px // scale,
                             args.join_gap_src_px // scale, args.min_rule_src_px // scale)
    v_rules = chain_segments(v_segments, "v", args.join_perp_src_px // scale,
                             args.join_gap_src_px // scale, args.min_rule_src_px // scale)

    # merge parallel duplicates: Hough fires on both edges of one thick rule
    def merge_parallel(rules, perp_tol):
        rules = sorted(rules, key=lambda r: r[2])
        merged = []
        for rule in rules:
            if merged:
                last = merged[-1]
                overlap = min(rule[1], last[1]) - max(rule[0], last[0])
                shorter = max(1, min(rule[1] - rule[0], last[1] - last[0]))
                if abs(rule[2] - last[2]) <= perp_tol and overlap / shorter >= 0.5:
                    last[0] = min(last[0], rule[0])
                    last[1] = max(last[1], rule[1])
                    last[2] = (last[2] + rule[2]) / 2
                    last[3] += rule[3]
                    continue
            merged.append(list(rule))
        return merged
    perp_tol = max(2, 44 // scale)
    h_rules = merge_parallel(h_rules, perp_tol)
    v_rules = merge_parallel(v_rules, perp_tol)

    # completion: masks decided a rule EXISTS; raw ink decides its EXTENT.
    # Follow the printed line through dashes and past masked stretches.
    ink_bool = ink > 0
    gap_tol = max(2, args.max_gap_src_px * 3 // scale)
    band = 2
    def complete(rules, axis):
        for rule in rules:
            perp = int(round(rule[2]))
            if axis == "h":
                strip = ink_bool[max(0, perp - band):perp + band + 1, :]
            else:
                strip = ink_bool[:, max(0, perp - band):perp + band + 1].T
            profile = strip.any(axis=0)
            n = profile.size
            lo, hi = int(rule[0]), int(rule[1])
            pos = lo
            gap = 0
            while pos - 1 >= 0 and gap <= gap_tol:
                pos -= 1
                gap = 0 if profile[pos] else gap + 1
            rule[0] = min(lo, pos + gap)
            pos = hi
            gap = 0
            while pos + 1 < n and gap <= gap_tol:
                pos += 1
                gap = 0 if profile[pos] else gap + 1
            rule[1] = max(hi, pos - gap)
        return rules
    h_rules = complete(h_rules, "h")
    v_rules = complete(v_rules, "v")

    # structural snapping: a rule's endpoint extends to the nearest crossing
    # perpendicular rule within snap range (newspaper rules terminate at rules)
    snap = max(args.endpoint_extend_src_px, 320) // scale
    def snap_rules(primary, cross, tol):
        for rule in primary:
            lo, hi, perp = rule[0], rule[1], rule[2]
            for c_lo, c_hi, c_perp, _ in cross:
                if c_lo - tol <= perp <= c_hi + tol:
                    if 0 < lo - c_perp <= snap:
                        rule[0] = c_perp
                    if 0 < c_perp - hi <= snap:
                        rule[1] = c_perp
        return primary
    tol = max(2, 40 // scale)
    for _ in range(2):
        h_rules = snap_rules(h_rules, v_rules, tol)
        v_rules = snap_rules(v_rules, h_rules, tol)

    separator = np.zeros(ink.shape, dtype=np.uint8)
    extend = max(2, 24 // scale)
    for lo, hi, perp, _ in h_rules:
        cv2.line(separator, (int(lo - extend), int(perp)), (int(hi + extend), int(perp)), 255, 3)
    for lo, hi, perp, _ in v_rules:
        cv2.line(separator, (int(perp), int(lo - extend)), (int(perp), int(hi + extend)), 255, 3)
    separator[0, :] = separator[-1, :] = 255
    separator[:, 0] = separator[:, -1] = 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (separator == 0).astype(np.uint8), connectivity=4)
    cells = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < args.min_cell_area_frac * labels.size:
            continue
        if min(w, h) * scale < args.min_cell_dim_src_px:
            continue
        cells.append({
            "cell_id": len(cells),
            "bbox_source_xyxy": [int(x * scale), int(y * scale),
                                 int((x + w) * scale), int((y + h) * scale)],
            "area_fraction": round(float(area) / labels.size, 5),
        })

    crops_dir = Path(args.crops_dir) if args.crops_dir else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
        for cell in cells:
            x1, y1, x2, y2 = cell["bbox_source_xyxy"]
            path = crops_dir / f"cell{cell['cell_id']:02d}_{x2-x1}x{y2-y1}.png"
            cv2.imwrite(str(path), source[y1:y2, x1:x2])
            cell["crop_path"] = str(path)

    proxy = cv2.imread(args.proxy, cv2.IMREAD_COLOR)
    px_ratio = proxy.shape[1] / width / scale
    py_ratio = proxy.shape[0] / height / scale
    for lo, hi, perp, _ in h_rules:
        cv2.line(proxy, (int(lo * scale * px_ratio), int(perp * scale * py_ratio)),
                 (int(hi * scale * px_ratio), int(perp * scale * py_ratio)), (60, 180, 60), 3)
    for lo, hi, perp, _ in v_rules:
        cv2.line(proxy, (int(perp * scale * px_ratio), int(lo * scale * py_ratio)),
                 (int(perp * scale * px_ratio), int(hi * scale * py_ratio)), (60, 180, 60), 3)
    for cell in cells:
        x1, y1, x2, y2 = cell["bbox_source_xyxy"]
        cv2.rectangle(proxy, (int(x1 * px_ratio), int(y1 * py_ratio)),
                      (int(x2 * px_ratio), int(y2 * py_ratio)), (30, 30, 220), 3)
        cv2.putText(proxy, str(cell["cell_id"]), (int(x1 * px_ratio) + 4, int(y1 * py_ratio) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 220), 2)
    Path(args.overlay_out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.overlay_out, proxy)

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_not_qualified",
        "semantic_boundary_authority": "none",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "page_image": {"path": args.image, "sha256": _sha256_file(Path(args.image))},
        "parameters": {key: getattr(args, key.replace("-", "_"))
                       for key in ("work_scale", "ink_threshold", "hough_threshold",
                                   "min_seg_src_px", "max_gap_src_px", "max_angle_deg",
                                   "max_text_support", "max_image_support",
                                   "join_perp_src_px", "join_gap_src_px", "min_rule_src_px",
                                   "endpoint_extend_src_px", "min_cell_dim_src_px")},
        "segments": {"raw": len(segments), "rejected_by_support": rejected,
                     "h_kept": len(h_segments), "v_kept": len(v_segments)},
        "rules": {"horizontal": len(h_rules), "vertical": len(v_rules)},
        "cells": {"count": len(cells), "list": cells},
        "overlay": {"path": args.overlay_out},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=1))
    print(json.dumps({"segments": len(segments), "rejected": rejected,
                      "h_rules": len(h_rules), "v_rules": len(v_rules), "cells": len(cells)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
