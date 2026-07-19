"""Detect printed boundary rules and propose rectangular cells (v7).

Attribute-based rule detection per the measured stroke taxonomy
(rule_attributes.measure_line): candidates from Hough chaining are accepted
only when they match gold-measured rule physics — object continuity (interior
gaps at print-damage scale <=12px, fill >=0.95), straightness, per-class
thickness/uniformity/flank profiles, and at least one blank flank
(flank_min <= 0.25; a separator separates content, figure texture is dense on
both sides — page-frame rules are exempt because the out-of-page void reads
as ink in the inverted render).

Accepted rules are seeded at their measured object extent, completed by an
object-physics walker (free continuation only across <=12px damage; a
13-90px break is crossed only when a dense collinear continuation lies
beyond it), merged (<=12px = same rule; 13-44px only when the pair is not
double-inked — a warped rule has one centerline, a real stepped pair keeps
both), junction-snapped, and flood-filled into cells.

Diagnostic proposals scored against an operator gold labeling when given;
no semantic authority. Hunyuan remains the sole transcription authority.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from boundary_hough import chain_segments  # noqa: E402
from rule_attributes import measure_line  # noqa: E402

SCHEMA_VERSION = "rule-detector-v7"


def classify_full(m, frame_zone=False):
    if not m or not m.get("present"):
        return ("no_ink", "")
    if m["length"] < 500:
        return ("rejected_short", "")
    if m["straightness_std"] is None or m["straightness_std"] > 4.0:
        return ("rejected_crooked", f"resid {m['straightness_std']}")
    # object-level continuity (operator definition): a rule's interior gaps
    # are print damage of a few px; fill near 1.0 by construction
    if m["fill"] < 0.95:
        return ("rejected_discontinuous", f"fill {m['fill']:.2f}")
    if (m["max_gap"] or 0) > 12:
        return ("rejected_large_gap", f"gap {m['max_gap']}px")
    # a separator has at least one blank side (it separates content); texture
    # embedded in a figure is dense on BOTH sides. Gold flank_min max 0.219;
    # p0308 sun-figure impostor 0.316. Page-frame rules are exempt: the
    # out-of-page void reads as ink in this inverted render.
    if not frame_zone and (m.get("flank_min") or 0) > 0.25:
        return ("rejected_embedded", f"flank_min {m['flank_min']:.2f}")
    u = m["thickness_uniformity"] or 9
    p90 = m["p90_thickness"] or 99
    if m["median_thickness"] <= 28 and p90 <= 42 and u <= 2.0 and m["flank_ink"] <= 0.45:
        return ("thin_rule" if m["median_thickness"] <= 12 else "medium_rule", "ok")
    if m["median_thickness"] > 28 and u <= 1.8 and m["flank_ink"] <= 0.25:
        return ("thick_band", "")
    # textured/brush-printed rule: uniformity fails on ragged edges, but a
    # long straight continuous ISOLATED stroke can only be a rule
    if m["median_thickness"] <= 45 and m["flank_ink"] <= 0.20:
        return ("rough_rule", "isolated textured")
    return ("rejected_text_signature", f"u{u:.2f} p90 {p90:.0f}")


def coverage(lo, hi, perp, axis, boxes):
    cov = 0
    for x1, y1, x2, y2 in boxes:
        if axis == "h" and y1 - 8 <= perp <= y2 + 8:
            cov += max(0, min(hi, x2) - max(lo, x1))
        if axis == "v" and x1 - 8 <= perp <= x2 + 8:
            cov += max(0, min(hi, y2) - max(lo, y1))
    return cov / max(1, hi - lo)


def walk(ink, axis, center_perp, col, direction,
         damage_gap=12, break_max=90, lookahead=80, look_fill=0.70):
    """Object-physics walker: free continuation only across damage-scale gaps;
    a larger break is crossed ONLY when a dense collinear continuation lies
    directly beyond it. Otherwise the object ends."""
    H, W = ink.shape
    n = W if axis == "h" else H
    moving = float(center_perp)
    gap = 0
    last = col

    def ink_at(cc, c):
        seg = ink[max(0, c - 4):c + 5, cc] if axis == "h" else ink[cc, max(0, c - 4):c + 5]
        return np.nonzero(seg)[0], max(0, c - 4)

    while 0 <= col + direction < n:
        col += direction
        c = int(round(moving))
        near, base = ink_at(col, c)
        if near.size:
            moving = 0.85 * moving + 0.15 * (near[len(near) // 2] + base)
            gap = 0
            last = col
            continue
        gap += 1
        if gap <= damage_gap:
            continue
        jumped = False
        probe = col
        while abs(probe - last) <= break_max and 0 <= probe + direction < n:
            probe += direction
            near2, _ = ink_at(probe, int(round(moving)))
            if near2.size:
                hits = 0
                total = 0
                pp = probe
                while total < lookahead and 0 <= pp + direction < n:
                    n3, _ = ink_at(pp, int(round(moving)))
                    hits += 1 if n3.size else 0
                    total += 1
                    pp += direction
                if total and hits / total >= look_fill:
                    col = probe
                    gap = 0
                    last = probe
                    jumped = True
                break
        if not jumped:
            break
    return last


def detect(src, ink, mask_boxes):
    ink4 = ((src[::4, ::4] > 128).astype(np.uint8)) * 255
    segs = cv2.HoughLinesP(ink4, 1, math.pi / 180, 60, minLineLength=50, maxLineGap=18)
    segs = [] if segs is None else [x[0].tolist() for x in segs]
    h_s, v_s = [], []
    for x1, y1, x2, y2 in segs:
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        if ang <= 4 or ang >= 176:
            h_s.append([x1, y1, x2, y2])
        elif abs(ang - 90) <= 4:
            v_s.append([x1, y1, x2, y2])
    counts = Counter()
    accepted = {"h": [], "v": []}
    for axis, ch in (("h", chain_segments(h_s, "h", 7, 22, 60)),
                     ("v", chain_segments(v_s, "v", 7, 22, 60))):
        for lo, hi, perp, _ in ch:
            LO, HI, P = int(lo * 4), int(hi * 4), int(perp * 4)
            if coverage(LO, HI, P, axis, mask_boxes) > 0.4:
                counts["masked"] += 1
                continue
            m = measure_line(ink, axis, P, LO, HI, band=40)
            dim = src.shape[0] if axis == "h" else src.shape[1]
            cls, _why = classify_full(m, frame_zone=(P < 250 or P > dim - 250))
            counts[cls] += 1
            if cls in ("thin_rule", "medium_rule", "thick_band", "rough_rule"):
                # seed = the measured continuous object, not the loose Hough span
                sLO = LO + m["obj_offset_lo"]
                sHI = LO + m["obj_offset_hi"]
                nlo = walk(ink, axis, P, sLO, -1)
                nhi = walk(ink, axis, P, sHI, +1)
                accepted[axis].append([nlo, nhi, P, cls])
    return accepted, counts


def double_ink(ink, axis, p1, p2, a, b):
    """Fraction of overlap columns with ink at BOTH perp positions: a warped
    single rule has one centerline (never both); a real stepped/double pair
    (p0308 y4457+y4492) has two. Decides same-rule vs distinct at merge."""
    both = 0
    n = 0
    for c in range(a, b, 4):
        n += 1
        if axis == "h":
            i1 = ink[max(0, p1 - 5):p1 + 6, c].any()
            i2 = ink[max(0, p2 - 5):p2 + 6, c].any()
        else:
            i1 = ink[c, max(0, p1 - 5):p1 + 6].any()
            i2 = ink[c, max(0, p2 - 5):p2 + 6].any()
        if i1 and i2:
            both += 1
    return (both / n) if n else 0.0


def merge(ink, rr, axis):
    # <=12px apart = damage scale, always same rule. 13-44px apart = same rule
    # ONLY if the pair is not double-inked (warp offset, not a real pair).
    # A merged line sits at the LONGEST member's perp position.
    rr = sorted(rr, key=lambda r: r[2])
    out = []
    for r in rr:
        if out:
            q = out[-1]
            ov_lo, ov_hi = max(q[0], r[0]), min(q[1], r[1])
            ov = ov_hi - ov_lo
            close = abs(q[2] - r[2]) <= 12 or (
                abs(q[2] - r[2]) <= 44 and ov > 0
                and double_ink(ink, axis, q[2], r[2], ov_lo, ov_hi) < 0.3)
            if close and ov > 0.3 * min(r[1] - r[0], q[1] - q[0]):
                if r[1] - r[0] > q[1] - q[0]:
                    q[2] = r[2]
                    q[3] = r[3]
                q[0] = min(q[0], r[0])
                q[1] = max(q[1], r[1])
                continue
        out.append(list(r))
    return out


def snap_all(accepted, snap=520, tol=60):
    # blind junction snap: no observed defect came from snap; corridor-gated
    # variants blocked legitimate closures (perpendicular crossing rules read
    # as strokes) and broke the grid (v7 calibration, 2026-07-19)
    def snap_rules(prim, cross):
        for r in prim:
            for c in cross:
                if c[0] - tol <= r[2] <= c[1] + tol:
                    if 0 < r[0] - c[2] <= snap:
                        r[0] = c[2]
                    if 0 < c[2] - r[1] <= snap:
                        r[1] = c[2]
        return prim
    for _ in range(2):
        accepted["h"] = snap_rules(accepted["h"], accepted["v"])
        accepted["v"] = snap_rules(accepted["v"], accepted["h"])
    return accepted


def cells_from(accepted, shape, S=4, min_dim_src=180):
    sep = np.zeros((shape[0] // S, shape[1] // S), np.uint8)
    for lo, hi, p, _ in accepted["h"]:
        cv2.line(sep, (int(lo / S) - 6, int(p / S)), (int(hi / S) + 6, int(p / S)), 255, 3)
    for lo, hi, p, _ in accepted["v"]:
        cv2.line(sep, (int(p / S), int(lo / S) - 6), (int(p / S), int(hi / S) + 6), 255, 3)
    sep[0:2, :] = sep[-2:, :] = 255
    sep[:, 0:2] = sep[:, -2:] = 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats((sep == 0).astype(np.uint8), connectivity=4)
    cells = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if min(w, h) * S < min_dim_src or w * h < 0.0004 * lab.size:
            continue
        # the background/margin component has a near-page bbox; it is free
        # space around the grid, not a crop region
        if w * h > 0.5 * lab.size:
            continue
        cells.append([int(x) * S, int(y) * S, int(x + w) * S, int(y + h) * S])
    # index in the paper's reading order: top-to-bottom bands, right-to-left
    cells.sort(key=lambda c: (c[1] // 300, -c[0]))
    return cells


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="lossless page image (inverted render)")
    parser.add_argument("--detections", default="", help="PP-DocLayoutV2 detections JSON (image/doc_title masks)")
    parser.add_argument("--detections-scale", type=float, default=4.0)
    parser.add_argument("--gold", default="", help="gold blocks JSON for scoring")
    parser.add_argument("--overlay-out", default="")
    parser.add_argument("--show-rules", action="store_true",
                        help="also draw accepted rules by class (diagnostic); default shows only final cells")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    ink = src > 128
    mask_boxes = []
    if args.detections:
        dets = json.loads(Path(args.detections).read_text())
        s = args.detections_scale
        mask_boxes = [[v * s for v in d["proxy_xyxy"]] for d in dets
                      if d["cls"] in ("image", "doc_title")]

    accepted, counts = detect(src, ink, mask_boxes)
    for ax in accepted:
        accepted[ax] = merge(ink, accepted[ax], ax)
    accepted = snap_all(accepted)
    cells = cells_from(accepted, src.shape)
    print(dict(counts))
    print(f"rules: {len(accepted['h'])} h + {len(accepted['v'])} v | cells: {len(cells)}")

    score = None
    if args.gold:
        gold = [g["bbox_source_xyxy"] for g in json.loads(Path(args.gold).read_text())["blocks"]]
        scores = [max((iou(g, c) for c in cells), default=0) for g in gold]
        score = {"gold_total": len(gold),
                 "iou_ge_06": sum(1 for v in scores if v >= 0.6),
                 "iou_ge_05": sum(1 for v in scores if v >= 0.5),
                 "median_iou": round(float(np.median(scores)), 3),
                 "per_block": [round(v, 3) for v in scores]}
        print(f"gold IoU>=0.6: {score['iou_ge_06']}/{len(gold)} | median {score['median_iou']}")

    if args.overlay_out:
        S = 4
        over = cv2.cvtColor(src[::S, ::S].copy(), cv2.COLOR_GRAY2BGR)
        if args.show_rules:
            CC = {"thin_rule": (60, 220, 60), "medium_rule": (255, 140, 0),
                  "thick_band": (0, 215, 255), "rough_rule": (0, 255, 255)}
            for ax in ("h", "v"):
                for lo, hi, p, cls in accepted[ax]:
                    col = CC[cls]
                    if ax == "h":
                        cv2.line(over, (lo // S, p // S), (hi // S, p // S), col, 3)
                    else:
                        cv2.line(over, (p // S, lo // S), (p // S, hi // S), col, 3)
        for i, c in enumerate(cells):
            cv2.rectangle(over, (c[0] // S, c[1] // S), (c[2] // S, c[3] // S), (30, 30, 220), 2)
            cx, cy = (c[0] + c[2]) // (2 * S), (c[1] + c[3]) // (2 * S)
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            cv2.rectangle(over, (cx - tw // 2 - 6, cy - th // 2 - 6),
                          (cx + tw // 2 + 6, cy + th // 2 + 6), (255, 255, 255), -1)
            cv2.putText(over, label, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 220), 3)
        Path(args.overlay_out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.overlay_out, over)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"schema_version": SCHEMA_VERSION,
         "status": "diagnostic_not_qualified",
         "semantic_boundary_authority": "none",
         "image": args.image,
         "counts": dict(counts),
         "rules": accepted,
         "cells": cells,
         "score_vs_gold": score}, indent=1, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
