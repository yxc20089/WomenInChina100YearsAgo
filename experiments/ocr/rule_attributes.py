"""Measure physical attributes of printed boundary rules vs confusers.

Hypothesis (operator, 2026-07-19): true boundary rules on this press share
measurable attributes (thickness class, fill/continuity, gap structure,
straightness) that separate them from look-alikes such as horizontal strokes
of display glyphs and solid regions of figures — even under period printing
imperfections. This tool measures, it does not decide.

Given a page image, a gold block labeling (edges = candidate rule positions)
and optional layout detections (doc_title/image boxes = confuser sources), it
reports per-line attributes and summary distributions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def edge_candidates_from_gold(blocks, snap=32):
    """Unique candidate rule segments from gold block edges (source px)."""
    h_edges, v_edges = [], []
    for b in blocks:
        x1, y1, x2, y2 = b
        h_edges.append((y1, x1, x2)); h_edges.append((y2, x1, x2))
        v_edges.append((x1, y1, y2)); v_edges.append((x2, y1, y2))
    def dedup(edges):
        out = []
        for p, a, b in sorted(edges):
            for o in out:
                if abs(o[0] - p) <= snap and min(o[2], b) - max(o[1], a) > 0.5 * min(b - a, o[2] - o[1]):
                    o[0] = (o[0] + p) / 2; o[1] = min(o[1], a); o[2] = max(o[2], b)
                    break
            else:
                out.append([p, a, b])
        return out
    return dedup(h_edges), dedup(v_edges)


def measure_line(ink, orient, pos, lo, hi, band=40):
    """Attributes of the best ink line near (pos) spanning [lo,hi]."""
    if orient == "h":
        window = ink[max(0, pos - band):pos + band, lo:hi]
    else:
        window = ink[lo:hi, max(0, pos - band):pos + band].T
    if window.size == 0:
        return None
    rows = window.mean(axis=1)
    center = int(np.argmax(rows))
    if rows[center] < 0.15:
        return {"present": False}
    # centerline trace: per-column nearest ink within a fixed window around
    # the straight center row. Rigid by design: this is the validated
    # measurement; warp is handled downstream by the completion walker.
    length = window.shape[1]
    trace = np.full(length, np.nan)
    occupied = np.zeros(length, dtype=bool)
    thicknesses = []
    for col in range(length):
        column = window[:, col]
        near = np.nonzero(column[max(0, center - 8):center + 9])[0]
        if near.size:
            row_idx = near[len(near) // 2] + max(0, center - 8)
            trace[col] = row_idx
            occupied[col] = True
            top = row_idx
            while top - 1 >= 0 and column[top - 1]:
                top -= 1
            bottom = row_idx
            while bottom + 1 < window.shape[0] and column[bottom + 1]:
                bottom += 1
            thicknesses.append(bottom - top + 1)
    # WARP CORRECTION (two-pass): fit a robust rolling-median curve through
    # the collected points, then re-measure occupancy within +/-4 px of the
    # curve. A physically continuous rule then measures continuous even when
    # the page warps; its true damage gaps are a few pixels (operator).
    occ0 = np.nonzero(occupied)[0]
    if occ0.size > 30:
        xs_all = np.arange(length)
        rough = np.interp(xs_all, occ0, trace[occ0])
        half_w = 100
        curve = np.empty(length)
        for i in range(length):
            seg = rough[max(0, i - half_w):i + half_w + 1]
            curve[i] = np.median(seg)
        occupied = np.zeros(length, dtype=bool)
        thicknesses = []
        for col in range(length):
            c = int(round(curve[col]))
            lo_c = max(0, c - 4)
            column = window[:, col]
            near = np.nonzero(column[lo_c:c + 5])[0]
            if near.size:
                occupied[col] = True
                row_idx = near[len(near) // 2] + lo_c
                top = row_idx
                while top - 1 >= 0 and column[top - 1]:
                    top -= 1
                bottom = row_idx
                while bottom + 1 < window.shape[0] and column[bottom + 1]:
                    bottom += 1
                thicknesses.append(bottom - top + 1)
        trace_fit = trace.copy()
        valid0 = ~np.isnan(trace_fit)
        warp_amplitude = float(curve.max() - curve.min())
    else:
        warp_amplitude = 0.0

    # trim to actual ink extent: fill and gap statistics describe the rule's
    # INTERIOR; overhang beyond the last ink is not a gap, it is where the
    # object ends (operator definition, 2026-07-19)
    occ_idx = np.nonzero(occupied)[0]
    if occ_idx.size == 0:
        return {"present": False}
    # OBJECT SEGMENTATION (operator definition): ink positions chain into one
    # object only across gaps <= GAP_JOIN px (print damage scale); larger
    # whitespace splits objects. The measured candidate IS the longest object.
    GAP_JOIN = 12
    objects = []
    start = prev = int(occ_idx[0])
    for i in occ_idx[1:]:
        i = int(i)
        if i - prev - 1 > GAP_JOIN:
            objects.append((start, prev))
            start = i
        prev = i
    objects.append((start, prev))
    first, last = max(objects, key=lambda o: o[1] - o[0])
    # object dominance: fraction of ALL occupied ink held by the longest
    # object. A true rule's window holds one dominant object plus stray
    # specks; figure texture fragments into several comparable chunks.
    occupied_total = int(occ_idx.size)
    dominant_ink = int(occupied[first:last + 1].sum())
    obj_dominance = round(dominant_ink / max(1, occupied_total), 3)
    span_fraction = (last - first + 1) / length
    # object-population statistics: a printed chain border is a row of
    # IDENTICAL cast motifs (object lengths tight, spacing tight); character
    # pseudo-lines fragment into stroke chunks of highly variable length
    obj_lens = np.array([b - a + 1 for a, b in objects], dtype=float)
    obj_len_cv = round(float(obj_lens.std() / obj_lens.mean()), 2) if len(objects) >= 4 else None
    ospan = objects[-1][1] - objects[0][0] + 1
    obj_span_coverage = round(float(obj_lens.sum() / max(1, ospan)), 3)
    obj_gaps = np.array([objects[i + 1][0] - objects[i][1] - 1 for i in range(len(objects) - 1)], dtype=float)
    obj_gap_cv = (round(float(obj_gaps.std() / obj_gaps.mean()), 2)
                  if len(obj_gaps) >= 3 and obj_gaps.mean() > 0 else None)
    interior = occupied[first:last + 1]
    fill = float(interior.mean())
    gaps = []
    run = 0
    for value in interior:
        if not value:
            run += 1
        elif run:
            gaps.append(run); run = 0
    valid = ~np.isnan(trace)
    straightness = float(np.std(trace[valid])) if valid.sum() > 8 else None
    med_t = float(np.median(thicknesses)) if thicknesses else None
    p90_t = float(np.percentile(thicknesses, 90)) if thicknesses else None
    uniformity = round(p90_t / med_t, 2) if med_t else None
    # gap periodicity: regular spacing = character pitch, irregular = damage
    gap_starts = []
    prev = True
    for i, v in enumerate(occupied):
        if prev and not v:
            gap_starts.append(i)
        prev = v
    periodicity_cv = None
    if len(gap_starts) >= 4:
        spacing = np.diff(gap_starts)
        if spacing.mean() > 0:
            periodicity_cv = round(float(spacing.std() / spacing.mean()), 2)
    # flank clearance: ink just beyond the line body; rules sit in blank gutters
    half = int((med_t or 4) / 2)
    flank_lo = max(0, center - half - 10)
    flank_hi = min(window.shape[0], center + half + 11)
    top_band = window[flank_lo:max(flank_lo, center - half - 2), :]
    bot_band = window[min(flank_hi, center + half + 3):flank_hi, :]
    flank_ink = round(float(np.concatenate([
        top_band.ravel(), bot_band.ravel()]).mean()) if (top_band.size + bot_band.size) else 0.0, 3)
    # per-side flanks: a separator has at least one blank side (it separates
    # content); texture embedded in a figure is dense on BOTH sides.
    flank_a = float(top_band.mean()) if top_band.size else 0.0
    flank_b = float(bot_band.mean()) if bot_band.size else 0.0
    flank_min = round(min(flank_a, flank_b), 3)
    return {
        "present": True,
        "length": int(last - first + 1),
        "overhang": int(length - (last - first + 1)),
        "warp_amplitude": round(warp_amplitude, 1),
        "span_fraction": round(span_fraction, 3),
        "obj_offset_lo": int(first),
        "obj_offset_hi": int(last),
        "n_objects": len(objects),
        "obj_dominance": obj_dominance,
        "obj_len_cv": obj_len_cv,
        "obj_span_coverage": obj_span_coverage,
        "obj_gap_cv": obj_gap_cv,
        "fill": round(fill, 3),
        "max_gap": int(max(gaps)) if gaps else 0,
        "n_gaps_over_8px": int(sum(1 for g in gaps if g > 8)),
        "median_thickness": med_t,
        "p90_thickness": p90_t,
        "thickness_uniformity": uniformity,
        "gap_periodicity_cv": periodicity_cv,
        "flank_ink": flank_ink,
        "flank_min": flank_min,
        "straightness_std": round(straightness, 2) if straightness is not None else None,
    }


def confuser_lines(ink, boxes, orient, min_run, limit=40):
    """Long ink runs inside given boxes (display strokes / figure masses)."""
    out = []
    for x1, y1, x2, y2 in boxes:
        region = ink[y1:y2, x1:x2]
        axis_len = region.shape[1] if orient == "h" else region.shape[0]
        if axis_len < min_run:
            continue
        data = region if orient == "h" else region.T
        step = max(1, data.shape[0] // 40)
        for row in range(0, data.shape[0], step):
            line = data[row]
            run = best_start = cur_start = 0; best = 0
            for i, v in enumerate(line):
                if v:
                    if run == 0: cur_start = i
                    run += 1
                    if run > best: best, best_start = run, cur_start
                else:
                    run = 0
            if best >= min_run:
                pos = (y1 + row) if orient == "h" else (x1 + row)
                lo = (x1 + best_start) if orient == "h" else (y1 + best_start)
                m = measure_line(ink, orient, pos, lo, lo + best, band=24)
                if m and m.get("present"):
                    m["source"] = "confuser"
                    out.append(m)
                if len(out) >= limit:
                    return out
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gold", default="", help="gold blocks JSON (bbox_source_xyxy)")
    parser.add_argument("--detections", default="", help="V2 detections for confusers")
    parser.add_argument("--detections-scale", type=float, default=4.0)
    parser.add_argument("--ink-threshold", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    ink = source > args.ink_threshold

    report = {"image": args.image, "gold_edges": [], "confusers": []}
    if args.gold:
        blocks = [b["bbox_source_xyxy"] for b in json.loads(Path(args.gold).read_text())["blocks"]]
        h_edges, v_edges = edge_candidates_from_gold(blocks)
        for pos, lo, hi in h_edges:
            m = measure_line(ink, "h", int(pos), int(lo), int(hi))
            if m: m.update({"orient": "h", "pos": int(pos)}); report["gold_edges"].append(m)
        for pos, lo, hi in v_edges:
            m = measure_line(ink, "v", int(pos), int(lo), int(hi))
            if m: m.update({"orient": "v", "pos": int(pos)}); report["gold_edges"].append(m)

    if args.detections:
        dets = json.loads(Path(args.detections).read_text())
        s = args.detections_scale
        titles = [[int(v * s) for v in d["proxy_xyxy"]] for d in dets if d["cls"] == "doc_title"]
        images = [[int(v * s) for v in d["proxy_xyxy"]] for d in dets if d["cls"] == "image"]
        report["confusers"] = (
            confuser_lines(ink, titles, "h", 280) + confuser_lines(ink, titles, "v", 280)
            + confuser_lines(ink, images, "h", 280) + confuser_lines(ink, images, "v", 280)
        )

    def summarize(items, label):
        present = [i for i in items if i.get("present")]
        if not present:
            print(f"{label}: none"); return
        def col(key):
            vals = [i[key] for i in present if i.get(key) is not None]
            return (np.percentile(vals, [10, 50, 90]).round(1).tolist()) if vals else None
        print(f"{label}: n={len(present)} | thickness p10/50/90 {col('median_thickness')} | "
              f"fill {col('fill')} | max_gap {col('max_gap')} | straightness {col('straightness_std')}")

    n_absent = sum(1 for e in report["gold_edges"] if not e.get("present"))
    print(f"gold edges measured: {len(report['gold_edges'])} ({n_absent} with no printed rule)")
    summarize(report["gold_edges"], "GOLD RULES")
    summarize(report["confusers"], "CONFUSERS")
    Path(args.output).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
