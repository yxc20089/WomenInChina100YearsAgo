"""Build a labeled stroke taxonomy and fit an interpretable rule classifier.

Long straight ink runs on a broadsheet page fall into classes with distinct
physical attribute profiles. This tool builds a LABELED dataset on the page
with operator gold (labels derived from independent sources: gold block edges
= rule; OCR text-det boxes = text stroke; layout doc_title boxes = glyph
stroke; layout image boxes = halftone/figure), measures every candidate's
attributes, reports per-class profiles, and fits a shallow decision tree that
is printed as explicit production rules. The tree is documentation of the
decision boundary — production code re-implements it as plain conditions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from boundary_hough import chain_segments  # noqa: E402
from rule_attributes import measure_line, edge_candidates_from_gold  # noqa: E402

FEATURES = ["length", "median_thickness", "p90_thickness", "thickness_uniformity",
            "fill", "max_gap", "n_gaps_over_8px", "gap_periodicity_cv",
            "flank_ink", "straightness_std"]


def coverage(lo, hi, perp, orient, boxes):
    """Fraction of the candidate span covered by any of the boxes (with the
    perpendicular coordinate inside the box)."""
    if not boxes:
        return 0.0
    covered = 0
    for x1, y1, x2, y2 in boxes:
        if orient == "h":
            if y1 - 8 <= perp <= y2 + 8:
                covered += max(0, min(hi, x2) - max(lo, x1))
        else:
            if x1 - 8 <= perp <= x2 + 8:
                covered += max(0, min(hi, y2) - max(lo, y1))
    return min(1.0, covered / max(1, hi - lo))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--text-regions", required=True, help="OCR det artifact")
    parser.add_argument("--detections", required=True, help="V2 detections JSON")
    parser.add_argument("--detections-scale", type=float, default=4.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    ink = source > 128
    ink4 = ((source[::4, ::4] > 128).astype(np.uint8)) * 255

    gold_blocks = [b["bbox_source_xyxy"] for b in json.loads(Path(args.gold).read_text())["blocks"]]
    gh, gv = edge_candidates_from_gold(gold_blocks)
    text_boxes = []
    for region in json.loads(Path(args.text_regions).read_text())["regions"]:
        xs = [p["x"] for p in region["polygon"]["points"]]
        ys = [p["y"] for p in region["polygon"]["points"]]
        text_boxes.append([min(xs), min(ys), max(xs), max(ys)])
    dets = json.loads(Path(args.detections).read_text())
    s = args.detections_scale
    title_boxes = [[v * s for v in d["proxy_xyxy"]] for d in dets if d["cls"] == "doc_title"]
    image_boxes = [[v * s for v in d["proxy_xyxy"]] for d in dets if d["cls"] == "image"]

    # loose candidate pool
    segs = cv2.HoughLinesP(ink4, 1, math.pi / 180, 60, minLineLength=50, maxLineGap=18)
    segs = [] if segs is None else [x[0].tolist() for x in segs]
    h_s, v_s = [], []
    for x1, y1, x2, y2 in segs:
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        if ang <= 4 or ang >= 176:
            h_s.append([x1, y1, x2, y2])
        elif abs(ang - 90) <= 4:
            v_s.append([x1, y1, x2, y2])
    candidates = ([("h", r) for r in chain_segments(h_s, "h", 7, 55, 60)]
                  + [("v", r) for r in chain_segments(v_s, "v", 7, 55, 60)])

    def near_gold(orient, perp, lo, hi):
        edges = gh if orient == "h" else gv
        for pos, a, b in edges:
            if abs(pos - perp) <= 40 and min(hi, b) - max(lo, a) >= 0.4 * (hi - lo):
                return True
        return False

    samples = []
    for orient, (lo4, hi4, perp4, _) in candidates:
        lo, hi, perp = int(lo4 * 4), int(hi4 * 4), int(perp4 * 4)
        m = measure_line(ink, orient, perp, lo, hi, band=40)
        if not m or not m.get("present"):
            continue
        if near_gold(orient, perp, lo, hi):
            label = "rule"
        elif coverage(lo, hi, perp, orient, image_boxes) > 0.6:
            label = "halftone_or_figure"
        elif coverage(lo, hi, perp, orient, title_boxes) > 0.6:
            label = "glyph_stroke"
        elif coverage(lo, hi, perp, orient, text_boxes) > 0.55:
            label = "text_stroke"
        else:
            label = "unlabeled"
        m.update({"orient": orient, "pos": perp, "lo": lo, "hi": hi, "label": label})
        samples.append(m)

    # add confirmed gold edges directly (measured at their exact positions)
    for orient, edges in (("h", gh), ("v", gv)):
        for pos, a, b in edges:
            m = measure_line(ink, orient, int(pos), int(a), int(b), band=40)
            if m and m.get("present"):
                m.update({"orient": orient, "pos": int(pos), "lo": int(a), "hi": int(b),
                          "label": "rule"})
                samples.append(m)

    labeled = [x for x in samples if x["label"] != "unlabeled"]
    print(f"candidates: {len(samples)} | labeled: {len(labeled)}")
    from collections import Counter
    print("label counts:", dict(Counter(x["label"] for x in labeled)))

    def profile(label):
        rows = [x for x in labeled if x["label"] == label]
        out = {}
        for f in FEATURES:
            vals = [r[f] for r in rows if r.get(f) is not None]
            if vals:
                out[f] = np.percentile(vals, [10, 50, 90]).round(2).tolist()
        return out
    profiles = {label: profile(label) for label in
                ("rule", "text_stroke", "glyph_stroke", "halftone_or_figure")}
    for label, prof in profiles.items():
        print(f"\n[{label}]")
        for f, v in prof.items():
            print(f"  {f}: {v}")

    # interpretable classifier: binary rule / not-rule
    from sklearn.tree import DecisionTreeClassifier, export_text
    X, y = [], []
    for r in labeled:
        X.append([r[f] if r.get(f) is not None else -1 for f in FEATURES])
        y.append(1 if r["label"] == "rule" else 0)
    X, y = np.array(X), np.array(y)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8,
                                  class_weight="balanced", random_state=0)
    tree.fit(X, y)
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(tree, X, y, cv=5, scoring="f1")
    print(f"\nrule-vs-rest decision tree (5-fold F1 {cv.mean():.3f} +/- {cv.std():.3f}):")
    print(export_text(tree, feature_names=FEATURES, max_depth=4))

    Path(args.output).write_text(json.dumps(
        {"schema_version": "stroke-taxonomy-v1",
         "profiles": profiles,
         "n_samples": len(labeled),
         "samples": labeled}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
