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
    # dotted/perforated hairline (operator, 2026-07-19: "the lines are
    # thinner and the gap measurement should be proportionally smaller"):
    # a row of many collinear dots too thin to be characters. p90 thickness
    # is the discriminator — text pseudo-lines never measure below ~22px
    # (p10=22, n=811 p0308 candidates); dotted rules measure 6-9px.
    nobj = m.get("n_objects") or 1
    ext = (m.get("all_obj_hi") or 0) - (m.get("all_obj_lo") or 0)
    if (m["length"] < 500 and nobj >= 8 and ext >= 600
            and (m["p90_thickness"] or 99) <= 10
            and (m.get("obj_span_coverage") or 0) >= 0.4
            and (m.get("hollow_fraction") or 0) <= 0.15
            and m["straightness_std"] is not None and m["straightness_std"] <= 4.0
            and m["flank_ink"] <= 0.30 and (m.get("flank_min") or 0) <= 0.25):
        return ("dotted_rule", "perforated hairline")
    # ornamental motif chain (p0367 borders): rows of larger cast stamps.
    # Discriminator is SPARSITY — motif chains cover 0.35-0.41 of their
    # extent, text pseudo-lines >=0.60 (p10, n=145 long candidates).
    if (m["length"] < 500 and nobj >= 10 and ext >= 1200
            and (m.get("obj_span_coverage") or 1) <= 0.52
            and (m["median_thickness"] or 99) <= 16
            and m["straightness_std"] is not None and m["straightness_std"] <= 4.0
            and m["flank_ink"] <= 0.30 and (m.get("flank_min") or 0) <= 0.25):
        return ("motif_rule", "ornament chain")
    if m["length"] < 500:
        return ("rejected_short", "")
    # 4.6 ceiling: p0367's real bead-adjacent band boundaries measure
    # 4.08-4.46 (multi-object chains warp more than solid rules)
    if m["straightness_std"] is None or m["straightness_std"] > 4.6:
        return ("rejected_crooked", f"resid {m['straightness_std']}")
    # object-level continuity (operator definition): a rule's interior gaps
    # are print damage of a few px; fill near 1.0 by construction. Hairline
    # rules (<=6px) lose proportionally more ink to the same damage: the
    # p0308 zhabei boundary measures 0.947 while every thicker gold rule
    # sits at 0.96-1.0, so the floor is class-dependent.
    # 0.94 general floor: p0367's real 14px inner frame measures 0.944
    # beside its bead border (gold p10 was 0.96; max_gap<=12 remains the
    # hard continuity gate)
    fill_floor = 0.93 if (m["median_thickness"] or 99) <= 6 else 0.94
    if m["fill"] < fill_floor:
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
    # frame-zone rules carry the page's bead border on their outer flank, so
    # flank_ink relaxes there (flank_min already exempt); u 2.15 admits the
    # p0367 photo-block top edge (2.09) with margin below text strokes (~3.0)
    flank_cap = 0.85 if frame_zone else 0.45
    if m["median_thickness"] <= 28 and p90 <= 42 and u <= 2.15 and m["flank_ink"] <= flank_cap:
        return ("thin_rule" if m["median_thickness"] <= 12 else "medium_rule", "ok")
    if m["median_thickness"] > 28 and u <= 1.8 and m["flank_ink"] <= (0.85 if frame_zone else 0.25):
        return ("thick_band", "")
    # textured/brush-printed rule: uniformity fails on ragged edges, but a
    # long straight continuous ISOLATED stroke can only be a rule
    if m["median_thickness"] <= 45 and m["flank_ink"] <= 0.20:
        return ("rough_rule", "isolated textured")
    return ("rejected_text_signature", f"u{u:.2f} p90 {p90:.0f}")


def coverage(lo, hi, perp, axis, boxes, pad=8):
    cov = 0
    for x1, y1, x2, y2 in boxes:
        if axis == "h" and y1 - pad <= perp <= y2 + pad:
            cov += max(0, min(hi, x2) - max(lo, x1))
        if axis == "v" and x1 - pad <= perp <= x2 + pad:
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


RULE_CLASSES = ("thin_rule", "medium_rule", "thick_band", "rough_rule", "dotted_rule", "motif_rule")

TITLE_BOXES = []  # doc_title boxes dilated by 120px, set by main()


def title_contained(axis, perp, olo, ohi):
    """A measured object floating entirely inside a title's neighborhood is
    cartouche/ornament decoration, not a boundary: real boundary rules span
    their whole block and poke out of any title box (p0308 zhongyang bottom
    border [3116,4401] sits 90px below the V2 box, past coverage dilation)."""
    for x1, y1, x2, y2 in TITLE_BOXES:
        if axis == "h":
            if x1 <= olo and ohi <= x2 and y1 <= perp <= y2:
                return True
        else:
            if y1 <= olo and ohi <= y2 and x1 <= perp <= x2:
                return True
    return False


def rescue_fragments(src, ink, fragments, accepted, counts, mask_boxes, image_boxes):
    """Candidate-generation rescue: heavy page warp shatters a rule's Hough
    chains into short fragments (the p0308 zhabei rule drifts ~40px over
    1400px and died entirely as rejected_short). Collinear fragment clusters
    are re-measured over their union span — measure_line's warp-corrected
    trace follows what chaining could not. Full classification still applies,
    so text-row fragment clusters die on fill/gap as usual."""
    for axis in ("h", "v"):
        frs = sorted(fragments[axis], key=lambda f: f[0])
        used = [False] * len(frs)
        for i, (P, sLO, sHI) in enumerate(frs):
            if used[i]:
                continue
            cP, clo, chi = [P], sLO, sHI
            for j in range(i + 1, len(frs)):
                Pj, lj, hj = frs[j]
                if used[j] or Pj - np.median(cP) > 60:
                    continue
                # generous span holes: middle fragments of a damaged rule
                # often fail the fragment gate entirely; the cluster is only
                # a hypothesis, re-measured strictly below
                if lj - chi <= 1500 and clo - hj <= 1500:
                    used[j] = True
                    cP.append(Pj)
                    clo, chi = min(clo, lj), max(chi, hj)
            if chi - clo < 600:
                continue
            medP = int(np.median(cP))
            dim = src.shape[0] if axis == "h" else src.shape[1]
            best = None
            for pos in range(medP - 40, medP + 81, 20):
                m = measure_line(ink, axis, pos, clo, chi, band=60)
                if not m or not m.get("present"):
                    continue
                cls, _why = classify_full(m, frame_zone=(pos < 250 or pos > dim - 250))
                if cls in RULE_CLASSES and (best is None or m["length"] > best[2]["length"]):
                    best = (pos, cls, m)
            if best:
                pos, cls, m = best
                if cls == "motif_rule":
                    # cluster unions are sparse by construction — the
                    # sparsity-gated class may never come from rescue
                    continue
                if cls == "dotted_rule":
                    olo, ohi = clo + m["all_obj_lo"], clo + m["all_obj_hi"]
                else:
                    olo, ohi = clo + m["obj_offset_lo"], clo + m["obj_offset_hi"]
                dup = any(abs(p - pos) <= 60 and min(hi, ohi) > max(lo, olo)
                          for lo, hi, p, _cls in accepted[axis])
                if (not dup and not title_contained(axis, pos, olo, ohi)
                        and not masked(olo, ohi, pos, axis, mask_boxes, image_boxes)):
                    counts["rescued_" + cls] += 1
                    accepted[axis].append([olo, ohi, pos, cls])


def masked(lo, hi, perp, axis, mask_boxes, image_boxes):
    # note: a wide perpendicular pad on image boxes was tried against the
    # cartouche ornament border and rejected — the p0308 kiss-me-again box
    # sits in the identical geometric relation to a REAL boundary rule.
    # Ornament-only regions are handled at cell level (merge_boxed_cells).
    return (coverage(lo, hi, perp, axis, mask_boxes) > 0.4
            or coverage(lo, hi, perp, axis, image_boxes) > 0.4)


def detect(src, ink, mask_boxes, image_boxes, chain_views=False):
    ink4 = ((src[::4, ::4] > 128).astype(np.uint8)) * 255
    # candidate hypotheses from three views: raw ink, and directionally
    # closed ink (Hough barely fires on sparse motif/dotted chains — p0367's
    # 5400px mid divider yielded one 288px fragment; closing 20 quarter-px
    # gaps makes chains vote). Measurement always runs on the ORIGINAL ink,
    # so closing adds hypotheses, never evidence.
    views = [(ink4, "hv")]
    if chain_views:
        close_h = cv2.morphologyEx(ink4, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1)))
        close_v = cv2.morphologyEx(ink4, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21)))
        views += [(close_h, "h"), (close_v, "v")]
    h_s, v_s = [], []
    h_closed, v_closed = [], []
    for img, want in views:
        segs = cv2.HoughLinesP(img, 1, math.pi / 180, 60, minLineLength=50, maxLineGap=18)
        closed = img is not ink4
        for x1, y1, x2, y2 in ([] if segs is None else [x[0].tolist() for x in segs]):
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            if (ang <= 4 or ang >= 176) and "h" in want:
                (h_closed if closed else h_s).append([x1, y1, x2, y2])
            elif abs(ang - 90) <= 4 and "v" in want:
                (v_closed if closed else v_s).append([x1, y1, x2, y2])
    counts = Counter()
    accepted = {"h": [], "v": []}
    fragments = {"h": [], "v": []}
    # closed-view candidates exist to surface chain classes; a solid-class
    # match from a closed view is a shifted duplicate of a raw candidate
    for axis, ch, chain_only in (
            ("h", chain_segments(h_s, "h", 7, 22, 60), False),
            ("v", chain_segments(v_s, "v", 7, 22, 60), False),
            ("h", chain_segments(h_closed, "h", 7, 22, 60), True),
            ("v", chain_segments(v_closed, "v", 7, 22, 60), True)):
        for lo, hi, perp, _ in ch:
            LO, HI, P = int(lo * 4), int(hi * 4), int(perp * 4)
            if masked(LO, HI, P, axis, mask_boxes, image_boxes):
                counts["masked"] += 1
                continue
            m = measure_line(ink, axis, P, LO, HI, band=40)
            dim = src.shape[0] if axis == "h" else src.shape[1]
            cls, _why = classify_full(m, frame_zone=(P < 250 or P > dim - 250))
            if chain_only and cls in RULE_CLASSES and cls not in ("dotted_rule", "motif_rule"):
                # closed-view solids are kept only when raw Hough missed the
                # rule entirely (p0309 x3592); otherwise they are shifted
                # duplicates of raw walls
                dup = any(abs(r[2] - P) <= 20 and min(r[1], HI) - max(r[0], LO) > 0.3 * (HI - LO)
                          for r in accepted[axis])
                if dup:
                    counts["closed_view_dup_skipped"] += 1
                    continue
            counts[cls] += 1
            if cls in RULE_CLASSES:
                if cls in ("dotted_rule", "motif_rule"):
                    # a dotted/motif rule IS its row of dots: the multi-object
                    # extent is the physical extent; the solid-ink walker
                    # cannot ride dots and is skipped
                    sLO = LO + m["all_obj_lo"]
                    sHI = LO + m["all_obj_hi"]
                    if not title_contained(axis, P, sLO, sHI):
                        accepted[axis].append([sLO, sHI, P, cls])
                    else:
                        counts["masked_title_contained"] += 1
                    continue
                # seed = the measured continuous object, not the loose Hough
                # span. DASHED rules (many collinear thin objects, p0309
                # column rules: nobj to 41, cov>=0.6) seed at the full object
                # row — the solid-ink walker cannot ride their dots.
                dashed = ((m.get("n_objects") or 1) >= 5
                          and (m.get("obj_span_coverage") or 0) >= 0.6
                          and (m["p90_thickness"] or 99) <= 14
                          and (m.get("hollow_fraction") or 0) <= 0.15
                          and (m["all_obj_hi"] - m["all_obj_lo"]) >= 1.3 * m["length"])
                if dashed:
                    sLO = LO + m["all_obj_lo"]
                    sHI = LO + m["all_obj_hi"]
                else:
                    sLO = LO + m["obj_offset_lo"]
                    sHI = LO + m["obj_offset_hi"]
                if title_contained(axis, P, sLO, sHI):
                    counts["masked_title_contained"] += 1
                    continue
                nlo = walk(ink, axis, P, sLO, -1)
                nhi = walk(ink, axis, P, sHI, +1)
                accepted[axis].append([nlo, nhi, P, cls])
            elif (not chain_only and cls == "rejected_short" and m.get("present")
                  and m["length"] >= 80 and (m["median_thickness"] or 99) <= 28
                  and m["fill"] >= 0.7
                  and m["straightness_std"] is not None and m["straightness_std"] <= 4.0):
                fragments[axis].append((P, LO + m["obj_offset_lo"], LO + m["obj_offset_hi"]))
    # DENSE SWEEP: Hough repeatedly proved candidate-incomplete (warp
    # shatter, sparse chains, dashed columns — p0309 x3592 had no candidate
    # at all). measure_line is the validated detector: probe a 40px perp
    # grid in overlapping 1600px windows with the full classifier;
    # duplicates of existing walls are dropped.
    for axis in ("h", "v"):
        dim_perp = src.shape[0] if axis == "h" else src.shape[1]
        dim_along = src.shape[1] if axis == "h" else src.shape[0]
        for P in range(60, dim_perp - 60, 40):
            for wlo in range(0, max(1, dim_along - 800), 800):
                whi = min(dim_along, wlo + 1600)
                m = measure_line(ink, axis, P, wlo, whi, band=45)
                if not m or not m.get("present"):
                    continue
                cls, _why = classify_full(m, frame_zone=(P < 250 or P > dim_perp - 250))
                # sweep recovers page-scale structure only: the sparsity-gated
                # motif class collapses in confined windows (135 spurious hits
                # on p0309), and short internal rules stay Hough's job
                if cls not in RULE_CLASSES or cls == "motif_rule":
                    continue
                if cls == "dotted_rule" or (
                        (m.get("n_objects") or 1) >= 5
                        and (m.get("obj_span_coverage") or 0) >= 0.6
                        and (m["p90_thickness"] or 99) <= 14
                        and (m.get("hollow_fraction") or 0) <= 0.15
                        and (m["all_obj_hi"] - m["all_obj_lo"]) >= 1.3 * m["length"]):
                    sLO, sHI = wlo + m["all_obj_lo"], wlo + m["all_obj_hi"]
                    walkable = cls != "dotted_rule"
                else:
                    sLO, sHI = wlo + m["obj_offset_lo"], wlo + m["obj_offset_hi"]
                    walkable = True
                if sHI - sLO < 1200:
                    continue
                if masked(sLO, sHI, P, axis, mask_boxes, image_boxes) or title_contained(axis, P, sLO, sHI):
                    continue
                dup = any(abs(r[2] - P) <= 45 and min(r[1], sHI) - max(r[0], sLO) > 0.5 * (sHI - sLO)
                          for r in accepted[axis])
                if dup:
                    continue
                counts["sweep_" + cls] += 1
                if walkable:
                    accepted[axis].append([walk(ink, axis, P, sLO, -1), walk(ink, axis, P, sHI, +1), P, cls])
                else:
                    accepted[axis].append([sLO, sHI, P, cls])
    # chain classes mark isolated boundary ornament only. Catalog dot-leaders
    # (p0308 baidai ad) match the gates but come in STACKS: measured leader
    # pitch 118px vs the closest real boundary pair 128px (p0367 box edge +
    # mid divider), so parallel chain siblings within 120px are typography.
    for axis in ("h", "v"):
        chains = [r for r in accepted[axis] if r[3] in ("motif_rule", "dotted_rule")]
        drop = set()
        for i, a in enumerate(chains):
            for b in chains[i + 1:]:
                if abs(a[2] - b[2]) <= 120 and min(a[1], b[1]) - max(a[0], b[0]) > 0:
                    drop.add(id(a))
                    drop.add(id(b))
        if drop:
            counts["chain_leader_stack_dropped"] += len(drop)
            accepted[axis] = [r for r in accepted[axis] if id(r) not in drop]
    rescue_fragments(src, ink, fragments, accepted, counts, mask_boxes, image_boxes)
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
    # collinear-gap chaining: same physical rule broken by damage/junction —
    # p0367's left frame split into segments 8px apart in perp with a 140px
    # span hole, which overlap-gated merging can never join
    out.sort(key=lambda r: (r[2], r[0]))
    chained = []
    for r in out:
        if chained:
            q = chained[-1]
            if abs(q[2] - r[2]) <= 12 and 0 <= r[0] - q[1] <= 400:
                q[1] = max(q[1], r[1])
                if r[1] - r[0] > q[1] - q[0]:
                    q[2] = r[2]
                continue
        chained.append(r)
    return chained


def extend_walls(accepted, ink, shape):
    """Iterative wall extension: sweep/Hough windows repeatedly missed the
    continuation of walls they found elsewhere (p0367 band rule accepted as
    [4791,6185] where the printed rule spans [3256,6177] — the confined gold
    probe classified the full span cleanly). Anchor a probe window just
    beyond each dangling end at the wall's own perp; extend while the
    classifier keeps accepting rule ink that reaches the end."""
    for axis in ("h", "v"):
        dim_perp = shape[0] if axis == "h" else shape[1]
        dim_along = shape[1] if axis == "h" else shape[0]
        for r in accepted[axis]:
            for side in (0, 1):
                for _ in range(4):
                    if side == 0:
                        wlo, whi = max(0, r[0] - 1400), min(dim_along, r[0] + 200)
                    else:
                        wlo, whi = max(0, r[1] - 200), min(dim_along, r[1] + 1400)
                    if whi - wlo < 400:
                        break
                    m = measure_line(ink, axis, r[2], wlo, whi, band=45)
                    if not m or not m.get("present"):
                        break
                    cls, _why = classify_full(
                        m, frame_zone=(r[2] < 250 or r[2] > dim_perp - 250))
                    if cls not in RULE_CLASSES:
                        break
                    alo, ahi = wlo + m["all_obj_lo"], wlo + m["all_obj_hi"]
                    if side == 0:
                        if ahi < r[0] - 100 or alo >= r[0] - 50:
                            break
                        r[0] = alo
                    else:
                        if alo > r[1] + 100 or ahi <= r[1] + 50:
                            break
                        r[1] = ahi
    return accepted


def snap_all(accepted, shape, snap=520, frame_snap=900, tol=60):
    # blind junction snap: no observed defect came from snap; corridor-gated
    # variants blocked legitimate closures (perpendicular crossing rules read
    # as strokes) and broke the grid (v7 calibration, 2026-07-19). Long snaps
    # are allowed only TOWARD page-frame rules (blank margins — p0367's mid
    # divider ends 824px from the right frame); interior junctions keep the
    # proven 520 cap (900 globally re-broke p0308).
    def snap_rules(prim, cross, cross_dim):
        for r in prim:
            for c in cross:
                cap = frame_snap if (c[2] < 250 or c[2] > cross_dim - 250) else snap
                if c[0] - tol <= r[2] <= c[1] + tol:
                    if 0 < r[0] - c[2] <= cap:
                        r[0] = c[2]
                    if 0 < c[2] - r[1] <= cap:
                        r[1] = c[2]
        return prim
    for _ in range(2):
        accepted["h"] = snap_rules(accepted["h"], accepted["v"], shape[1])
        accepted["v"] = snap_rules(accepted["v"], accepted["h"], shape[0])
    return accepted


CELL_CUTTING = ("thin_rule", "medium_rule", "thick_band", "rough_rule")
# dotted/motif detections are reported but never cut cells: no measured
# attribute separates printed dotted boundaries from incidental dot
# alignments inside text bands (p0338 offenders vs kade/p0309 frames overlap
# on pitch, coverage, len_cv, hollow), and the operator's priority is
# explicit — a missed ribbon boundary is acceptable, a cut through words is
# not (2026-07-20)


def wall_geometry(accepted, ink):
    """Re-measure each final wall to recover its true centerline: physical
    rules warp (p0308 zhabei drifts 3246-3315 over 1400px), and a wall cut
    at one straight coordinate clips content on warped pages (operator
    report, three pages). Returns per-wall curve samples in source px."""
    walls = []
    for axis in ("h", "v"):
        for lo, hi, p, cls in accepted[axis]:
            if cls not in CELL_CUTTING:
                continue
            m = measure_line(ink, axis, p, lo, hi, band=70)
            curve = None
            if m and m.get("present") and m.get("centerline"):
                curve = [[lo + a, pp] for a, pp in m["centerline"]]
            walls.append({"axis": axis, "lo": lo, "hi": hi, "P": p, "cls": cls,
                          "curve": curve})
    return walls


def cells_from(walls, shape, S=4, min_dim_src=180):  # noqa: C901
    sep = np.zeros((shape[0] // S, shape[1] // S), np.uint8)
    for w in walls:
        lo, hi, p = w["lo"], w["hi"], w["P"]
        if w["curve"]:
            pts = [[lo - 24, w["curve"][0][1]]] + w["curve"] + [[hi + 24, w["curve"][-1][1]]]
        else:
            pts = [[lo - 24, p], [hi + 24, p]]
        if w["axis"] == "h":
            arr = np.array([[a // S, pp // S] for a, pp in pts], np.int32)
        else:
            arr = np.array([[pp // S, a // S] for a, pp in pts], np.int32)
        cv2.polylines(sep, [arr], False, 255, 3)
    sep[0:2, :] = sep[-2:, :] = 255
    sep[:, 0:2] = sep[:, -2:] = 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats((sep == 0).astype(np.uint8), connectivity=4)
    cells = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if min(w, h) * S < min_dim_src or w * h < 0.0004 * lab.size:
            continue
        # the background/margin component is a thin RING around the grid: a
        # near-page bbox that its own pixels barely fill. A legitimate large
        # cell (p0367's half-page ad) fills its bbox solidly.
        if w * h > 0.5 * lab.size and _area / (w * h) < 0.5:
            continue
        cells.append([int(x) * S, int(y) * S, int(x + w) * S, int(y + h) * S])
    # crops cover their boundary line: each edge abutting a wall extends to
    # the wall's local extreme (toward the cell) so no content is clipped and
    # adjacent crops meet with no padding — under warp they overlap slightly
    # instead of sharing one exact coordinate
    def local_range(w, a1, a2):
        if not w["curve"]:
            return w["P"], w["P"]
        perps = [pp for a, pp in w["curve"] if a1 - 32 <= a <= a2 + 32] or [pp for _, pp in w["curve"]]
        return min(perps), max(perps)

    for c in cells:
        for w in walls:
            if w["axis"] == "v" and min(c[3], w["hi"]) > max(c[1], w["lo"]):
                pmin, pmax = local_range(w, c[1], c[3])
                if pmin - 24 <= c[0] <= pmax + 24:
                    c[0] = pmin - 4
                if pmin - 24 <= c[2] <= pmax + 24:
                    c[2] = pmax + 4
            if w["axis"] == "h" and min(c[2], w["hi"]) > max(c[0], w["lo"]):
                pmin, pmax = local_range(w, c[0], c[2])
                if pmin - 24 <= c[1] <= pmax + 24:
                    c[1] = pmin - 4
                if pmin - 24 <= c[3] <= pmax + 24:
                    c[3] = pmax + 4
    # index in the paper's reading order: top-to-bottom bands, right-to-left
    cells.sort(key=lambda c: (c[1] // 300, -c[0]))
    return cells


def merge_boxed_cells(cells, ink, boxes):
    """A SMALL cell whose ink is dominated by title/image detections is a
    title cartouche or standalone ornament, not a content region — its
    printed border is real, but the crop belongs with the neighboring ad
    body (p0308 zhongyang cartouche strip vs its body cell). Merge into the
    neighbor sharing the longest wall. The size gate matters: display-glyph
    ads (xinshijie, dashijie) are title-dominated WHOLE blocks and must
    stay standalone."""
    changed = True
    while changed:
        changed = False
        for i, c in enumerate(cells):
            if min(c[2] - c[0], c[3] - c[1]) >= 600:
                continue
            win = ink[c[1]:c[3]:4, c[0]:c[2]:4]
            total = int(win.sum())
            if not total:
                continue
            m = np.zeros_like(win)
            for x1, y1, x2, y2 in boxes:
                xa = max(0, int((x1 - 20 - c[0]) / 4))
                ya = max(0, int((y1 - 20 - c[1]) / 4))
                xb = max(0, int((x2 + 20 - c[0]) / 4))
                yb = max(0, int((y2 + 20 - c[1]) / 4))
                m[ya:yb, xa:xb] = True
            if int((win & m).sum()) / total < 0.6:
                continue
            best = None
            for j, o in enumerate(cells):
                if j == i:
                    continue
                if c[0] == o[2] or c[2] == o[0]:
                    sh = min(c[3], o[3]) - max(c[1], o[1])
                    if sh > 0 and (best is None or sh > best[1]):
                        best = (j, sh)
                if c[1] == o[3] or c[3] == o[1]:
                    sh = min(c[2], o[2]) - max(c[0], o[0])
                    if sh > 0 and (best is None or sh > best[1]):
                        best = (j, sh)
            if best:
                o = cells[best[0]]
                o[0], o[1] = min(c[0], o[0]), min(c[1], o[1])
                o[2], o[3] = max(c[2], o[2]), max(c[3], o[3])
                del cells[i]
                changed = True
                break
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
    parser.add_argument("--chain-views", action="store_true",
                        help="add gap-closed candidate views (superseded by the dense sweep; off by default)")
    parser.add_argument("--show-rules", action="store_true",
                        help="also draw accepted rules by class (diagnostic); default shows only final cells")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    ink = src > 128
    mask_boxes = []
    image_boxes = []
    if args.detections:
        dets = json.loads(Path(args.detections).read_text())
        s = args.detections_scale
        for d in dets:
            if d["cls"] == "image":
                image_boxes.append([v * s for v in d["proxy_xyxy"]])
            elif d["cls"] == "doc_title":
                # ornamental cartouche borders hug their title box but extend
                # beyond the detection; near overlaps use coverage (+40),
                # floating ornament lines use containment (+120, see
                # title_contained)
                x1, y1, x2, y2 = (v * s for v in d["proxy_xyxy"])
                mask_boxes.append([x1 - 40, y1 - 40, x2 + 40, y2 + 40])
                TITLE_BOXES.append([x1 - 120, y1 - 120, x2 + 120, y2 + 120])

    accepted, counts = detect(src, ink, mask_boxes, image_boxes, chain_views=args.chain_views)
    for ax in accepted:
        accepted[ax] = merge(ink, accepted[ax], ax)
    accepted = extend_walls(accepted, ink, src.shape)
    accepted = snap_all(accepted, src.shape)
    walls = wall_geometry(accepted, ink)
    cells = cells_from(walls, src.shape)
    cells = merge_boxed_cells(cells, ink, image_boxes + mask_boxes)
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
