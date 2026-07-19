"""Block-level region proposals via rule- and whitespace-aware recursive XY-cut.

Target granularity: the visually distinct advertisement/article blocks a human
would crop — not text lines. Cuts are placed only where a printed rule or a
sufficiently wide whitespace gutter crosses the whole span; recursion stops at
minimum block size. Crops are written in ORIGINAL polarity for human review;
polarity inversion happens only at OCR request time.

Diagnostic proposals only; no semantic authority.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from region_cell_geometry import _erode_1d, _dilate_1d, _label  # noqa: E402

SCHEMA_VERSION = "block-xycut-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_masks(page: np.ndarray, artifact: dict, ink_threshold: int, scale: int):
    small = page[::scale, ::scale]
    ink = small > ink_threshold

    text_mask = np.zeros_like(ink)
    for region in artifact["regions"]:
        xs = [p["x"] for p in region["polygon"]["points"]]
        ys = [p["y"] for p in region["polygon"]["points"]]
        text_mask[max(0, min(ys) // scale - 1) : -(-max(ys) // scale) + 1,
                  max(0, min(xs) // scale - 1) : -(-max(xs) // scale) + 1] = True
    text_free = ink & ~text_mask

    run_v = max(3, 500 // scale)
    run_h = max(3, 280 // scale)
    gap = max(1, 48 // scale)
    thick = max(2, 120 // scale)

    bridged_h = _dilate_1d(text_free, gap, axis=1)
    horizontal = _dilate_1d(_erode_1d(bridged_h, run_h, axis=1), run_h, axis=1)
    h_thick = _dilate_1d(_erode_1d(horizontal, thick, axis=0), thick, axis=0)
    horizontal &= ~h_thick

    bridged_v = _dilate_1d(text_free, gap, axis=0)
    vertical = _dilate_1d(_erode_1d(bridged_v, run_v, axis=0), run_v, axis=0)
    v_thick = _dilate_1d(_erode_1d(vertical, thick, axis=1), thick, axis=1)
    vertical &= ~v_thick

    return ink, horizontal | vertical


def find_cuts(ink, rules, box, cut_axis, min_gap_cells, rule_frac, ink_eps):
    """Interior separator intervals along `cut_axis` ('y' = horizontal cuts)."""
    x1, y1, x2, y2 = box
    window_ink = ink[y1:y2, x1:x2]
    window_rules = rules[y1:y2, x1:x2]
    mean_axis = 1 if cut_axis == "y" else 0  # 'y': one value per row
    ink_profile = window_ink.mean(axis=mean_axis)
    rule_profile = window_rules.mean(axis=mean_axis)
    open_positions = (ink_profile <= ink_eps) | (rule_profile >= rule_frac)

    gaps = []
    start = None
    for position, is_open in enumerate(open_positions):
        if is_open and start is None:
            start = position
        elif not is_open and start is not None:
            gaps.append((start, position))
            start = None
    if start is not None:
        gaps.append((start, len(open_positions)))
    # tolerate a single crossing cell (a headline crossing a gutter) between gaps
    merged = []
    for a, b in gaps:
        if merged and a - merged[-1][1] <= 1:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [
        (a, b) for a, b in merged
        if b - a >= min_gap_cells and a > 0 and b < len(open_positions)
    ]


def xycut(ink, rules, box, min_block_cells, min_gap_cells, rule_frac, ink_eps, depth=0):
    x1, y1, x2, y2 = box
    if depth >= 8:
        return [box]
    candidates = {}
    for cut_axis in ("y", "x"):
        length = (y2 - y1) if cut_axis == "y" else (x2 - x1)
        if length < 2 * min_block_cells:
            continue
        cuts = find_cuts(ink, rules, box, cut_axis, min_gap_cells[cut_axis], rule_frac, ink_eps)
        cuts = [(a, b) for a, b in cuts if min(a, length - b) >= min_block_cells // 2]
        if cuts:
            candidates[cut_axis] = (max(b - a for a, b in cuts), cuts, length)
    if not candidates:
        return [box]
    # cut along the axis with the widest separator
    cut_axis = max(candidates, key=lambda k: candidates[k][0])
    _, cuts, length = candidates[cut_axis]
    segments = []
    previous = 0
    for a, b in cuts:
        segments.append((previous, a))
        previous = b
    segments.append((previous, length))
    segments = [(a, b) for a, b in segments if b - a >= 2]
    if len(segments) < 2:
        return [box]
    blocks = []
    for a, b in segments:
        child = (x1, y1 + a, x2, y1 + b) if cut_axis == "y" else (x1 + a, y1, x1 + b, y2)
        blocks.extend(
            xycut(ink, rules, child, min_block_cells, min_gap_cells,
                  rule_frac, ink_eps, depth + 1)
        )
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--ink-threshold", type=int, default=128)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--min-gap-y-px", type=int, default=24, help="horizontal-cut gutter width, source px")
    parser.add_argument("--min-gap-x-px", type=int, default=40,
                        help="vertical-cut gutter width; must exceed text-column gutters")
    parser.add_argument("--min-block-px", type=int, default=360, help="minimum block side, source px")
    parser.add_argument("--rule-frac", type=float, default=0.25)
    parser.add_argument("--ink-eps", type=float, default=0.04)
    parser.add_argument("--min-ink-fraction", type=float, default=0.01)
    parser.add_argument("--crops-dir", default="")
    parser.add_argument("--overlay-out", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact = json.loads(Path(args.artifact).read_text())
    page = np.asarray(Image.open(args.image).convert("L"))
    page_sha = _sha256_file(Path(args.image))
    if page_sha != artifact["image_sha256"]:
        raise SystemExit("page hash does not match artifact image_sha256; refusing")

    scale = args.scale
    ink, rules = build_masks(page, artifact, args.ink_threshold, scale)
    height, width = ink.shape
    xycut_args = dict(
        min_block_cells=args.min_block_px // scale,
        min_gap_cells={"y": max(2, args.min_gap_y_px // scale), "x": max(2, args.min_gap_x_px // scale)},
        rule_frac=args.rule_frac,
        ink_eps=args.ink_eps,
    )

    # Frames first: closed-frame cells are atomic blocks (a frame is the
    # boundary; interior whitespace must not subdivide an ad). Only the
    # unframed remainder is XY-cut.
    cs = 2  # label at scale*cs = 16
    sep_h = rules.shape[0] - rules.shape[0] % cs
    sep_w = rules.shape[1] - rules.shape[1] % cs
    rules_padded = _dilate_1d(_dilate_1d(rules, 2, axis=0), 2, axis=1)
    blocked = rules_padded[:sep_h, :sep_w].reshape(sep_h // cs, cs, sep_w // cs, cs).any(axis=(1, 3))
    blocked[0, :] = blocked[-1, :] = True
    blocked[:, 0] = blocked[:, -1] = True
    labels, count = _label(~blocked)
    page_cells = labels.size
    blocks = []
    for value in range(1, count + 1):
        ys, xs = np.nonzero(labels == value)
        if ys.size < 0.0004 * page_cells:
            continue
        bx1, by1 = int(xs.min() * cs), int(ys.min() * cs)
        bx2, by2 = int((xs.max() + 1) * cs), int((ys.max() + 1) * cs)
        area_fraction = ys.size / page_cells
        ring = 2
        border = np.zeros_like(blocked)
        y1r, y2r = max(0, ys.min() - ring), min(blocked.shape[0], ys.max() + 1 + ring)
        x1r, x2r = max(0, xs.min() - ring), min(blocked.shape[1], xs.max() + 1 + ring)
        border[y1r:y2r, x1r:x2r] = True
        border[ys.min():ys.max()+1, xs.min():xs.max()+1] = False
        framedness = float(blocked[border].mean()) if border.any() else 0.0
        cell_box = (bx1, by1, min(bx2, width), min(by2, height))
        if area_fraction <= 0.15 and framedness >= 0.5:
            blocks.append(cell_box)  # atomic framed ad
        else:
            blocks.extend(xycut(ink, rules, cell_box, **xycut_args))

    kept = []
    for x1, y1, x2, y2 in blocks:
        window = ink[y1:y2, x1:x2]
        if window.size == 0 or window.mean() < args.min_ink_fraction:
            continue
        ys, xs = np.nonzero(window)
        margin = max(1, 8 // scale)
        tx1 = max(0, x1 + int(xs.min()) - margin)
        ty1 = max(0, y1 + int(ys.min()) - margin)
        tx2 = min(width, x1 + int(xs.max()) + 1 + margin)
        ty2 = min(height, y1 + int(ys.max()) + 1 + margin)
        kept.append(
            {
                "block_id": len(kept),
                "bbox_source_xyxy": [tx1 * scale, ty1 * scale, tx2 * scale, ty2 * scale],
                "ink_fraction": round(float(window.mean()), 4),
            }
        )

    crops_dir = Path(args.crops_dir) if args.crops_dir else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
        source = Image.open(args.image).convert("L")
        for block in kept:
            x1, y1, x2, y2 = block["bbox_source_xyxy"]
            crop = source.crop((x1, y1, x2, y2))  # original polarity for review
            path = crops_dir / f"block{block['block_id']:03d}.png"
            crop.save(path)
            block["crop_path"] = str(path)
            block["crop_sha256"] = _sha256_file(path)

    proxy = Image.open(args.proxy).convert("RGB")
    sx = proxy.width / (width * scale)
    sy = proxy.height / (height * scale)
    draw = ImageDraw.Draw(proxy)
    for block in kept:
        x1, y1, x2, y2 = block["bbox_source_xyxy"]
        draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=(200, 30, 160), width=3)
        draw.text((x1 * sx + 4, y1 * sy + 2), str(block["block_id"]), fill=(200, 30, 160))
    overlay_path = Path(args.overlay_out)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    proxy.save(overlay_path)

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_not_qualified",
        "semantic_boundary_authority": "none",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_artifact": {"path": args.artifact, "run": artifact.get("run", {})},
        "page_image": {"path": args.image, "sha256": page_sha},
        "parameters": {
            key: getattr(args, key)
            for key in ("ink_threshold", "scale", "min_gap_y_px", "min_gap_x_px", "min_block_px",
                        "rule_frac", "ink_eps", "min_ink_fraction")
        },
        "blocks": {"count": len(kept), "list": kept},
        "overlay": {"path": str(overlay_path)},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=1))
    print(json.dumps({"blocks": len(kept)}))
    print(f"[artifact] {out_path}")
    print(f"[overlay] {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
