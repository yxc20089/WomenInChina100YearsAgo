"""Compose rule-traced cells with whitespace XY-cut into a full block set.

Stage 1 (input): boundary_lines cells — regions enclosed by detected printed
rules. Framed ads arrive complete; the frameless remainder arrives as one or
few oversized cells.
Stage 2: any cell larger than --max-direct-area-frac is subdivided by
whitespace XY-cut restricted to that cell's own ink (other cells' regions are
masked out), using the detected rules as additional separators.
Stage 3: blocks are trimmed to ink extent; slivers and empty leaves dropped.

Outputs block crops (ORIGINAL polarity), overlay, and JSON. Diagnostic
proposals for review; no semantic authority.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from block_xycut import xycut  # noqa: E402

SCHEMA_VERSION = "compose-blocks-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--boundary", required=True, help="boundary_lines output JSON")
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--ink-threshold", type=int, default=128)
    parser.add_argument("--endpoint-extend-src-px", type=int, default=72)
    parser.add_argument("--max-direct-area-frac", type=float, default=0.08)
    parser.add_argument("--min-block-src-px", type=int, default=200)
    parser.add_argument("--min-gap-y-px", type=int, default=24)
    parser.add_argument("--min-gap-x-px", type=int, default=40)
    parser.add_argument("--ink-eps", type=float, default=0.04)
    parser.add_argument("--min-ink-fraction", type=float, default=0.015)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--crops-dir", required=True)
    parser.add_argument("--overlay-out", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    boundary = json.loads(Path(args.boundary).read_text())
    source = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    scale = args.scale
    small = source[::scale, ::scale]
    ink = small > args.ink_threshold
    height, width = ink.shape

    # rasterize detected rules at working scale, with endpoint extension
    work_scale = boundary["parameters"]["work_scale"]
    factor = work_scale / scale
    rules = np.zeros_like(ink, dtype=np.uint8)
    for x1, y1, x2, y2 in boundary["lines"]["horizontal_boxes_workscale"]:
        rules[int(y1 * factor):int(y2 * factor) + 1,
              int(x1 * factor):int(x2 * factor) + 1] = 255
    for x1, y1, x2, y2 in boundary["lines"]["vertical_boxes_workscale"]:
        rules[int(y1 * factor):int(y2 * factor) + 1,
              int(x1 * factor):int(x2 * factor) + 1] = 255
    extend = max(1, args.endpoint_extend_src_px // scale)
    h_ext = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * extend + 1, 3)))
    v_ext = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2 * extend + 1)))
    separator = cv2.bitwise_or(h_ext, v_ext)
    separator[0, :] = separator[-1, :] = 255
    separator[:, 0] = separator[:, -1] = 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (separator == 0).astype(np.uint8), connectivity=4)
    total = labels.size

    blocks = []
    rules_bool = separator > 0
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 0.0004 * total:
            continue
        if min(w, h) * scale < args.min_block_src_px:
            continue  # gutter / rule-band sliver
        if area <= args.max_direct_area_frac * total:
            leaves = [(x, y, x + w, y + h)]
            origin = "rule_frame"
        else:
            cell_ink = ink & (labels == index)
            leaves = xycut(
                cell_ink, rules_bool, (x, y, x + w, y + h),
                min_block_cells=max(4, args.min_block_src_px // scale),
                min_gap_cells={"y": max(2, args.min_gap_y_px // scale),
                               "x": max(2, args.min_gap_x_px // scale)},
                rule_frac=0.25,
                ink_eps=args.ink_eps,
            )
            origin = "whitespace_cut"
        for lx1, ly1, lx2, ly2 in leaves:
            window = (ink & (labels == index))[ly1:ly2, lx1:lx2]
            if window.size == 0 or window.mean() < args.min_ink_fraction:
                continue
            ys, xs = np.nonzero(window)
            tx1, ty1 = int(lx1) + int(xs.min()), int(ly1) + int(ys.min())
            tx2, ty2 = int(lx1) + int(xs.max()) + 1, int(ly1) + int(ys.max()) + 1
            if min(tx2 - tx1, ty2 - ty1) * scale < args.min_block_src_px:
                continue
            blocks.append({
                "block_id": len(blocks),
                "origin": origin,
                "cell_label": int(index),
                "bbox_source_xyxy": [tx1 * scale, ty1 * scale, tx2 * scale, ty2 * scale],
                "ink_fraction": round(float(window.mean()), 4),
            })

    crops_dir = Path(args.crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        x1, y1, x2, y2 = block["bbox_source_xyxy"]
        crop = source[y1:y2, x1:x2]
        path = crops_dir / f"block{block['block_id']:03d}_{x2-x1}x{y2-y1}.png"
        cv2.imwrite(str(path), crop)
        block["crop_path"] = str(path)

    proxy = cv2.imread(args.proxy, cv2.IMREAD_COLOR)
    px = proxy.shape[1] / source.shape[1]
    py = proxy.shape[0] / source.shape[0]
    for block in blocks:
        x1, y1, x2, y2 = block["bbox_source_xyxy"]
        color = (30, 30, 220) if block["origin"] == "rule_frame" else (220, 120, 30)
        cv2.rectangle(proxy, (int(x1 * px), int(y1 * py)), (int(x2 * px), int(y2 * py)), color, 3)
        cv2.putText(proxy, str(block["block_id"]), (int(x1 * px) + 4, int(y1 * py) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    Path(args.overlay_out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.overlay_out, proxy)

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_not_qualified",
        "semantic_boundary_authority": "none",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "page_image": {"path": args.image, "sha256": _sha256_file(Path(args.image))},
        "boundary_artifact": args.boundary,
        "parameters": {
            key: getattr(args, key)
            for key in ("scale", "ink_threshold", "endpoint_extend_src_px",
                        "max_direct_area_frac", "min_block_src_px", "min_gap_y_px",
                        "min_gap_x_px", "ink_eps", "min_ink_fraction")
        },
        "blocks": {"count": len(blocks), "list": blocks},
        "overlay": {"path": args.overlay_out},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=1))
    from collections import Counter
    print(json.dumps({"blocks": len(blocks),
                      "by_origin": dict(Counter(b["origin"] for b in blocks))}))
    print(f"[artifact] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
