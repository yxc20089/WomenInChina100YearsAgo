"""RLSA-style content-blob block detection with rule removal.

Operator-proposed method (morphological merge + external contours), adapted:
printed rules CONNECT neighboring blocks, so detected rule pixels are removed
first (line detection in a removal role needs no closure), then remaining ink
is closed into solid blobs; external contour bounding boxes are the blocks.
Complementary to line-tracing: works where gutters, not rules, separate
content. Parameters are swept against an operator gold labeling when given.

Diagnostic proposals; no semantic authority.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SCHEMA_VERSION = "rlsa-blocks-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_mask(ink, min_len_cells):
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len_cells, 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len_cells))
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_k)
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_k)
    return cv2.bitwise_or(h, v)


def blocks_for(ink, rules, close_cells, min_dim_cells, min_ink_fraction):
    content = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(
        rules, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_cells, close_cells))
    blob = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < min_dim_cells:
            continue
        window = content[y:y + h, x:x + w]
        if window.size == 0 or (window > 0).mean() < min_ink_fraction:
            continue
        out.append((x, y, x + w, y + h))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--ink-threshold", type=int, default=128)
    parser.add_argument("--line-len-src-px", type=int, default=800)
    parser.add_argument("--close-src-px", type=int, default=0,
                        help="0 = sweep against gold and pick best")
    parser.add_argument("--sweep", default="24,32,40,48,64,80,96",
                        help="close sizes (source px) to sweep when close=0")
    parser.add_argument("--min-dim-src-px", type=int, default=160)
    parser.add_argument("--min-ink-fraction", type=float, default=0.02)
    parser.add_argument("--gold", default="", help="gold blocks JSON for scoring/sweeping")
    parser.add_argument("--match-iou", type=float, default=0.6)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--crops-dir", default="")
    parser.add_argument("--overlay-out", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    scale = args.scale
    ink = ((source[::scale, ::scale] > args.ink_threshold).astype(np.uint8)) * 255
    rules = line_mask(ink, max(8, args.line_len_src_px // scale))

    gold = None
    if args.gold:
        gold = [g["bbox_source_xyxy"] for g in json.loads(Path(args.gold).read_text())["blocks"]]

    def score(cells_src):
        if not gold:
            return None
        matched = sum(1 for g in gold if max((iou(g, c) for c in cells_src), default=0) >= args.match_iou)
        spurious = sum(1 for c in cells_src if max((iou(g, c) for g in gold), default=0) < args.match_iou)
        return {"gold_matched": matched, "gold_total": len(gold),
                "spurious": spurious, "detected": len(cells_src)}

    candidates = ([args.close_src_px] if args.close_src_px
                  else [int(v) for v in args.sweep.split(",")])
    best = None
    for close_src in candidates:
        cells = blocks_for(ink, rules, max(3, close_src // scale),
                           max(2, args.min_dim_src_px // scale), args.min_ink_fraction)
        cells_src = [[v * scale for v in c] for c in cells]
        s = score(cells_src)
        rank = (s["gold_matched"], -s["spurious"]) if s else (len(cells_src),)
        print(json.dumps({"close_src_px": close_src, **(s or {"detected": len(cells_src)})}))
        if best is None or rank > best[0]:
            best = (rank, close_src, cells_src, s)
    _, close_chosen, cells_src, final_score = best

    crops_dir = Path(args.crops_dir) if args.crops_dir else None
    blocks = []
    for i, (x1, y1, x2, y2) in enumerate(sorted(cells_src, key=lambda c: (c[1], c[0]))):
        block = {"block_id": i, "bbox_source_xyxy": [x1, y1, x2, y2]}
        if crops_dir:
            crops_dir.mkdir(parents=True, exist_ok=True)
            path = crops_dir / f"block{i:02d}_{x2-x1}x{y2-y1}.png"
            cv2.imwrite(str(path), source[y1:y2, x1:x2])
            block["crop_path"] = str(path)
        blocks.append(block)

    proxy = cv2.imread(args.proxy, cv2.IMREAD_COLOR)
    px = proxy.shape[1] / source.shape[1]
    py = proxy.shape[0] / source.shape[0]
    if gold:
        for g in gold:
            cv2.rectangle(proxy, (int(g[0] * px), int(g[1] * py)),
                          (int(g[2] * px), int(g[3] * py)), (60, 180, 60), 2)
    for block in blocks:
        x1, y1, x2, y2 = block["bbox_source_xyxy"]
        cv2.rectangle(proxy, (int(x1 * px), int(y1 * py)), (int(x2 * px), int(y2 * py)),
                      (30, 30, 220), 3)
        cv2.putText(proxy, str(block["block_id"]), (int(x1 * px) + 4, int(y1 * py) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 220), 2)
    Path(args.overlay_out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.overlay_out, proxy)

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_not_qualified",
        "semantic_boundary_authority": "none",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "page_image": {"path": args.image, "sha256": _sha256_file(Path(args.image))},
        "parameters": {"scale": scale, "line_len_src_px": args.line_len_src_px,
                       "close_src_px_chosen": close_chosen,
                       "min_dim_src_px": args.min_dim_src_px,
                       "min_ink_fraction": args.min_ink_fraction},
        "score_vs_gold": final_score,
        "blocks": {"count": len(blocks), "list": blocks},
        "overlay": {"path": args.overlay_out},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=1))
    print(json.dumps({"chosen_close_src_px": close_chosen, "blocks": len(blocks),
                      "score": final_score}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
