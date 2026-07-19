"""Validate display/figure and column crop proposals through crop OCR.

Reads `display_or_figure` proposals and column proposals from a
region_cell_geometry artifact, crops them from the immutable lossless page
(padded, polarity-inverted), and sends each through the qualified crop-level
``spotting_json`` path. Diagnostic only; no gold accuracy claim.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import hashlib
import io
import json
import sys
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from region_crop_validation import (  # noqa: E402
    _load_spotting_prompt,
    _parse_spotting,
    _boxes_in_range,
    _request_one,
    _sha256_bytes,
    _sha256_file,
)

SCHEMA_VERSION = "display-crop-validation-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, help="region_cell_geometry artifact")
    parser.add_argument("--image", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="HYVL-F16")
    parser.add_argument("--model-gguf", required=True)
    parser.add_argument("--mmproj-gguf", required=True)
    parser.add_argument("--config", default="config/pipeline-models.toml")
    parser.add_argument("--pad", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--column-sample", type=int, default=30,
                        help="largest multi-member columns to include")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    geometry = json.loads(Path(args.geometry).read_text())
    page = Image.open(args.image).convert("L")
    page_sha = _sha256_file(Path(args.image))
    if page_sha != geometry["page_image"]["sha256"]:
        raise SystemExit("page hash mismatch with geometry artifact; refusing")
    prompt = _load_spotting_prompt(Path(args.config))

    units = [
        {"unit_kind": "display_or_figure", "bbox": p["bbox_source_xyxy"], "cell_id": p["cell_id"]}
        for p in geometry["display_proposals"]["list"]
    ]
    columns = [c for c in geometry["columns"]["list"] if c["member_regions"] >= 2]
    columns.sort(key=lambda c: c["member_regions"], reverse=True)
    units += [
        {"unit_kind": "column", "bbox": c["bbox_source_xyxy"], "cell_id": c["cell_id"]}
        for c in columns[: args.column_sample]
    ]

    prepared = []
    for index, unit in enumerate(units):
        x1, y1, x2, y2 = unit["bbox"]
        box = (
            max(0, x1 - args.pad), max(0, y1 - args.pad),
            min(page.width, x2 + args.pad), min(page.height, y2 + args.pad),
        )
        crop = Image.eval(page.crop(box), lambda px: 255 - px)
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        prepared.append((index, unit, box, buffer.getvalue()))

    results = []
    with httpx.Client(timeout=600.0) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(_request_one, client, args.endpoint, args.model, prompt,
                            crop_png, args.max_tokens): (index, unit, box, crop_png)
                for index, unit, box, crop_png in prepared
            }
            for future in concurrent.futures.as_completed(futures):
                index, unit, box, crop_png = futures[future]
                try:
                    outcome = future.result()
                    error = None
                except Exception as exc:  # noqa: BLE001 - recorded, not retried
                    outcome, error = None, repr(exc)
                entry = {
                    "unit_index": index,
                    "unit_kind": unit["unit_kind"],
                    "cell_id": unit["cell_id"],
                    "crop_bbox_xyxy": list(box),
                    "crop_sha256": _sha256_bytes(crop_png),
                    "error": error,
                }
                if outcome is not None:
                    items, strict = _parse_spotting(outcome["content"])
                    entry.update(
                        {
                            "finish_reason": outcome["finish_reason"],
                            "completion_tokens": outcome["usage"].get("completion_tokens"),
                            "content": outcome["content"],
                            "strict_json": strict,
                            "item_count": len(items),
                            "boxes_in_range": _boxes_in_range(items),
                            "text": "".join(item.get("text", "") for item in items),
                        }
                    )
                results.append(entry)

    results.sort(key=lambda item: item["unit_index"])
    completed = [r for r in results if r.get("error") is None]

    def _bucket(kind):
        rows = [r for r in completed if r["unit_kind"] == kind]
        return {
            "count": len(rows),
            "stopped_normally": sum(1 for r in rows if r.get("finish_reason") == "stop"),
            "strict_json": sum(1 for r in rows if r.get("strict_json")),
            "non_empty": sum(1 for r in rows if r.get("item_count", 0) > 0),
            "boxes_in_range": sum(1 for r in rows if r.get("boxes_in_range")),
        }

    summary = {
        "units": len(prepared),
        "completed": len(completed),
        "transport_errors": len(results) - len(completed),
        "display_or_figure": _bucket("display_or_figure"),
        "column": _bucket("column"),
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_not_qualified",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "geometry_artifact": args.geometry,
        "page_image": {"path": args.image, "sha256": page_sha},
        "model": {
            "endpoint": args.endpoint,
            "alias": args.model,
            "model_gguf_sha256": _sha256_file(Path(args.model_gguf)),
            "mmproj_gguf_sha256": _sha256_file(Path(args.mmproj_gguf)),
        },
        "request": {
            "task": "spotting_json",
            "prompt": prompt,
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "seed": 42,
            "max_tokens": args.max_tokens,
            "cache_prompt": False,
            "client_concurrency": args.concurrency,
            "pad_px": args.pad,
            "polarity_transform": "invert_rgb_255_minus_channel_after_crop",
        },
        "summary": summary,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))
    print(f"[artifact] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
