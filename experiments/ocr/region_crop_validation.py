"""Validate the detect -> crop -> transcribe loop on one page.

Reads text-detection polygons from an existing OCR artifact (PP-OCRv6 pilot),
crops each region from the immutable lossless page with padding, applies the
recorded reversible polarity transform, sends every crop to a running
llama.cpp HunyuanOCR server as an official crop-level ``spotting_json``
request, and records per-crop validity and agreement diagnostics.

This is a diagnostic for region-proposal machinery. It claims no gold
accuracy: the detector boxes are unreviewed proposals and the comparison
text is unreviewed machine OCR. Outputs are experiment artifacts and must
never be ingested as reviewed evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import base64
import datetime as _dt
import hashlib
import io
import json
import re
import tomllib
from pathlib import Path

import httpx
from PIL import Image

SCHEMA_VERSION = "region-crop-validation-v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_spotting_prompt(config_path: Path) -> str:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    return config["ocr"]["spotting_prompt"]


def _polygon_bbox(polygon: dict) -> tuple[int, int, int, int]:
    xs = [point["x"] for point in polygon["points"]]
    ys = [point["y"] for point in polygon["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def _padded_bbox(
    bbox: tuple[int, int, int, int], pad_fraction: float, width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad = max(12, round(pad_fraction * min(x2 - x1, y2 - y1)))
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


_SPOTTING_ITEM = re.compile(
    r"\{\s*\"box\"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*,"
    r"\s*\"text\"\s*:\s*\"(.*?)\"\s*\}",
    re.DOTALL,
)


def _parse_spotting(content: str) -> tuple[list[dict], bool]:
    """Parse spotting output; returns (items, strict_json)."""
    try:
        loaded = json.loads(content)
        if isinstance(loaded, list) and all(
            isinstance(item, dict) and "box" in item and "text" in item
            for item in loaded
        ):
            return loaded, True
    except json.JSONDecodeError:
        pass
    items = [
        {
            "box": [int(match.group(i)) for i in range(1, 5)],
            "text": match.group(5),
        }
        for match in _SPOTTING_ITEM.finditer(content)
    ]
    return items, False


def _char_f1(reference: str, hypothesis: str) -> float:
    """Order-insensitive character F1; a fuzzy agreement signal only."""
    ref = [c for c in reference if not c.isspace()]
    hyp = [c for c in hypothesis if not c.isspace()]
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    from collections import Counter

    overlap = sum((Counter(ref) & Counter(hyp)).values())
    precision = overlap / len(hyp)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _boxes_in_range(items: list[dict]) -> bool:
    for item in items:
        box = item.get("box", [])
        if len(box) != 4:
            return False
        x1, y1, x2, y2 = box
        if not (0 <= x1 <= 1000 and 0 <= y1 <= 1000 and x1 <= x2 <= 1000 and y1 <= y2 <= 1000):
            return False
    return True


def _request_one(
    client: httpx.Client,
    endpoint: str,
    model: str,
    prompt: str,
    crop_png: bytes,
    max_tokens: int,
) -> dict:
    payload = {
        "model": model,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "seed": 42,
        "max_tokens": max_tokens,
        "cache_prompt": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(crop_png).decode("ascii")
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    response = client.post(f"{endpoint}/v1/chat/completions", json=payload)
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    return {
        "content": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="OCR artifact with det polygons")
    parser.add_argument("--image", required=True, help="lossless page image")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="HYVL-F16")
    parser.add_argument("--model-gguf", required=True)
    parser.add_argument("--mmproj-gguf", required=True)
    parser.add_argument("--config", default="config/pipeline-models.toml")
    parser.add_argument("--pad-fraction", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 = all regions")
    parser.add_argument("--crops-dir", default="", help="optional dir to keep crop PNGs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact = json.loads(Path(args.artifact).read_text())
    regions = artifact["regions"]
    if args.limit:
        step = max(1, len(regions) // args.limit)
        regions = regions[::step][: args.limit]

    page = Image.open(args.image).convert("L")
    page_sha = _sha256_file(Path(args.image))
    if page_sha != artifact["image_sha256"]:
        raise SystemExit(
            f"page hash {page_sha} does not match artifact image_sha256 "
            f"{artifact['image_sha256']}; refusing to crop"
        )
    prompt = _load_spotting_prompt(Path(args.config))

    crops_dir = Path(args.crops_dir) if args.crops_dir else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    for region in regions:
        bbox = _polygon_bbox(region["polygon"])
        padded = _padded_bbox(bbox, args.pad_fraction, page.width, page.height)
        crop = page.crop(padded)
        crop = Image.eval(crop, lambda px: 255 - px)  # match qualified black-on-white regime
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        crop_png = buffer.getvalue()
        if crops_dir:
            (crops_dir / f"{region['region_id']}.png").write_bytes(crop_png)
        prepared.append((region, bbox, padded, crop_png))

    results = []
    with httpx.Client(timeout=300.0) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    _request_one,
                    client,
                    args.endpoint,
                    args.model,
                    prompt,
                    crop_png,
                    args.max_tokens,
                ): (region, bbox, padded, crop_png)
                for region, bbox, padded, crop_png in prepared
            }
            for future in concurrent.futures.as_completed(futures):
                region, bbox, padded, crop_png = futures[future]
                try:
                    outcome = future.result()
                    error = None
                except Exception as exc:  # noqa: BLE001 - recorded, not retried
                    outcome = None
                    error = repr(exc)
                entry = {
                    "region_id": region["region_id"],
                    "reading_order": region["reading_order"],
                    "det_bbox_xyxy": list(bbox),
                    "crop_bbox_xyxy": list(padded),
                    "crop_sha256": _sha256_bytes(crop_png),
                    "paddle_raw_text": region.get("raw_text"),
                    "error": error,
                }
                if outcome is not None:
                    items, strict = _parse_spotting(outcome["content"])
                    joined = "".join(item.get("text", "") for item in items)
                    entry.update(
                        {
                            "finish_reason": outcome["finish_reason"],
                            "completion_tokens": outcome["usage"].get("completion_tokens"),
                            "content": outcome["content"],
                            "content_sha256": _sha256_bytes(
                                outcome["content"].encode("utf-8")
                            ),
                            "strict_json": strict,
                            "item_count": len(items),
                            "boxes_in_range": _boxes_in_range(items),
                            "hunyuan_text": joined,
                            "char_f1_vs_paddle": round(
                                _char_f1(region.get("raw_text") or "", joined), 4
                            ),
                        }
                    )
                results.append(entry)

    results.sort(key=lambda item: item["reading_order"])
    completed = [r for r in results if r.get("error") is None]
    summary = {
        "regions_requested": len(prepared),
        "completed": len(completed),
        "transport_errors": len(results) - len(completed),
        "strict_json": sum(1 for r in completed if r.get("strict_json")),
        "parseable": sum(1 for r in completed if r.get("item_count", 0) > 0),
        "empty_output": sum(1 for r in completed if r.get("item_count", 0) == 0),
        "stopped_normally": sum(
            1 for r in completed if r.get("finish_reason") == "stop"
        ),
        "hit_token_cap": sum(
            1 for r in completed if r.get("finish_reason") == "length"
        ),
        "boxes_in_range": sum(1 for r in completed if r.get("boxes_in_range")),
        "median_char_f1_vs_paddle": None,
    }
    f1s = sorted(
        r["char_f1_vs_paddle"] for r in completed if r.get("char_f1_vs_paddle") is not None
    )
    if f1s:
        summary["median_char_f1_vs_paddle"] = f1s[len(f1s) // 2]

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_not_qualified",
        "purpose": (
            "Detect->crop->transcribe loop validation over unreviewed detector "
            "proposals; not gold accuracy evidence."
        ),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_artifact": {
            "path": args.artifact,
            "run": artifact.get("run", {}),
            "image_sha256": artifact["image_sha256"],
        },
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
            "pad_fraction": args.pad_fraction,
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
