"""Short-lived PaddleOCR tile worker used to isolate native runtime state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .ocr_pipeline import PaddleOCRPredictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="ch")
    args = parser.parse_args()
    with Image.open(args.input) as image:
        lines = PaddleOCRPredictor(args.language)(image.convert("RGB"))
    payload = [
        {
            "text": line.text,
            "confidence": line.confidence,
            "points": line.points,
            "engine_payload": line.engine_payload,
        }
        for line in lines
    ]
    args.output.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
