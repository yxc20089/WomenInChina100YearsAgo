"""Process a bounded batch of OCR tiles in one short-lived Paddle process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .ocr_pipeline import PaddleOCRPredictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="ch")
    args = parser.parse_args()
    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    predictor = PaddleOCRPredictor(args.language)
    output = []
    for item in items:
        with Image.open(item["path"]) as image:
            lines = predictor(image.convert("RGB"))
        output.append(
            {
                "tile_index": item["tile_index"],
                "lines": [
                    {
                        "text": line.text,
                        "confidence": line.confidence,
                        "points": line.points,
                        "engine_payload": line.engine_payload,
                    }
                    for line in lines
                ],
            }
        )
    args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
