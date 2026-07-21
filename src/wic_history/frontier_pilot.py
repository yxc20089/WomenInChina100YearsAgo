"""Run the two one-time frontier passes over one page's detector blocks.

Pilot for unit economics and extraction validation: block cells (region
detector output) -> frontier OCR -> broad extraction v2, all results in one
provenance-carrying artifact. Abstentions are listed, never papered over.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .frontier_extraction import (
    ExtractionAbstention,
    extract_article,
    extraction_identity,
)
from .frontier_ocr import transcribe_page_blocks
from .model_config import load_pipeline_model_configuration


def _cells_from_detector(path: Path) -> list[tuple[int, int, int, int]]:
    payload = json.loads(path.read_text())
    return [tuple(cell) for cell in payload["cells"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="lossless page image")
    parser.add_argument(
        "--cells", default=None, help="rule-detector output JSON (cells list)"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N blocks only")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument(
        "--ocr-artifact",
        default=None,
        help="reuse an existing frontier-ocr artifact instead of re-transcribing",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)
    if args.ocr_artifact:
        ocr_artifact = json.loads(Path(args.ocr_artifact).read_text())
        cells = [tuple(b["bbox_source_xyxy"]) for b in ocr_artifact["blocks"]]
    else:
        cells = _cells_from_detector(Path(args.cells))
        ocr_artifact = transcribe_page_blocks(
            image_path,
            cells,
            output_path.with_suffix(".ocr.json"),
            limit=args.limit,
        )
    page_sha256 = ocr_artifact["page_image"]["sha256"]

    extractions = []
    extraction_abstained = []
    configuration = load_pipeline_model_configuration()
    if not args.skip_extraction:
        import os

        from .generation import OpenAICompatibleGenerator

        semantic = configuration.semantic
        generator = OpenAICompatibleGenerator(
            semantic.base_url,
            semantic.served_model,
            api_key=os.environ[semantic.api_key_environment_variable],
            model_revision=semantic.model_revision_status,
            timeout_seconds=min(semantic.timeout_seconds, 300.0),
            max_output_tokens=semantic.max_output_tokens,
            seed=semantic.seed,
            allow_remote=True,
        )
        for block in ocr_artifact["blocks"]:
            region_id = uuid5(
                NAMESPACE_URL,
                f"wic-frontier-block:{page_sha256}:{block['block_index']}",
            )
            try:
                result = extract_article(generator, block["text"], region_id)
            except ExtractionAbstention as error:
                extraction_abstained.append(
                    {"block_index": block["block_index"], "reason": error.reason}
                )
                continue
            extractions.append(
                {
                    "block_index": block["block_index"],
                    "region_id": str(region_id),
                    "extraction": result.model_dump(mode="json"),
                }
            )

    artifact = {
        "schema_version": "frontier-pilot/v1",
        "status": "machine_tier",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_image": ocr_artifact["page_image"],
        "ocr_model_identity": ocr_artifact["model_identity"],
        "extraction_identity": extraction_identity(
            configuration.semantic.provenance_identity()
        ),
        "configuration_sha256": configuration.sha256,
        "blocks_transcribed": len(ocr_artifact["blocks"]),
        "ocr_abstained": ocr_artifact["abstained"],
        "extractions": extractions,
        "extraction_abstained": extraction_abstained,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=1))
    print(
        json.dumps(
            {
                "blocks": len(cells),
                "transcribed": len(ocr_artifact["blocks"]),
                "ocr_abstained": len(ocr_artifact["abstained"]),
                "extracted": len(extractions),
                "extraction_abstained": len(extraction_abstained),
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
