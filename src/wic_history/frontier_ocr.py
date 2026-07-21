"""Frontier-model OCR over article-block crops (one-time ingestion pass).

Owner ruling 2026-07-20: OCR and entity extraction are the only two LLM
passes in ingestion, each run exactly once, via a remote provider pinned in
config/pipeline-models.toml. This module transcribes one block crop per
call. A failed, truncated, or implausible response abstains for that block —
no retry with another model, no partial fabrication; abstained blocks are
listed in the output artifact for later explicit re-runs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, final

from .model_config import FrontierOCRModel, load_pipeline_model_configuration
from .semantic_tasks import REMOTE_CONSENT_ENVIRONMENT_VARIABLE

_MAX_CROP_BYTES = 8 * 1024 * 1024


@final
class FrontierOCRAbstention(RuntimeError):
    """This block's transcription is unusable and must not be persisted."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class BlockTranscription:
    block_index: int
    bbox_source_xyxy: tuple[int, int, int, int]
    crop_sha256: str
    text: str
    text_sha256: str
    finish_reason: str | None
    usage: dict[str, Any]


def _consented_api_key(model: FrontierOCRModel) -> str:
    def _true(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() == "true"

    if not _true(REMOTE_CONSENT_ENVIRONMENT_VARIABLE):
        raise RuntimeError(
            "frontier OCR sends page-image crops to a remote endpoint; set "
            f"{REMOTE_CONSENT_ENVIRONMENT_VARIABLE}=true to consent"
        )
    api_key = os.environ.get(model.api_key_environment_variable, "")
    if not api_key:
        raise RuntimeError(
            f"{model.api_key_environment_variable} is required for frontier "
            "OCR; the key is read from the environment and never written "
            "into configuration or artifacts"
        )
    return api_key


def build_frontier_generator(model: FrontierOCRModel):
    from .generation import OpenAICompatibleGenerator

    return OpenAICompatibleGenerator(
        model.base_url,
        model.served_model,
        api_key=_consented_api_key(model),
        model_revision=model.model_revision_status,
        timeout_seconds=min(model.timeout_seconds, 300.0),
        max_output_tokens=model.max_output_tokens,
        seed=None,
        allow_remote=True,
    )


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿" or ch == "□")
    visible = sum(1 for ch in text if not ch.isspace())
    return (cjk / visible) if visible else 0.0


def validate_transcription(text: str) -> str:
    """Plausibility gate: fluent off-domain hallucination must abstain.

    Measured failure mode (p0308 display-crop validation): near-blank crops
    can elicit fluent unrelated text that passes JSON/termination gates. A
    Shen Bao block transcription is overwhelmingly CJK; anything else is not
    a transcription of this crop.
    """
    stripped = text.strip()
    if not stripped:
        raise FrontierOCRAbstention("empty transcription")
    if _cjk_ratio(stripped) < 0.5:
        raise FrontierOCRAbstention("transcription is not predominantly CJK")
    lines = stripped.splitlines()
    if len(lines) > 3 and len(set(lines)) <= max(1, len(lines) // 4):
        raise FrontierOCRAbstention("transcription degenerated into repetition")
    return stripped


def transcribe_block(
    generator: Any,
    model: FrontierOCRModel,
    crop_png: bytes,
    *,
    block_index: int,
    bbox: tuple[int, int, int, int],
) -> BlockTranscription:
    if len(crop_png) > _MAX_CROP_BYTES:
        raise FrontierOCRAbstention("crop image exceeds the size limit")
    encoded = base64.b64encode(crop_png).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": model.prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]
    try:
        completion = generator.complete(messages)
    except RuntimeError as exc:
        raise FrontierOCRAbstention(f"provider call failed: {exc}") from exc
    finish_reason = getattr(completion, "finish_reason", None)
    content = getattr(completion, "content", completion)
    if finish_reason not in {None, "stop", "end_turn"}:
        raise FrontierOCRAbstention(f"non-terminal finish reason {finish_reason!r}")
    text = validate_transcription(str(content))
    usage = getattr(completion, "usage", None) or {}
    return BlockTranscription(
        block_index=block_index,
        bbox_source_xyxy=bbox,
        crop_sha256=hashlib.sha256(crop_png).hexdigest(),
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        finish_reason=finish_reason,
        usage=dict(usage) if isinstance(usage, dict) else {},
    )


def transcribe_page_blocks(
    page_image_path: Path,
    cells: list[tuple[int, int, int, int]],
    output_path: Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Crop each detector cell and transcribe it; write the page artifact."""
    import cv2

    configuration = load_pipeline_model_configuration()
    model = configuration.frontier_ocr
    if model is None:
        raise RuntimeError(
            "the pinned pipeline configuration declares no [frontier_ocr] model"
        )
    generator = build_frontier_generator(model)
    page = cv2.imread(str(page_image_path), cv2.IMREAD_GRAYSCALE)
    if page is None:
        raise ValueError(f"cannot read page image {page_image_path}")
    page_sha256 = hashlib.sha256(page_image_path.read_bytes()).hexdigest()
    blocks: list[dict[str, Any]] = []
    abstained: list[dict[str, Any]] = []
    selected = cells[:limit] if limit else cells
    for index, (x1, y1, x2, y2) in enumerate(selected):
        crop = page[max(0, y1):y2, max(0, x1):x2]
        # source pages render inverted (white text on black); frontier vision
        # models read black-on-white, and this inversion is presentation
        # only — the stored provenance stays the original image hash
        ok, buffer = cv2.imencode(".png", 255 - crop)
        if not ok:
            abstained.append({"block_index": index, "reason": "crop encode failed"})
            continue
        try:
            result = transcribe_block(
                generator,
                model,
                buffer.tobytes(),
                block_index=index,
                bbox=(x1, y1, x2, y2),
            )
        except FrontierOCRAbstention as error:
            abstained.append({"block_index": index, "reason": error.reason})
            continue
        blocks.append(asdict(result))
    artifact = {
        "schema_version": "frontier-ocr/v1",
        "status": "machine_tier",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_image": {"path": str(page_image_path), "sha256": page_sha256},
        "model_identity": model.provenance_identity(),
        "configuration_sha256": configuration.sha256,
        "blocks": blocks,
        "abstained": abstained,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=1))
    return artifact
