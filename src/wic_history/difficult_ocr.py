"""Pinned HunyuanOCR-1.5 inference and strict paired-output validation.

The old difficult-region router has deliberately been removed. HunyuanOCR is
the sole learned OCR/layout model. Its official ``spotting_json`` and
``layout_parse`` tasks are run on the same immutable page. Invalid output or
text disagreement produces an abstention requiring review; no alternate model,
geometric sorter, OCR repair, or confidence estimate is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from PIL import Image
from pydantic import Field, model_validator

from .evidence import StrictModel
from .model_config import (
    LAYOUT_PARSE_PROMPT,
    SPOTTING_JSON_PROMPT,
    HunyuanOCRModel,
    load_pipeline_model_configuration,
)
from .render_samples import sha256_file


class HunyuanOutputError(ValueError):
    """The model output does not satisfy the pinned task contract."""


class HunyuanTaskBundle(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spotting_task: Literal["spotting_json"] = "spotting_json"
    spotting_prompt: Literal[SPOTTING_JSON_PROMPT] = SPOTTING_JSON_PROMPT
    spotting_raw_output: str
    spotting_raw_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spotting_visual_output_id: UUID
    layout_task: Literal["layout_parse"] = "layout_parse"
    layout_prompt: Literal[LAYOUT_PARSE_PROMPT] = LAYOUT_PARSE_PROMPT
    layout_raw_output: str
    layout_raw_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    layout_visual_output_id: UUID
    confidence_status: Literal["not_emitted_by_model"] = "not_emitted_by_model"
    confidence_calibration: Literal["not_available"] = "not_available"

    @model_validator(mode="after")
    def validate_hashes(self) -> "HunyuanTaskBundle":
        if _sha256(self.spotting_raw_output) != self.spotting_raw_output_sha256:
            raise ValueError("spotting raw-output hash does not match its bytes")
        if _sha256(self.layout_raw_output) != self.layout_raw_output_sha256:
            raise ValueError("layout raw-output hash does not match its bytes")
        expected_spotting_id = stable_visual_output_id(
            self.image_sha256, self.spotting_task, self.spotting_raw_output_sha256
        )
        expected_layout_id = stable_visual_output_id(
            self.image_sha256, self.layout_task, self.layout_raw_output_sha256
        )
        if self.spotting_visual_output_id != expected_spotting_id:
            raise ValueError("spotting visual-output ID is not content-derived")
        if self.layout_visual_output_id != expected_layout_id:
            raise ValueError("layout visual-output ID is not content-derived")
        return self


@dataclass(frozen=True, slots=True)
class SpottingLine:
    """One strict official spotting item, before layout-order alignment."""

    text: str
    normalized_box: tuple[float, float, float, float]
    page_points: tuple[tuple[float, float], ...]
    raw_item: dict[str, Any]


@dataclass(frozen=True, slots=True)
class HunyuanOutputAssessment:
    status: Literal["accepted", "abstained_invalid", "abstained_disagreement"]
    review_required: bool
    reason: str
    spotting_lines: tuple[SpottingLine, ...]
    spotting_comparison_text: str
    layout_comparison_text: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_visual_output_id(
    image_sha256: str, task: str, raw_output_sha256: str
) -> UUID:
    """Derive a retry-stable identifier from immutable input/task/output bytes."""
    return uuid5(
        NAMESPACE_URL,
        f"wic-hunyuanocr-1.5:{image_sha256}:{task}:{raw_output_sha256}",
    )


def _number(value: Any, *, item_index: int, coordinate_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HunyuanOutputError(
            f"spotting item {item_index} coordinate {coordinate_index} is not numeric"
        )
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1000:
        raise HunyuanOutputError(
            f"spotting item {item_index} coordinate {coordinate_index} is outside [0, 1000]"
        )
    return number


def parse_spotting_json(raw_output: str, width: int, height: int) -> tuple[SpottingLine, ...]:
    """Parse the official JSON format without fences, extraction, or repair."""
    if width <= 0 or height <= 0:
        raise ValueError("page dimensions must be positive")
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise HunyuanOutputError("spotting_json output is not direct valid JSON") from exc
    if not isinstance(payload, list):
        raise HunyuanOutputError("spotting_json output must be one JSON array")
    lines: list[SpottingLine] = []
    for item_index, item in enumerate(payload):
        if not isinstance(item, dict) or "box" not in item or "text" not in item:
            raise HunyuanOutputError(
                f"spotting item {item_index} must contain box and text"
            )
        box = item["box"]
        text = item["text"]
        if not isinstance(box, list) or len(box) != 4:
            raise HunyuanOutputError(f"spotting item {item_index} box must have four values")
        if not isinstance(text, str) or not text.strip():
            raise HunyuanOutputError(f"spotting item {item_index} text must be nonblank")
        left, top, right, bottom = tuple(
            _number(value, item_index=item_index, coordinate_index=coordinate_index)
            for coordinate_index, value in enumerate(box)
        )
        if right <= left or bottom <= top:
            raise HunyuanOutputError(
                f"spotting item {item_index} box must have positive area"
            )
        page_left = left * width / 1000.0
        page_top = top * height / 1000.0
        page_right = right * width / 1000.0
        page_bottom = bottom * height / 1000.0
        lines.append(
            SpottingLine(
                text=text,
                normalized_box=(left, top, right, bottom),
                page_points=(
                    (page_left, page_top),
                    (page_right, page_top),
                    (page_right, page_bottom),
                    (page_left, page_bottom),
                ),
                raw_item=dict(item),
            )
        )
    return tuple(lines)


_TRAILING_COORDINATES = re.compile(
    r"(?:\s*\(?\d{1,4}\s*,\s*\d{1,4}\)?\s*,\s*"
    r"\(?\d{1,4}\s*,\s*\d{1,4}\)?\s*)+$"
)
_HTML_TAG = re.compile(r"<[^>]+>")


def comparison_text(value: str) -> str:
    """Derive a comparison-only character sequence; never publish this as OCR."""
    lines = [_TRAILING_COORDINATES.sub("", line) for line in value.splitlines()]
    without_markup = _HTML_TAG.sub("", "\n".join(lines))
    normalized = unicodedata.normalize("NFC", without_markup)
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] not in {"C", "P", "S", "Z"}
    )


def _order_spotting_from_layout(
    lines: tuple[SpottingLine, ...], layout_text: str
) -> tuple[SpottingLine, ...]:
    """Map boxes to the unique text sequence proposed by ``layout_parse``.

    This is exact cross-output validation, not geometric reading-order logic.
    Repeated or partially unmatched text is intentionally ambiguous and must be
    reviewed rather than resolved with a heuristic.
    """
    positioned: list[tuple[int, int, SpottingLine]] = []
    for index, line in enumerate(lines):
        token = comparison_text(line.text)
        if not token:
            raise HunyuanOutputError(
                f"spotting item {index} has no comparable characters"
            )
        start = layout_text.find(token)
        if start < 0:
            raise HunyuanOutputError(
                f"spotting item {index} is absent from layout_parse text"
            )
        if layout_text.find(token, start + 1) >= 0:
            raise HunyuanOutputError(
                f"spotting item {index} is not uniquely locatable in layout_parse text"
            )
        positioned.append((start, start + len(token), line))

    positioned.sort(key=lambda item: item[0])
    cursor = 0
    for start, end, _line in positioned:
        if start != cursor:
            raise HunyuanOutputError(
                "spotting lines do not exactly and uniquely cover layout_parse order"
            )
        cursor = end
    if cursor != len(layout_text):
        raise HunyuanOutputError(
            "spotting lines do not exactly and uniquely cover layout_parse order"
        )
    return tuple(item[2] for item in positioned)


def assess_hunyuan_outputs(
    bundle: HunyuanTaskBundle,
    *,
    width: int,
    height: int,
) -> HunyuanOutputAssessment:
    """Require valid spotting and text agreement between both official tasks."""
    try:
        lines = parse_spotting_json(bundle.spotting_raw_output, width, height)
    except HunyuanOutputError as exc:
        return HunyuanOutputAssessment(
            "abstained_invalid", True, str(exc), (), "", ""
        )
    if not lines:
        if bundle.layout_raw_output.strip():
            return HunyuanOutputAssessment(
                "abstained_disagreement",
                True,
                "spotting_json reported no text but layout_parse returned content",
                (),
                "",
                comparison_text(bundle.layout_raw_output),
            )
        return HunyuanOutputAssessment(
            "accepted", False, "both official tasks reported an empty page", (), "", ""
        )
    if not bundle.layout_raw_output.strip():
        return HunyuanOutputAssessment(
            "abstained_invalid",
            True,
            "layout_parse output is blank",
            lines,
            comparison_text("\n".join(item.text for item in lines)),
            "",
        )
    spotting_text = comparison_text("\n".join(item.text for item in lines))
    layout_text = comparison_text(bundle.layout_raw_output)
    if not layout_text:
        return HunyuanOutputAssessment(
            "abstained_invalid",
            True,
            "layout_parse contains no comparable text",
            lines,
            spotting_text,
            layout_text,
        )
    try:
        ordered_lines = _order_spotting_from_layout(lines, layout_text)
    except HunyuanOutputError as exc:
        return HunyuanOutputAssessment(
            "abstained_disagreement",
            True,
            str(exc),
            lines,
            spotting_text,
            layout_text,
        )
    return HunyuanOutputAssessment(
        "accepted",
        False,
        "layout_parse supplies an exact unique order for every spotting line",
        ordered_lines,
        comparison_text("\n".join(item.text for item in ordered_lines)),
        layout_text,
    )


def visual_model_output_records(
    bundle: HunyuanTaskBundle,
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Expose stable migration-021-shaped records for worker persistence.

    Artifact URI/SHA and archive occurrence IDs are assigned by the worker
    after it atomically writes the containing artifact. Exact task responses,
    response hashes, stable output IDs, and lossless structured parses are
    already explicit here rather than hidden in an opaque engine payload.
    """
    try:
        lines = parse_spotting_json(bundle.spotting_raw_output, width, height)
    except HunyuanOutputError as exc:
        spotting_structured: dict[str, Any] = {
            "task": bundle.spotting_task,
            "parse_status": "invalid",
            "parse_error": str(exc),
            "items": [],
        }
    else:
        spotting_structured = {
            "task": bundle.spotting_task,
            "parse_status": "valid",
            "coordinate_space": "source_page_pixels",
            "items": [
                {
                    "raw_item": line.raw_item,
                    "normalized_box_0_1000": list(line.normalized_box),
                    "page_points": [list(point) for point in line.page_points],
                }
                for line in lines
            ],
        }
    return [
        {
            "visual_output_id": str(bundle.spotting_visual_output_id),
            "output_kind": "spotting",
            "task": bundle.spotting_task,
            "prompt": bundle.spotting_prompt,
            "raw_output": bundle.spotting_raw_output,
            "raw_output_sha256": bundle.spotting_raw_output_sha256,
            "structured_output": spotting_structured,
            "confidence": None,
            "confidence_status": "not_reported",
            "calibration_id": None,
        },
        {
            "visual_output_id": str(bundle.layout_visual_output_id),
            "output_kind": "layout",
            "task": bundle.layout_task,
            "prompt": bundle.layout_prompt,
            "raw_output": bundle.layout_raw_output,
            "raw_output_sha256": bundle.layout_raw_output_sha256,
            "structured_output": {
                "task": bundle.layout_task,
                "parse_status": "retained_as_model_markdown",
                "format": "markdown_with_html_tables_and_latex",
                "comparison_only_text": comparison_text(bundle.layout_raw_output),
            },
            "confidence": None,
            "confidence_status": "not_reported",
            "calibration_id": None,
        },
    ]


class HunyuanTransformersRunner:
    """Pinned official Transformers/CUDA path; it has no backend fallback."""

    def __init__(self, selected: HunyuanOCRModel):
        try:
            import torch
            import transformers
            from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
        except (ImportError, AttributeError) as exc:  # pragma: no cover - CUDA env
            raise RuntimeError(
                "Use the dedicated HunyuanOCR CUDA environment with the pinned "
                "Transformers runtime"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The authoritative HunyuanOCR runtime requires CUDA; no CPU, MPS, "
                "Paddle, or other fallback is allowed"
            )
        self.selected = selected
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.processor = AutoProcessor.from_pretrained(
            selected.model_name,
            revision=selected.model_revision,
            use_fast=False,
        )
        dtype = getattr(torch, selected.dtype)
        self.model = HunYuanVLForConditionalGeneration.from_pretrained(
            selected.model_name,
            revision=selected.model_revision,
            dtype=dtype,
            attn_implementation="eager",
        ).to(selected.device)
        self.model.eval()

    def __call__(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "immutable-page"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[rendered], images=image, padding=True, return_tensors="pt"
        ).to(self.selected.device)
        input_ids = inputs["input_ids"] if "input_ids" in inputs else inputs["inputs"]
        generation: dict[str, Any] = {
            "max_new_tokens": self.selected.max_new_tokens,
            "do_sample": False,
            "repetition_penalty": self.selected.repetition_penalty,
            "use_cache": True,
        }
        tokenizer = self.processor.tokenizer
        if tokenizer.eos_token_id is not None:
            generation["eos_token_id"] = tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            generation["pad_token_id"] = tokenizer.pad_token_id
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation)
        trimmed = [
            output[len(source) :]
            for source, output in zip(input_ids, generated, strict=True)
        ]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""


def run_hunyuan_tasks(
    image: Image.Image,
    selected: HunyuanOCRModel,
    runner: Callable[[Image.Image, str], str] | None = None,
    *,
    image_sha256: str,
) -> HunyuanTaskBundle:
    """Run exactly the two pinned tasks; never recover through another model."""
    adapter = runner or HunyuanTransformersRunner(selected)
    spotting = adapter(image, selected.spotting_prompt)
    layout = adapter(image, selected.layout_prompt)
    return HunyuanTaskBundle(
        image_sha256=image_sha256,
        spotting_raw_output=spotting,
        spotting_raw_output_sha256=_sha256(spotting),
        spotting_visual_output_id=stable_visual_output_id(
            image_sha256, selected.spotting_task, _sha256(spotting)
        ),
        layout_raw_output=layout,
        layout_raw_output_sha256=_sha256(layout),
        layout_visual_output_id=stable_visual_output_id(
            image_sha256, selected.layout_task, _sha256(layout)
        ),
        confidence_status=selected.confidence_status,
        confidence_calibration=selected.confidence_calibration,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        help="Complete model configuration; individual model overrides are forbidden",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configuration = load_pipeline_model_configuration(args.model_config)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    bundle = run_hunyuan_tasks(
        image,
        configuration.ocr,
        image_sha256=sha256_file(args.image),
    )
    assessment = assess_hunyuan_outputs(bundle, width=image.width, height=image.height)
    payload = {
        **bundle.model_dump(mode="json"),
        "visual_model_outputs": visual_model_output_records(
            bundle, width=image.width, height=image.height
        ),
        "assessment": {
            "status": assessment.status,
            "review_required": assessment.review_required,
            "reason": assessment.reason,
        },
        "model_identity": {
            "model_name": configuration.ocr.model_name,
            "model_revision": configuration.ocr.model_revision,
            "toolkit_name": configuration.ocr.toolkit_name,
            "toolkit_revision": configuration.ocr.toolkit_revision,
            "pipeline_model_configuration_sha256": configuration.sha256,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "status": assessment.status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
