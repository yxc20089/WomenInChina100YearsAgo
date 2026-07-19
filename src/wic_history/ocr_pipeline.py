"""Materialize OCR evidence from HunyuanOCR's immutable paired task outputs.

HunyuanOCR-1.5 is the sole learned OCR/layout model. This stage does not run a
second recognizer, tile the page, repair output, estimate confidence, or infer
reading order. It revalidates the exact ``spotting_json`` and ``layout_parse``
outputs embedded by the layout stage and publishes text only when they agree.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .difficult_ocr import assess_hunyuan_outputs, visual_model_output_records
from .evidence import (
    LayoutPageArtifact,
    LayoutRegion,
    LayoutRegionKind,
    OCRPageArtifact,
    OCRRegion,
    Point,
    Polygon,
    ProcessingRun,
    RegionKind,
    RunKind,
)
from .layout_pipeline import task_bundle_from_layout
from .model_config import (
    HunyuanOCRModel,
    PipelineModelConfiguration,
    load_pipeline_model_configuration,
)
from .render_provenance import resolve_render_provenance
from .render_samples import sha256_file


def normalize_ocr_text(text: str) -> str:
    """Preserve historical glyphs; apply Unicode canonical composition only."""
    return unicodedata.normalize("NFC", text).strip()


def _model_identity(selected: HunyuanOCRModel) -> dict[str, object]:
    return {
        "engine": selected.engine,
        "model_name": selected.model_name,
        "model_revision": selected.model_revision,
        "toolkit_name": selected.toolkit_name,
        "toolkit_revision": selected.toolkit_revision,
        "pipeline": selected.pipeline,
        "language": selected.language,
        "runtime": selected.runtime,
        "dtype": selected.dtype,
        "device": selected.device,
        "temperature": selected.temperature,
        "top_p": selected.top_p,
        "max_new_tokens": selected.max_new_tokens,
        "repetition_penalty": selected.repetition_penalty,
    }


def _layout_matches_configuration(
    layout: LayoutPageArtifact,
    selected: HunyuanOCRModel,
    configuration_sha256: str,
) -> None:
    expected_run = (selected.engine, selected.model_name, selected.model_revision)
    actual_run = (layout.run.engine, layout.run.model_name, layout.run.model_revision)
    if actual_run != expected_run:
        raise ValueError("layout artifact was not produced by the configured Hunyuan model")
    for key, expected in _model_identity(selected).items():
        if layout.run.configuration.get(key) != expected:
            raise ValueError(f"layout artifact model identity disagrees at {key}")
    if (
        layout.run.configuration.get("pipeline_model_configuration_sha256")
        != configuration_sha256
    ):
        raise ValueError("layout artifact belongs to a different complete model configuration")
    if layout.run.configuration.get("fallback_allowed") is not False:
        raise ValueError("layout artifact does not prove the no-fallback contract")


def _text_targets(layout: LayoutPageArtifact) -> list[LayoutRegion]:
    targets = [
        region
        for region in layout.regions
        if region.kind == LayoutRegionKind.TEXT_GROUP
    ]
    # This sort only indexes already-declared model order. It never uses page
    # geometry to manufacture a reading sequence.
    return sorted(
        targets,
        key=lambda region: (
            region.reading_order is None,
            region.reading_order if region.reading_order is not None else 0,
        ),
    )


def _same_polygon(
    region: LayoutRegion,
    expected: tuple[tuple[float, float], ...],
) -> bool:
    actual = tuple((point.x, point.y) for point in region.polygon.points)
    return len(actual) == len(expected) and all(
        abs(actual_x - expected_x) <= 1e-9 and abs(actual_y - expected_y) <= 1e-9
        for (actual_x, actual_y), (expected_x, expected_y) in zip(
            actual, expected, strict=True
        )
    )


def _materialization_targets_are_valid(
    targets: list[LayoutRegion],
    lines: tuple[Any, ...],
) -> tuple[bool, str]:
    if len(targets) != len(lines):
        return False, "layout text-region count disagrees with the immutable spotting output"
    for index, (target, line) in enumerate(zip(targets, lines, strict=True), 1):
        if target.reading_order != index:
            return False, "layout text regions do not preserve model-supplied array order"
        if not _same_polygon(target, line.page_points):
            return False, "layout text-region coordinates disagree with spotting output"
        if target.engine_payload.get("spotting_raw_item") != line.raw_item:
            return False, "layout text-region raw payload disagrees with spotting output"
    return True, "layout regions exactly materialize the immutable spotting output"


def create_layout_aware_ocr_artifact(
    *,
    image_path: Path,
    layout: LayoutPageArtifact,
    layout_artifact_sha256: str,
    pipeline_model_configuration_sha256: str,
    model_configuration: PipelineModelConfiguration | None = None,
) -> OCRPageArtifact:
    """Publish paired Hunyuan text or an explicit review-required abstention."""
    started_at = datetime.now(timezone.utc)
    configuration = model_configuration or load_pipeline_model_configuration()
    if pipeline_model_configuration_sha256 != configuration.sha256:
        raise ValueError("supplied model-configuration hash does not match its bytes")
    _layout_matches_configuration(layout, configuration.ocr, configuration.sha256)

    image_sha256 = sha256_file(image_path)
    with Image.open(image_path) as source_image:
        width, height = source_image.size
        dpi_value = source_image.info.get("dpi", (None, None))[0]
    if image_sha256 != layout.image_sha256:
        raise ValueError("layout artifact and OCR image bytes differ")
    if (width, height) != (layout.width, layout.height):
        raise ValueError("layout artifact and OCR image dimensions differ")

    bundle = task_bundle_from_layout(layout)
    assessment = assess_hunyuan_outputs(bundle, width=width, height=height)
    layout_status = layout.run.configuration.get("assessment_status")
    if layout_status != assessment.status:
        raise ValueError("layout assessment status disagrees with its immutable raw outputs")

    targets = _text_targets(layout)
    materialization_valid, materialization_reason = _materialization_targets_are_valid(
        targets, assessment.spotting_lines
    )
    publish = assessment.status == "accepted" and materialization_valid
    materialization_status = (
        "accepted" if publish else (
            "abstained_invalid_layout" if assessment.status == "accepted" else assessment.status
        )
    )
    selected = configuration.ocr
    regions = []
    if publish:
        regions = [
            OCRRegion(
                layout_region_id=target.layout_region_id,
                kind=RegionKind.TEXT,
                polygon=Polygon(
                    points=[Point(x=x, y=y) for x, y in line.page_points]
                ),
                reading_order=index,
                raw_text=line.text,
                normalized_text=normalize_ocr_text(line.text),
                confidence=None,
                language=selected.language,
                direction="unknown",
                engine_payload={
                    "spotting_raw_item": line.raw_item,
                    "source_normalized_box_0_1000": list(line.normalized_box),
                    "spotting_output_sha256": bundle.spotting_raw_output_sha256,
                    "layout_parse_output_sha256": bundle.layout_raw_output_sha256,
                    "confidence_status": selected.confidence_status,
                    "confidence_calibration": selected.confidence_calibration,
                    "publication_status": "candidate",
                },
            )
            for index, (target, line) in enumerate(
                zip(targets, assessment.spotting_lines, strict=True)
            )
        ]

    review_required = assessment.review_required or not materialization_valid
    run_configuration = {
        **_model_identity(selected),
        "role": "materialize_accepted_hunyuan_spotting_json",
        "official_tasks": [selected.spotting_task, selected.layout_task],
        "official_prompts": {
            selected.spotting_task: selected.spotting_prompt,
            selected.layout_task: selected.layout_prompt,
        },
        "raw_task_bundle": bundle.model_dump(mode="json"),
        "visual_model_outputs": visual_model_output_records(
            bundle, width=width, height=height
        ),
        "paired_assessment_status": assessment.status,
        "paired_assessment_reason": assessment.reason,
        "materialization_status": materialization_status,
        "materialization_reason": materialization_reason,
        "review_required": review_required,
        "fallback_allowed": False,
        "reinference_performed": False,
        "tiling_used": False,
        "output_repair_used": False,
        "deterministic_reading_order_used": False,
        "coordinate_space": "source_page_pixels",
        "confidence_status": selected.confidence_status,
        "confidence_calibration": selected.confidence_calibration,
        "layout_run_id": str(layout.run.run_id),
        "layout_artifact_sha256": layout_artifact_sha256,
        "pipeline_model_configuration_sha256": configuration.sha256,
        "evidence_tier": layout.source.evidence_tier or "unreviewed_input",
        "render_manifest": layout.run.configuration.get("render_manifest"),
    }
    warnings = [
        "OCR text and page coordinates come from the immutable Hunyuan "
        "spotting_json output; numeric model confidence is unavailable and "
        "uncalibrated."
    ]
    if not publish:
        warnings.append(
            "Hunyuan evidence did not satisfy the publication gate; OCR "
            "abstained and requires review."
        )
    return OCRPageArtifact(
        source=layout.source,
        image_uri=str(image_path),
        image_sha256=image_sha256,
        width=width,
        height=height,
        dpi=int(round(dpi_value)) if dpi_value else layout.dpi,
        run=ProcessingRun(
            kind=RunKind.OCR,
            engine=selected.engine,
            model_name=selected.model_name,
            model_revision=selected.model_revision,
            software_version=f"toolkit-{selected.toolkit_revision}",
            configuration=run_configuration,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        ),
        regions=regions,
        warnings=warnings,
    )


def create_ocr_artifact(*args: object, **kwargs: object) -> OCRPageArtifact:
    """Reject the removed independent/Paddle OCR execution path."""
    del args, kwargs
    raise RuntimeError(
        "independent OCR is disabled; provide a paired Hunyuan layout artifact"
    )


class PaddleOCRPredictor:
    """Import-compatible tombstone for removed workers; it cannot run."""

    def __init__(self, *args: object, **kwargs: object):
        del args, kwargs
        raise RuntimeError("PaddleOCR is not an allowed pipeline fallback")


def sha256_argument(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--layout-artifact", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        help="Complete model configuration; individual model overrides are forbidden",
    )
    parser.add_argument("--model-config-sha256", type=sha256_argument)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configuration = load_pipeline_model_configuration(args.model_config)
    if (
        args.model_config_sha256 is not None
        and args.model_config_sha256 != configuration.sha256
    ):
        raise SystemExit("--model-config-sha256 does not match the complete configuration")
    layout = LayoutPageArtifact.model_validate_json(
        args.layout_artifact.read_text(encoding="utf-8")
    )
    artifact = create_layout_aware_ocr_artifact(
        image_path=args.image,
        layout=layout,
        layout_artifact_sha256=sha256_file(args.layout_artifact),
        pipeline_model_configuration_sha256=configuration.sha256,
        model_configuration=configuration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "regions": len(artifact.regions),
                "materialization_status": artifact.run.configuration[
                    "materialization_status"
                ],
                "review_required": artifact.run.configuration["review_required"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
