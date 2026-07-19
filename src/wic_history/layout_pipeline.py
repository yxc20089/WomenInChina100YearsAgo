"""HunyuanOCR-1.5 paired-task layout proposals with no fallback authority."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image

from .difficult_ocr import (
    HunyuanTaskBundle,
    assess_hunyuan_outputs,
    run_hunyuan_tasks,
    visual_model_output_records,
)
from .evidence import (
    LayoutPageArtifact,
    LayoutRegion,
    LayoutRegionKind,
    Point,
    Polygon,
    ProcessingRun,
    RunKind,
    SourcePointer,
)
from .model_config import (
    HunyuanOCRModel,
    PipelineModelConfiguration,
    load_pipeline_model_configuration,
)
from .render_provenance import resolve_render_provenance
from .render_samples import sha256_file


TaskRunner = Callable[[Image.Image, str], str]


def _page_polygon(width: int, height: int) -> Polygon:
    return Polygon(
        points=[
            Point(x=0, y=0),
            Point(x=width, y=0),
            Point(x=width, y=height),
            Point(x=0, y=height),
        ]
    )


def _line_polygon(points: tuple[tuple[float, float], ...]) -> Polygon:
    return Polygon(points=[Point(x=x, y=y) for x, y in points])


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


def create_layout_artifact(
    *,
    image_path: Path,
    source_uri: str,
    page_number: int,
    volume_number: int | None,
    publication_year: int | None,
    source_sha256: str | None,
    evidence_tier: str,
    render_manifest_path: str | None,
    model_configuration: PipelineModelConfiguration,
    predictor: TaskRunner | None = None,
) -> LayoutPageArtifact:
    """Run both official tasks and publish boxes only when their texts agree."""
    started_at = datetime.now(timezone.utc)
    image_sha256 = sha256_file(image_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        dpi_value = source.info.get("dpi", (None, None))[0]
    selected = model_configuration.ocr
    bundle = run_hunyuan_tasks(
        image,
        selected,
        predictor,
        image_sha256=image_sha256,
    )
    assessment = assess_hunyuan_outputs(bundle, width=image.width, height=image.height)
    root = LayoutRegion(
        kind=LayoutRegionKind.PAGE,
        polygon=_page_polygon(image.width, image.height),
        reading_order=0 if assessment.status == "accepted" else None,
        direction="unknown",
        source_method="hunyuanocr-1.5:spotting_json+layout_parse",
        confidence=None,
        boundary_evidence={
            "coordinate_space": "source_page_pixels",
            "geometry_source": "immutable_page_dimensions",
        },
        engine_payload={
            "raw_task_bundle": bundle.model_dump(mode="json"),
            "assessment_status": assessment.status,
            "review_required": assessment.review_required,
            "assessment_reason": assessment.reason,
            "parsed_spotting_candidates": [
                {
                    "raw_item": line.raw_item,
                    "normalized_box": list(line.normalized_box),
                    "page_points": [list(point) for point in line.page_points],
                }
                for line in assessment.spotting_lines
            ],
            "confidence_status": selected.confidence_status,
            "confidence_calibration": selected.confidence_calibration,
        },
    )
    regions = [root]
    if assessment.status == "accepted":
        regions.extend(
            LayoutRegion(
                parent_layout_region_id=root.layout_region_id,
                kind=LayoutRegionKind.TEXT_GROUP,
                polygon=_line_polygon(line.page_points),
                reading_order=index,
                direction="unknown",
                source_method="hunyuanocr-1.5:spotting_json+layout_parse",
                confidence=None,
                boundary_evidence={
                    "coordinate_space": "source_page_pixels",
                    "source_normalized_box_0_1000": list(line.normalized_box),
                },
                engine_payload={
                    "spotting_raw_item": line.raw_item,
                    "spotting_output_sha256": bundle.spotting_raw_output_sha256,
                    "layout_parse_output_sha256": bundle.layout_raw_output_sha256,
                    "confidence_status": selected.confidence_status,
                    "confidence_calibration": selected.confidence_calibration,
                    "publication_status": "candidate",
                },
            )
            for index, line in enumerate(assessment.spotting_lines, 1)
        )
    run_configuration = {
        **_model_identity(selected),
        "role": "sole_layout_and_ocr_model",
        "official_tasks": [selected.spotting_task, selected.layout_task],
        "official_prompts": {
            selected.spotting_task: selected.spotting_prompt,
            selected.layout_task: selected.layout_prompt,
        },
        "raw_task_bundle": bundle.model_dump(mode="json"),
        "visual_model_outputs": visual_model_output_records(
            bundle, width=image.width, height=image.height
        ),
        "assessment_status": assessment.status,
        "review_required": assessment.review_required,
        "assessment_reason": assessment.reason,
        "fallback_allowed": False,
        "deterministic_reading_order_used": False,
        "coordinate_space": "source_page_pixels",
        "confidence_status": selected.confidence_status,
        "confidence_calibration": selected.confidence_calibration,
        "evidence_tier": evidence_tier,
        "render_manifest": render_manifest_path,
        "pipeline_model_configuration_sha256": model_configuration.sha256,
    }
    warnings = [
        "HunyuanOCR layout is a machine proposal; model confidence is unavailable and uncalibrated."
    ]
    if assessment.status != "accepted":
        warnings.append(
            "Paired Hunyuan outputs did not satisfy the publication gate; "
            "layout abstained and requires review."
        )
    return LayoutPageArtifact(
        source=SourcePointer(
            source_uri=source_uri,
            source_sha256=source_sha256,
            evidence_tier=evidence_tier,
            volume_number=volume_number,
            publication_year=publication_year,
            page_number=page_number,
        ),
        image_uri=str(image_path),
        image_sha256=image_sha256,
        width=image.width,
        height=image.height,
        dpi=int(round(dpi_value)) if dpi_value else None,
        run=ProcessingRun(
            kind=RunKind.LAYOUT,
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


def task_bundle_from_layout(layout: LayoutPageArtifact) -> HunyuanTaskBundle:
    """Recover and revalidate the immutable raw outputs embedded in layout."""
    raw = layout.run.configuration.get("raw_task_bundle")
    if not isinstance(raw, dict):
        raise ValueError("layout artifact lacks its immutable Hunyuan task bundle")
    bundle = HunyuanTaskBundle.model_validate(raw)
    if bundle.image_sha256 != layout.image_sha256:
        raise ValueError("layout task bundle belongs to different image bytes")
    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-sha256")
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--volume", type=int)
    parser.add_argument("--year", type=int)
    parser.add_argument("--model-config")
    parser.add_argument("--model-config-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configuration = load_pipeline_model_configuration(args.model_config)
    if (
        args.model_config_sha256 is not None
        and args.model_config_sha256 != configuration.sha256
    ):
        raise SystemExit("--model-config-sha256 does not match the complete model configuration")
    source_sha256, evidence_tier = resolve_render_provenance(
        args.image,
        args.render_manifest,
        source_uri=args.source_uri,
        page_number=args.page,
        volume_number=args.volume,
        publication_year=args.year,
        supplied_source_sha256=args.source_sha256,
        artifact_root=Path.cwd(),
    )
    artifact = create_layout_artifact(
        image_path=args.image,
        source_uri=args.source_uri,
        page_number=args.page,
        volume_number=args.volume,
        publication_year=args.year,
        source_sha256=source_sha256,
        evidence_tier=evidence_tier,
        render_manifest_path=str(args.render_manifest),
        model_configuration=configuration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "regions": len(artifact.regions),
                "assessment_status": artifact.run.configuration["assessment_status"],
                "review_required": artifact.run.configuration["review_required"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
