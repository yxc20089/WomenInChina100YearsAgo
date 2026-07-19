from __future__ import annotations

from pathlib import Path

from PIL import Image

from wic_history.evidence import LayoutRegionKind
from wic_history.layout_pipeline import build_parser, create_layout_artifact
from wic_history.model_config import (
    LAYOUT_PARSE_PROMPT,
    SPOTTING_JSON_PROMPT,
    load_pipeline_model_configuration,
)


def _artifact(tmp_path: Path, spotting: str, layout_text: str):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    configuration = load_pipeline_model_configuration()
    observed: list[str] = []

    def predictor(_image: Image.Image, prompt: str) -> str:
        observed.append(prompt)
        return spotting if prompt == SPOTTING_JSON_PROMPT else layout_text

    artifact = create_layout_artifact(
        image_path=image_path,
        source_uri="s3://example/volume.pdf",
        page_number=3,
        volume_number=219,
        publication_year=1925,
        source_sha256="a" * 64,
        evidence_tier="unreviewed_input",
        render_manifest_path="artifacts/render-manifest.jsonl",
        model_configuration=configuration,
        predictor=predictor,
    )
    return image_path, configuration, artifact, observed


def _bounds(region) -> tuple[float, float, float, float]:
    xs = [point.x for point in region.polygon.points]
    ys = [point.y for point in region.polygon.points]
    return min(xs), min(ys), max(xs), max(ys)


def test_layout_uses_layout_parse_order_and_exact_source_page_coordinates(
    tmp_path: Path,
) -> None:
    spotting = (
        '[{"box":[100,700,300,900],"text":"第一","raw":"kept"},'
        '{"box":[700,100,900,300],"text":"第二"}]'
    )
    _, configuration, artifact, observed = _artifact(
        tmp_path, spotting, "第二\n第一"
    )

    assert observed == [SPOTTING_JSON_PROMPT, LAYOUT_PARSE_PROMPT]
    assert artifact.run.engine == "transformers"
    assert artifact.run.model_name == "tencent/HunyuanOCR"
    assert artifact.run.model_revision == configuration.ocr.model_revision
    assert artifact.run.configuration["assessment_status"] == "accepted"
    assert artifact.run.configuration["fallback_allowed"] is False
    assert artifact.run.configuration["deterministic_reading_order_used"] is False
    assert artifact.run.configuration["raw_task_bundle"]["spotting_raw_output"] == spotting
    visual_outputs = artifact.run.configuration["visual_model_outputs"]
    assert [item["output_kind"] for item in visual_outputs] == ["spotting", "layout"]
    assert visual_outputs[0]["raw_output"] == spotting
    assert len(visual_outputs[0]["visual_output_id"]) == 36
    text_regions = [
        region for region in artifact.regions if region.kind == LayoutRegionKind.TEXT_GROUP
    ]
    assert [_bounds(region) for region in text_regions] == [
        (140.0, 10.0, 180.0, 30.0),
        (20.0, 70.0, 60.0, 90.0),
    ]
    assert [region.reading_order for region in text_regions] == [1, 2]
    assert all(region.direction == "unknown" for region in text_regions)
    assert all(region.confidence is None for region in artifact.regions)
    assert text_regions[1].engine_payload["spotting_raw_item"]["raw"] == "kept"
    assert text_regions[0].engine_payload["confidence_status"] == "not_emitted_by_model"


def test_disagreement_retains_raw_candidates_but_publishes_only_page_root(
    tmp_path: Path,
) -> None:
    spotting = '[{"box":[100,100,300,300],"text":"中央大戲院"}]'
    _, _, artifact, _ = _artifact(tmp_path, spotting, "中央大影院")

    assert artifact.run.configuration["assessment_status"] == "abstained_disagreement"
    assert artifact.run.configuration["review_required"] is True
    assert len(artifact.regions) == 1
    root = artifact.regions[0]
    assert root.kind == LayoutRegionKind.PAGE
    assert root.reading_order is None
    assert (
        root.engine_payload["parsed_spotting_candidates"][0]["raw_item"]["text"]
        == "中央大戲院"
    )
    assert root.engine_payload["raw_task_bundle"]["layout_raw_output"] == "中央大影院"
    assert any("abstained" in warning for warning in artifact.warnings)


def test_invalid_spotting_never_falls_back_to_layout_parse(tmp_path: Path) -> None:
    _, _, artifact, _ = _artifact(
        tmp_path,
        '```json\n[{"box":[1,2,3,4],"text":"婦女"}]\n```',
        "婦女",
    )

    assert artifact.run.configuration["assessment_status"] == "abstained_invalid"
    assert artifact.run.configuration["fallback_allowed"] is False
    assert len(artifact.regions) == 1


def test_layout_cli_has_only_complete_model_configuration_override() -> None:
    options = {
        option for action in build_parser()._actions for option in action.option_strings
    }

    assert "--model-config" in options
    assert "--render-manifest" in options
    assert "--evidence-tier" not in options
    assert "--model" not in options
    assert "--revision" not in options
    assert "--engine" not in options
