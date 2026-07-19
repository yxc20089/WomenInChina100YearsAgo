from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from wic_history.layout_pipeline import create_layout_artifact
from wic_history.model_config import (
    SPOTTING_JSON_PROMPT,
    load_pipeline_model_configuration,
)
from wic_history.ocr_pipeline import (
    build_parser,
    create_layout_aware_ocr_artifact,
    resolve_render_provenance,
)
from wic_history.render_samples import sha256_file


def _layout(tmp_path: Path, spotting: str, layout_text: str):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    configuration = load_pipeline_model_configuration()

    def predictor(_image: Image.Image, prompt: str) -> str:
        return spotting if prompt == SPOTTING_JSON_PROMPT else layout_text

    layout = create_layout_artifact(
        image_path=image_path,
        source_uri="s3://example/volume.pdf",
        page_number=3,
        volume_number=219,
        publication_year=1925,
        source_sha256="a" * 64,
        evidence_tier="unreviewed_input",
        render_manifest_path=None,
        model_configuration=configuration,
        predictor=predictor,
    )
    return image_path, configuration, layout


def _materialize(image_path, configuration, layout):
    return create_layout_aware_ocr_artifact(
        image_path=image_path,
        layout=layout,
        layout_artifact_sha256="b" * 64,
        pipeline_model_configuration_sha256=configuration.sha256,
        model_configuration=configuration,
    )


def _bounds(region) -> tuple[float, float, float, float]:
    xs = [point.x for point in region.polygon.points]
    ys = [point.y for point in region.polygon.points]
    return min(xs), min(ys), max(xs), max(ys)


def test_ocr_materializes_exact_accepted_raw_text_coordinates_and_identity(
    tmp_path: Path,
) -> None:
    raw_text = "女e\u0301學生"
    spotting = json.dumps(
        [{"box": [100, 200, 400, 600], "text": raw_text, "source": "exact"}],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    image_path, configuration, layout = _layout(tmp_path, spotting, raw_text)
    artifact = _materialize(image_path, configuration, layout)

    assert len(artifact.regions) == 1
    region = artifact.regions[0]
    assert region.raw_text == raw_text
    assert region.normalized_text == "女é學生"
    assert _bounds(region) == (20.0, 20.0, 80.0, 60.0)
    assert region.layout_region_id == layout.regions[1].layout_region_id
    assert region.reading_order == 0
    assert region.direction == "unknown"
    assert region.confidence is None
    assert region.engine_payload["spotting_raw_item"]["source"] == "exact"
    assert region.engine_payload["confidence_status"] == "not_emitted_by_model"
    assert artifact.run.engine == "transformers"
    assert artifact.run.model_name == "tencent/HunyuanOCR"
    assert artifact.run.configuration["raw_task_bundle"]["spotting_raw_output"] == spotting
    assert artifact.run.configuration["visual_model_outputs"][0]["raw_output"] == spotting
    assert artifact.run.configuration["materialization_status"] == "accepted"
    assert artifact.run.configuration["reinference_performed"] is False
    assert artifact.run.configuration["tiling_used"] is False
    assert artifact.run.configuration["fallback_allowed"] is False
    assert artifact.run.configuration["confidence_calibration"] == "not_available"


def test_disagreement_produces_review_required_empty_ocr_artifact(tmp_path: Path) -> None:
    spotting = '[{"box":[100,200,400,600],"text":"中央大戲院"}]'
    image_path, configuration, layout = _layout(tmp_path, spotting, "中央大影院")
    artifact = _materialize(image_path, configuration, layout)

    assert artifact.regions == []
    assert artifact.run.configuration["materialization_status"] == "abstained_disagreement"
    assert artifact.run.configuration["review_required"] is True
    assert artifact.run.configuration["raw_task_bundle"]["spotting_raw_output"] == spotting
    assert any("abstained" in warning for warning in artifact.warnings)


def test_tampered_layout_coordinates_abstain_instead_of_recomputing(
    tmp_path: Path,
) -> None:
    spotting = '[{"box":[100,200,400,600],"text":"婦女報"}]'
    image_path, configuration, layout = _layout(tmp_path, spotting, "婦女報")
    layout.regions[1].polygon.points[0].x += 1
    artifact = _materialize(image_path, configuration, layout)

    assert artifact.regions == []
    assert artifact.run.configuration["materialization_status"] == "abstained_invalid_layout"
    assert artifact.run.configuration["review_required"] is True
    assert "coordinates disagree" in artifact.run.configuration["materialization_reason"]


def test_lossless_manifest_supplies_source_hash_and_evidence_tier(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("L", (20, 30), "white").save(image_path)
    manifest_path = tmp_path / "manifest.jsonl"
    source_sha256 = "c" * 64
    manifest_path.write_text(
        json.dumps(
            {
                "status": "rendered",
                "render_path": str(image_path),
                "render_sha256": sha256_file(image_path),
                "source_object_sha256": source_sha256,
                "source_uri": "s3://bucket/volume.pdf",
                "volume_number": 219,
                "publication_year": 1925,
                "page_number": 308,
                "selection": {"gold_status": "non_gold_pilot"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    resolved_hash, tier = resolve_render_provenance(
        image_path,
        manifest_path,
        source_uri="s3://bucket/volume.pdf",
        page_number=308,
        volume_number=219,
        publication_year=1925,
    )

    assert resolved_hash == source_sha256
    assert tier == "non_gold_lossless_pilot"


def test_ocr_cli_requires_layout_and_has_no_paddle_or_model_override_flags() -> None:
    options = {
        option for action in build_parser()._actions for option in action.option_strings
    }

    assert "--layout-artifact" in options
    assert "--model-config" in options
    assert "--language" not in options
    assert "--tile-size" not in options
    assert "--overlap" not in options
    assert "--model" not in options
