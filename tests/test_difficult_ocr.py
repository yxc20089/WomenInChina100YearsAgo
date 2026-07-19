from __future__ import annotations

import hashlib

import pytest
from PIL import Image

from wic_history.difficult_ocr import (
    HunyuanOutputError,
    HunyuanTaskBundle,
    assess_hunyuan_outputs,
    build_parser,
    parse_spotting_json,
    run_hunyuan_tasks,
    stable_visual_output_id,
    visual_model_output_records,
)
from wic_history.model_config import (
    LAYOUT_PARSE_PROMPT,
    SPOTTING_JSON_PROMPT,
    load_pipeline_model_configuration,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bundle(spotting: str, layout: str) -> HunyuanTaskBundle:
    return HunyuanTaskBundle(
        image_sha256="a" * 64,
        spotting_raw_output=spotting,
        spotting_raw_output_sha256=_hash(spotting),
        spotting_visual_output_id=stable_visual_output_id(
            "a" * 64, "spotting_json", _hash(spotting)
        ),
        layout_raw_output=layout,
        layout_raw_output_sha256=_hash(layout),
        layout_visual_output_id=stable_visual_output_id(
            "a" * 64, "layout_parse", _hash(layout)
        ),
    )


def test_spotting_json_preserves_raw_item_and_maps_to_source_page_coordinates() -> None:
    raw = '[{"box":[100,200,400,600],"text":"中央大戲院","note":"raw"}]'
    lines = parse_spotting_json(raw, width=2000, height=1000)

    assert len(lines) == 1
    assert lines[0].text == "中央大戲院"
    assert lines[0].normalized_box == (100.0, 200.0, 400.0, 600.0)
    assert lines[0].page_points == (
        (200.0, 200.0),
        (800.0, 200.0),
        (800.0, 600.0),
        (200.0, 600.0),
    )
    assert lines[0].raw_item == {
        "box": [100, 200, 400, 600],
        "text": "中央大戲院",
        "note": "raw",
    }


def test_spotting_json_rejects_fences_or_repair() -> None:
    with pytest.raises(HunyuanOutputError, match="direct valid JSON"):
        parse_spotting_json('```json\n[{"box":[1,2,3,4],"text":"女"}]\n```', 100, 100)


def test_paired_tasks_accept_only_agreeing_character_sequences() -> None:
    spotting = (
        '[{"box":[1,2,100,50],"text":"中央大戲院"},'
        '{"box":[2,60,100,100],"text":"婦女報"}]'
    )
    accepted = assess_hunyuan_outputs(
        _bundle(spotting, "# 中央大戲院\n婦女報"), width=1000, height=1000
    )
    disagreed = assess_hunyuan_outputs(
        _bundle(spotting, "中央大戲院\n婦人報"), width=1000, height=1000
    )

    assert accepted.status == "accepted"
    assert accepted.review_required is False
    assert disagreed.status == "abstained_disagreement"
    assert disagreed.review_required is True
    assert disagreed.spotting_lines[1].text == "婦女報"


def test_layout_parse_order_reorders_spotting_boxes_without_geometry_heuristics() -> None:
    spotting = (
        '[{"box":[1,2,100,50],"text":"左欄"},'
        '{"box":[800,2,999,50],"text":"右欄"}]'
    )

    assessment = assess_hunyuan_outputs(
        _bundle(spotting, "右欄\n左欄"), width=1000, height=1000
    )

    assert assessment.status == "accepted"
    assert [line.text for line in assessment.spotting_lines] == ["右欄", "左欄"]


def test_repeated_layout_text_abstains_instead_of_guessing_box_order() -> None:
    spotting = (
        '[{"box":[1,2,100,50],"text":"廣告"},'
        '{"box":[800,2,999,50],"text":"廣告"}]'
    )

    assessment = assess_hunyuan_outputs(
        _bundle(spotting, "廣告\n廣告"), width=1000, height=1000
    )

    assert assessment.status == "abstained_disagreement"
    assert "not uniquely locatable" in assessment.reason


def test_invalid_spotting_abstains_without_using_layout_as_fallback() -> None:
    assessment = assess_hunyuan_outputs(
        _bundle("not json", "中央大戲院"), width=100, height=100
    )

    assert assessment.status == "abstained_invalid"
    assert assessment.spotting_lines == ()
    assert assessment.review_required is True


def test_runner_invokes_exact_pinned_tasks_and_keeps_raw_outputs() -> None:
    selected = load_pipeline_model_configuration().ocr
    observed: list[str] = []

    def runner(_image: Image.Image, prompt: str) -> str:
        observed.append(prompt)
        return (
            '[{"box":[1,2,3,4],"text":"女學校"}]'
            if prompt == SPOTTING_JSON_PROMPT
            else "女學校"
        )

    bundle = run_hunyuan_tasks(
        Image.new("RGB", (10, 10)), selected, runner, image_sha256="b" * 64
    )

    assert observed == [SPOTTING_JSON_PROMPT, LAYOUT_PARSE_PROMPT]
    assert bundle.spotting_raw_output == '[{"box":[1,2,3,4],"text":"女學校"}]'
    assert bundle.layout_raw_output == "女學校"
    assert bundle.confidence_status == "not_emitted_by_model"
    assert bundle.confidence_calibration == "not_available"
    records = visual_model_output_records(bundle, width=10, height=10)
    assert [record["output_kind"] for record in records] == ["spotting", "layout"]
    assert records[0]["visual_output_id"] == str(bundle.spotting_visual_output_id)
    assert records[0]["raw_output"] == bundle.spotting_raw_output
    assert records[0]["raw_output_sha256"] == bundle.spotting_raw_output_sha256
    assert records[0]["structured_output"]["parse_status"] == "valid"
    assert records[0]["confidence_status"] == "not_reported"


def test_hunyuan_cli_has_no_independent_model_or_fallback_flags() -> None:
    options = {
        option for action in build_parser()._actions for option in action.option_strings
    }

    assert "--model-config" in options
    assert "--model" not in options
    assert "--revision" not in options
    assert "--fallback" not in options
