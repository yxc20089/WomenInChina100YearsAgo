from __future__ import annotations

import hashlib
from uuid import UUID
from unittest.mock import patch

from fastapi.testclient import TestClient

from wic_history.api import create_app
from wic_history.generation import TextCompletion
from wic_history.result_explanation import (
    ExplanationAuthority,
    ExplanationStatus,
    ExplanationTarget,
    _evidence,
    explain_result,
    prepare_explanation_messages,
)
from wic_history.semantic_inputs import CoherentTextBundle, CoherentTextSegment


REVISION_ID = UUID("00000000-0000-0000-0000-000000000010")
REGION_IDS = (
    UUID("00000000-0000-0000-0000-000000000011"),
    UUID("00000000-0000-0000-0000-000000000012"),
)


def target() -> ExplanationTarget:
    segments = (
        CoherentTextSegment(
            sequence_number=0,
            region_id=REGION_IDS[0],
            page_id=UUID("00000000-0000-0000-0000-000000000020"),
            text_version_id=UUID("00000000-0000-0000-0000-000000000021"),
            selection_id=UUID("00000000-0000-0000-0000-000000000022"),
            text_start=0,
            text_end=3,
            composite_start=0,
            composite_end=3,
            text="女學生",
            role="body",
            polygon=None,
        ),
        CoherentTextSegment(
            sequence_number=1,
            region_id=REGION_IDS[1],
            page_id=UUID("00000000-0000-0000-0000-000000000020"),
            text_version_id=UUID("00000000-0000-0000-0000-000000000023"),
            selection_id=UUID("00000000-0000-0000-0000-000000000024"),
            text_start=0,
            text_end=4,
            composite_start=4,
            composite_end=8,
            text="入校讀書",
            role="body",
            polygon=None,
        ),
    )
    content = "女學生\n入校讀書"
    bundle = CoherentTextBundle(
        coherent_unit_revision_id=REVISION_ID,
        content=content,
        input_sha256="a" * 64,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        multimodal_input_sha256="b" * 64,
        segments=segments,
        page_images=(),
    )
    return ExplanationTarget(
        bundle,
        ExplanationAuthority.EXPERIMENTAL,
        "Experimental/non-gold passage.",
    )


class FakeGenerator:
    model_identity = "local/model@revision-1"
    model_revision = "revision-1"
    provider_kind = "test"
    generation_configuration_sha256 = "c" * 64

    def complete(self, messages):
        assert "E1" in messages[1]["content"]
        assert "E2" in messages[1]["content"]
        return TextCompletion(
            content=(
                '{"plain_language_gloss":"女学生进入学校读书。",'
                '"ambiguous_phrases":[],"limitations":["OCR断句可能有误。"],'
                '"evidence_ids":["E1","E2"]}'
            ),
            raw_content_sha256="d" * 64,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )


def test_explain_result_maps_request_scoped_evidence_to_region_ids() -> None:
    result = explain_result(target(), "女学生教育", FakeGenerator())

    assert result.status == ExplanationStatus.COMPLETED
    assert result.authority == ExplanationAuthority.EXPERIMENTAL
    assert result.plain_language_gloss == "女学生进入学校读书。"
    assert [item.evidence_id for item in result.evidence] == ["E1", "E2"]
    assert [item.region_id for item in result.evidence] == list(REGION_IDS)
    assert result.total_tokens == 30


def test_explain_result_rejects_unknown_evidence_ids() -> None:
    class InvalidGenerator(FakeGenerator):
        def complete(self, messages):
            return (
                '{"plain_language_gloss":"unsupported",'
                '"ambiguous_phrases":[],"limitations":["uncertain"],'
                '"evidence_ids":["E99"]}'
            )

    result = explain_result(target(), "女学生教育", InvalidGenerator())

    assert result.status == ExplanationStatus.REJECTED
    assert result.plain_language_gloss == ""
    assert "E99" in result.validation_errors[0]


def test_explain_result_reports_unconfigured_generator_without_invocation() -> None:
    result = explain_result(target(), "女学生教育", None)

    assert result.status == ExplanationStatus.UNAVAILABLE
    assert result.model is None
    assert result.evidence[0].region_id == REGION_IDS[0]


def test_explain_result_api_loads_target_by_revision_id() -> None:
    with patch(
        "wic_history.api.load_explanation_target", return_value=target()
    ) as loader:
        response = TestClient(
            create_app(
                database_url="postgresql://example",
                generator_factory=FakeGenerator,
            )
        ).post(
            "/api/explain-result",
            json={"revision_id": str(REVISION_ID), "query": "女学生教育"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["evidence"][0]["evidence_id"] == "E1"
    loader.assert_called_once_with("postgresql://example", REVISION_ID)


def test_model_context_excludes_real_uuids() -> None:
    passage = target()
    messages, _, _ = prepare_explanation_messages(
        passage, "女学生教育", _evidence(passage)
    )
    prompt = messages[1]["content"]
    assert str(REVISION_ID) not in prompt
    for region_id in REGION_IDS:
        assert str(region_id) not in prompt
    assert "E1" in prompt and "E2" in prompt


def test_explain_result_rejects_duplicate_ids_in_ambiguous_phrases() -> None:
    class DuplicateGenerator(FakeGenerator):
        def complete(self, messages):
            return (
                '{"plain_language_gloss":"g",'
                '"ambiguous_phrases":[{"phrase":"x","explanation":"y",'
                '"evidence_ids":["E1","E1"]}],'
                '"limitations":["l"],"evidence_ids":["E1"]}'
            )

    result = explain_result(target(), "女学生教育", DuplicateGenerator())

    assert result.status == ExplanationStatus.REJECTED
    assert "duplicate" in result.validation_errors[0].lower()
