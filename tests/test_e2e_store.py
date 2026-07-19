from __future__ import annotations

import hashlib
import json
from uuid import uuid4

import pytest

from wic_history.e2e_store import (
    ARTICLE_LOCAL_IDENTITY_EVIDENCE_SQL,
    ENTITY_REVERSE_EVIDENCE_SQL,
    EVENT_REVERSE_EVIDENCE_SQL,
    ConfidenceCalibrationSpec,
    EventParticipantSpec,
    LocalIdentityClusterSpec,
    VisualEvidencePathSpec,
    VisualModelOutputSpec,
    alignment_operations,
    locate_unique_surface,
    review_local_identity_cluster,
)


def test_context_disambiguates_repeated_surface_and_code_owns_offsets() -> None:
    text = "霍爾平曾入英宮。英皇時召霍臨宮中。"
    located = locate_unique_surface(text, "霍", left_context="時召", right_context="臨宮中")
    assert text[located.text_start : located.text_end] == "霍"
    assert located.text_start == text.rindex("霍")


def test_ambiguous_or_missing_surface_abstains() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        locate_unique_surface("霍與霍", "霍")
    with pytest.raises(ValueError, match="found 0"):
        locate_unique_surface("霍", "雷")


def test_alignment_is_reversible_and_does_not_overwrite_raw_ocr() -> None:
    operations = alignment_operations("英皇時召雷臨宮中", "英皇時召霍臨宮中")
    replacement = next(item for item in operations if item["operation"] == "replace")
    assert replacement["source_text"] == "雷"
    assert replacement["target_text"] == "霍"
    assert replacement["source_start"] == replacement["target_start"] == 4


def test_event_participant_requires_exactly_one_identity_form() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        EventParticipantSpec(participant_role="invitee")


def test_reverse_indexes_follow_mentions_and_events_to_materials() -> None:
    for sql in (ENTITY_REVERSE_EVIDENCE_SQL, EVENT_REVERSE_EVIDENCE_SQL):
        assert "evidence.evidence_span" in sql
        assert "evidence.text_version" in sql
        assert "archive.source_object" in sql
    assert "evidence.mention_resolution" in ENTITY_REVERSE_EVIDENCE_SQL
    assert "resolution.superseded_at IS NULL" in ENTITY_REVERSE_EVIDENCE_SQL
    assert "event.event_status = 'reviewed'" in EVENT_REVERSE_EVIDENCE_SQL


def test_visual_output_requires_exact_path_and_explicit_confidence_provenance() -> None:
    evidence_path = VisualEvidencePathSpec(
        evidence_path_id=uuid4(),
        path_role="input_crop",
        source_object_id=uuid4(),
        page_id=uuid4(),
        derivative_id=uuid4(),
        source_uri="s3://bucket/volume.pdf",
        image_uri="artifacts/pages/v219-p0308.png",
        image_sha256="a" * 64,
        crop_uri="artifacts/crops/hunyuan-c09.png",
        crop_sha256="b" * 64,
        crop_bounds={"left": 10, "top": 20, "right": 30, "bottom": 80},
    )
    output = VisualModelOutputSpec(
        visual_output_id=uuid4(),
        output_kind="spotting",
        artifact_uri="artifacts/hunyuan/structured-parse.json",
        artifact_sha256="c" * 64,
        raw_output='{"text":"英皇時召霍臨宮中"}\n',
        evidence_paths=(evidence_path,),
        confidence=0.91,
        confidence_status="uncalibrated",
    )
    assert output.raw_output_sha256 == hashlib.sha256(
        output.raw_output.encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="score/calibration"):
        VisualModelOutputSpec(
            visual_output_id=uuid4(),
            output_kind="layout",
            artifact_uri="artifact.json",
            artifact_sha256="d" * 64,
            raw_output="{}",
            evidence_paths=(evidence_path,),
            confidence=0.91,
            confidence_status="calibrated",
        )
    with pytest.raises(ValueError, match="exact crop bytes"):
        VisualEvidencePathSpec(
            evidence_path_id=uuid4(),
            path_role="input_crop",
            source_object_id=uuid4(),
            page_id=uuid4(),
            derivative_id=uuid4(),
            source_uri="s3://bucket/volume.pdf",
            image_uri="page.png",
            image_sha256="e" * 64,
        )


def test_calibration_and_local_cluster_specs_preserve_occurrence_boundaries() -> None:
    calibration = ConfidenceCalibrationSpec(
        calibration_id=uuid4(),
        task_kind="ocr_spotting",
        model_name="tencent/HunyuanOCR",
        model_revision="model-revision",
        method="isotonic",
        dataset_id="historian-reviewed-difficult-glyph-v1",
        dataset_sha256="a" * 64,
        artifact_uri="artifacts/calibration/hunyuan-isotonic.json",
        artifact_sha256="b" * 64,
    )
    assert calibration.model_name == "tencent/HunyuanOCR"

    mention_id = uuid4()
    cluster = LocalIdentityClusterSpec(uuid4(), (mention_id,))
    assert cluster.mention_ids == (mention_id,)
    with pytest.raises(ValueError, match="only once"):
        LocalIdentityClusterSpec(uuid4(), (mention_id, mention_id))

    for fragment in (
        "evidence.local_coreference_member",
        "evidence.entity_mention",
        "evidence.evidence_span",
        "evidence.text_version",
        "evidence.ocr_run_input",
        "archive.page_derivative",
        "archive.source_object",
    ):
        assert fragment in ARTICLE_LOCAL_IDENTITY_EVIDENCE_SQL
    assert "evidence.entity_redirect" not in ARTICLE_LOCAL_IDENTITY_EVIDENCE_SQL


def test_local_identity_review_is_terminal_idempotent_and_keeps_members(monkeypatch) -> None:
    from wic_history import e2e_store

    cluster_id = uuid4()
    local_run_id = uuid4()
    revision_id = uuid4()
    review_id = uuid4()
    state = {
        "status": "candidate",
        "members": 2,
        "review": None,
    }

    class Result:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params):
            if "FROM evidence.local_coreference_cluster" in sql:
                return Result(
                    {
                        "local_cluster_id": cluster_id,
                        "local_coreference_run_id": local_run_id,
                        "coherent_unit_revision_id": revision_id,
                        "review_status": state["status"],
                        "created_at": "immutable",
                    }
                )
            if "SELECT count(*) AS total" in sql:
                return Result({"total": state["members"]})
            if "INSERT INTO evidence.review_decision" in sql:
                state["review"] = {
                    "review_id": params[0],
                    "reviewer": params[3],
                    "note": params[4],
                    "previous_value": json.loads(params[5]),
                    "new_value": json.loads(params[6]),
                }
                return Result({"review_id": params[0]})
            if "UPDATE evidence.local_coreference_cluster" in sql:
                state["status"] = params[0]
                return Result({"local_cluster_id": cluster_id})
            if "FROM evidence.review_decision" in sql:
                return Result(state["review"])
            raise AssertionError(sql)

    class Psycopg:
        @staticmethod
        def connect(*_args, **_kwargs):
            return Connection()

    monkeypatch.setattr(e2e_store, "_clients", lambda: (Psycopg, object()))
    first = review_local_identity_cluster(
        "postgresql://store-test",
        cluster_id,
        decision="accept",
        reviewer="historian:berta",
        note="Same person within this article revision.",
        review_id=review_id,
    )
    second = review_local_identity_cluster(
        "postgresql://store-test",
        cluster_id,
        decision="accept",
        reviewer="historian:berta",
        note="Same person within this article revision.",
        review_id=review_id,
    )
    assert first.review_status == second.review_status == "reviewed"
    assert first.memberships == second.memberships == 2
    assert first.reused is False
    assert second.reused is True
    assert state["members"] == 2
