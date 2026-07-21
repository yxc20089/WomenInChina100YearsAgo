from __future__ import annotations

import sys
from pathlib import Path

import pytest


E2E_DIR = Path(__file__).parents[1] / "experiments" / "e2e"
sys.path.insert(0, str(E2E_DIR))

from holbein_v219_p0308 import (  # noqa: E402
    EXPECTED_LABELS,
    SOURCE,
    exact_candidates,
    validate_response,
)
from load_holbein_candidates import (  # noqa: E402
    PILOT_MODEL_CONFIG_PATH,
    build_ner_artifact,
)
from wic_history.model_config import load_pipeline_model_configuration


def accepted_response() -> dict[str, object]:
    return {
        "decisions": [
            {"candidate_id": candidate_id, "label": label}
            for candidate_id, label in EXPECTED_LABELS.items()
        ]
    }


def model_artifact() -> dict[str, object]:
    configuration = load_pipeline_model_configuration(PILOT_MODEL_CONFIG_PATH)
    semantic = configuration.semantic
    return {
        "created_at": "2026-07-18T21:20:21.640716+00:00",
        "source": SOURCE,
        "model": semantic.served_model,
        "model_identity": {
            "model_name": semantic.model_name,
            "model_revision": semantic.model_revision,
            "ollama_manifest_digest": semantic.ollama_manifest_digest,
            "model_blob_sha256": semantic.model_blob_sha256,
            "quantization": semantic.quantization,
            "runtime_name": semantic.runtime_name,
            "runtime_version": semantic.runtime_version,
            "acceleration": semantic.acceleration,
        },
        "pipeline_model_configuration_sha256": configuration.sha256,
        "candidates": exact_candidates(SOURCE["region_text"]),
        "model_result": {
            "elapsed_seconds": 12.0,
            "raw_content_sha256": "a" * 64,
            "response": accepted_response(),
        },
        "accepted_for_ingestion": True,
    }


def test_candidates_have_deterministic_exact_offsets_and_context() -> None:
    candidates = {
        item["candidate_id"]: item
        for item in exact_candidates(SOURCE["region_text"])
    }
    assert (candidates["C1"]["start"], candidates["C1"]["end"]) == (7, 9)
    assert (candidates["C2"]["start"], candidates["C2"]["end"]) == (11, 12)
    assert (candidates["C3"]["start"], candidates["C3"]["end"]) == (11, 13)
    assert (candidates["C4"]["start"], candidates["C4"]["end"]) == (13, 15)
    for candidate in candidates.values():
        assert (
            SOURCE["region_text"][candidate["start"] : candidate["end"]]
            == candidate["surface"]
        )


def test_whole_response_validator_rejects_the_observed_08b_shape() -> None:
    observed_08b = [
        {"candidate_id": candidate_id, "label": "PERSON_REFERENCE"}
        for candidate_id in EXPECTED_LABELS
    ]
    errors = validate_response(observed_08b, exact_candidates(SOURCE["region_text"]))
    assert errors == ["response must be exactly one object containing decisions"]


def test_loader_emits_only_reviewable_mentions_and_never_model_offsets() -> None:
    artifact = build_ner_artifact(model_artifact())
    assert [(item.text, item.entity_type.value) for item in artifact.mentions] == [
        ("英皇", "role_title"),
        ("霍", "person"),
    ]
    assert [
        (item.source.text_start, item.source.text_end) for item in artifact.mentions
    ] == [(7, 9), (11, 12)]
    assert artifact.run.configuration["offsets_supplied_by_model"] is False
    assert artifact.mentions[1].attributes["do_not_register_as_global_alias"] is True


def test_loader_rejects_tampered_candidates() -> None:
    artifact = model_artifact()
    artifact["candidates"][1]["start"] = 10  # type: ignore[index]
    with pytest.raises(ValueError, match="deterministic candidates"):
        build_ner_artifact(artifact)
