from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID

import pytest

from wic_history.generation import TextCompletion
from wic_history.semantic_tasks import (
    IdentityPairResponse,
    LocalMentionInput,
    MentionCandidateInput,
    PageImageInput,
    ResolutionMentionInput,
    SemanticAbstention,
    SemanticTextSegmentInput,
    StructuredSemanticClient,
)


def _uuid(number: int) -> UUID:
    return UUID(int=number)


class FakeGenerator:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return TextCompletion(
            content=json.dumps(self.payload),
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )


def _multimodal_context(
    tmp_path: Path, *, image_sha256: str | None = None
) -> tuple[list[SemanticTextSegmentInput], list[PageImageInput]]:
    image_path = tmp_path / "page.png"
    image_bytes = b"immutable-page-image-bytes"
    image_path.write_bytes(image_bytes)
    region_id = _uuid(11)
    page_id = _uuid(12)
    return (
        [
            SemanticTextSegmentInput(
                region_id=region_id,
                page_id=page_id,
                text_version_id=_uuid(13),
                text_start=10,
                text_end=18,
                text="霍爾平曾赴上海。",
                role="body",
                polygon={
                    "points": [
                        {"x": 0, "y": 0},
                        {"x": 100, "y": 0},
                        {"x": 100, "y": 100},
                    ]
                },
            )
        ],
        [
            PageImageInput(
                page_id=page_id,
                derivative_id=_uuid(14),
                image_uri=str(image_path),
                image_sha256=image_sha256 or hashlib.sha256(image_bytes).hexdigest(),
                media_type="image/png",
                width=100,
                height=100,
                region_ids=[region_id],
            )
        ],
    )


def _extraction_payload(*, participant_key: str = "m1") -> dict[str, object]:
    return {
        "mentions": [
            {
                "mention_key": "m1",
                "region_id": str(_uuid(11)),
                "text_start": 10,
                "text_end": 13,
                "surface": "霍爾平",
                "entity_type": "person",
                "mention_form": "full_name",
            }
        ],
        "event_evidence": [
            {
                "evidence_key": "t1",
                "region_id": str(_uuid(11)),
                "text_start": 14,
                "text_end": 15,
                "surface": "赴",
                "evidence_role": "event_trigger",
            },
            {
                "evidence_key": "l1",
                "region_id": str(_uuid(11)),
                "text_start": 15,
                "text_end": 17,
                "surface": "上海",
                "evidence_role": "event_location",
            },
        ],
        "events": [
            {
                "event_key": "ev1",
                "event_type": "travel",
                "trigger_evidence_key": "t1",
                "participant_decisions": [
                    {"mention_key": participant_key, "participant_role": "traveler"}
                ],
                "evidence_keys": ["t1", "l1"],
                "date_evidence_key": None,
                "location_evidence_key": "l1",
                "aspect_evidence_key": None,
            }
        ],
    }


def test_combined_extraction_sends_verified_image_bytes_and_exact_evidence(
    tmp_path: Path,
) -> None:
    segments, images = _multimodal_context(tmp_path)
    generator = FakeGenerator(_extraction_payload())
    result = StructuredSemanticClient(
        generator, model_configuration_sha256="a" * 64
    ).extract_evidence(
        coherent_text="霍爾平曾赴上海。", segments=segments, page_images=images
    )
    assert result.response.mentions[0].surface == "霍爾平"
    user_content = generator.calls[0][0][1]["content"]
    assert isinstance(user_content, list)
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # Provenance remains in the logged/prompted JSON even though transport uses bytes.
    assert images[0].image_uri in user_content[0]["text"]
    assert images[0].image_sha256 in user_content[0]["text"]


def test_extraction_abstains_on_missing_or_hash_mismatched_image(
    tmp_path: Path,
) -> None:
    segments, images = _multimodal_context(tmp_path, image_sha256="0" * 64)
    client = StructuredSemanticClient(
        FakeGenerator(_extraction_payload()), model_configuration_sha256="a" * 64
    )
    with pytest.raises(SemanticAbstention, match="hash"):
        client.extract_evidence(
            coherent_text="霍爾平曾赴上海。", segments=segments, page_images=images
        )

    images[0] = images[0].model_copy(
        update={"image_uri": str(tmp_path / "missing.png")}
    )
    with pytest.raises(SemanticAbstention, match="missing"):
        client.extract_evidence(
            coherent_text="霍爾平曾赴上海。", segments=segments, page_images=images
        )


def test_extraction_abstains_on_unknown_local_ids_or_inexact_surface(
    tmp_path: Path,
) -> None:
    segments, images = _multimodal_context(tmp_path)
    unknown = StructuredSemanticClient(
        FakeGenerator(_extraction_payload(participant_key="invented")),
        model_configuration_sha256="a" * 64,
    )
    with pytest.raises(SemanticAbstention, match="local key"):
        unknown.extract_evidence(
            coherent_text="霍爾平曾赴上海。", segments=segments, page_images=images
        )

    payload = _extraction_payload()
    payload["mentions"][0]["surface"] = "霍爾乎"  # type: ignore[index]
    inexact = StructuredSemanticClient(
        FakeGenerator(payload), model_configuration_sha256="a" * 64
    )
    with pytest.raises(SemanticAbstention, match="exactly match"):
        inexact.extract_evidence(
            coherent_text="霍爾平曾赴上海。", segments=segments, page_images=images
        )


def test_second_call_accounts_for_only_validated_occurrence_ids(
    tmp_path: Path,
) -> None:
    segments, images = _multimodal_context(tmp_path)
    mentions = [
        ResolutionMentionInput(
            mention_id=_uuid(number),
            evidence_span_id=_uuid(number + 100),
            region_id=_uuid(11),
            page_id=_uuid(12),
            text_version_id=_uuid(13),
            text_start=start,
            text_end=end,
            surface=surface,
            entity_type="person",
            mention_form=form,
        )
        for number, start, end, surface, form in (
            (1, 10, 13, "霍爾平", "full_name"),
            (2, 12, 13, "平", "short_name"),
        )
    ]
    generator = FakeGenerator(
        {
            "clusters": [
                {
                    "mention_ids": [str(_uuid(1)), str(_uuid(2))],
                    "evidence_span_ids": [str(_uuid(101)), str(_uuid(102))],
                }
            ],
            "unresolved_mention_ids": [],
        }
    )
    result = StructuredSemanticClient(
        generator, model_configuration_sha256="a" * 64
    ).resolve_local_identities(
        coherent_text="霍爾平曾赴上海。",
        segments=segments,
        page_images=images,
        mentions=mentions,
    )
    assert result.response.clusters[0].mention_ids == [_uuid(1), _uuid(2)]
    assert generator.calls[0][0][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )

    omitted = StructuredSemanticClient(
        FakeGenerator(
            {
                "clusters": [],
                "unresolved_mention_ids": [str(_uuid(1))],
            }
        ),
        model_configuration_sha256="a" * 64,
    )
    with pytest.raises(SemanticAbstention, match="exactly once"):
        omitted.resolve_local_identities(
            coherent_text="霍爾平曾赴上海。",
            segments=segments,
            page_images=images,
            mentions=mentions,
        )


def test_candidate_classifier_requires_exact_id_coverage() -> None:
    candidates = [
        MentionCandidateInput(
            candidate_id=_uuid(1),
            evidence_span_id=_uuid(101),
            surface="霍",
            left_context="時召",
            right_context="臨宮中",
        )
    ]
    generator = FakeGenerator(
        {
            "decisions": [
                {
                    "candidate_id": str(_uuid(1)),
                    "decision": "KEEP",
                    "entity_type": "person",
                    "mention_form": "short_name",
                }
            ]
        }
    )
    result = StructuredSemanticClient(
        generator, model_configuration_sha256="a" * 64
    ).classify_mentions("英皇時召霍臨宮中", candidates)
    assert result.response.decisions[0].entity_type.value == "person"
    assert generator.calls[0][1]["response_format"]["type"] == "json_schema"

    missing = StructuredSemanticClient(
        FakeGenerator({"decisions": []}), model_configuration_sha256="a" * 64
    )
    with pytest.raises(SemanticAbstention, match="omitted"):
        missing.classify_mentions("英皇時召霍臨宮中", candidates)


def test_local_coreference_cannot_escape_supplied_mentions_or_evidence() -> None:
    mentions = [
        LocalMentionInput(
            mention_id=_uuid(1),
            evidence_span_id=_uuid(101),
            surface="霍爾平",
            entity_type="person",
        ),
        LocalMentionInput(
            mention_id=_uuid(2),
            evidence_span_id=_uuid(102),
            surface="霍",
            entity_type="person",
        ),
    ]
    valid = StructuredSemanticClient(
        FakeGenerator(
            {
                "clusters": [
                    {
                        "mention_ids": [str(_uuid(1)), str(_uuid(2))],
                        "evidence_span_ids": [str(_uuid(101)), str(_uuid(102))],
                    }
                ]
            }
        ),
        model_configuration_sha256="a" * 64,
    ).local_coreference("霍爾平……英皇時召霍臨宮中", mentions)
    assert len(valid.response.clusters) == 1

    invalid = StructuredSemanticClient(
        FakeGenerator(
            {
                "clusters": [
                    {
                        "mention_ids": [str(_uuid(1)), str(_uuid(3))],
                        "evidence_span_ids": [str(_uuid(101))],
                    }
                ]
            }
        ),
        model_configuration_sha256="a" * 64,
    )
    with pytest.raises(SemanticAbstention, match="boundaries"):
        invalid.local_coreference("text", mentions)


def test_invalid_named_entity_type_rejects_whole_structured_response() -> None:
    client = StructuredSemanticClient(
        FakeGenerator(
            {
                "decisions": [
                    {
                        "candidate_id": str(_uuid(1)),
                        "decision": "KEEP",
                        "entity_type": "role_title",
                        "mention_form": "title_reference",
                    }
                ]
            }
        ),
        model_configuration_sha256="a" * 64,
    )
    with pytest.raises(SemanticAbstention, match="invalid structured"):
        client.classify_mentions(
            "英皇",
            [
                MentionCandidateInput(
                    candidate_id=_uuid(1),
                    evidence_span_id=_uuid(101),
                    surface="英皇",
                )
            ],
        )


def test_identity_decisions_require_evidence_for_same_or_different() -> None:
    with pytest.raises(ValueError, match="supporting evidence"):
        IdentityPairResponse(
            decision="SAME",
            supporting_evidence_ids=[],
            contradiction_evidence_ids=[],
        )
    with pytest.raises(ValueError, match="contradiction evidence"):
        IdentityPairResponse(
            decision="DIFFERENT",
            supporting_evidence_ids=[],
            contradiction_evidence_ids=[],
        )


def test_identity_pair_cannot_cite_unsupplied_evidence() -> None:
    allowed = _uuid(101)
    client = StructuredSemanticClient(
        FakeGenerator(
            {
                "decision": "SAME",
                "supporting_evidence_ids": [str(_uuid(999))],
                "contradiction_evidence_ids": [],
            }
        ),
        model_configuration_sha256="a" * 64,
    )
    with pytest.raises(SemanticAbstention, match="unknown evidence"):
        client.identity_pair(
            left_profile={"name_surfaces": ["孫文"]},
            right_profile={"name_surfaces": ["孫中山"]},
            evidence_ids=[allowed],
        )
