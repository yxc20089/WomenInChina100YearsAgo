from __future__ import annotations

from uuid import UUID

import pytest

from wic_history.semantic_repository import (
    CoherentTextBundle,
    CoherentTextSegment,
    PageImageReference,
    _durable_mention_id,
    _map_boundary,
    _selected_interval,
    _semantic_run_id,
    semantic_multimodal_context,
)


def test_full_region_uses_entire_reviewed_correction() -> None:
    assert _selected_interval("英皇時召雷臨宮中", "英皇時召霍臨宮中", 0, 8, None) == (
        0,
        8,
    )


def test_partial_region_maps_only_unambiguous_alignment_boundaries() -> None:
    operations = [
        {
            "operation": "equal",
            "source_start": 0,
            "source_end": 4,
            "target_start": 0,
            "target_end": 4,
        },
        {
            "operation": "replace",
            "source_start": 4,
            "source_end": 5,
            "target_start": 4,
            "target_end": 5,
        },
        {
            "operation": "equal",
            "source_start": 5,
            "source_end": 8,
            "target_start": 5,
            "target_end": 8,
        },
    ]
    assert _map_boundary(operations, 4) == 4
    assert _selected_interval(
        "英皇時召雷臨宮中", "英皇時召霍臨宮中", 0, 4, operations
    ) == (
        0,
        4,
    )
    with pytest.raises(ValueError, match="ambiguous"):
        _map_boundary(
            [
                {
                    "operation": "replace",
                    "source_start": 0,
                    "source_end": 2,
                    "target_start": 0,
                    "target_end": 3,
                }
            ],
            1,
        )


def test_semantic_run_identity_changes_with_any_provenance_input() -> None:
    revision = UUID(int=1)
    base = _semantic_run_id(revision, "event_frames", "a" * 64, "b" * 64, "c" * 64)
    assert base == _semantic_run_id(
        revision, "event_frames", "a" * 64, "b" * 64, "c" * 64
    )
    assert base != _semantic_run_id(
        revision, "event_frames", "d" * 64, "b" * 64, "c" * 64
    )


def test_equal_surfaces_at_different_offsets_keep_distinct_occurrence_ids() -> None:
    segment = CoherentTextSegment(
        sequence_number=0,
        region_id=UUID(int=2),
        page_id=UUID(int=3),
        text_version_id=UUID(int=4),
        selection_id=UUID(int=5),
        text_start=0,
        text_end=3,
        composite_start=0,
        composite_end=3,
        text="霍與霍",
        role="body",
        polygon={
            "points": [
                {"x": 0, "y": 0},
                {"x": 10, "y": 0},
                {"x": 10, "y": 10},
            ]
        },
    )
    revision_id = UUID(int=1)
    assert _durable_mention_id(revision_id, segment, 0, 1) != _durable_mention_id(
        revision_id, segment, 2, 3
    )


def test_multimodal_context_binds_region_box_to_immutable_derivative() -> None:
    segment = CoherentTextSegment(
        sequence_number=0,
        region_id=UUID(int=2),
        page_id=UUID(int=3),
        text_version_id=UUID(int=4),
        selection_id=UUID(int=5),
        text_start=0,
        text_end=3,
        composite_start=0,
        composite_end=3,
        text="霍與霍",
        role="body",
        polygon={
            "points": [
                {"x": 0, "y": 0},
                {"x": 10, "y": 0},
                {"x": 10, "y": 10},
            ]
        },
    )
    image = PageImageReference(
        page_id=UUID(int=3),
        derivative_id=UUID(int=6),
        image_uri="artifacts/page.png",
        image_sha256="a" * 64,
        media_type="image/png",
        width=10,
        height=10,
        region_ids=(UUID(int=2),),
    )
    bundle = CoherentTextBundle(
        coherent_unit_revision_id=UUID(int=1),
        content="霍與霍",
        input_sha256="b" * 64,
        segments=(segment,),
        page_images=(image,),
    )
    segments, images = semantic_multimodal_context(bundle)
    assert segments[0].polygon.points[2].x == 10
    assert images[0].image_sha256 == "a" * 64
    assert images[0].region_ids == [UUID(int=2)]
