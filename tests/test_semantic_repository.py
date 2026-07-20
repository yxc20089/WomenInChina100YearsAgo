from __future__ import annotations

from dataclasses import replace
from types import TracebackType
from uuid import UUID

import pytest

from wic_history import semantic_repository
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


QueryRow = dict[str, UUID | str | int]


class _QueryResult:
    def __init__(
        self,
        *,
        row: QueryRow | None = None,
        rows: list[QueryRow] | None = None,
    ) -> None:
        self._row: QueryRow | None = row
        self._rows: list[QueryRow] = rows or []

    def fetchone(self) -> QueryRow | None:
        return self._row

    def fetchall(self) -> list[QueryRow]:
        return self._rows


class _MissingSelectionConnection:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def __enter__(self) -> _MissingSelectionConnection:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def execute(self, query: str, _parameters: tuple[UUID, ...]) -> _QueryResult:
        self.queries.append(query)
        if "SELECT revision.revision_id" in query:
            return _QueryResult(row={"revision_id": UUID(int=1), "unit_kind": "article"})
        if "SELECT count(*) AS count" in query:
            return _QueryResult(row={"count": 1})
        return _QueryResult(rows=[])


class _MissingSelectionPsycopg:
    def __init__(self, connection: _MissingSelectionConnection) -> None:
        self._connection: _MissingSelectionConnection = connection

    def connect(
        self, _database_url: str, *, row_factory: str
    ) -> _MissingSelectionConnection:
        del row_factory
        return self._connection


@pytest.mark.parametrize("selection_state", ["missing", "superseded"])
def test_loader_rejects_missing_or_superseded_reviewed_selection(
    monkeypatch: pytest.MonkeyPatch, selection_state: str
) -> None:
    # Given: one active article span whose selected reviewed text is unavailable.
    connection = _MissingSelectionConnection()
    psycopg = _MissingSelectionPsycopg(connection)
    monkeypatch.setattr(
        semantic_repository,
        "_clients",
        lambda: (psycopg, selection_state),
    )

    # When/Then: loading fails instead of materializing partial or stale text.
    with pytest.raises(ValueError, match="every coherent-unit region"):
        _ = semantic_repository.load_reviewed_coherent_text(
            "postgresql://unused", UUID(int=1)
        )

    reviewed_query = connection.queries[-1]
    assert "selection.superseded_at IS NULL" in reviewed_query
    assert "version.review_status = 'reviewed'" in reviewed_query


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


@pytest.mark.parametrize(
    "changed_segment",
    [
        {"role": "headline"},
        {
            "polygon": {
                "points": [
                    {"x": 1, "y": 0},
                    {"x": 10, "y": 0},
                    {"x": 10, "y": 10},
                ]
            }
        },
    ],
)
def test_multimodal_identity_changes_with_every_model_visible_segment_field(
    changed_segment: dict[str, str | dict[str, list[dict[str, int]]]],
) -> None:
    # Given: identical reviewed text and image identity with one segment field changed.
    segment = CoherentTextSegment(
        sequence_number=0,
        region_id=UUID(int=2),
        page_id=UUID(int=3),
        text_version_id=UUID(int=4),
        selection_id=UUID(int=5),
        text_start=0,
        text_end=1,
        composite_start=0,
        composite_end=1,
        text="霍",
        role="body",
        polygon={"points": [{"x": 0, "y": 0}]},
    )
    bundle = CoherentTextBundle(
        coherent_unit_revision_id=UUID(int=1),
        content="霍",
        input_sha256="b" * 64,
        segments=(segment,),
        page_images=(),
        content_sha256="c" * 64,
    )
    changed = replace(segment, **changed_segment)

    # When: multimodal identities are calculated for both model-visible contexts.
    original_identity = semantic_repository.semantic_multimodal_input_sha256(bundle)
    changed_identity = semantic_repository.semantic_multimodal_input_sha256(
        replace(bundle, segments=(changed,))
    )

    # Then: role and polygon changes cannot reuse the same semantic run identity.
    assert original_identity != changed_identity
