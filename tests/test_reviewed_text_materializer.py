from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from uuid import UUID

import pytest

from wic_history.reviewed_text_materializer import (
    MATERIALIZER_SCHEMA,
    SPAN_SEPARATOR,
    AlignmentOperation,
    ReviewedSpanInput,
    ReviewedTextMaterializationError,
    materialize_reviewed_article,
)


@pytest.fixture
def reviewed_spans() -> tuple[ReviewedSpanInput, ...]:
    operations: tuple[AlignmentOperation, ...] = (
        {
            "operation": "equal",
            "source_start": 0,
            "source_end": 1,
            "target_start": 0,
            "target_end": 1,
        },
        {
            "operation": "replace",
            "source_start": 1,
            "source_end": 2,
            "target_start": 1,
            "target_end": 2,
        },
        {
            "operation": "equal",
            "source_start": 2,
            "source_end": 3,
            "target_start": 2,
            "target_end": 3,
        },
    )
    return (
        ReviewedSpanInput(
            sequence_number=0,
            region_id=UUID(int=2),
            page_id=UUID(int=10),
            raw_text="甲雷乙",
            raw_start=0,
            raw_end=2,
            selected_text_version_id=UUID(int=102),
            selected_text_sha256=hashlib.sha256("甲霍乙".encode("utf-8")).hexdigest(),
            selection_id=UUID(int=202),
            selected_text="甲霍乙",
            role="body",
            alignment_operations=operations,
        ),
        ReviewedSpanInput(
            sequence_number=1,
            region_id=UUID(int=3),
            page_id=UUID(int=11),
            raw_text="丙",
            raw_start=0,
            raw_end=1,
            selected_text_version_id=UUID(int=103),
            selected_text_sha256=hashlib.sha256("丁".encode("utf-8")).hexdigest(),
            selection_id=UUID(int=203),
            selected_text="丁",
            role="body",
            alignment_operations=None,
        ),
    )


def test_materializer_preserves_cross_page_reviewed_offsets_and_exact_hashes(
    reviewed_spans: tuple[ReviewedSpanInput, ...],
) -> None:
    # Given: two ordered reviewed spans on different pages, including a correction.
    revision_id = UUID(int=1)

    # When: the article is materialized through the canonical text-only path.
    materialized = materialize_reviewed_article(
        revision_id, "article", reviewed_spans
    )

    # Then: content, offsets, provenance, and hashes match the exact selected text.
    assert materialized.content == "甲霍\n丁"
    assert materialized.content_sha256 == hashlib.sha256(
        materialized.content.encode("utf-8")
    ).hexdigest()
    assert [
        (span.selected_start, span.selected_end, span.composite_start, span.composite_end)
        for span in materialized.spans
    ] == [(0, 2, 0, 2), (0, 1, 3, 4)]
    identity = {
        "materializer_schema": MATERIALIZER_SCHEMA,
        "separator": SPAN_SEPARATOR,
        "coherent_unit_revision_id": str(revision_id),
        "spans": [
            {
                "sequence_number": span.sequence_number,
                "region_id": str(span.region_id),
                "page_id": str(span.page_id),
                "raw_start": span.raw_start,
                "raw_end": span.raw_end,
                "selection_id": str(span.selection_id),
                "selected_text_version_id": str(span.selected_text_version_id),
                "selected_text_sha256": span.selected_text_sha256,
                "selected_start": span.selected_start,
                "selected_end": span.selected_end,
                "materialized_text_sha256": hashlib.sha256(
                    span.text.encode("utf-8")
                ).hexdigest(),
            }
            for span in materialized.spans
        ],
    }
    expected_input = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert materialized.input_sha256 == expected_input


@pytest.mark.parametrize("unit_kind", ["column", "caption"])
def test_materializer_rejects_non_article_units(
    unit_kind: str,
    reviewed_spans: tuple[ReviewedSpanInput, ...],
) -> None:
    # Given: reviewed text belonging to a coherent unit that is not an article.
    # When/Then: canonical article materialization rejects the unit boundary.
    with pytest.raises(ReviewedTextMaterializationError, match="unit_kind article"):
        _ = materialize_reviewed_article(UUID(int=1), unit_kind, reviewed_spans)


def test_materializer_rejects_stale_selected_text_hash(
    reviewed_spans: tuple[ReviewedSpanInput, ...],
) -> None:
    # Given: selected text whose stored immutable hash no longer matches its content.
    stale = (replace(reviewed_spans[0], selected_text_sha256="0" * 64),)

    # When/Then: materialization fails before emitting misleading canonical output.
    with pytest.raises(ReviewedTextMaterializationError, match="hash differs"):
        _ = materialize_reviewed_article(UUID(int=1), "article", stale)


@pytest.mark.parametrize(
    ("raw_start", "raw_end"),
    [(-1, 1), (2, 1), (0, 4), (1, 1)],
)
def test_materializer_rejects_invalid_or_empty_raw_intervals(
    raw_start: int,
    raw_end: int,
    reviewed_spans: tuple[ReviewedSpanInput, ...],
) -> None:
    # Given: a raw coherent-unit span outside the non-empty source interval contract.
    invalid = (
        replace(reviewed_spans[0], raw_start=raw_start, raw_end=raw_end),
    )

    # When/Then: materialization rejects it before Python slicing can normalize it.
    with pytest.raises(ReviewedTextMaterializationError, match="raw interval"):
        _ = materialize_reviewed_article(UUID(int=1), "article", invalid)


@pytest.mark.parametrize(
    ("target_start", "target_end"),
    [(-1, 1), (0, 4), (1, 1), (2, 1)],
)
def test_materializer_rejects_invalid_or_empty_selected_intervals(
    target_start: int,
    target_end: int,
    reviewed_spans: tuple[ReviewedSpanInput, ...],
) -> None:
    # Given: alignment boundaries outside the non-empty selected-text contract.
    operations: tuple[AlignmentOperation, ...] = (
        {
            "operation": "equal",
            "source_start": 0,
            "source_end": 2,
            "target_start": target_start,
            "target_end": target_end,
        },
    )
    invalid = (replace(reviewed_spans[0], alignment_operations=operations),)

    # When/Then: materialization rejects the selected interval before slicing.
    with pytest.raises(ReviewedTextMaterializationError, match="selected interval"):
        _ = materialize_reviewed_article(UUID(int=1), "article", invalid)
