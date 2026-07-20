"""Canonical text-only materialization for reviewed article revisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, TypedDict
from uuid import UUID


MATERIALIZER_SCHEMA: Final = "reviewed-article-text/v1"
SPAN_SEPARATOR: Final = "\n"


class AlignmentOperation(TypedDict):
    operation: str
    source_start: int
    source_end: int
    target_start: int
    target_end: int


@dataclass(frozen=True, slots=True)
class ReviewedTextMaterializationError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class ReviewedSpanInput:
    sequence_number: int
    region_id: UUID
    page_id: UUID
    raw_text: str
    raw_start: int
    raw_end: int
    selected_text_version_id: UUID
    selected_text_sha256: str
    selection_id: UUID
    selected_text: str
    role: str
    alignment_operations: tuple[AlignmentOperation, ...] | None


@dataclass(frozen=True, slots=True)
class MaterializedReviewedSpan:
    sequence_number: int
    region_id: UUID
    page_id: UUID
    raw_start: int
    raw_end: int
    selected_text_version_id: UUID
    selected_text_sha256: str
    selection_id: UUID
    selected_start: int
    selected_end: int
    composite_start: int
    composite_end: int
    text: str
    role: str


@dataclass(frozen=True, slots=True)
class ReviewedArticleText:
    coherent_unit_revision_id: UUID
    content: str
    content_sha256: str
    input_sha256: str
    spans: tuple[MaterializedReviewedSpan, ...]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: dict[str, str | list[dict[str, str | int]]]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _map_boundary(
    operations: tuple[AlignmentOperation, ...] | list[AlignmentOperation],
    boundary: int,
) -> int:
    for operation in operations:
        source_start = operation["source_start"]
        source_end = operation["source_end"]
        target_start = operation["target_start"]
        target_end = operation["target_end"]
        if boundary == source_start:
            return target_start
        if boundary == source_end:
            return target_end
        if source_start < boundary < source_end:
            if operation["operation"] != "equal":
                raise ReviewedTextMaterializationError(
                    "coherent-unit split crosses an ambiguous text correction"
                )
            return target_start + (boundary - source_start)
    raise ReviewedTextMaterializationError(
        "coherent-unit offset is outside its text alignment"
    )


def _selected_interval(span: ReviewedSpanInput) -> tuple[int, int]:
    if span.raw_start == 0 and span.raw_end == len(span.raw_text):
        return 0, len(span.selected_text)
    if span.raw_text == span.selected_text:
        return span.raw_start, span.raw_end
    if span.alignment_operations is None:
        raise ReviewedTextMaterializationError(
            "partial corrected coherent-unit span lacks an alignment"
        )
    return _map_boundary(
        span.alignment_operations, span.raw_start
    ), _map_boundary(span.alignment_operations, span.raw_end)


def materialize_reviewed_article(
    coherent_unit_revision_id: UUID,
    unit_kind: str,
    spans: tuple[ReviewedSpanInput, ...],
) -> ReviewedArticleText:
    """Assemble one reviewed article with deterministic offsets and text identity."""
    if unit_kind != "article":
        raise ReviewedTextMaterializationError(
            "canonical reviewed text requires unit_kind article"
        )
    if not spans:
        raise ReviewedTextMaterializationError(
            "canonical reviewed text requires at least one selected span"
        )
    ordered = tuple(sorted(spans, key=lambda span: span.sequence_number))
    if tuple(span.sequence_number for span in ordered) != tuple(range(len(ordered))):
        raise ReviewedTextMaterializationError(
            "reviewed article spans require contiguous unique sequence numbers"
        )

    materialized: list[MaterializedReviewedSpan] = []
    pieces: list[str] = []
    position = 0
    for source in ordered:
        if _sha256_text(source.selected_text) != source.selected_text_sha256:
            raise ReviewedTextMaterializationError(
                f"selected text hash differs for region {source.region_id}"
            )
        if not 0 <= source.raw_start < source.raw_end <= len(source.raw_text):
            raise ReviewedTextMaterializationError(
                f"invalid non-empty raw interval for region {source.region_id}"
            )
        selected_start, selected_end = _selected_interval(source)
        if not 0 <= selected_start < selected_end <= len(source.selected_text):
            raise ReviewedTextMaterializationError(
                f"invalid non-empty selected interval for region {source.region_id}"
            )
        text = source.selected_text[selected_start:selected_end]
        if materialized:
            pieces.append(SPAN_SEPARATOR)
            position += len(SPAN_SEPARATOR)
        composite_start = position
        pieces.append(text)
        position += len(text)
        materialized.append(
            MaterializedReviewedSpan(
                sequence_number=source.sequence_number,
                region_id=source.region_id,
                page_id=source.page_id,
                raw_start=source.raw_start,
                raw_end=source.raw_end,
                selected_text_version_id=source.selected_text_version_id,
                selected_text_sha256=source.selected_text_sha256,
                selection_id=source.selection_id,
                selected_start=selected_start,
                selected_end=selected_end,
                composite_start=composite_start,
                composite_end=position,
                text=text,
                role=source.role,
            )
        )

    content = "".join(pieces)
    identity_spans: list[dict[str, str | int]] = [
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
            "materialized_text_sha256": _sha256_text(span.text),
        }
        for span in materialized
    ]
    input_sha256 = _canonical_sha256(
        {
            "materializer_schema": MATERIALIZER_SCHEMA,
            "separator": SPAN_SEPARATOR,
            "coherent_unit_revision_id": str(coherent_unit_revision_id),
            "spans": identity_spans,
        }
    )
    return ReviewedArticleText(
        coherent_unit_revision_id=coherent_unit_revision_id,
        content=content,
        content_sha256=_sha256_text(content),
        input_sha256=input_sha256,
        spans=tuple(materialized),
    )
