"""Typed researcher-UI contract for segmentation proposals and approvals."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from .evidence import StrictModel
from .segmentation import (
    SegmentationActivationResult,
    SegmentationProposalResult,
    SegmentationReviewResult,
    activate_segmentation,
    export_segmentation_artifact,
    import_historian_segmentation,
    review_segmentation,
    validate_span_coverage,
)


class SegmentationQueueItem(StrictModel):
    run_id: UUID
    page_id: UUID
    source_ocr_run_id: UUID
    source_ocr_selection_id: UUID
    input_sha256: str
    proposal_sha256: str
    proposal_kind: Literal["machine", "historian_authored"]
    method: str
    method_version: str
    proposed_by: str
    created_at: datetime
    volume_number: int
    publication_year: int
    page_number: int
    derivative_id: UUID
    image_sha256: str
    image_width: int
    image_height: int
    evidence_tier: str
    source_selection_active: bool
    reviewable: bool
    units: int
    member_spans: int
    source_regions: int
    review_count: int
    review_counts: dict[str, int]
    active_issue_id: UUID | None = None
    latest_review_id: UUID | None = None
    latest_decision: Literal["accept", "reject", "needs_revision"] | None = None
    latest_reviewer: str | None = None
    active_selection_id: UUID | None = None
    current_page_selection_id: UUID | None = None
    approved_units: int


class SegmentationQueueResponse(StrictModel):
    total: int
    limit: int
    offset: int
    items: list[SegmentationQueueItem]
    warnings: list[str]


class SegmentationReviewRecord(StrictModel):
    review_id: UUID
    decision: Literal["accept", "reject", "needs_revision"]
    reviewer: str
    note: str | None = None
    reviewed_at: datetime
    activated_selection_id: UUID | None = None
    selection_active: bool = False


class SegmentationSpanPreview(StrictModel):
    region_id: UUID
    reading_order: int
    text_start: int
    text_end: int
    text: str
    polygon: dict[str, Any]


class SegmentationUnitPreview(StrictModel):
    ordinal: int
    title: str | None = None
    unit_kind: str
    region_spans: int
    text: str
    spans: list[SegmentationSpanPreview]


class SegmentationDetailResponse(StrictModel):
    summary: SegmentationQueueItem
    reviews: list[SegmentationReviewRecord]
    editable_artifact: dict[str, Any]
    units: list[SegmentationUnitPreview]
    source_regions: int
    covered_regions: int
    member_spans: int
    coverage_complete: bool
    scan_available: bool
    reviewable: bool
    review_blockers: list[str]
    warnings: list[str]


class SegmentationImportRequest(StrictModel):
    artifact: "SegmentationEditArtifact"
    proposed_by: str = Field(min_length=1, max_length=200)
    confirmation: Literal["CREATE_UNAPPROVED_PROPOSAL"]


class SegmentationReviewRequest(StrictModel):
    review_id: UUID
    decision: Literal["accept", "reject", "needs_revision"]
    reviewer: str = Field(min_length=1, max_length=200)
    note: str | None = Field(default=None, max_length=5000)
    expected_proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checked_all_units: bool
    confirmation: Literal["RECORD_REVIEW_WITHOUT_ACTIVATION"]

    @model_validator(mode="after")
    def require_complete_acceptance_check(self) -> "SegmentationReviewRequest":
        if self.decision == "accept" and not self.checked_all_units:
            raise ValueError("accept requires checked_all_units=true")
        return self


class SegmentationActivationRequest(StrictModel):
    selected_by: str = Field(min_length=1, max_length=200)
    expected_previous_selection_id: UUID | None
    expected_proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation: Literal["ACTIVATE_ACCEPTED_SEGMENTATION"]


class SegmentationEditSpan(StrictModel):
    region_id: UUID
    text_start: int = Field(ge=0)
    text_end: int = Field(ge=0)
    role: str = Field(default="body", min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_offsets(self) -> "SegmentationEditSpan":
        if self.text_end < self.text_start:
            raise ValueError("text_end cannot precede text_start")
        return self


class SegmentationEditUnit(StrictModel):
    ordinal: int = Field(ge=0)
    title: str | None = Field(default=None, max_length=1000)
    unit_kind: Literal[
        "unknown", "article", "column", "caption", "advertisement",
        "classified", "table", "other",
    ]
    confidence: float | None = Field(default=None, ge=0, le=1)
    spans: list[SegmentationEditSpan] = Field(min_length=1, max_length=5000)


class SegmentationEditArtifact(StrictModel):
    schema_version: Literal["1.0"]
    status: Literal["segmentation_proposal_edit"]
    source_proposal_run_id: UUID
    source_proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_id: UUID
    source_ocr_run_id: UUID
    source_ocr_selection_id: UUID
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    instructions: list[str] = Field(default_factory=list, max_length=20)
    units: list[SegmentationEditUnit] = Field(min_length=1, max_length=5000)


SegmentationImportRequest.model_rebuild()


class SegmentationProposalResultView(StrictModel):
    run_id: UUID
    page_id: UUID
    source_ocr_run_id: UUID
    proposal_sha256: str
    units: int
    regions: int
    reused: bool


class SegmentationReviewResultView(StrictModel):
    review_id: UUID
    run_id: UUID
    decision: Literal["accept", "reject", "needs_revision"]
    reviewer: str


class SegmentationActivationResultView(StrictModel):
    selection_id: UUID
    run_id: UUID
    page_id: UUID
    review_id: UUID
    superseded_selections: int
    approved_units: int
    reused: bool


QUEUE_SQL = """
SELECT segmentation.run_id, segmentation.page_id,
       segmentation.source_ocr_run_id, segmentation.source_ocr_selection_id,
       segmentation.input_sha256, segmentation.proposal_sha256,
       segmentation.proposal_kind, segmentation.method,
       segmentation.method_version, segmentation.proposed_by,
       segmentation.created_at, volume.volume_number,
       volume.publication_year, page.page_number,
       ocr_selection.derivative_id, derivative.image_sha256,
       derivative.width AS image_width, derivative.height AS image_height,
       derivative.evidence_tier,
       (ocr_selection.superseded_at IS NULL) AS source_selection_active,
       (ocr_selection.superseded_at IS NULL) AS reviewable,
       (SELECT count(*) FROM evidence.article article
        WHERE article.run_id = segmentation.run_id) AS units,
       (SELECT count(*) FROM evidence.article_region member
        WHERE member.run_id = segmentation.run_id) AS member_spans,
       (SELECT count(*) FROM evidence.ocr_region source_region
        WHERE source_region.page_id = segmentation.page_id
          AND source_region.run_id = segmentation.source_ocr_run_id) AS source_regions,
       (SELECT count(*) FROM evidence.article_segmentation_review review_count
        WHERE review_count.run_id = segmentation.run_id) AS review_count,
       latest.review_id AS latest_review_id,
       latest.decision AS latest_decision,
       latest.reviewer AS latest_reviewer,
       jsonb_build_object(
         'accept', (SELECT count(*) FROM evidence.article_segmentation_review r
                    WHERE r.run_id = segmentation.run_id AND r.decision = 'accept'),
         'reject', (SELECT count(*) FROM evidence.article_segmentation_review r
                    WHERE r.run_id = segmentation.run_id AND r.decision = 'reject'),
         'needs_revision', (SELECT count(*) FROM evidence.article_segmentation_review r
                    WHERE r.run_id = segmentation.run_id AND r.decision = 'needs_revision')
       ) AS review_counts,
       active.selection_id AS active_selection_id,
       current_page_active.selection_id AS current_page_selection_id,
       issue_assignment.issue_id AS active_issue_id,
       (SELECT count(*) FROM evidence.coherent_unit_revision revision
        WHERE revision.approval_selection_id = active.selection_id
          AND revision.superseded_at IS NULL) AS approved_units,
       count(*) OVER () AS total
FROM evidence.article_segmentation segmentation
JOIN archive.page page USING (page_id)
JOIN archive.volume volume USING (volume_id)
JOIN evidence.page_ocr_selection ocr_selection
  ON ocr_selection.selection_id = segmentation.source_ocr_selection_id
JOIN archive.page_derivative derivative
  ON derivative.derivative_id = ocr_selection.derivative_id
 AND derivative.page_id = ocr_selection.page_id
LEFT JOIN LATERAL (
    SELECT review.review_id, review.decision, review.reviewer
    FROM evidence.article_segmentation_review review
    WHERE review.run_id = segmentation.run_id
    ORDER BY review.reviewed_at DESC, review.review_id DESC
    LIMIT 1
) latest ON true
LEFT JOIN evidence.page_article_segmentation_selection active
  ON active.run_id = segmentation.run_id
 AND active.superseded_at IS NULL
LEFT JOIN evidence.page_article_segmentation_selection current_page_active
  ON current_page_active.page_id = segmentation.page_id
 AND current_page_active.superseded_at IS NULL
LEFT JOIN archive.page_issue_assignment issue_assignment
  ON issue_assignment.page_id = segmentation.page_id
 AND issue_assignment.superseded_at IS NULL
ORDER BY (active.selection_id IS NOT NULL) DESC,
         segmentation.created_at DESC, segmentation.run_id
LIMIT %(limit)s OFFSET %(offset)s
"""


def _psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    return psycopg, dict_row


def list_segmentation_queue(
    database_url: str, *, limit: int = 25, offset: int = 0
) -> SegmentationQueueResponse:
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("limit must be 1–100 and offset nonnegative")
    psycopg, dict_row = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        total = connection.execute(
            "SELECT count(*) AS count FROM evidence.article_segmentation"
        ).fetchone()["count"]
        rows = connection.execute(QUEUE_SQL, {"limit": limit, "offset": offset}).fetchall()
    items = [
        SegmentationQueueItem.model_validate(
            {key: value for key, value in row.items() if key != "total"}
        )
        for row in rows
    ]
    return SegmentationQueueResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=items,
        warnings=[
            "Proposal counts are not historical article counts.",
            "Accepting a review does not activate or publish coherent units.",
        ],
    )


def segmentation_detail(database_url: str, run_id: UUID) -> SegmentationDetailResponse:
    psycopg, dict_row = _psycopg()
    artifact = export_segmentation_artifact(database_url, run_id)
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        summary_row = connection.execute(
            f"SELECT * FROM ({QUEUE_SQL.replace('LIMIT %(limit)s OFFSET %(offset)s', '')}) queue WHERE run_id = %(run_id)s",
            {"limit": 100, "offset": 0, "run_id": run_id},
        ).fetchone()
        if not summary_row:
            raise ValueError(f"unknown segmentation run {run_id}")
        reviews = connection.execute(
            """
            SELECT review.review_id, review.decision, review.reviewer,
                   review.note, review.reviewed_at,
                   selection.selection_id AS activated_selection_id,
                   (selection.selection_id IS NOT NULL
                    AND selection.superseded_at IS NULL) AS selection_active
            FROM evidence.article_segmentation_review review
            LEFT JOIN evidence.page_article_segmentation_selection selection
              ON selection.review_id = review.review_id
            WHERE review.run_id = %s
            ORDER BY review.reviewed_at, review.review_id
            """,
            (run_id,),
        ).fetchall()
        region_rows = connection.execute(
            """
            SELECT region_id, raw_text, reading_order, polygon
            FROM evidence.ocr_region
            WHERE run_id = %s
            """,
            (UUID(artifact["source_ocr_run_id"]),),
        ).fetchall()
        image_row = connection.execute(
            """
            SELECT image_uri FROM archive.page_derivative WHERE derivative_id = %s
            """,
            (summary_row["derivative_id"],),
        ).fetchone()
    summary = SegmentationQueueItem.model_validate(
        {key: value for key, value in summary_row.items() if key != "total"}
    )
    source_by_region = {str(row["region_id"]): row for row in region_rows}
    previews = []
    member_rows = []
    for unit in artifact["units"]:
        pieces = []
        span_previews = []
        for span in unit["spans"]:
            source = source_by_region[span["region_id"]]
            text = source["raw_text"][span["text_start"] : span["text_end"]]
            pieces.append(text)
            span_previews.append(
                SegmentationSpanPreview(
                    region_id=span["region_id"],
                    reading_order=source["reading_order"],
                    text_start=span["text_start"],
                    text_end=span["text_end"],
                    text=text,
                    polygon=source["polygon"],
                )
            )
            member_rows.append(
                {
                    "region_id": UUID(span["region_id"]),
                    "text_start": span["text_start"],
                    "text_end": span["text_end"],
                }
            )
        previews.append(
            SegmentationUnitPreview(
                ordinal=unit["ordinal"],
                title=unit["title"],
                unit_kind=unit["unit_kind"],
                region_spans=len(unit["spans"]),
                text="\n".join(pieces),
                spans=span_previews,
            )
        )
    blockers = []
    coverage_complete = True
    try:
        validate_span_coverage(region_rows, member_rows)
    except ValueError as exc:
        coverage_complete = False
        blockers.append(str(exc))
    if not summary.source_selection_active:
        blockers.append("Source OCR selection is stale; create a proposal over the active OCR run.")
    scan_available = False
    if image_row:
        candidate = Path(image_row["image_uri"])
        path = (candidate if candidate.is_absolute() else Path.cwd() / candidate).resolve()
        artifact_root = (Path.cwd() / "artifacts").resolve()
        if path.is_relative_to(artifact_root) and path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            scan_available = digest.hexdigest() == summary.image_sha256
    if not scan_available:
        blockers.append("The exact registered scan is missing or fails its SHA-256 check.")
    return SegmentationDetailResponse(
        summary=summary,
        reviews=[SegmentationReviewRecord.model_validate(row) for row in reviews],
        editable_artifact=artifact,
        units=previews,
        source_regions=len(region_rows),
        covered_regions=len({row["region_id"] for row in member_rows}),
        member_spans=len(member_rows),
        coverage_complete=coverage_complete,
        scan_available=scan_available,
        reviewable=not blockers,
        review_blockers=blockers,
        warnings=[
            "Inspect every boundary against the cited scan before accepting.",
            "Edit/import creates another proposal; it never changes this immutable record.",
            "Activation is a separate explicit operation available only for an accepted review.",
        ],
    )


def import_segmentation_edit(
    database_url: str, request: SegmentationImportRequest
) -> SegmentationProposalResult:
    return import_historian_segmentation(
        database_url,
        request.artifact.model_dump(mode="json"),
        proposed_by=request.proposed_by,
    )


def record_segmentation_review(
    database_url: str, run_id: UUID, request: SegmentationReviewRequest
) -> SegmentationReviewResult:
    return review_segmentation(
        database_url,
        run_id,
        decision=request.decision,
        reviewer=request.reviewer,
        note=request.note,
        review_id=request.review_id,
        expected_proposal_sha256=request.expected_proposal_sha256,
        expected_input_sha256=request.expected_input_sha256,
    )


def activate_reviewed_segmentation(
    database_url: str, review_id: UUID, request: SegmentationActivationRequest
) -> SegmentationActivationResult:
    return activate_segmentation(
        database_url,
        review_id,
        selected_by=request.selected_by,
        expected_previous_selection_id=request.expected_previous_selection_id,
        expected_proposal_sha256=request.expected_proposal_sha256,
    )
