"""Versioned, review-gated article/coherent-unit segmentation.

Machine output is stored only as an immutable proposal.  A separate accepted
review and explicit activation are required before downstream tools can consume
the units as historian-reviewed segmentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5


METHOD = "reading_order_window"
METHOD_VERSION = "1"


@dataclass(frozen=True, slots=True)
class SegmentationProposalResult:
    run_id: str
    page_id: str
    source_ocr_run_id: str
    proposal_sha256: str
    units: int
    regions: int
    reused: bool


@dataclass(frozen=True, slots=True)
class SegmentationReviewResult:
    review_id: str
    run_id: str
    decision: str
    reviewer: str


@dataclass(frozen=True, slots=True)
class SegmentationActivationResult:
    selection_id: str
    run_id: str
    page_id: str
    review_id: str
    superseded_selections: int
    approved_units: int
    reused: bool


@dataclass(frozen=True, slots=True)
class IssueResult:
    issue_id: str
    publication_date: str | None
    issue_number: str | None


@dataclass(frozen=True, slots=True)
class PageIssueAssignmentResult:
    assignment_id: str
    issue_id: str
    page_id: str
    sequence_number: int
    superseded_assignments: int


ACTIVE_REGIONS_SQL = """
SELECT region.region_id, region.page_id, region.run_id AS source_ocr_run_id,
       selection.selection_id AS source_ocr_selection_id,
       region.reading_order, region.raw_text, region.normalized_text,
       region.polygon, region.region_kind, region.confidence,
       page.page_number, volume.volume_number
FROM evidence.ocr_region region
JOIN archive.page page USING (page_id)
JOIN archive.volume volume USING (volume_id)
JOIN evidence.page_ocr_selection selection
  ON selection.page_id = region.page_id
 AND selection.run_id = region.run_id
 AND selection.superseded_at IS NULL
WHERE (CAST(%(volume_number)s AS integer) IS NULL
       OR volume.volume_number = CAST(%(volume_number)s AS integer))
  AND (CAST(%(page_number)s AS integer) IS NULL
       OR page.page_number = CAST(%(page_number)s AS integer))
ORDER BY volume.volume_number, page.page_number,
         region.reading_order, region.region_id
"""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def group_reading_order_windows(
    rows: Iterable[dict[str, Any]], *, max_regions: int, max_characters: int
) -> list[list[dict[str, Any]]]:
    """Create reproducible annotation-sized candidates, never inferred articles."""
    if max_regions < 1:
        raise ValueError("max_regions must be positive")
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    ordered = sorted(rows, key=lambda row: (row["reading_order"], str(row["region_id"])))
    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_characters = 0
    for row in ordered:
        text = (row.get("normalized_text") or row.get("raw_text") or "").strip()
        added_characters = len(text) + (1 if current and text else 0)
        if current and (
            len(current) >= max_regions
            or (current_characters + added_characters > max_characters and text)
        ):
            windows.append(current)
            current = []
            current_characters = 0
            added_characters = len(text)
        current.append(row)
        current_characters += added_characters
    if current:
        windows.append(current)
    return windows


def _input_identity(rows: list[dict[str, Any]]) -> str:
    return _canonical_sha256(
        [
            {
                "region_id": row["region_id"],
                "reading_order": row["reading_order"],
                "raw_text": row["raw_text"],
                "normalized_text": row.get("normalized_text"),
                "polygon": row["polygon"],
            }
            for row in rows
        ]
    )


def _proposal_identity(
    page_id: UUID,
    source_ocr_run_id: UUID,
    input_sha256: str,
    windows: list[list[dict[str, Any]]],
    configuration: dict[str, Any],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "1.0",
            "page_id": page_id,
            "source_ocr_run_id": source_ocr_run_id,
            "input_sha256": input_sha256,
            "method": METHOD,
            "method_version": METHOD_VERSION,
            "configuration": configuration,
            "units": [[row["region_id"] for row in window] for window in windows],
        }
    )


def _psycopg() -> tuple[Any, Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    return psycopg, dict_row, Jsonb


def validate_span_coverage(
    source_rows: Iterable[dict[str, Any]], member_rows: Iterable[dict[str, Any]]
) -> None:
    """Require exact, gap-free, non-overlapping coverage of every OCR string."""
    source = {row["region_id"]: row["raw_text"] for row in source_rows}
    intervals: dict[UUID, list[tuple[int, int]]] = defaultdict(list)
    for member in member_rows:
        region_id = member["region_id"]
        if region_id not in source:
            raise ValueError(f"segmentation references a foreign OCR region {region_id}")
        start = member["text_start"]
        end = member["text_end"]
        if start is None or end is None:
            raise ValueError("every segmentation membership requires exact text offsets")
        if start < 0 or end < start or end > len(source[region_id]):
            raise ValueError(f"invalid offsets for OCR region {region_id}")
        intervals[region_id].append((start, end))
    if set(intervals) != set(source):
        missing = sorted(str(region_id) for region_id in set(source) - set(intervals))
        raise ValueError(f"segmentation omits source OCR regions: {', '.join(missing[:5])}")
    for region_id, text in source.items():
        cursor = 0
        spans = sorted(intervals[region_id])
        if not text and spans != [(0, 0)]:
            raise ValueError(f"empty OCR region {region_id} requires one accounting span")
        for start, end in spans:
            if start != cursor:
                raise ValueError(f"segmentation has a gap or overlap in OCR region {region_id}")
            if start == end and text:
                raise ValueError(f"segmentation has an empty span in nonempty OCR region {region_id}")
            cursor = end
        if cursor != len(text):
            raise ValueError(f"segmentation does not cover all text in OCR region {region_id}")


def export_segmentation_artifact(database_url: str, run_id: UUID) -> dict[str, Any]:
    """Export an editable proposal while retaining its immutable source identity."""
    psycopg, dict_row, _ = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        segmentation = connection.execute(
            "SELECT * FROM evidence.article_segmentation WHERE run_id = %s", (run_id,)
        ).fetchone()
        if not segmentation:
            raise ValueError(f"unknown segmentation run {run_id}")
        rows = connection.execute(
            """
            SELECT article.article_id, article.ordinal, article.title,
                   article.article_type AS unit_kind, article.confidence,
                   member.region_id, member.sequence_number, member.text_start,
                   member.text_end, member.role
            FROM evidence.article article
            JOIN evidence.article_region member USING (article_id)
            WHERE article.run_id = %s
            ORDER BY article.ordinal, member.sequence_number
            """,
            (run_id,),
        ).fetchall()
    units_by_id: dict[UUID, dict[str, Any]] = {}
    for row in rows:
        unit = units_by_id.setdefault(
            row["article_id"],
            {
                "ordinal": row["ordinal"],
                "title": row["title"],
                "unit_kind": row["unit_kind"],
                "confidence": row["confidence"],
                "spans": [],
            },
        )
        unit["spans"].append(
            {
                "region_id": str(row["region_id"]),
                "text_start": row["text_start"],
                "text_end": row["text_end"],
                "role": row["role"],
            }
        )
    return {
        "schema_version": "1.0",
        "status": "segmentation_proposal_edit",
        "source_proposal_run_id": str(run_id),
        "source_proposal_sha256": segmentation["proposal_sha256"],
        "page_id": str(segmentation["page_id"]),
        "source_ocr_run_id": str(segmentation["source_ocr_run_id"]),
        "source_ocr_selection_id": str(segmentation["source_ocr_selection_id"]),
        "input_sha256": segmentation["input_sha256"],
        "instructions": [
            "Historian may merge, split, reorder, retitle, and retype units.",
            "Every OCR character must be covered exactly once; offsets are end-exclusive.",
            "Import creates a new immutable proposal and does not approve it.",
        ],
        "units": sorted(units_by_id.values(), key=lambda unit: unit["ordinal"]),
    }


def import_historian_segmentation(
    database_url: str, artifact: dict[str, Any], *, proposed_by: str
) -> SegmentationProposalResult:
    """Validate a historian edit and store it as a new unapproved proposal."""
    if artifact.get("schema_version") != "1.0":
        raise ValueError("historian segmentation artifact must use schema version 1.0")
    if not proposed_by.strip():
        raise ValueError("proposed_by is required")
    page_id = UUID(artifact["page_id"])
    source_ocr_run_id = UUID(artifact["source_ocr_run_id"])
    source_ocr_selection_id = UUID(artifact["source_ocr_selection_id"])
    units = artifact.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("historian segmentation requires at least one unit")
    psycopg, dict_row, Jsonb = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        source_proposal_run_id = artifact.get("source_proposal_run_id")
        source_proposal_sha256 = artifact.get("source_proposal_sha256")
        if source_proposal_run_id and source_proposal_sha256:
            source_proposal = connection.execute(
                """
                SELECT proposal_sha256 FROM evidence.article_segmentation
                WHERE run_id = %s
                """,
                (UUID(source_proposal_run_id),),
            ).fetchone()
            if (
                not source_proposal
                or source_proposal["proposal_sha256"] != source_proposal_sha256
            ):
                raise ValueError("source proposal hash does not match immutable database state")
        source_rows = connection.execute(
            """
            SELECT region.region_id, region.raw_text, region.normalized_text,
                   region.reading_order, region.polygon,
                   selection.selection_id AS source_ocr_selection_id
            FROM evidence.ocr_region region
            JOIN evidence.page_ocr_selection selection
              ON selection.page_id = region.page_id
             AND selection.run_id = region.run_id
             AND selection.superseded_at IS NULL
            WHERE region.page_id = %s AND region.run_id = %s
            ORDER BY region.reading_order, region.region_id
            """,
            (page_id, source_ocr_run_id),
        ).fetchall()
        if not source_rows:
            raise ValueError("artifact source OCR run is not the active page selection")
        if source_rows[0]["source_ocr_selection_id"] != source_ocr_selection_id:
            raise ValueError("artifact source OCR selection was superseded")
        input_sha256 = _input_identity(source_rows)
        if input_sha256 != artifact.get("input_sha256"):
            raise ValueError("artifact input hash does not match active OCR evidence")

        member_rows = []
        normalized_units = []
        seen_ordinals: set[int] = set()
        for unit in units:
            ordinal = int(unit["ordinal"])
            if ordinal < 0 or ordinal in seen_ordinals:
                raise ValueError("unit ordinals must be unique nonnegative integers")
            seen_ordinals.add(ordinal)
            spans = unit.get("spans")
            if not isinstance(spans, list) or not spans:
                raise ValueError(f"unit {ordinal} has no spans")
            normalized_spans = []
            seen_unit_regions: set[UUID] = set()
            for span in spans:
                region_id = UUID(span["region_id"])
                if region_id in seen_unit_regions:
                    raise ValueError("one unit may reference an OCR region only once")
                seen_unit_regions.add(region_id)
                normalized = {
                    "region_id": region_id,
                    "text_start": int(span["text_start"]),
                    "text_end": int(span["text_end"]),
                    "role": str(span.get("role") or "body"),
                }
                normalized_spans.append(normalized)
                member_rows.append(normalized)
            normalized_units.append(
                {
                    "ordinal": ordinal,
                    "title": unit.get("title"),
                    "unit_kind": str(unit.get("unit_kind") or "other"),
                    "confidence": unit.get("confidence"),
                    "spans": normalized_spans,
                }
            )
        validate_span_coverage(source_rows, member_rows)
        normalized_units.sort(key=lambda unit: unit["ordinal"])
        proposal_sha256 = _canonical_sha256(
            {
                "schema_version": "1.0",
                "page_id": page_id,
                "source_ocr_run_id": source_ocr_run_id,
                "source_ocr_selection_id": source_ocr_selection_id,
                "input_sha256": input_sha256,
                "method": "historian_json_edit",
                "units": normalized_units,
            }
        )
        existing = connection.execute(
            "SELECT run_id FROM evidence.article_segmentation WHERE proposal_sha256 = %s",
            (proposal_sha256,),
        ).fetchone()
        if existing:
            return SegmentationProposalResult(
                str(existing["run_id"]), str(page_id), str(source_ocr_run_id),
                proposal_sha256, len(normalized_units), len(source_rows), True,
            )
        run_id = uuid5(NAMESPACE_URL, f"wic-segmentation:{proposal_sha256}")
        configuration = {
            "artifact_schema_version": "1.0",
            "source_proposal_run_id": artifact.get("source_proposal_run_id"),
            "source_proposal_sha256": artifact.get("source_proposal_sha256"),
        }
        now = datetime.now(timezone.utc)
        connection.execute(
            """
            INSERT INTO evidence.processing_run (
                run_id, kind, engine, model_name, model_revision,
                software_version, configuration, status, started_at, completed_at
            ) VALUES (%s, 'layout', 'wic_history.segmentation',
                      'historian-json-edit', '1', '0.1.0', %s,
                      'completed', %s, %s)
            """,
            (run_id, Jsonb(configuration), now, now),
        )
        connection.execute(
            """
            INSERT INTO evidence.article_segmentation (
                run_id, page_id, source_ocr_run_id, source_ocr_selection_id,
                input_sha256, proposal_sha256, method, method_version,
                proposal_kind, configuration, proposed_by
            ) VALUES (%s, %s, %s, %s, %s, %s, 'historian_json_edit', '1',
                      'historian_authored', %s, %s)
            """,
            (
                run_id, page_id, source_ocr_run_id, source_ocr_selection_id,
                input_sha256, proposal_sha256, Jsonb(configuration), proposed_by,
            ),
        )
        source_by_id = {row["region_id"]: row for row in source_rows}
        for unit in normalized_units:
            article_id = uuid5(run_id, f"unit:{unit['ordinal']}")
            connection.execute(
                """
                INSERT INTO evidence.article (
                    article_id, run_id, page_id, ordinal, title,
                    article_type, confidence, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    article_id, run_id, page_id, unit["ordinal"], unit["title"],
                    unit["unit_kind"], unit["confidence"],
                    Jsonb({"unit_basis": "historian_authored_proposal"}),
                ),
            )
            for sequence_number, span in enumerate(unit["spans"]):
                if span["region_id"] not in source_by_id:
                    raise ValueError("historian span references a foreign region")
                connection.execute(
                    """
                    INSERT INTO evidence.article_region (
                        article_id, region_id, sequence_number, page_id, run_id,
                        source_ocr_run_id, text_start, text_end, role
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        article_id, span["region_id"], sequence_number, page_id,
                        run_id, source_ocr_run_id, span["text_start"],
                        span["text_end"], span["role"],
                    ),
                )
    return SegmentationProposalResult(
        str(run_id), str(page_id), str(source_ocr_run_id), proposal_sha256,
        len(normalized_units), len(source_rows), False,
    )


def propose_segmentations(
    database_url: str,
    *,
    volume_number: int | None = None,
    page_number: int | None = None,
    max_regions: int = 24,
    max_characters: int = 600,
    proposed_by: str = "wic-segment",
) -> list[SegmentationProposalResult]:
    """Persist deterministic machine candidates over each active OCR page."""
    if page_number is not None and volume_number is None:
        raise ValueError("page_number requires volume_number")
    psycopg, dict_row, Jsonb = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            ACTIVE_REGIONS_SQL,
            {"volume_number": volume_number, "page_number": page_number},
        ).fetchall()
        if not rows:
            raise ValueError("no active OCR regions match the requested scope")
        by_page: dict[tuple[UUID, UUID, UUID], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_page[
                (row["page_id"], row["source_ocr_run_id"], row["source_ocr_selection_id"])
            ].append(row)

        output = []
        for (page_id, source_ocr_run_id, source_ocr_selection_id), page_rows in by_page.items():
            windows = group_reading_order_windows(
                page_rows, max_regions=max_regions, max_characters=max_characters
            )
            input_sha256 = _input_identity(page_rows)
            configuration = {
                "max_regions": max_regions,
                "max_characters": max_characters,
                "semantic_claim": "annotation-sized candidate only; not an inferred article",
            }
            proposal_sha256 = _proposal_identity(
                page_id, source_ocr_run_id, input_sha256, windows, configuration
            )
            existing = connection.execute(
                """
                SELECT run_id FROM evidence.article_segmentation
                WHERE proposal_sha256 = %s
                """,
                (proposal_sha256,),
            ).fetchone()
            if existing:
                run_id = existing["run_id"]
                output.append(
                    SegmentationProposalResult(
                        str(run_id), str(page_id), str(source_ocr_run_id),
                        proposal_sha256, len(windows), len(page_rows), True
                    )
                )
                continue

            run_id = uuid5(NAMESPACE_URL, f"wic-segmentation:{proposal_sha256}")
            now = datetime.now(timezone.utc)
            connection.execute(
                """
                INSERT INTO evidence.processing_run (
                    run_id, kind, engine, model_name, model_revision,
                    software_version, configuration, status, started_at, completed_at
                ) VALUES (%s, 'layout', %s, %s, %s, %s, %s, 'completed', %s, %s)
                """,
                (
                    run_id, "wic_history.segmentation", METHOD, METHOD_VERSION,
                    "0.1.0", Jsonb(configuration), now, now,
                ),
            )
            connection.execute(
                """
                INSERT INTO evidence.article_segmentation (
                    run_id, page_id, source_ocr_run_id, source_ocr_selection_id, input_sha256,
                    proposal_sha256, method, method_version, proposal_kind,
                    configuration, proposed_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'machine', %s, %s)
                """,
                (
                    run_id, page_id, source_ocr_run_id, source_ocr_selection_id, input_sha256,
                    proposal_sha256, METHOD, METHOD_VERSION,
                    Jsonb(configuration), proposed_by,
                ),
            )
            for ordinal, window in enumerate(windows):
                article_id = uuid5(run_id, f"unit:{ordinal}")
                confidences = [row["confidence"] for row in window if row["confidence"] is not None]
                confidence = sum(confidences) / len(confidences) if confidences else None
                connection.execute(
                    """
                    INSERT INTO evidence.article (
                        article_id, run_id, page_id, ordinal, article_type,
                        confidence, metadata
                    ) VALUES (%s, %s, %s, %s, 'unknown', %s, %s)
                    """,
                    (
                        article_id, run_id, page_id, ordinal, confidence,
                        Jsonb({"unit_basis": "reading_order_window_candidate"}),
                    ),
                )
                for sequence_number, row in enumerate(window):
                    connection.execute(
                        """
                        INSERT INTO evidence.article_region (
                            article_id, region_id, sequence_number, page_id,
                            run_id, source_ocr_run_id, text_start, text_end, role
                        ) VALUES (%s, %s, %s, %s, %s, %s, 0, %s, 'body')
                        """,
                        (
                            article_id, row["region_id"], sequence_number,
                            page_id, run_id, source_ocr_run_id, len(row["raw_text"]),
                        ),
                    )
            output.append(
                SegmentationProposalResult(
                    str(run_id), str(page_id), str(source_ocr_run_id),
                    proposal_sha256, len(windows), len(page_rows), False
                )
            )
        return output


def _verify_current_complete_segmentation(connection: Any, run_id: UUID) -> dict[str, Any]:
    segmentation = connection.execute(
        """
        SELECT segmentation.run_id, segmentation.page_id,
               segmentation.source_ocr_run_id, segmentation.source_ocr_selection_id,
               selection.run_id AS active_ocr_run_id,
               selection.selection_id AS active_ocr_selection_id
        FROM evidence.article_segmentation segmentation
        LEFT JOIN evidence.page_ocr_selection selection
          ON selection.page_id = segmentation.page_id
         AND selection.superseded_at IS NULL
        WHERE segmentation.run_id = %s
        """,
        (run_id,),
    ).fetchone()
    if not segmentation:
        raise ValueError(f"unknown segmentation run {run_id}")
    if segmentation["active_ocr_run_id"] != segmentation["source_ocr_run_id"]:
        raise ValueError("segmentation source OCR run is no longer the active page selection")
    if segmentation["active_ocr_selection_id"] != segmentation["source_ocr_selection_id"]:
        raise ValueError("segmentation source OCR selection was superseded")
    source_rows = connection.execute(
        """
        SELECT region_id, raw_text
        FROM evidence.ocr_region
        WHERE page_id = %s AND run_id = %s
        """,
        (segmentation["page_id"], segmentation["source_ocr_run_id"]),
    ).fetchall()
    member_rows = connection.execute(
        """
        SELECT region_id, text_start, text_end
        FROM evidence.article_region WHERE run_id = %s
        """,
        (run_id,),
    ).fetchall()
    units = connection.execute(
        "SELECT count(*) AS count FROM evidence.article WHERE run_id = %s", (run_id,)
    ).fetchone()["count"]
    if units < 1:
        raise ValueError("segmentation has no coherent-unit proposals")
    validate_span_coverage(source_rows, member_rows)
    return {
        **segmentation,
        "source_regions": len(source_rows),
        "member_spans": len(member_rows),
        "units": units,
    }


def review_segmentation(
    database_url: str,
    run_id: UUID,
    *,
    decision: str,
    reviewer: str,
    note: str | None = None,
    review_id: UUID | None = None,
    expected_proposal_sha256: str | None = None,
    expected_input_sha256: str | None = None,
) -> SegmentationReviewResult:
    if decision not in {"accept", "reject", "needs_revision"}:
        raise ValueError("decision must be accept, reject, or needs_revision")
    if not reviewer.strip():
        raise ValueError("reviewer is required")
    psycopg, dict_row, _ = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        review_id = review_id or uuid4()
        stored = connection.execute(
            """
            SELECT run_id, decision, reviewer, note
            FROM evidence.article_segmentation_review WHERE review_id = %s
            """,
            (review_id,),
        ).fetchone()
        expected = {
            "run_id": run_id,
            "decision": decision,
            "reviewer": reviewer,
            "note": note,
        }
        if stored:
            if stored != expected:
                raise ValueError(f"review UUID {review_id} already has different content")
            return SegmentationReviewResult(str(review_id), str(run_id), decision, reviewer)
        identity = connection.execute(
            """
            SELECT proposal_sha256, input_sha256
            FROM evidence.article_segmentation WHERE run_id = %s
            """,
            (run_id,),
        ).fetchone()
        if not identity:
            raise ValueError(f"unknown segmentation run {run_id}")
        if (
            expected_proposal_sha256 is not None
            and identity["proposal_sha256"] != expected_proposal_sha256
        ):
            raise ValueError("segmentation proposal hash changed; reload before review")
        if (
            expected_input_sha256 is not None
            and identity["input_sha256"] != expected_input_sha256
        ):
            raise ValueError("segmentation input hash changed; reload before review")
        _verify_current_complete_segmentation(connection, run_id)
        connection.execute(
            """
            INSERT INTO evidence.article_segmentation_review (
                review_id, run_id, decision, reviewer, note
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (review_id) DO NOTHING
            """,
            (review_id, run_id, decision, reviewer, note),
        )
        stored = connection.execute(
            """
            SELECT run_id, decision, reviewer, note
            FROM evidence.article_segmentation_review WHERE review_id = %s
            """,
            (review_id,),
        ).fetchone()
        if stored != expected:
            raise ValueError(f"review UUID {review_id} already has different content")
    return SegmentationReviewResult(str(review_id), str(run_id), decision, reviewer)


def activate_segmentation(
    database_url: str,
    review_id: UUID,
    *,
    selected_by: str,
    expected_previous_selection_id: UUID | None,
    expected_proposal_sha256: str | None = None,
) -> SegmentationActivationResult:
    if not selected_by.strip():
        raise ValueError("selected_by is required")
    psycopg, dict_row, _ = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        review = connection.execute(
            """
            SELECT review.review_id, review.run_id, review.decision,
                   review.reviewed_at, segmentation.page_id,
                   segmentation.proposal_sha256
            FROM evidence.article_segmentation_review review
            JOIN evidence.article_segmentation segmentation USING (run_id)
            WHERE review.review_id = %s
            """,
            (review_id,),
        ).fetchone()
        if not review:
            raise ValueError(f"unknown segmentation review {review_id}")
        if review["decision"] != "accept":
            raise ValueError("only an accepted segmentation review can be activated")
        if (
            expected_proposal_sha256 is not None
            and review["proposal_sha256"] != expected_proposal_sha256
        ):
            raise ValueError("segmentation proposal hash changed; reload before activation")
        existing = connection.execute(
            """
            SELECT selection_id, page_id, run_id
            FROM evidence.page_article_segmentation_selection
            WHERE review_id = %s
            """,
            (review_id,),
        ).fetchone()
        if existing:
            approved_units = connection.execute(
                """
                SELECT count(*) AS count FROM evidence.coherent_unit_revision
                WHERE approval_selection_id = %s
                """,
                (existing["selection_id"],),
            ).fetchone()["count"]
            return SegmentationActivationResult(
                str(existing["selection_id"]), str(existing["run_id"]),
                str(existing["page_id"]), str(review_id), 0, approved_units, True,
            )
        _verify_current_complete_segmentation(connection, review["run_id"])
        connection.execute(
            "SELECT page_id FROM archive.page WHERE page_id = %s FOR UPDATE",
            (review["page_id"],),
        ).fetchone()
        later_blocker = connection.execute(
            """
            SELECT review_id, decision
            FROM evidence.article_segmentation_review
            WHERE run_id = %s
              AND (reviewed_at, review_id) > (%s, %s)
              AND decision IN ('reject', 'needs_revision')
            ORDER BY reviewed_at DESC, review_id DESC
            LIMIT 1
            """,
            (review["run_id"], review["reviewed_at"], review_id),
        ).fetchone()
        if later_blocker:
            raise ValueError(
                "accepted review is superseded by a later "
                f"{later_blocker['decision']} decision {later_blocker['review_id']}"
            )
        current_selection = connection.execute(
            """
            SELECT selection_id
            FROM evidence.page_article_segmentation_selection
            WHERE page_id = %s AND superseded_at IS NULL
            """,
            (review["page_id"],),
        ).fetchone()
        current_selection_id = (
            current_selection["selection_id"] if current_selection else None
        )
        if current_selection_id != expected_previous_selection_id:
            raise ValueError(
                "active page segmentation changed; reload before activation "
                f"(expected {expected_previous_selection_id}, found {current_selection_id})"
            )
        old_selection_ids = [
            row["selection_id"]
            for row in connection.execute(
                """
                SELECT selection_id
                FROM evidence.page_article_segmentation_selection
                WHERE page_id = %s AND superseded_at IS NULL
                """,
                (review["page_id"],),
            ).fetchall()
        ]
        if old_selection_ids:
            connection.execute(
                """
                UPDATE evidence.coherent_unit_revision
                SET superseded_at = now()
                WHERE approval_selection_id = ANY(%s::uuid[])
                  AND superseded_at IS NULL
                """,
                (old_selection_ids,),
            )
        superseded = connection.execute(
            """
            UPDATE evidence.page_article_segmentation_selection
            SET superseded_at = now()
            WHERE page_id = %s AND superseded_at IS NULL
            """,
            (review["page_id"],),
        ).rowcount
        selection_id = uuid4()
        connection.execute(
            """
            INSERT INTO evidence.page_article_segmentation_selection (
                selection_id, page_id, run_id, review_id,
                selection_basis, selected_by
            ) VALUES (%s, %s, %s, %s, 'historian_approved', %s)
            """,
            (
                selection_id, review["page_id"], review["run_id"],
                review_id, selected_by,
            ),
        )
        issue_row = connection.execute(
            """
            SELECT issue_id
            FROM archive.page_issue_assignment
            WHERE page_id = %s AND superseded_at IS NULL
            """,
            (review["page_id"],),
        ).fetchone()
        issue_id = issue_row["issue_id"] if issue_row else None
        proposals = connection.execute(
            """
            SELECT article.article_id, article.title, article.article_type,
                   member.region_id, member.sequence_number, member.text_start,
                   member.text_end, member.role, region.raw_text
            FROM evidence.article article
            JOIN evidence.article_region member USING (article_id)
            JOIN evidence.ocr_region region USING (region_id)
            WHERE article.run_id = %s
            ORDER BY article.ordinal, member.sequence_number
            """,
            (review["run_id"],),
        ).fetchall()
        by_article: dict[UUID, list[dict[str, Any]]] = defaultdict(list)
        for proposal in proposals:
            by_article[proposal["article_id"]].append(proposal)
        for article_id, members in by_article.items():
            unit_id = uuid5(NAMESPACE_URL, f"wic-coherent-unit:{article_id}")
            connection.execute(
                "INSERT INTO evidence.coherent_unit (unit_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (unit_id,),
            )
            revision_number = connection.execute(
                """
                SELECT COALESCE(max(revision_number), 0) + 1 AS revision_number
                FROM evidence.coherent_unit_revision WHERE unit_id = %s
                """,
                (unit_id,),
            ).fetchone()["revision_number"]
            content = []
            for member in members:
                start = member["text_start"] if member["text_start"] is not None else 0
                end = (
                    member["text_end"]
                    if member["text_end"] is not None
                    else len(member["raw_text"])
                )
                if end > len(member["raw_text"]):
                    raise ValueError("proposal member offsets exceed immutable OCR text")
                content.append(
                    {
                        "region_id": member["region_id"],
                        "text_start": start,
                        "text_end": end,
                        "text": member["raw_text"][start:end],
                        "role": member["role"],
                    }
                )
            content_sha256 = _canonical_sha256(content)
            revision_id = uuid5(selection_id, f"revision:{article_id}")
            unit_kind = members[0]["article_type"]
            if unit_kind not in {
                "article", "column", "caption", "advertisement",
                "classified", "table", "other",
            }:
                unit_kind = "other"
            connection.execute(
                """
                INSERT INTO evidence.coherent_unit_revision (
                    revision_id, unit_id, revision_number, issue_id, unit_kind,
                    title, source_proposal_article_id, approval_selection_id,
                    content_sha256, approved_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    revision_id, unit_id, revision_number, issue_id, unit_kind,
                    members[0]["title"], article_id, selection_id,
                    content_sha256, selected_by,
                ),
            )
            for sequence_number, item in enumerate(content):
                connection.execute(
                    """
                    INSERT INTO evidence.coherent_unit_span (
                        revision_id, region_id, sequence_number,
                        text_start, text_end, role
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        revision_id, item["region_id"], sequence_number,
                        item["text_start"], item["text_end"], item["role"],
                    ),
                )
    return SegmentationActivationResult(
        str(selection_id), str(review["run_id"]), str(review["page_id"]),
        str(review_id), superseded, len(by_article), False,
    )


def create_issue(
    database_url: str,
    *,
    publication_date: date | None,
    issue_number: str | None,
    edition_label: str | None,
    created_by: str,
) -> IssueResult:
    if publication_date is None and not (issue_number or "").strip():
        raise ValueError("publication_date or issue_number is required")
    if not created_by.strip():
        raise ValueError("created_by is required")
    psycopg, dict_row, Jsonb = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        issue_id = uuid4()
        connection.execute(
            """
            INSERT INTO archive.issue (
                issue_id, publication_date, issue_number, edition_label,
                metadata, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                issue_id, publication_date, issue_number, edition_label,
                Jsonb({"assertion_basis": "historian_assignment"}), created_by,
            ),
        )
    return IssueResult(
        str(issue_id), publication_date.isoformat() if publication_date else None,
        issue_number,
    )


def assign_page_to_issue(
    database_url: str,
    issue_id: UUID,
    *,
    volume_number: int,
    page_number: int,
    sequence_number: int,
    assigned_by: str,
    note: str | None = None,
) -> PageIssueAssignmentResult:
    if sequence_number < 0:
        raise ValueError("sequence_number must be nonnegative")
    if not assigned_by.strip():
        raise ValueError("assigned_by is required")
    psycopg, dict_row, _ = _psycopg()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        page = connection.execute(
            """
            SELECT page.page_id
            FROM archive.page page
            JOIN archive.volume volume USING (volume_id)
            WHERE volume.volume_number = %s AND page.page_number = %s
            """,
            (volume_number, page_number),
        ).fetchone()
        if not page:
            raise ValueError(f"unknown page v{volume_number} p{page_number}")
        if not connection.execute(
            "SELECT 1 FROM archive.issue WHERE issue_id = %s", (issue_id,)
        ).fetchone():
            raise ValueError(f"unknown issue {issue_id}")
        superseded = connection.execute(
            """
            UPDATE archive.page_issue_assignment
            SET superseded_at = now()
            WHERE page_id = %s AND superseded_at IS NULL
            """,
            (page["page_id"],),
        ).rowcount
        assignment_id = uuid4()
        connection.execute(
            """
            INSERT INTO archive.page_issue_assignment (
                assignment_id, page_id, issue_id, sequence_number,
                assigned_by, note
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                assignment_id, page["page_id"], issue_id, sequence_number,
                assigned_by, note,
            ),
        )
    return PageIssueAssignmentResult(
        str(assignment_id), str(issue_id), str(page["page_id"]),
        sequence_number, superseded,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    propose = subparsers.add_parser("propose")
    propose.add_argument("--volume", type=int)
    propose.add_argument("--page", type=int)
    propose.add_argument("--max-regions", type=int, default=24)
    propose.add_argument("--max-characters", type=int, default=600)
    propose.add_argument("--proposed-by", default="wic-segment")

    export = subparsers.add_parser("export")
    export.add_argument("--run-id", type=UUID, required=True)
    export.add_argument("--output", type=Path, required=True)

    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--input", type=Path, required=True)
    import_parser.add_argument("--proposed-by", required=True)

    review = subparsers.add_parser("review")
    review.add_argument("--run-id", type=UUID, required=True)
    review.add_argument("--decision", choices=("accept", "reject", "needs_revision"), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--note")

    activate = subparsers.add_parser("activate")
    activate.add_argument("--review-id", type=UUID, required=True)
    activate.add_argument("--selected-by", required=True)
    activate.add_argument("--expected-previous-selection-id", type=UUID)

    issue = subparsers.add_parser("create-issue")
    issue.add_argument("--publication-date", type=date.fromisoformat)
    issue.add_argument("--issue-number")
    issue.add_argument("--edition-label")
    issue.add_argument("--created-by", required=True)

    assignment = subparsers.add_parser("assign-page")
    assignment.add_argument("--issue-id", type=UUID, required=True)
    assignment.add_argument("--volume", type=int, required=True)
    assignment.add_argument("--page", type=int, required=True)
    assignment.add_argument("--sequence", type=int, required=True)
    assignment.add_argument("--assigned-by", required=True)
    assignment.add_argument("--note")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("--database-url or DATABASE_URL is required")
    if args.command == "propose":
        result: Any = propose_segmentations(
            args.database_url,
            volume_number=args.volume,
            page_number=args.page,
            max_regions=args.max_regions,
            max_characters=args.max_characters,
            proposed_by=args.proposed_by,
        )
        payload = [asdict(item) for item in result]
    elif args.command == "export":
        payload = export_segmentation_artifact(args.database_url, args.run_id)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        payload = {
            "output": str(args.output),
            "source_proposal_run_id": payload["source_proposal_run_id"],
            "units": len(payload["units"]),
        }
    elif args.command == "import":
        artifact = json.loads(args.input.read_text(encoding="utf-8"))
        payload = asdict(
            import_historian_segmentation(
                args.database_url, artifact, proposed_by=args.proposed_by
            )
        )
    elif args.command == "review":
        payload = asdict(
            review_segmentation(
                args.database_url, args.run_id, decision=args.decision,
                reviewer=args.reviewer, note=args.note,
            )
        )
    elif args.command == "activate":
        payload = asdict(
            activate_segmentation(
                args.database_url,
                args.review_id,
                selected_by=args.selected_by,
                expected_previous_selection_id=args.expected_previous_selection_id,
            )
        )
    elif args.command == "create-issue":
        payload = asdict(
            create_issue(
                args.database_url,
                publication_date=args.publication_date,
                issue_number=args.issue_number,
                edition_label=args.edition_label,
                created_by=args.created_by,
            )
        )
    else:
        payload = asdict(
            assign_page_to_issue(
                args.database_url,
                args.issue_id,
                volume_number=args.volume,
                page_number=args.page,
                sequence_number=args.sequence,
                assigned_by=args.assigned_by,
                note=args.note,
            )
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
