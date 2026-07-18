"""Build provenance-locked NER annotation packets from active OCR selections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, ValidationError, model_validator

from .evidence import Polygon, RegionKind, SourcePointer, StrictModel
from .exploration import THEMES
from .ner_gold import (
    GoldAdjudication,
    GoldSnippet,
    NERGoldSet,
    ReviewerAnnotation,
)


class SamplingReason(StrEnum):
    WOMEN_THEME = "women_theme"
    NER_DISAGREEMENT = "ner_disagreement"
    NER_CANDIDATE = "ner_candidate"
    NO_CANDIDATE_BASELINE = "no_candidate_baseline"
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    MEDIUM_OCR_CONFIDENCE = "medium_ocr_confidence"
    HIGH_OCR_CONFIDENCE = "high_ocr_confidence"
    UNKNOWN_OCR_CONFIDENCE = "unknown_ocr_confidence"


class PacketSamplingConfig(StrictModel):
    max_units: int = Field(gt=0, le=5000)
    context_radius: int = Field(ge=0, le=20)
    volume_number: int | None = Field(default=None, ge=1)
    page_number: int | None = Field(default=None, ge=1)
    strata: list[SamplingReason]
    deterministic_order: Literal["sha256(dataset_id, source_ocr_region_id)"] = (
        "sha256(dataset_id, source_ocr_region_id)"
    )


class PacketOCRRegion(StrictModel):
    source_ocr_run_id: UUID
    source_ocr_region_id: UUID
    kind: RegionKind
    reading_order: int = Field(ge=0)
    polygon: Polygon
    raw_text: str
    normalized_text: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    direction: Literal["vertical", "horizontal", "mixed", "unknown"]


class PacketPage(StrictModel):
    page_id: UUID
    page_key: str
    issue_id: str | None = None
    source: SourcePointer
    derivative_id: UUID
    image_uri: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    dpi: int | None = Field(default=None, gt=0)
    media_type: str = Field(pattern=r"^image/")
    evidence_tier: str
    render_manifest_uri: str | None = None
    source_ocr_run_id: UUID
    selection_basis: str


class NERAnnotationUnit(StrictModel):
    unit_id: UUID
    page_key: str
    source: SourcePointer
    target: PacketOCRRegion
    context_before: list[PacketOCRRegion] = Field(default_factory=list)
    context_after: list[PacketOCRRegion] = Field(default_factory=list)
    selection_reasons: list[SamplingReason] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target_source(self) -> "NERAnnotationUnit":
        if self.source.region_id != self.target.source_ocr_region_id:
            raise ValueError("annotation-unit source must cite its target OCR region")
        if self.source.polygon != self.target.polygon:
            raise ValueError("annotation-unit source polygon must equal its target polygon")
        return self


class PacketCoverage(StrictModel):
    units: int
    pages: int
    volumes: int
    decades: list[str]
    known_issues: int
    by_reason: dict[str, int]


class NERAnnotationPacket(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    packet_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: str = Field(min_length=1, max_length=300)
    status: Literal["annotation_candidate"] = "annotation_candidate"
    generated_at: datetime
    ontology_version: str = Field(min_length=1, max_length=100)
    sampling: PacketSamplingConfig
    coverage: PacketCoverage
    benchmark_eligible: bool
    eligibility_failures: list[str]
    pages: list[PacketPage] = Field(min_length=1)
    units: list[NERAnnotationUnit] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_packet(self) -> "NERAnnotationPacket":
        page_keys = [page.page_key for page in self.pages]
        unit_ids = [unit.unit_id for unit in self.units]
        if len(set(page_keys)) != len(page_keys):
            raise ValueError("packet page keys must be unique")
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("packet annotation-unit IDs must be unique")
        page_key_set = set(page_keys)
        if any(unit.page_key not in page_key_set for unit in self.units):
            raise ValueError("every annotation unit must reference a packet page")
        if self.benchmark_eligible == bool(self.eligibility_failures):
            raise ValueError(
                "benchmark_eligible must be true exactly when eligibility_failures is empty"
            )
        return self


class PacketUnitAdjudication(GoldAdjudication):
    gold_region_id: UUID
    page_genre: Literal[
        "news_editorial",
        "advertisement_classified",
        "mixed",
        "photograph_caption",
        "table_market_schedule",
        "front_matter_index",
        "blank_other",
    ]
    layout: Literal["vertical", "horizontal", "mixed", "unknown"]
    scan_quality: Literal["clean", "moderate", "poor", "unusable"]


class PacketUnitAnnotation(StrictModel):
    unit_id: UUID
    reviews: list[ReviewerAnnotation] = Field(min_length=2)
    adjudication: PacketUnitAdjudication


class PacketAnnotationSubmission(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    packet_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    instructions: list[str] = Field(default_factory=list)
    units: list[PacketUnitAnnotation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_units(self) -> "PacketAnnotationSubmission":
        unit_ids = [unit.unit_id for unit in self.units]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("annotation submission contains duplicate unit IDs")
        return self


ACTIVE_REGION_SQL = """
SELECT region.region_id, region.run_id AS source_ocr_run_id, region.page_id,
       region.region_kind, region.reading_order, region.polygon,
       region.raw_text, region.normalized_text, region.confidence,
       region.direction, page.page_number, volume.volume_number,
       volume.publication_year, source.source_uri,
       source.sha256 AS source_sha256, derivative.derivative_id,
       derivative.image_uri, derivative.image_sha256, derivative.width,
       derivative.height, derivative.dpi, derivative.media_type,
       derivative.evidence_tier, derivative.render_manifest_uri,
       selection.selection_basis
FROM evidence.ocr_region region
JOIN archive.page page USING (page_id)
JOIN archive.volume volume USING (volume_id)
JOIN archive.source_object source USING (source_object_id)
JOIN evidence.page_ocr_selection selection
  ON selection.page_id = region.page_id
 AND selection.run_id = region.run_id
 AND selection.superseded_at IS NULL
JOIN archive.page_derivative derivative
  ON derivative.derivative_id = selection.derivative_id
WHERE (CAST(%s AS integer) IS NULL OR volume.volume_number = CAST(%s AS integer))
  AND (CAST(%s AS integer) IS NULL OR page.page_number = CAST(%s AS integer))
ORDER BY volume.publication_year, volume.volume_number, page.page_number,
         region.reading_order, region.region_id
"""


MENTION_SQL = """
SELECT mention.region_id, mention.run_id, run.model_name,
       mention.entity_type, mention.text_start, mention.text_end
FROM evidence.entity_mention mention
JOIN evidence.processing_run run USING (run_id)
WHERE mention.region_id = ANY(%s::uuid[])
  AND mention.mention_status = 'candidate'
ORDER BY mention.region_id, run.model_name, mention.entity_type,
         mention.text_start, mention.text_end
"""


REASON_ORDER = (
    SamplingReason.WOMEN_THEME,
    SamplingReason.NER_DISAGREEMENT,
    SamplingReason.LOW_OCR_CONFIDENCE,
    SamplingReason.MEDIUM_OCR_CONFIDENCE,
    SamplingReason.HIGH_OCR_CONFIDENCE,
    SamplingReason.NO_CANDIDATE_BASELINE,
)


def _page_key(row: dict[str, Any]) -> str:
    return f"v{int(row['volume_number']):03d}-p{int(row['page_number']):04d}"


def _packet_region(row: dict[str, Any]) -> PacketOCRRegion:
    return PacketOCRRegion(
        source_ocr_run_id=row["source_ocr_run_id"],
        source_ocr_region_id=row["region_id"],
        kind=row["region_kind"],
        reading_order=row["reading_order"],
        polygon=row["polygon"],
        raw_text=row["raw_text"],
        normalized_text=row["normalized_text"],
        confidence=row["confidence"],
        direction=row["direction"],
    )


def _sampling_reasons(
    row: dict[str, Any],
    learned_run_sets: dict[str, set[tuple[str, int, int]]],
) -> list[SamplingReason]:
    text = row["normalized_text"] or row["raw_text"]
    reasons: list[SamplingReason] = []
    if any(re.search(pattern, text) for _, _, pattern, _ in THEMES):
        reasons.append(SamplingReason.WOMEN_THEME)
    nonempty_sets = [frozenset(values) for values in learned_run_sets.values()]
    if len(nonempty_sets) >= 2 and len(set(nonempty_sets)) > 1:
        reasons.append(SamplingReason.NER_DISAGREEMENT)
    if any(nonempty_sets):
        reasons.append(SamplingReason.NER_CANDIDATE)
    else:
        reasons.append(SamplingReason.NO_CANDIDATE_BASELINE)
    confidence = row["confidence"]
    if confidence is None:
        reasons.append(SamplingReason.UNKNOWN_OCR_CONFIDENCE)
    elif confidence < 0.5:
        reasons.append(SamplingReason.LOW_OCR_CONFIDENCE)
    elif confidence < 0.8:
        reasons.append(SamplingReason.MEDIUM_OCR_CONFIDENCE)
    else:
        reasons.append(SamplingReason.HIGH_OCR_CONFIDENCE)
    return reasons


def _stable_key(dataset_id: str, region_id: UUID) -> str:
    return hashlib.sha256(f"{dataset_id}:{region_id}".encode()).hexdigest()


def _select_rows(
    candidates: list[tuple[dict[str, Any], list[SamplingReason]]],
    dataset_id: str,
    max_units: int,
) -> list[tuple[dict[str, Any], list[SamplingReason]]]:
    ordered = sorted(
        candidates, key=lambda item: _stable_key(dataset_id, item[0]["region_id"])
    )
    quota = max(1, math.ceil(max_units / len(REASON_ORDER)))
    selected: list[tuple[dict[str, Any], list[SamplingReason]]] = []
    selected_ids: set[UUID] = set()
    for reason in REASON_ORDER:
        matches = [item for item in ordered if reason in item[1]]
        for item in matches[:quota]:
            region_id = item[0]["region_id"]
            if region_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(region_id)
            if len(selected) >= max_units:
                return selected
    for item in ordered:
        region_id = item[0]["region_id"]
        if region_id not in selected_ids:
            selected.append(item)
            selected_ids.add(region_id)
        if len(selected) >= max_units:
            break
    return selected


def _identity_payload(packet: NERAnnotationPacket) -> dict[str, Any]:
    return {
        "schema_version": packet.schema_version,
        "dataset_id": packet.dataset_id,
        "status": packet.status,
        "ontology_version": packet.ontology_version,
        "sampling": packet.sampling.model_dump(mode="json"),
        "coverage": packet.coverage.model_dump(mode="json"),
        "benchmark_eligible": packet.benchmark_eligible,
        "eligibility_failures": packet.eligibility_failures,
        "pages": [page.model_dump(mode="json") for page in packet.pages],
        "units": [unit.model_dump(mode="json") for unit in packet.units],
    }


def packet_identity(packet: NERAnnotationPacket) -> str:
    return hashlib.sha256(
        json.dumps(
            _identity_payload(packet),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def validate_packet_identity(packet: NERAnnotationPacket) -> None:
    expected = packet_identity(packet)
    if packet.packet_id != expected:
        raise ValueError(
            f"annotation packet identity mismatch: recorded {packet.packet_id}, expected {expected}"
        )


def build_packet_from_rows(
    rows: list[dict[str, Any]],
    mention_rows: list[dict[str, Any]],
    *,
    dataset_id: str,
    ontology_version: str,
    max_units: int,
    context_radius: int,
    volume_number: int | None = None,
    page_number: int | None = None,
    generated_at: datetime | None = None,
) -> NERAnnotationPacket:
    if not rows:
        raise ValueError("no active text-bearing OCR regions matched the packet scope")
    if page_number is not None and volume_number is None:
        raise ValueError("page_number requires volume_number")
    if not 1 <= max_units <= 5000:
        raise ValueError("max_units must be between 1 and 5000")
    if not 0 <= context_radius <= 20:
        raise ValueError("context_radius must be between 0 and 20")

    mention_sets: dict[UUID, dict[str, set[tuple[str, int, int]]]] = {}
    learned_model_names = {
        mention["model_name"]
        for mention in mention_rows
        if mention["model_name"] != "historical-women-zh-rules"
    }
    for mention in mention_rows:
        if mention["model_name"] == "historical-women-zh-rules":
            continue
        mention_sets.setdefault(mention["region_id"], {}).setdefault(
            mention["model_name"], set()
        ).add(
            (
                mention["entity_type"],
                mention["text_start"],
                mention["text_end"],
            )
        )

    text_rows = [row for row in rows if row["raw_text"].strip()]
    candidates = [
        (
            row,
            _sampling_reasons(
                row,
                {
                    model_name: mention_sets.get(row["region_id"], {}).get(
                        model_name, set()
                    )
                    for model_name in learned_model_names
                },
            ),
        )
        for row in text_rows
    ]
    selected = _select_rows(candidates, dataset_id, max_units)
    if not selected:
        raise ValueError("no text-bearing OCR regions are available for annotation")

    rows_by_page: dict[UUID, list[dict[str, Any]]] = {}
    for row in text_rows:
        rows_by_page.setdefault(row["page_id"], []).append(row)
    for page_rows in rows_by_page.values():
        page_rows.sort(key=lambda row: (row["reading_order"], str(row["region_id"])))
    indexes = {
        row["region_id"]: index
        for page_rows in rows_by_page.values()
        for index, row in enumerate(page_rows)
    }

    units = []
    selected_page_ids: set[UUID] = set()
    for row, reasons in selected:
        selected_page_ids.add(row["page_id"])
        page_rows = rows_by_page[row["page_id"]]
        index = indexes[row["region_id"]]
        target = _packet_region(row)
        source = SourcePointer(
            source_uri=row["source_uri"],
            source_sha256=row["source_sha256"],
            derivative_id=row["derivative_id"],
            image_sha256=row["image_sha256"],
            evidence_tier=row["evidence_tier"],
            volume_number=row["volume_number"],
            publication_year=row["publication_year"],
            page_number=row["page_number"],
            region_id=row["region_id"],
            polygon=row["polygon"],
            text_start=0,
            text_end=len(row["raw_text"]),
        )
        units.append(
            NERAnnotationUnit(
                unit_id=uuid5(
                    NAMESPACE_URL,
                    f"wic-annotation-unit:{dataset_id}:{row['source_ocr_run_id']}:{row['region_id']}",
                ),
                page_key=_page_key(row),
                source=source,
                target=target,
                context_before=[
                    _packet_region(context)
                    for context in page_rows[max(0, index - context_radius) : index]
                ],
                context_after=[
                    _packet_region(context)
                    for context in page_rows[index + 1 : index + 1 + context_radius]
                ],
                selection_reasons=reasons,
            )
        )

    first_by_page = {}
    for row in rows:
        if row["page_id"] in selected_page_ids:
            first_by_page.setdefault(row["page_id"], row)
    pages = []
    for row in sorted(
        first_by_page.values(),
        key=lambda item: (
            item["publication_year"],
            item["volume_number"],
            item["page_number"],
        ),
    ):
        pages.append(
            PacketPage(
                page_id=row["page_id"],
                page_key=_page_key(row),
                source=SourcePointer(
                    source_uri=row["source_uri"],
                    source_sha256=row["source_sha256"],
                    derivative_id=row["derivative_id"],
                    image_sha256=row["image_sha256"],
                    evidence_tier=row["evidence_tier"],
                    volume_number=row["volume_number"],
                    publication_year=row["publication_year"],
                    page_number=row["page_number"],
                ),
                derivative_id=row["derivative_id"],
                image_uri=row["image_uri"],
                image_sha256=row["image_sha256"],
                width=row["width"],
                height=row["height"],
                dpi=row["dpi"],
                media_type=row["media_type"],
                evidence_tier=row["evidence_tier"],
                render_manifest_uri=row["render_manifest_uri"],
                source_ocr_run_id=row["source_ocr_run_id"],
                selection_basis=row["selection_basis"],
            )
        )

    by_reason = {
        reason.value: sum(reason in unit.selection_reasons for unit in units)
        for reason in SamplingReason
    }
    by_reason = {key: value for key, value in by_reason.items() if value}
    years = {
        page.source.publication_year
        for page in pages
        if page.source.publication_year is not None
    }
    known_issues = len({page.issue_id for page in pages if page.issue_id})
    failures = []
    if len(units) < 500:
        failures.append(
            f"Packet has {len(units)} units; the frozen benchmark protocol requires at least 500."
        )
    if known_issues < 30:
        failures.append(
            "Issue/article identifiers are unavailable or cover fewer than 30 issues; issue-level splitting is not yet possible."
        )
    if len({(year // 10) * 10 for year in years}) < 3:
        failures.append("Packet covers fewer than three publication decades.")

    sampling = PacketSamplingConfig(
        max_units=max_units,
        context_radius=context_radius,
        volume_number=volume_number,
        page_number=page_number,
        strata=list(REASON_ORDER),
    )
    coverage = PacketCoverage(
        units=len(units),
        pages=len(pages),
        volumes=len({page.source.volume_number for page in pages}),
        decades=sorted({f"{(year // 10) * 10}s" for year in years}),
        known_issues=known_issues,
        by_reason=by_reason,
    )
    packet = NERAnnotationPacket(
        packet_id="0" * 64,
        dataset_id=dataset_id,
        generated_at=generated_at or datetime.now(timezone.utc),
        ontology_version=ontology_version,
        sampling=sampling,
        coverage=coverage,
        benchmark_eligible=not failures,
        eligibility_failures=failures,
        pages=pages,
        units=units,
        warnings=[
            "This is an annotation candidate packet, not gold and not a model-quality result.",
            "Independent reviewers must not see model outputs or each other's annotations.",
            "Selection is deliberately stratified; publish stratum counts and retain an unbiased baseline.",
            "Finalization requires two distinct reviews, adjudication, and a model-independent gold region UUID.",
        ],
    )
    return packet.model_copy(update={"packet_id": packet_identity(packet)})


def load_active_rows(
    database_url: str,
    *,
    volume_number: int | None = None,
    page_number: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            ACTIVE_REGION_SQL,
            (volume_number, volume_number, page_number, page_number),
        ).fetchall()
        region_ids = [row["region_id"] for row in rows]
        mentions = (
            connection.execute(MENTION_SQL, (region_ids,)).fetchall()
            if region_ids
            else []
        )
    return rows, mentions


def verify_packet_files(packet: NERAnnotationPacket, workspace_root: Path) -> None:
    validate_packet_identity(packet)
    for page in packet.pages:
        candidate = Path(page.image_uri)
        path = (
            candidate if candidate.is_absolute() else workspace_root / candidate
        ).resolve()
        artifact_root = (workspace_root / "artifacts").resolve()
        if not path.is_relative_to(artifact_root) or not path.is_file():
            raise ValueError(f"packet image is outside local artifacts or absent: {page.image_uri}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != page.image_sha256:
            raise ValueError(f"packet image hash mismatch: {page.image_uri}")


def annotation_template(packet: NERAnnotationPacket) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "packet_id": packet.packet_id,
        "instructions": [
            "Create two independent review records without showing either reviewer model output or the other review.",
            "Use exact Unicode character offsets into each review's corrected_text and, when recoverable, raw OCR offsets.",
            "The adjudicator must assign a new gold_region_id that is not the source_ocr_region_id.",
            "Replace every null or empty placeholder before finalization.",
        ],
        "units": [
            {
                "unit_id": str(unit.unit_id),
                "reviews": [
                    {
                        "reviewer": "",
                        "corrected_text": "",
                        "entities": [],
                        "annotated_at": None,
                        "notes": None,
                    },
                    {
                        "reviewer": "",
                        "corrected_text": "",
                        "entities": [],
                        "annotated_at": None,
                        "notes": None,
                    },
                ],
                "adjudication": {
                    "adjudicator": "",
                    "gold_region_id": None,
                    "corrected_text": "",
                    "entities": [],
                    "adjudicated_at": None,
                    "page_genre": None,
                    "layout": None,
                    "scan_quality": None,
                    "notes": None,
                },
            }
            for unit in packet.units
        ],
    }


def blinded_reviewer_view(packet: NERAnnotationPacket) -> dict[str, Any]:
    """Return evidence/context for reviewers without sampling/model signals."""
    validate_packet_identity(packet)
    return {
        "schema_version": "1.0",
        "packet_id": packet.packet_id,
        "status": "blinded_annotation_view",
        "dataset_id": packet.dataset_id,
        "ontology_version": packet.ontology_version,
        "instructions": [
            "Annotate only the target text; use adjacent regions and the cited scan for context.",
            "Do not seek model output or the other reviewer's work before submitting your independent pass.",
            "Treat raw OCR as a fallible transcript and preserve exact corrected/raw Unicode offsets.",
        ],
        "pages": [page.model_dump(mode="json") for page in packet.pages],
        "units": [
            {
                "unit_id": str(unit.unit_id),
                "page_key": unit.page_key,
                "source": unit.source.model_dump(mode="json"),
                "target": unit.target.model_dump(mode="json"),
                "context_before": [
                    region.model_dump(mode="json") for region in unit.context_before
                ],
                "context_after": [
                    region.model_dump(mode="json") for region in unit.context_after
                ],
            }
            for unit in packet.units
        ],
    }


def finalize_packet(
    packet: NERAnnotationPacket,
    submission: PacketAnnotationSubmission,
) -> NERGoldSet:
    validate_packet_identity(packet)
    if submission.packet_id != packet.packet_id:
        raise ValueError("annotation submission targets a different packet")
    units_by_id = {unit.unit_id: unit for unit in packet.units}
    submitted_by_id = {unit.unit_id: unit for unit in submission.units}
    missing = sorted(str(unit_id) for unit_id in units_by_id.keys() - submitted_by_id.keys())
    extra = sorted(str(unit_id) for unit_id in submitted_by_id.keys() - units_by_id.keys())
    if missing or extra:
        raise ValueError(f"annotation submission unit mismatch: missing={missing}, extra={extra}")

    snippets = []
    for unit in packet.units:
        annotation = submitted_by_id[unit.unit_id]
        if annotation.adjudication.gold_region_id == unit.target.source_ocr_region_id:
            raise ValueError(
                f"unit {unit.unit_id} reuses its model OCR region UUID as gold identity"
            )
        snippets.append(
            GoldSnippet(
                snippet_id=str(unit.unit_id),
                gold_region_id=annotation.adjudication.gold_region_id,
                source_ocr_run_id=unit.target.source_ocr_run_id,
                source_ocr_region_id=unit.target.source_ocr_region_id,
                source=unit.source,
                raw_ocr_text=unit.target.raw_text,
                page_genre=annotation.adjudication.page_genre,
                layout=annotation.adjudication.layout,
                scan_quality=annotation.adjudication.scan_quality,
                reviews=annotation.reviews,
                adjudication=GoldAdjudication(
                    adjudicator=annotation.adjudication.adjudicator,
                    corrected_text=annotation.adjudication.corrected_text,
                    entities=annotation.adjudication.entities,
                    adjudicated_at=annotation.adjudication.adjudicated_at,
                    notes=annotation.adjudication.notes,
                ),
            )
        )
    return NERGoldSet(
        schema_version="1.1",
        dataset_id=packet.dataset_id,
        created_at=max(snippet.adjudication.adjudicated_at for snippet in snippets),
        ontology_version=packet.ontology_version,
        snippets=snippets,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build an annotation-candidate packet")
    build.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--ontology-version", default="women-history-zh-v1")
    build.add_argument("--max-units", type=int, default=50)
    build.add_argument("--context-radius", type=int, default=2)
    build.add_argument("--volume", type=int)
    build.add_argument("--page", type=int)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--template", type=Path, required=True)
    build.add_argument("--reviewer-view", type=Path, required=True)
    finalize = subparsers.add_parser("finalize", help="Validate reviews and create NER gold")
    finalize.add_argument("--packet", type=Path, required=True)
    finalize.add_argument("--annotations", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        if not args.database_url:
            raise SystemExit("--database-url or DATABASE_URL is required")
        rows, mentions = load_active_rows(
            args.database_url, volume_number=args.volume, page_number=args.page
        )
        packet = build_packet_from_rows(
            rows,
            mentions,
            dataset_id=args.dataset_id,
            ontology_version=args.ontology_version,
            max_units=args.max_units,
            context_radius=args.context_radius,
            volume_number=args.volume,
            page_number=args.page,
        )
        verify_packet_files(packet, Path.cwd())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.template.parent.mkdir(parents=True, exist_ok=True)
        args.reviewer_view.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            packet.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        args.template.write_text(
            json.dumps(
                annotation_template(packet),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        args.reviewer_view.write_text(
            json.dumps(
                blinded_reviewer_view(packet),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "packet": str(args.output),
                    "template": str(args.template),
                    "reviewer_view": str(args.reviewer_view),
                    "packet_id": packet.packet_id,
                    "coverage": packet.coverage.model_dump(mode="json"),
                    "benchmark_eligible": packet.benchmark_eligible,
                },
                sort_keys=True,
            )
        )
        return 0

    try:
        packet = NERAnnotationPacket.model_validate_json(
            args.packet.read_text(encoding="utf-8")
        )
        submission = PacketAnnotationSubmission.model_validate_json(
            args.annotations.read_text(encoding="utf-8")
        )
    except ValidationError as exc:
        examples = [
            {
                "location": ".".join(str(part) for part in error["loc"]),
                "message": error["msg"],
            }
            for error in exc.errors()[:10]
        ]
        raise SystemExit(
            json.dumps(
                {
                    "error": "annotation packet or submission is incomplete or invalid",
                    "validation_error_count": exc.error_count(),
                    "examples": examples,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        ) from None
    gold = finalize_packet(packet, submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(gold.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dataset_id": gold.dataset_id,
                "snippets": len(gold.snippets),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
