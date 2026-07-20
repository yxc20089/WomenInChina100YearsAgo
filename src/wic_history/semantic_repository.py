"""Article-scoped semantic inputs and persistence for the selected Qwen model."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import NAMESPACE_URL, UUID, uuid5

from .e2e_store import (
    EventEvidenceSpec,
    EventParticipantSpec,
    EventResult,
    propose_event,
)
from .model_config import load_pipeline_model_configuration
from .reviewed_text_materializer import (
    ReviewedSpanInput,
    materialize_reviewed_article,
)
from .semantic_tasks import (
    EventFrameResponse,
    LiteralInput,
    LocalMentionInput,
    LocalCoreferenceResponse,
    LocalResolutionResponse,
    MentionCandidateInput,
    MentionClassificationResponse,
    MentionDiscoveryItem,
    PageImageInput,
    ResolutionMentionInput,
    SemanticExtractionResponse,
    SemanticTextSegmentInput,
    SemanticTaskResult,
    TriggerInput,
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _clients() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    return psycopg, dict_row


@dataclass(frozen=True, slots=True)
class CoherentTextSegment:
    sequence_number: int
    region_id: UUID
    page_id: UUID
    text_version_id: UUID
    selection_id: UUID
    text_start: int
    text_end: int
    composite_start: int
    composite_end: int
    text: str
    role: str
    polygon: Any


@dataclass(frozen=True, slots=True)
class PageImageReference:
    page_id: UUID
    derivative_id: UUID
    image_uri: str
    image_sha256: str
    media_type: str
    width: int
    height: int
    region_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class CoherentTextBundle:
    coherent_unit_revision_id: UUID
    content: str
    input_sha256: str
    segments: tuple[CoherentTextSegment, ...]
    page_images: tuple[PageImageReference, ...]
    content_sha256: str = ""
    multimodal_input_sha256: str = ""


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    candidate: MentionCandidateInput
    region_id: UUID
    text_version_id: UUID
    text_start: int
    text_end: int


@dataclass(frozen=True, slots=True)
class CandidateRoster:
    coherent_unit_revision_id: UUID
    candidates: tuple[MentionCandidateInput, ...]
    records: tuple[CandidateRecord, ...]
    input_sha256: str

    def by_id(self) -> dict[UUID, CandidateRecord]:
        return {record.candidate.candidate_id: record for record in self.records}


@dataclass(frozen=True, slots=True)
class SemanticPersistResult:
    run_id: str
    records: int
    reused: bool


@dataclass(frozen=True, slots=True)
class SemanticExtractionPersistResult:
    run: SemanticPersistResult
    mention_ids: tuple[UUID, ...]
    events: tuple[EventResult, ...]


@dataclass(frozen=True, slots=True)
class EventInputBundle:
    triggers: tuple[TriggerInput, ...]
    literals: tuple[LiteralInput, ...]
    evidence_span_ids: tuple[UUID, ...]


def semantic_multimodal_input_sha256(bundle: CoherentTextBundle) -> str:
    """Hash every text-segment and page-image field visible to the semantic model."""
    identity = {
        "reviewed_text_input_sha256": bundle.input_sha256,
        "segments": [
            {
                "region_id": str(item.region_id),
                "page_id": str(item.page_id),
                "text_version_id": str(item.text_version_id),
                "text_start": item.text_start,
                "text_end": item.text_end,
                "text": item.text,
                "role": item.role,
                "polygon": item.polygon,
            }
            for item in bundle.segments
        ],
        "page_images": [
            {
                "page_id": str(item.page_id),
                "derivative_id": str(item.derivative_id),
                "image_uri": item.image_uri,
                "image_sha256": item.image_sha256,
                "media_type": item.media_type,
                "width": item.width,
                "height": item.height,
                "region_ids": [str(value) for value in item.region_ids],
            }
            for item in bundle.page_images
        ],
    }
    return _canonical_sha256(identity)


def _map_boundary(operations: list[dict[str, Any]], boundary: int) -> int:
    for operation in operations:
        source_start = int(operation["source_start"])
        source_end = int(operation["source_end"])
        target_start = int(operation["target_start"])
        target_end = int(operation["target_end"])
        if boundary == source_start:
            return target_start
        if boundary == source_end:
            return target_end
        if source_start < boundary < source_end:
            if operation["operation"] != "equal":
                raise ValueError(
                    "coherent-unit split crosses an ambiguous text correction"
                )
            return target_start + (boundary - source_start)
    raise ValueError("coherent-unit offset is outside its text alignment")


def _selected_interval(
    raw_text: str,
    selected_text: str,
    raw_start: int,
    raw_end: int,
    operations: list[dict[str, Any]] | None,
) -> tuple[int, int]:
    if raw_start == 0 and raw_end == len(raw_text):
        return 0, len(selected_text)
    if raw_text == selected_text:
        return raw_start, raw_end
    if operations is None:
        raise ValueError("partial corrected coherent-unit span lacks an alignment")
    return _map_boundary(operations, raw_start), _map_boundary(operations, raw_end)


def load_reviewed_coherent_text(
    database_url: str, coherent_unit_revision_id: UUID
) -> CoherentTextBundle:
    """Materialize one active unit from explicitly selected reviewed text versions."""
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        revision = connection.execute(
            """
            SELECT revision.revision_id, revision.unit_kind
            FROM evidence.coherent_unit_revision revision
            JOIN evidence.page_article_segmentation_selection segmentation_selection
              ON segmentation_selection.selection_id = revision.approval_selection_id
             AND segmentation_selection.superseded_at IS NULL
            WHERE revision.revision_id = %s
              AND revision.superseded_at IS NULL
              AND revision.unit_kind = 'article'
            """,
            (coherent_unit_revision_id,),
        ).fetchone()
        if revision is None:
            raise ValueError(
                "semantic tasks require an active reviewed article revision"
            )
        span_count = connection.execute(
            "SELECT count(*) AS count FROM evidence.coherent_unit_span WHERE revision_id = %s",
            (coherent_unit_revision_id,),
        ).fetchone()["count"]
        rows = connection.execute(
            """
            SELECT span.sequence_number, span.region_id, region.page_id,
                   span.text_start AS raw_start, span.text_end AS raw_end,
                   span.role, region.raw_text, region.polygon,
                   selection.selection_id, version.text_version_id,
                   version.text_content, version.text_sha256,
                   raw_version.text_version_id AS raw_text_version_id,
                   alignment.operations,
                   derivative.derivative_id, derivative.image_uri,
                   derivative.image_sha256, derivative.media_type,
                   derivative.width, derivative.height
            FROM evidence.coherent_unit_span span
            JOIN evidence.ocr_region region USING (region_id)
            JOIN evidence.ocr_run_input ocr_input
              ON ocr_input.run_id = region.run_id
             AND ocr_input.page_id = region.page_id
            JOIN archive.page_derivative derivative
              ON derivative.derivative_id = ocr_input.derivative_id
             AND derivative.page_id = region.page_id
            JOIN evidence.region_text_selection selection
              ON selection.region_id = span.region_id
             AND selection.superseded_at IS NULL
            JOIN evidence.text_version version
              ON version.text_version_id = selection.text_version_id
             AND version.review_status = 'reviewed'
            JOIN evidence.text_version raw_version
              ON raw_version.region_id = span.region_id
             AND raw_version.variant = 'raw_ocr'
             AND raw_version.text_content = region.raw_text
            LEFT JOIN evidence.text_version_alignment alignment
              ON alignment.source_text_version_id = raw_version.text_version_id
             AND alignment.target_text_version_id = version.text_version_id
            WHERE span.revision_id = %s
            ORDER BY span.sequence_number
            """,
            (coherent_unit_revision_id,),
        ).fetchall()
    if len(rows) != span_count:
        raise ValueError(
            "every coherent-unit region requires selected text and its exact immutable OCR image"
        )
    if not rows:
        raise ValueError("semantic tasks require at least one reviewed text region")
    sources = tuple(
        ReviewedSpanInput(
            sequence_number=row["sequence_number"],
            region_id=row["region_id"],
            page_id=row["page_id"],
            raw_text=row["raw_text"],
            raw_start=row["raw_start"],
            raw_end=row["raw_end"],
            selected_text_version_id=row["text_version_id"],
            selected_text_sha256=row["text_sha256"],
            selection_id=row["selection_id"],
            selected_text=row["text_content"],
            role=row["role"],
            alignment_operations=(
                tuple(row["operations"]) if row["operations"] is not None else None
            ),
        )
        for row in rows
    )
    canonical = materialize_reviewed_article(
        coherent_unit_revision_id,
        revision["unit_kind"],
        sources,
    )
    rows_by_sequence = {row["sequence_number"]: row for row in rows}
    segments: list[CoherentTextSegment] = []
    image_rows: dict[UUID, dict[str, Any]] = {}
    image_regions: dict[UUID, list[UUID]] = {}
    for span in canonical.spans:
        row = rows_by_sequence[span.sequence_number]
        segments.append(
            CoherentTextSegment(
                sequence_number=span.sequence_number,
                region_id=span.region_id,
                page_id=span.page_id,
                text_version_id=span.selected_text_version_id,
                selection_id=span.selection_id,
                text_start=span.selected_start,
                text_end=span.selected_end,
                composite_start=span.composite_start,
                composite_end=span.composite_end,
                text=span.text,
                role=span.role,
                polygon=row["polygon"],
            )
        )
        image_rows.setdefault(row["derivative_id"], row)
        region_ids = image_regions.setdefault(row["derivative_id"], [])
        if row["region_id"] not in region_ids:
            region_ids.append(row["region_id"])
    page_images = tuple(
        PageImageReference(
            page_id=row["page_id"],
            derivative_id=derivative_id,
            image_uri=row["image_uri"],
            image_sha256=row["image_sha256"],
            media_type=row["media_type"],
            width=row["width"],
            height=row["height"],
            region_ids=tuple(image_regions[derivative_id]),
        )
        for derivative_id, row in image_rows.items()
    )
    bundle = CoherentTextBundle(
        coherent_unit_revision_id=coherent_unit_revision_id,
        content=canonical.content,
        input_sha256=canonical.input_sha256,
        segments=tuple(segments),
        page_images=page_images,
        content_sha256=canonical.content_sha256,
    )
    return replace(
        bundle,
        multimodal_input_sha256=semantic_multimodal_input_sha256(bundle),
    )


def _segment_for_composite_span(
    bundle: CoherentTextBundle, start: int, end: int
) -> CoherentTextSegment:
    matches = [
        segment
        for segment in bundle.segments
        if segment.composite_start <= start and end <= segment.composite_end
    ]
    if len(matches) != 1:
        raise ValueError("candidate crosses a coherent-unit region boundary")
    return matches[0]


def semantic_multimodal_context(
    bundle: CoherentTextBundle,
) -> tuple[list[SemanticTextSegmentInput], list[PageImageInput]]:
    """Serialize reviewed text boxes and their exact immutable source derivatives."""
    segments = [
        SemanticTextSegmentInput(
            region_id=item.region_id,
            page_id=item.page_id,
            text_version_id=item.text_version_id,
            text_start=item.text_start,
            text_end=item.text_end,
            text=item.text,
            role=item.role,
            polygon=item.polygon,
        )
        for item in bundle.segments
    ]
    images = [
        PageImageInput(
            page_id=item.page_id,
            derivative_id=item.derivative_id,
            image_uri=item.image_uri,
            image_sha256=item.image_sha256,
            media_type=item.media_type,
            width=item.width,
            height=item.height,
            region_ids=list(item.region_ids),
        )
        for item in bundle.page_images
    ]
    return segments, images


def _segment_for_exact_span(
    bundle: CoherentTextBundle,
    *,
    region_id: UUID,
    text_start: int,
    text_end: int,
    surface: str,
) -> CoherentTextSegment:
    matches: list[CoherentTextSegment] = []
    for segment in bundle.segments:
        if (
            segment.region_id == region_id
            and segment.text_start <= text_start
            and text_end <= segment.text_end
        ):
            relative_start = text_start - segment.text_start
            relative_end = text_end - segment.text_start
            if segment.text[relative_start:relative_end] == surface:
                matches.append(segment)
    if len(matches) != 1:
        raise ValueError("semantic span must exactly match one reviewed text region")
    return matches[0]


def build_candidate_roster(
    database_url: str,
    bundle: CoherentTextBundle,
    discovered: Iterable[MentionDiscoveryItem],
    *,
    context_characters: int = 16,
) -> CandidateRoster:
    """Turn model surface/context into deterministic occurrence IDs and offsets."""
    from .e2e_store import locate_unique_surface

    if context_characters < 1:
        raise ValueError("context_characters must be positive")
    psycopg, dict_row = _clients()
    records: list[CandidateRecord] = []
    seen: set[tuple[UUID, int, int]] = set()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        for proposal in discovered:
            located = locate_unique_surface(
                bundle.content,
                proposal.surface,
                left_context=proposal.left_context,
                right_context=proposal.right_context,
            )
            segment = _segment_for_composite_span(
                bundle, located.text_start, located.text_end
            )
            local_start = (
                segment.text_start + located.text_start - segment.composite_start
            )
            local_end = local_start + len(proposal.surface)
            key = (segment.text_version_id, local_start, local_end)
            if key in seen:
                raise ValueError("candidate discovery duplicated one exact occurrence")
            seen.add(key)
            evidence_span_id = connection.execute(
                """
                INSERT INTO evidence.evidence_span (
                    text_version_id, text_start, text_end, surface_text,
                    surface_sha256, span_role
                ) VALUES (%s, %s, %s, %s, %s, 'mention_candidate')
                ON CONFLICT (text_version_id, text_start, text_end, surface_sha256)
                DO UPDATE SET surface_text = EXCLUDED.surface_text
                RETURNING evidence_span_id
                """,
                (
                    segment.text_version_id,
                    local_start,
                    local_end,
                    proposal.surface,
                    hashlib.sha256(proposal.surface.encode("utf-8")).hexdigest(),
                ),
            ).fetchone()["evidence_span_id"]
            candidate_id = uuid5(
                NAMESPACE_URL,
                f"wic-mention-candidate:{segment.text_version_id}:{local_start}:{local_end}",
            )
            candidate = MentionCandidateInput(
                candidate_id=candidate_id,
                evidence_span_id=evidence_span_id,
                surface=proposal.surface,
                left_context=bundle.content[
                    max(
                        segment.composite_start, located.text_start - context_characters
                    ) : located.text_start
                ],
                right_context=bundle.content[
                    located.text_end : min(
                        segment.composite_end, located.text_end + context_characters
                    )
                ],
            )
            records.append(
                CandidateRecord(
                    candidate,
                    segment.region_id,
                    segment.text_version_id,
                    local_start,
                    local_end,
                )
            )
    candidates = tuple(record.candidate for record in records)
    roster_identity = {
        "coherent_input_sha256": bundle.input_sha256,
        "candidates": [item.model_dump(mode="json") for item in candidates],
    }
    return CandidateRoster(
        bundle.coherent_unit_revision_id,
        candidates,
        tuple(records),
        _canonical_sha256(roster_identity),
    )


def _semantic_run_id(
    coherent_unit_revision_id: UUID,
    task: str,
    input_sha256: str,
    configuration_sha256: str,
    prompt_schema_sha256: str,
) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        ":".join(
            (
                "wic-semantic-run",
                str(coherent_unit_revision_id),
                task,
                input_sha256,
                configuration_sha256,
                prompt_schema_sha256,
            )
        ),
    )


def _register_semantic_run(
    connection: Any,
    *,
    coherent_unit_revision_id: UUID,
    task: str,
    input_sha256: str,
    result: SemanticTaskResult[Any],
    model_config_path: str | None,
    artifact_uri: str | None,
) -> tuple[UUID, bool]:
    configuration = load_pipeline_model_configuration(model_config_path)
    model = configuration.semantic
    run_id = _semantic_run_id(
        coherent_unit_revision_id,
        task,
        input_sha256,
        configuration.sha256,
        result.prompt_schema_sha256,
    )
    existing = connection.execute(
        """
        SELECT run.run_id, input.raw_output_sha256,
               input.metadata ->> 'prompt_sha256' AS prompt_sha256
        FROM evidence.processing_run run
        JOIN evidence.semantic_run_input input USING (run_id)
        WHERE run.run_id = %s
        """,
        (run_id,),
    ).fetchone()
    if existing is not None:
        if (
            existing["raw_output_sha256"] != result.raw_output_sha256
            or existing["prompt_sha256"] != result.prompt_sha256
        ):
            raise ValueError(
                "retry output differs from the immutable semantic task run"
            )
        return run_id, True
    now = datetime.now(timezone.utc)
    model_identity = model.provenance_identity()
    run_configuration = {
        "task": task,
        "pipeline_model_configuration_sha256": configuration.sha256,
        **model_identity,
        "temperature": model.temperature,
        "seed": model.seed,
        "prompt_sha256": result.prompt_sha256,
        "prompt_schema_sha256": result.prompt_schema_sha256,
        "response_format_sha256": result.response_format_sha256,
        "raw_output_sha256": result.raw_output_sha256,
        "finish_reason": result.finish_reason,
        "token_usage": {
            "prompt": result.prompt_tokens,
            "completion": result.completion_tokens,
            "total": result.total_tokens,
        },
        "whole_response_validation": "passed",
    }
    run_kind = (
        "ner"
        if task in {"mention_discovery", "mention_classification"}
        else ("relation" if task == "event_frames" else "entity_link")
    )
    connection.execute(
        """
        INSERT INTO evidence.processing_run (
            run_id, kind, engine, model_name, model_revision, software_version,
            configuration, status, started_at, completed_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, 'completed', %s, %s)
        """,
        (
            run_id,
            run_kind,
            f"structured-semantic:{model.provider}",
            model_identity["model_name"],
            model_identity["model_revision"],
            model_identity["runtime_version"],
            json.dumps(run_configuration, ensure_ascii=False),
            now,
            now,
        ),
    )
    connection.execute(
        """
        INSERT INTO evidence.semantic_run_input (
            run_id, coherent_unit_revision_id, task_kind, input_sha256,
            configuration_sha256, prompt_schema_sha256, artifact_uri, metadata,
            raw_output, raw_output_sha256
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
        """,
        (
            run_id,
            coherent_unit_revision_id,
            task,
            input_sha256,
            configuration.sha256,
            result.prompt_schema_sha256,
            artifact_uri,
            json.dumps(
                {
                    "semantic_task": result.task,
                    "prompt_sha256": result.prompt_sha256,
                    "raw_output_sha256": result.raw_output_sha256,
                }
            ),
            result.raw_output,
            result.raw_output_sha256,
        ),
    )
    return run_id, False


def persist_semantic_task_audit(
    database_url: str,
    bundle: CoherentTextBundle,
    result: SemanticTaskResult[Any],
    *,
    input_sha256: str,
    model_config_path: str | None = None,
    artifact_uri: str | None = None,
) -> SemanticPersistResult:
    """Register a validated discovery/semantic response even when it creates no facts."""
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        run_id, reused = _register_semantic_run(
            connection,
            coherent_unit_revision_id=bundle.coherent_unit_revision_id,
            task=result.task,
            input_sha256=input_sha256,
            result=result,
            model_config_path=model_config_path,
            artifact_uri=artifact_uri,
        )
    return SemanticPersistResult(str(run_id), 0, reused)


def load_local_mentions(
    database_url: str, bundle: CoherentTextBundle
) -> list[LocalMentionInput]:
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            """
            SELECT mention.mention_id, mention.entity_type, mention.mention_text,
                   span.evidence_span_id, span.text_start, span.text_end,
                   version.text_content
            FROM evidence.entity_mention mention
            JOIN evidence.evidence_span span USING (evidence_span_id)
            JOIN evidence.text_version version USING (text_version_id)
            WHERE mention.coherent_unit_revision_id = %s
              AND mention.mention_status <> 'rejected'
            ORDER BY mention.region_id, span.text_start, mention.mention_id
            """,
            (bundle.coherent_unit_revision_id,),
        ).fetchall()
    return [
        LocalMentionInput(
            mention_id=row["mention_id"],
            evidence_span_id=row["evidence_span_id"],
            surface=row["mention_text"],
            entity_type=row["entity_type"],
            left_context=row["text_content"][
                max(0, row["text_start"] - 24) : row["text_start"]
            ],
            right_context=row["text_content"][row["text_end"] : row["text_end"] + 24],
        )
        for row in rows
    ]


def load_resolution_mentions(
    database_url: str,
    bundle: CoherentTextBundle,
    mention_ids: Iterable[UUID],
) -> list[ResolutionMentionInput]:
    """Load exactly the durable occurrences accepted from extraction call 1."""
    expected = tuple(mention_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("validated extraction mention IDs must be unique")
    if not expected:
        return []
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            """
            SELECT mention.mention_id, mention.entity_type, mention.mention_text,
                   mention.mention_form, mention.region_id, region.page_id,
                   span.evidence_span_id, span.text_start, span.text_end,
                   span.text_version_id, version.text_content
            FROM evidence.entity_mention mention
            JOIN evidence.ocr_region region USING (region_id)
            JOIN evidence.evidence_span span USING (evidence_span_id)
            JOIN evidence.text_version version USING (text_version_id)
            WHERE mention.coherent_unit_revision_id = %s
              AND mention.mention_id = ANY(%s)
              AND mention.mention_status <> 'rejected'
            ORDER BY mention.region_id, span.text_start, mention.mention_id
            """,
            (bundle.coherent_unit_revision_id, list(expected)),
        ).fetchall()
    observed = {row["mention_id"] for row in rows}
    if observed != set(expected) or len(rows) != len(expected):
        raise ValueError(
            "local resolution input differs from validated extraction occurrences"
        )
    return [
        ResolutionMentionInput(
            mention_id=row["mention_id"],
            evidence_span_id=row["evidence_span_id"],
            region_id=row["region_id"],
            page_id=row["page_id"],
            text_version_id=row["text_version_id"],
            text_start=row["text_start"],
            text_end=row["text_end"],
            surface=row["mention_text"],
            entity_type=row["entity_type"],
            mention_form=row["mention_form"],
            left_context=row["text_content"][
                max(0, row["text_start"] - 24) : row["text_start"]
            ],
            right_context=row["text_content"][row["text_end"] : row["text_end"] + 24],
        )
        for row in rows
    ]


def _insert_span(
    connection: Any,
    segment: CoherentTextSegment,
    local_start: int,
    local_end: int,
    surface: str,
    role: str,
) -> UUID:
    return connection.execute(
        """
        INSERT INTO evidence.evidence_span (
            text_version_id, text_start, text_end, surface_text,
            surface_sha256, span_role
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (text_version_id, text_start, text_end, surface_sha256)
        DO UPDATE SET surface_text = EXCLUDED.surface_text
        RETURNING evidence_span_id
        """,
        (
            segment.text_version_id,
            local_start,
            local_end,
            surface,
            hashlib.sha256(surface.encode("utf-8")).hexdigest(),
            role,
        ),
    ).fetchone()["evidence_span_id"]


def build_event_inputs(
    database_url: str,
    bundle: CoherentTextBundle,
    mentions: Iterable[LocalMentionInput],
) -> EventInputBundle:
    """Deterministically enumerate bounded triggers, literals, and sentence evidence."""
    trigger_pattern = re.compile(r"召|入|至|往|任|生|卒|婚|演|刊|居|訪|赴|聘|就讀")
    date_pattern = re.compile(
        r"(?:民國|光緒|宣統)?[〇零一二三四五六七八九十百廿卅元]+年"
        r"(?:[〇零一二三四五六七八九十百廿卅元]+月)?"
        r"(?:[〇零一二三四五六七八九十百廿卅初]+日)?"
    )
    location_pattern = re.compile(
        r"宮中|[\u3400-\u9fff]{1,10}(?:戲院|學校|女校|公司|公會|會館|醫院|法院|市|縣|省)"
    )
    aspect_pattern = re.compile(r"時|曾|屢|常|再|正在|已|未")
    triggers: list[TriggerInput] = []
    literals: list[LiteralInput] = []
    evidence_ids: set[UUID] = {item.evidence_span_id for item in mentions}
    literal_seen: set[tuple[UUID, int, int, str]] = set()
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        for segment in bundle.segments:
            for sentence_match in re.finditer(r"[^。！？\n]+[。！？]?", segment.text):
                surface = sentence_match.group(0)
                if not surface.strip():
                    continue
                start = segment.text_start + sentence_match.start()
                end = segment.text_start + sentence_match.end()
                evidence_ids.add(
                    _insert_span(
                        connection, segment, start, end, surface, "event_context"
                    )
                )
            for trigger_match in trigger_pattern.finditer(segment.text):
                start = segment.text_start + trigger_match.start()
                end = segment.text_start + trigger_match.end()
                surface = trigger_match.group(0)
                evidence_span_id = _insert_span(
                    connection, segment, start, end, surface, "event_trigger"
                )
                evidence_ids.add(evidence_span_id)
                triggers.append(
                    TriggerInput(
                        trigger_id=uuid5(
                            NAMESPACE_URL,
                            f"wic-trigger:{segment.text_version_id}:{start}:{end}",
                        ),
                        evidence_span_id=evidence_span_id,
                        surface=surface,
                        left_context=segment.text[
                            max(0, trigger_match.start() - 16) : trigger_match.start()
                        ],
                        right_context=segment.text[
                            trigger_match.end() : trigger_match.end() + 16
                        ],
                    )
                )
            for pattern, kind in (
                (date_pattern, "date"),
                (location_pattern, "location"),
                (aspect_pattern, "aspect"),
            ):
                for match in pattern.finditer(segment.text):
                    start = segment.text_start + match.start()
                    end = segment.text_start + match.end()
                    key = (segment.text_version_id, start, end, kind)
                    if key in literal_seen:
                        continue
                    literal_seen.add(key)
                    surface = match.group(0)
                    evidence_span_id = _insert_span(
                        connection, segment, start, end, surface, f"event_{kind}"
                    )
                    evidence_ids.add(evidence_span_id)
                    literals.append(
                        LiteralInput(
                            literal_id=uuid5(
                                NAMESPACE_URL,
                                f"wic-literal:{kind}:{segment.text_version_id}:{start}:{end}",
                            ),
                            evidence_span_id=evidence_span_id,
                            surface=surface,
                            literal_kind=kind,
                        )
                    )
    return EventInputBundle(
        tuple(triggers), tuple(literals), tuple(sorted(evidence_ids))
    )


def _durable_mention_id(
    coherent_unit_revision_id: UUID,
    segment: CoherentTextSegment,
    text_start: int,
    text_end: int,
) -> UUID:
    """One durable ID per reviewed-unit occurrence; equal surfaces stay separate."""
    return uuid5(
        NAMESPACE_URL,
        (
            "wic-semantic-mention:"
            f"{coherent_unit_revision_id}:{segment.text_version_id}:{text_start}:{text_end}"
        ),
    )


def persist_semantic_extraction(
    database_url: str,
    bundle: CoherentTextBundle,
    result: SemanticTaskResult[SemanticExtractionResponse],
    *,
    model_config_path: str | None = None,
    artifact_uri: str | None = None,
) -> SemanticExtractionPersistResult:
    """Persist call 1 only after independently rechecking every exact span."""
    if result.task != "semantic_extraction":
        raise ValueError("expected the combined semantic extraction task")
    response = result.response
    mention_keys = [item.mention_key for item in response.mentions]
    evidence_keys = [item.evidence_key for item in response.event_evidence]
    if len(mention_keys) != len(set(mention_keys)):
        raise ValueError("semantic extraction mention keys must be unique")
    if len(evidence_keys) != len(set(evidence_keys)):
        raise ValueError("semantic extraction evidence keys must be unique")

    mention_segments = {
        item.mention_key: _segment_for_exact_span(
            bundle,
            region_id=item.region_id,
            text_start=item.text_start,
            text_end=item.text_end,
            surface=item.surface,
        )
        for item in response.mentions
    }
    evidence_segments = {
        item.evidence_key: _segment_for_exact_span(
            bundle,
            region_id=item.region_id,
            text_start=item.text_start,
            text_end=item.text_end,
            surface=item.surface,
        )
        for item in response.event_evidence
    }
    mention_occurrences = [
        (item.region_id, item.text_start, item.text_end) for item in response.mentions
    ]
    if len(mention_occurrences) != len(set(mention_occurrences)):
        raise ValueError("semantic extraction duplicated one exact mention occurrence")

    event_evidence_by_key = {
        item.evidence_key: item for item in response.event_evidence
    }
    mention_key_set = set(mention_keys)
    referenced_evidence: set[str] = set()
    for event in response.events:
        participant_keys = [item.mention_key for item in event.participant_decisions]
        if (
            len(participant_keys) != len(set(participant_keys))
            or not set(participant_keys) <= mention_key_set
            or len(event.evidence_keys) != len(set(event.evidence_keys))
            or not set(event.evidence_keys) <= set(event_evidence_by_key)
            or event.trigger_evidence_key not in event.evidence_keys
            or event_evidence_by_key[event.trigger_evidence_key].evidence_role
            != "event_trigger"
        ):
            raise ValueError("semantic event contains an invalid response-local key")
        for key, role in (
            (event.date_evidence_key, "event_date"),
            (event.location_evidence_key, "event_location"),
            (event.aspect_evidence_key, "event_aspect"),
        ):
            if key is not None and (
                key not in event.evidence_keys
                or event_evidence_by_key[key].evidence_role != role
            ):
                raise ValueError(
                    "semantic event literal has the wrong exact evidence role"
                )
        referenced_evidence.update(event.evidence_keys)
    if referenced_evidence != set(event_evidence_by_key):
        raise ValueError("semantic extraction event evidence must all be referenced")

    psycopg, dict_row = _clients()
    mention_ids: dict[str, UUID] = {}
    mention_evidence: dict[str, UUID] = {}
    event_evidence: dict[str, UUID] = {}
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        # The current append-only schema names this broad task kind
        # mention_classification; metadata preserves the combined call's true task.
        run_id, reused = _register_semantic_run(
            connection,
            coherent_unit_revision_id=bundle.coherent_unit_revision_id,
            task="mention_classification",
            input_sha256=(
                bundle.multimodal_input_sha256 or bundle.input_sha256
            ),
            result=result,
            model_config_path=model_config_path,
            artifact_uri=artifact_uri,
        )
        for mention in response.mentions:
            segment = mention_segments[mention.mention_key]
            evidence_span_id = _insert_span(
                connection,
                segment,
                mention.text_start,
                mention.text_end,
                mention.surface,
                "mention",
            )
            mention_id = _durable_mention_id(
                bundle.coherent_unit_revision_id,
                segment,
                mention.text_start,
                mention.text_end,
            )
            mention_ids[mention.mention_key] = mention_id
            mention_evidence[mention.mention_key] = evidence_span_id
            connection.execute(
                """
                INSERT INTO evidence.entity_mention (
                    mention_id, region_id, run_id, entity_type, mention_text,
                    text_start, text_end, confidence, mention_status,
                    attributes, evidence_span_id, coherent_unit_revision_id,
                    mention_form
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, NULL, 'candidate',
                    %s::jsonb, %s, %s, %s
                )
                ON CONFLICT (mention_id) DO NOTHING
                """,
                (
                    mention_id,
                    mention.region_id,
                    run_id,
                    mention.entity_type.value,
                    mention.surface,
                    mention.text_start,
                    mention.text_end,
                    json.dumps(
                        {
                            "response_local_mention_key": mention.mention_key,
                            "prompt_sha256": result.prompt_sha256,
                            "raw_output_sha256": result.raw_output_sha256,
                            "occurrence_only": True,
                            "global_alias_or_merge_proposed": False,
                        },
                        ensure_ascii=False,
                    ),
                    evidence_span_id,
                    bundle.coherent_unit_revision_id,
                    mention.mention_form,
                ),
            )
        for item in response.event_evidence:
            event_evidence[item.evidence_key] = _insert_span(
                connection,
                evidence_segments[item.evidence_key],
                item.text_start,
                item.text_end,
                item.surface,
                item.evidence_role,
            )

    event_results: list[EventResult] = []
    for event in response.events:
        location_literal = (
            event_evidence_by_key[event.location_evidence_key].surface
            if event.location_evidence_key is not None
            else None
        )
        aspect_literal = (
            event_evidence_by_key[event.aspect_evidence_key].surface
            if event.aspect_evidence_key is not None
            else None
        )
        date_literal = (
            event_evidence_by_key[event.date_evidence_key].surface
            if event.date_evidence_key is not None
            else None
        )
        event_results.append(
            propose_event(
                database_url,
                run_id=run_id,
                event_type=event.event_type,
                trigger_evidence_span_id=event_evidence[event.trigger_evidence_key],
                participants=[
                    EventParticipantSpec(
                        participant_role=item.participant_role,
                        mention_id=mention_ids[item.mention_key],
                        evidence_span_id=mention_evidence[item.mention_key],
                    )
                    for item in event.participant_decisions
                ],
                evidence=[
                    EventEvidenceSpec(evidence_span_id=event_evidence[key])
                    for key in event.evidence_keys
                ],
                coherent_unit_revision_id=bundle.coherent_unit_revision_id,
                location_literal=location_literal,
                aspect=aspect_literal,
                attributes={
                    "date_literal": date_literal,
                    "response_local_event_key": event.event_key,
                    "prompt_sha256": result.prompt_sha256,
                    "raw_output_sha256": result.raw_output_sha256,
                },
                event_id=uuid5(run_id, f"event:{event.event_key}"),
            )
        )
    ordered_ids = tuple(mention_ids[item.mention_key] for item in response.mentions)
    return SemanticExtractionPersistResult(
        run=SemanticPersistResult(str(run_id), len(ordered_ids), reused),
        mention_ids=ordered_ids,
        events=tuple(event_results),
    )


def persist_mention_classification(
    database_url: str,
    bundle: CoherentTextBundle,
    roster: CandidateRoster,
    result: SemanticTaskResult[MentionClassificationResponse],
    *,
    model_config_path: str | None = None,
    artifact_uri: str | None = None,
) -> SemanticPersistResult:
    decisions = {item.candidate_id: item for item in result.response.decisions}
    records = roster.by_id()
    if decisions.keys() != records.keys():
        raise ValueError("classification result and roster IDs differ")
    psycopg, dict_row = _clients()
    kept = [item for item in result.response.decisions if item.decision == "KEEP"]
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        run_id, reused = _register_semantic_run(
            connection,
            coherent_unit_revision_id=bundle.coherent_unit_revision_id,
            task="mention_classification",
            input_sha256=roster.input_sha256,
            result=result,
            model_config_path=model_config_path,
            artifact_uri=artifact_uri,
        )
        for decision in kept:
            record = records[decision.candidate_id]
            connection.execute(
                """
                INSERT INTO evidence.entity_mention (
                    mention_id, region_id, run_id, entity_type, mention_text,
                    text_start, text_end, confidence, mention_status,
                    attributes, evidence_span_id, coherent_unit_revision_id,
                    mention_form
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, NULL, 'candidate',
                    %s::jsonb, %s, %s, %s
                )
                ON CONFLICT (mention_id) DO NOTHING
                """,
                (
                    decision.candidate_id,
                    record.region_id,
                    run_id,
                    decision.entity_type.value,
                    record.candidate.surface,
                    record.text_start,
                    record.text_end,
                    json.dumps(
                        {
                            "candidate_roster_sha256": roster.input_sha256,
                            "prompt_sha256": result.prompt_sha256,
                            "raw_output_sha256": result.raw_output_sha256,
                            "candidate_only": True,
                        }
                    ),
                    record.candidate.evidence_span_id,
                    bundle.coherent_unit_revision_id,
                    decision.mention_form,
                ),
            )
    return SemanticPersistResult(str(run_id), len(kept), reused)


def persist_local_coreference(
    database_url: str,
    bundle: CoherentTextBundle,
    result: SemanticTaskResult[LocalCoreferenceResponse],
    *,
    model_config_path: str | None = None,
    artifact_uri: str | None = None,
) -> SemanticPersistResult:
    input_sha256 = result.prompt_sha256
    psycopg, dict_row = _clients()
    memberships = sum(len(cluster.mention_ids) for cluster in result.response.clusters)
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        supplied = {
            row["mention_id"]
            for row in connection.execute(
                """
                SELECT mention_id FROM evidence.entity_mention
                WHERE coherent_unit_revision_id = %s
                """,
                (bundle.coherent_unit_revision_id,),
            )
        }
        observed = {
            mention_id
            for cluster in result.response.clusters
            for mention_id in cluster.mention_ids
        }
        if not observed <= supplied:
            raise ValueError(
                "coreference result contains a mention outside its coherent unit"
            )
        run_id, reused = _register_semantic_run(
            connection,
            coherent_unit_revision_id=bundle.coherent_unit_revision_id,
            task="local_coreference",
            input_sha256=input_sha256,
            result=result,
            model_config_path=model_config_path,
            artifact_uri=artifact_uri,
        )
        local_run_id = uuid5(run_id, "local-coreference")
        connection.execute(
            """
            INSERT INTO evidence.local_coreference_run (
                local_coreference_run_id, processing_run_id,
                coherent_unit_revision_id, input_sha256, configuration_sha256
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (
                coherent_unit_revision_id, input_sha256, configuration_sha256
            ) DO NOTHING
            """,
            (
                local_run_id,
                run_id,
                bundle.coherent_unit_revision_id,
                input_sha256,
                load_pipeline_model_configuration(model_config_path).sha256,
            ),
        )
        for index, cluster in enumerate(result.response.clusters):
            cluster_id = uuid5(local_run_id, f"cluster:{index}")
            connection.execute(
                """
                INSERT INTO evidence.local_coreference_cluster (
                    local_cluster_id, local_coreference_run_id, review_status
                ) VALUES (%s, %s, 'candidate')
                ON CONFLICT (local_cluster_id) DO NOTHING
                """,
                (cluster_id, local_run_id),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO evidence.local_coreference_member (
                        local_cluster_id, local_coreference_run_id, mention_id
                    ) VALUES (%s, %s, %s)
                    ON CONFLICT (local_cluster_id, mention_id) DO NOTHING
                    """,
                    [
                        (cluster_id, local_run_id, mention_id)
                        for mention_id in cluster.mention_ids
                    ],
                )
    return SemanticPersistResult(str(run_id), memberships, reused)


def persist_local_resolution(
    database_url: str,
    bundle: CoherentTextBundle,
    result: SemanticTaskResult[LocalResolutionResponse],
    *,
    mention_ids: Iterable[UUID],
    model_config_path: str | None = None,
    artifact_uri: str | None = None,
) -> SemanticPersistResult:
    """Persist only article-scoped clusters; unresolved occurrences stay intact."""
    if result.task != "local_resolution":
        raise ValueError("expected the bounded local-resolution task")
    expected = tuple(mention_ids)
    expected_set = set(expected)
    memberships = [
        mention_id
        for cluster in result.response.clusters
        for mention_id in cluster.mention_ids
    ]
    unresolved = result.response.unresolved_mention_ids
    accounted = [*memberships, *unresolved]
    if (
        len(expected) != len(expected_set)
        or len(accounted) != len(set(accounted))
        or set(accounted) != expected_set
    ):
        raise ValueError(
            "local resolution must account for validated mentions exactly once"
        )

    input_sha256 = result.prompt_sha256
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        supplied = {
            row["mention_id"]
            for row in connection.execute(
                """
                SELECT mention_id FROM evidence.entity_mention
                WHERE coherent_unit_revision_id = %s
                  AND mention_id = ANY(%s)
                  AND mention_status <> 'rejected'
                """,
                (bundle.coherent_unit_revision_id, list(expected)),
            )
        }
        if supplied != expected_set:
            raise ValueError(
                "local resolution mentions differ from persisted extraction occurrences"
            )
        run_id, reused = _register_semantic_run(
            connection,
            coherent_unit_revision_id=bundle.coherent_unit_revision_id,
            task="local_coreference",
            input_sha256=input_sha256,
            result=result,
            model_config_path=model_config_path,
            artifact_uri=artifact_uri,
        )
        local_run_id = uuid5(run_id, "local-resolution")
        connection.execute(
            """
            INSERT INTO evidence.local_coreference_run (
                local_coreference_run_id, processing_run_id,
                coherent_unit_revision_id, input_sha256, configuration_sha256
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (
                coherent_unit_revision_id, input_sha256, configuration_sha256
            ) DO NOTHING
            """,
            (
                local_run_id,
                run_id,
                bundle.coherent_unit_revision_id,
                input_sha256,
                load_pipeline_model_configuration(model_config_path).sha256,
            ),
        )
        for index, cluster in enumerate(result.response.clusters):
            cluster_id = uuid5(
                local_run_id,
                f"cluster:{index}:"
                + ":".join(str(value) for value in cluster.mention_ids),
            )
            connection.execute(
                """
                INSERT INTO evidence.local_coreference_cluster (
                    local_cluster_id, local_coreference_run_id, review_status
                ) VALUES (%s, %s, 'candidate')
                ON CONFLICT (local_cluster_id) DO NOTHING
                """,
                (cluster_id, local_run_id),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO evidence.local_coreference_member (
                        local_cluster_id, local_coreference_run_id, mention_id
                    ) VALUES (%s, %s, %s)
                    ON CONFLICT (local_cluster_id, mention_id) DO NOTHING
                    """,
                    [
                        (cluster_id, local_run_id, mention_id)
                        for mention_id in cluster.mention_ids
                    ],
                )
    return SemanticPersistResult(str(run_id), len(memberships), reused)


def persist_event_frames(
    database_url: str,
    bundle: CoherentTextBundle,
    result: SemanticTaskResult[EventFrameResponse],
    *,
    triggers: Iterable[TriggerInput],
    literals: Iterable[LiteralInput],
    model_config_path: str | None = None,
    artifact_uri: str | None = None,
) -> tuple[SemanticPersistResult, tuple[EventResult, ...]]:
    trigger_map = {item.trigger_id: item for item in triggers}
    literal_map = {item.literal_id: item for item in literals}
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        run_id, reused = _register_semantic_run(
            connection,
            coherent_unit_revision_id=bundle.coherent_unit_revision_id,
            task="event_frames",
            input_sha256=result.prompt_sha256,
            result=result,
            model_config_path=model_config_path,
            artifact_uri=artifact_uri,
        )
        mention_evidence = {
            row["mention_id"]: row["evidence_span_id"]
            for row in connection.execute(
                """
                SELECT mention_id, evidence_span_id
                FROM evidence.entity_mention
                WHERE coherent_unit_revision_id = %s
                """,
                (bundle.coherent_unit_revision_id,),
            )
        }
    events: list[EventResult] = []
    for index, frame in enumerate(result.response.events):
        trigger = trigger_map[frame.trigger_id]
        location_literal = (
            literal_map[frame.location_literal_id].surface
            if frame.location_literal_id is not None
            else None
        )
        date_literal = (
            literal_map[frame.date_literal_id].surface
            if frame.date_literal_id is not None
            else None
        )
        aspect_literal = (
            literal_map[frame.aspect_literal_id].surface
            if frame.aspect_literal_id is not None
            else None
        )
        event_id = uuid5(run_id, f"event:{index}:{frame.trigger_id}")
        events.append(
            propose_event(
                database_url,
                run_id=run_id,
                event_type=frame.event_type,
                trigger_evidence_span_id=trigger.evidence_span_id,
                participants=[
                    EventParticipantSpec(
                        participant_role=item.participant_role,
                        mention_id=item.mention_id,
                        evidence_span_id=mention_evidence[item.mention_id],
                    )
                    for item in frame.participant_decisions
                ],
                evidence=[
                    EventEvidenceSpec(evidence_span_id=evidence_id)
                    for evidence_id in frame.evidence_span_ids
                ],
                coherent_unit_revision_id=bundle.coherent_unit_revision_id,
                location_literal=location_literal,
                aspect=aspect_literal,
                attributes={
                    "date_literal": date_literal,
                    "prompt_sha256": result.prompt_sha256,
                    "raw_output_sha256": result.raw_output_sha256,
                },
                event_id=event_id,
            )
        )
    return SemanticPersistResult(str(run_id), len(events), reused), tuple(events)
