"""Versioned contracts shared by OCR, extraction, storage, and retrieval.

These models intentionally keep observation, machine candidate, and reviewed
assertion states distinct. They are the serialization boundary between pipeline
stages and must remain backward compatible once artifacts are produced.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any, Literal, assert_never
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)


SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point(StrictModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class Polygon(StrictModel):
    points: list[Point] = Field(min_length=3)


class SourcePointer(StrictModel):
    source_uri: str
    source_sha256: str | None = None
    page_id: UUID | None = None
    derivative_id: UUID | None = None
    image_uri: str | None = None
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_tier: Literal[
        "screening_derivative",
        "unreviewed_input",
        "non_gold_lossless_pilot",
        "historian_selected_gold",
    ] | None = None
    volume_number: int | None = Field(default=None, ge=1)
    publication_year: int | None = Field(default=None, ge=1800, le=2100)
    page_number: int = Field(ge=1)
    region_id: UUID | None = None
    text_version_id: UUID | None = None
    text_selection_id: UUID | None = None
    polygon: Polygon | None = None
    text_start: int | None = Field(default=None, ge=0)
    text_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "SourcePointer":
        if (self.text_start is None) != (self.text_end is None):
            raise ValueError("text_start and text_end must be provided together")
        if (
            self.text_start is not None
            and self.text_end is not None
            and self.text_end < self.text_start
        ):
            raise ValueError("text_end must be greater than or equal to text_start")
        return self


class RunKind(StrEnum):
    RENDER = "render"
    OCR = "ocr"
    LAYOUT = "layout"
    NORMALIZE = "normalize"
    NER = "ner"
    ENTITY_LINK = "entity_link"
    RELATION = "relation"
    EMBEDDING = "embedding"
    INDEX = "index"
    GRAPH_PROJECTION = "graph_projection"


class ProcessingRun(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    run_id: UUID = Field(default_factory=uuid4)
    kind: RunKind
    engine: str
    model_name: str
    model_revision: str
    software_version: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_timestamps(self) -> "ProcessingRun":
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self


class RegionKind(StrEnum):
    PAGE = "page"
    ARTICLE = "article"
    HEADLINE = "headline"
    TEXT = "text"
    ADVERTISEMENT = "advertisement"
    CLASSIFIED = "classified"
    TABLE = "table"
    PHOTOGRAPH = "photograph"
    ILLUSTRATION = "illustration"
    CAPTION = "caption"
    MARGINALIA = "marginalia"
    UNKNOWN = "unknown"


class LayoutRegionKind(StrEnum):
    PAGE = "page"
    PANEL = "panel"
    COLUMN = "column"
    TEXT_GROUP = "text_group"
    IMAGE = "image"
    TABLE = "table"
    RULE = "rule"
    OTHER = "other"


class LayoutRegion(StrictModel):
    layout_region_id: UUID = Field(default_factory=uuid4)
    parent_layout_region_id: UUID | None = None
    kind: LayoutRegionKind
    polygon: Polygon
    reading_order: int | None = Field(default=None, ge=0)
    direction: Literal["vertical", "horizontal", "mixed", "unknown"] = "unknown"
    source_method: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    boundary_evidence: dict[str, Any] = Field(default_factory=dict)
    engine_payload: dict[str, Any] = Field(default_factory=dict)


class LayoutPageArtifact(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    artifact_id: UUID = Field(default_factory=uuid4)
    source: SourcePointer
    image_uri: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    dpi: int | None = Field(default=None, gt=0)
    run: ProcessingRun
    regions: list[LayoutRegion]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_layout(self) -> "LayoutPageArtifact":
        if self.run.kind != RunKind.LAYOUT:
            raise ValueError("layout artifact processing run must have kind=layout")
        ids = {region.layout_region_id for region in self.regions}
        if len(ids) != len(self.regions):
            raise ValueError("layout region UUIDs must be unique")
        if any(
            region.parent_layout_region_id is not None
            and region.parent_layout_region_id not in ids
            for region in self.regions
        ):
            raise ValueError("layout parents must belong to the same artifact")
        return self


class OCRRegion(StrictModel):
    region_id: UUID = Field(default_factory=uuid4)
    parent_region_id: UUID | None = None
    layout_region_id: UUID | None = None
    kind: RegionKind
    polygon: Polygon
    reading_order: int = Field(ge=0)
    raw_text: str
    normalized_text: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    language: str = "zh-Hant"
    direction: Literal["vertical", "horizontal", "mixed", "unknown"] = "unknown"
    engine_payload: dict[str, Any] = Field(default_factory=dict)


class OCRPageArtifact(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    artifact_id: UUID = Field(default_factory=uuid4)
    source: SourcePointer
    image_uri: str
    image_sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    dpi: int | None = Field(default=None, gt=0)
    run: ProcessingRun
    regions: list[OCRRegion]
    warnings: list[str] = Field(default_factory=list)


class EntityType(StrEnum):
    PERSON = "person"
    ALIAS = "alias"
    KINSHIP_TERM = "kinship_term"
    PLACE = "place"
    ADDRESS = "address"
    ORGANIZATION = "organization"
    SCHOOL = "school"
    OCCUPATION = "occupation"
    ROLE_TITLE = "role_title"
    PUBLICATION = "publication"
    EVENT = "event"
    DATE = "date"
    PRODUCT = "product"
    ADVERTISEMENT = "advertisement"


class EntityMentionCandidate(StrictModel):
    mention_id: UUID = Field(default_factory=uuid4)
    entity_type: EntityType
    text: str = Field(min_length=1)
    normalized_text: str | None = None
    source: SourcePointer
    confidence: float | None = Field(default=None, ge=0, le=1)
    run_id: UUID
    attributes: dict[str, Any] = Field(default_factory=dict)


class NERArtifact(StrictModel):
    schema_version: Literal["1.0", "1.1"] = SCHEMA_VERSION
    artifact_id: UUID = Field(default_factory=uuid4)
    source_ocr_run_id: UUID
    input_variant: Literal["raw_ocr", "corrected_text", "multimodal_transcript"] | None = None
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dataset_id: str | None = Field(default=None, min_length=1, max_length=300)
    split_id: str | None = Field(default=None, min_length=1, max_length=100)
    ontology_version: str | None = Field(default=None, min_length=1, max_length=100)
    adapter_id: str | None = Field(default=None, min_length=1, max_length=200)
    prompt_schema_revision: str | None = Field(default=None, min_length=1, max_length=200)
    run: ProcessingRun
    mentions: list[EntityMentionCandidate]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_run_links(self) -> "NERArtifact":
        if self.run.kind != RunKind.NER:
            raise ValueError("NER artifact processing run must have kind=ner")
        if any(mention.run_id != self.run.run_id for mention in self.mentions):
            raise ValueError("all mentions must reference the artifact processing run")
        if any(mention.source.region_id is None for mention in self.mentions):
            raise ValueError("all mentions must reference an OCR region")
        if self.schema_version == "1.1" and any(
            value is None
            for value in (
                self.input_variant,
                self.input_sha256,
                self.dataset_id,
                self.split_id,
                self.ontology_version,
                self.adapter_id,
            )
        ):
            raise ValueError(
                "NER artifact schema 1.1 requires input identity, dataset/split, ontology and adapter"
            )
        return self


class EntityLinkCandidate(StrictModel):
    link_id: UUID = Field(default_factory=uuid4)
    mention_id: UUID
    entity_id: UUID | None = None
    authority_uri: str | None = None
    canonical_name: str
    entity_type: EntityType
    score: float = Field(ge=0, le=1)
    features: dict[str, float | str | bool | None] = Field(default_factory=dict)
    nil_candidate: bool = False
    run_id: UUID

    @model_validator(mode="after")
    def validate_target(self) -> "EntityLinkCandidate":
        if self.nil_candidate and (self.entity_id is not None or self.authority_uri is not None):
            raise ValueError("NIL link candidates cannot target an entity or authority URI")
        if not self.nil_candidate and self.entity_id is None and self.authority_uri is None:
            raise ValueError("non-NIL link candidates require an entity or authority URI")
        return self


class EntityLinkArtifact(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    artifact_id: UUID = Field(default_factory=uuid4)
    source_ner_run_id: UUID
    run: ProcessingRun
    links: list[EntityLinkCandidate]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_run_links(self) -> "EntityLinkArtifact":
        if self.run.kind != RunKind.ENTITY_LINK:
            raise ValueError("entity-link artifact processing run must have kind=entity_link")
        if any(link.run_id != self.run.run_id for link in self.links):
            raise ValueError("all link candidates must reference the artifact processing run")
        return self


class ClaimStatus(StrEnum):
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    DISPUTED = "disputed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ClaimCandidate(StrictModel):
    claim_id: UUID = Field(default_factory=uuid4)
    subject_entity_id: UUID
    predicate: str = Field(min_length=1)
    object_entity_id: UUID | None = None
    object_literal: dict[str, Any] | None = None
    event_date_start: date | None = None
    event_date_end: date | None = None
    status: ClaimStatus = ClaimStatus.CANDIDATE
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[SourcePointer] = Field(min_length=1)
    supporting_quote: str = Field(min_length=1)
    run_id: UUID

    @model_validator(mode="after")
    def exactly_one_object(self) -> "ClaimCandidate":
        if (self.object_entity_id is None) == (self.object_literal is None):
            raise ValueError("exactly one of object_entity_id and object_literal is required")
        if self.event_date_start and self.event_date_end and self.event_date_end < self.event_date_start:
            raise ValueError("event_date_end cannot precede event_date_start")
        return self


class ClaimArtifact(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    artifact_id: UUID = Field(default_factory=uuid4)
    run: ProcessingRun
    claims: list[ClaimCandidate]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_run_links(self) -> "ClaimArtifact":
        if self.run.kind != RunKind.RELATION:
            raise ValueError("claim artifact processing run must have kind=relation")
        if any(claim.run_id != self.run.run_id for claim in self.claims):
            raise ValueError("all claims must reference the artifact processing run")
        return self


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    LEXICAL = "lexical"
    DENSE = "dense"
    GRAPH = "graph"


class RetrievalSourceSpan(StrictModel):
    citation_id: str = ""
    document_id: UUID | None
    sequence_number: int = Field(ge=0)
    document_start: int = Field(ge=0)
    document_end: int = Field(ge=0)
    role: str = Field(min_length=1)
    source: SourcePointer

    @model_validator(mode="after")
    def validate_document_offsets(self) -> "RetrievalSourceSpan":
        if self.document_end < self.document_start:
            raise ValueError(
                "document_end must be greater than or equal to document_start"
            )
        citation_id = _retrieval_citation_id(
            self.document_id, self.sequence_number, self.source
        )
        if self.citation_id and self.citation_id != citation_id:
            raise ValueError(
                "citation_id must match deterministic provenance identity"
            )
        self.citation_id = citation_id
        return self


def _retrieval_citation_id(
    document_id: UUID | None, sequence_number: int, source: SourcePointer
) -> str:
    identity = "|".join(
        (
            str(document_id) if document_id is not None else "legacy-region",
            str(sequence_number),
            str(source.region_id) if source.region_id is not None else "no-region",
            (
                str(source.text_version_id)
                if source.text_version_id is not None
                else "no-text-version"
            ),
            source.source_uri,
            str(source.page_number),
            str(source.text_start) if source.text_start is not None else "no-start",
            str(source.text_end) if source.text_end is not None else "no-end",
        )
    )
    return f"citation:{uuid5(NAMESPACE_URL, identity)}"


class RetrievalHit(StrictModel):
    rank: int = Field(ge=1)
    score: float
    target_kind: Literal["region", "reviewed_coherent_unit"] = "region"
    target_id: UUID | None = None
    coherent_unit_id: UUID | None = None
    source: SourcePointer | None = None
    sources: list[RetrievalSourceSpan] = Field(default_factory=list)
    text: str
    normalized_text: str | None = None
    entity_ids: list[UUID] = Field(default_factory=list)
    claim_ids: list[UUID] = Field(default_factory=list)
    explanation: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_target_provenance(self) -> "RetrievalHit":
        match self.target_kind:
            case "region":
                if self.source is None:
                    raise ValueError("region hits require a singular source")
                if self.target_id is None and self.source.region_id is not None:
                    self.target_id = self.source.region_id
                if (
                    self.source.region_id is not None
                    and self.target_id != self.source.region_id
                ):
                    raise ValueError("region target_id must match source.region_id")
                if self.coherent_unit_id is not None:
                    raise ValueError("region hits cannot have a coherent_unit_id")
                canonical_source = RetrievalSourceSpan(
                    document_id=self.target_id,
                    sequence_number=0,
                    document_start=0,
                    document_end=len(self.text),
                    role="region",
                    source=self.source,
                )
                if not self.sources:
                    self.sources = [canonical_source]
                if self.sources != [canonical_source]:
                    raise ValueError("region hits require one canonical source span")
            case "reviewed_coherent_unit":
                if self.target_id is None or self.coherent_unit_id is None:
                    raise ValueError(
                        "reviewed coherent-unit hits require target_id and coherent_unit_id"
                    )
                if self.source is not None:
                    raise ValueError(
                        "reviewed coherent-unit hits must not have a singular source"
                    )
                if not self.sources:
                    raise ValueError(
                        "reviewed coherent-unit hits require nonempty ordered sources"
                    )
                if any(span.document_id != self.target_id for span in self.sources):
                    raise ValueError(
                        "all coherent-unit sources must identify the target document"
                    )
                if any(
                    span.document_end == span.document_start for span in self.sources
                ):
                    raise ValueError("coherent-unit sources must have nonempty spans")
                citation_ids = [span.citation_id for span in self.sources]
                if len(set(citation_ids)) != len(citation_ids):
                    raise ValueError("coherent-unit citation_id values must be unique")
                sequence_numbers = [span.sequence_number for span in self.sources]
                if sequence_numbers != list(range(len(self.sources))):
                    raise ValueError(
                        "coherent-unit sources must be ordered and nonoverlapping"
                    )
                if any(
                    current.document_start < previous.document_end
                    for previous, current in zip(self.sources, self.sources[1:])
                ):
                    raise ValueError(
                        "coherent-unit sources must be ordered and nonoverlapping"
                    )
            case unreachable:
                assert_never(unreachable)
        return self


def serialize_legacy_sources(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_fields = {"page_id", "image_uri", "text_version_id", "text_selection_id"}
    for source in sources:
        for field in source_fields:
            source.pop(field, None)
    return sources


def _serialize_legacy_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hit_fields = {"target_kind", "target_id", "coherent_unit_id", "sources"}
    for hit in hits:
        for field in hit_fields:
            hit.pop(field, None)
        hit["source"] = serialize_legacy_sources([hit["source"]])[0]
    return hits


class RetrievalResponse(StrictModel):
    schema_version: Literal["1.0", "1.1"] = SCHEMA_VERSION
    query: str
    mode: RetrievalMode
    hits: list[RetrievalHit]
    generated_answer: str | None = None
    answer_model: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_schema_features(self) -> "RetrievalResponse":
        if self.schema_version == "1.0" and any(
            hit.target_kind == "reviewed_coherent_unit" for hit in self.hits
        ):
            raise ValueError("reviewed coherent-unit hits require response schema 1.1")
        return self

    @model_serializer(mode="wrap")
    def serialize_versioned(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        payload: dict[str, Any] = handler(self)
        match self.schema_version:
            case "1.1":
                return payload
            case "1.0":
                pass
            case unreachable:
                assert_never(unreachable)
        payload["hits"] = _serialize_legacy_hits(payload["hits"])
        return payload


class ScenarioEvidenceItem(StrictModel):
    statement: str
    epistemic_label: Literal["directly_evidenced", "plausible_inference", "speculative"]
    sources: list[SourcePointer]
    claim_ids: list[UUID] = Field(default_factory=list)


class ScenarioContextBundle(StrictModel):
    """Portable handoff to a chatbot or scenario-generation model."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    research_query: str
    evidence_items: list[ScenarioEvidenceItem]
    retrieved_context: list[RetrievalHit]
    warnings: list[str] = Field(default_factory=list)
    required_model_instruction: str = (
        "Keep directly evidenced facts, plausible inference, and speculation visibly separate. "
        "Never present generated details as recovered historical fact."
    )

    @model_serializer(mode="wrap")
    def serialize_legacy_retrieval(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        payload: dict[str, Any] = handler(self)
        payload["retrieved_context"] = _serialize_legacy_hits(
            payload["retrieved_context"]
        )
        for item in payload["evidence_items"]:
            item["sources"] = serialize_legacy_sources(item["sources"])
        return payload
