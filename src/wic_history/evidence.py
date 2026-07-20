"""Versioned contracts shared by OCR, extraction, storage, and retrieval.

These models intentionally keep observation, machine candidate, and reviewed
assertion states distinct. They are the serialization boundary between pipeline
stages and must remain backward compatible once artifacts are produced.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Literal, TypeAlias, TypedDict
from uuid import UUID, uuid4

from pydantic import (
    Field,
    model_serializer,
    model_validator,
)

from .retrieval_contracts import (
    LegacyHitPayload,
    LegacySourcePayload,
    RetrievalHit as RetrievalHit,
    RetrievalMode as RetrievalMode,
    RetrievalResponse as RetrievalResponse,
    RetrievalSourceSpan as RetrievalSourceSpan,
    SerializedValue,
    serialize_legacy_hits as serialize_legacy_hits,
    serialize_legacy_sources as serialize_legacy_sources,
)
from .source_provenance import (
    SCHEMA_VERSION as SCHEMA_VERSION,
    Point as Point,
    Polygon as Polygon,
    SourcePointer as SourcePointer,
    StrictModel as StrictModel,
)


class _ScenarioEvidencePayload(TypedDict):
    sources: list[LegacySourcePayload]


class _ScenarioContextPayload(TypedDict):
    retrieved_context: list[LegacyHitPayload]
    evidence_items: list[_ScenarioEvidencePayload]


_ScenarioContextSerializer: TypeAlias = Callable[
    ["ScenarioContextBundle"],
    _ScenarioContextPayload,
]


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

    @model_serializer(mode="wrap", return_type=dict[str, SerializedValue])
    def serialize_legacy_retrieval(
        self,
        handler: _ScenarioContextSerializer,
    ) -> _ScenarioContextPayload:
        payload = handler(self)
        payload["retrieved_context"] = serialize_legacy_hits(
            payload["retrieved_context"]
        )
        for item in payload["evidence_items"]:
            item["sources"] = serialize_legacy_sources(item["sources"])
        return payload
