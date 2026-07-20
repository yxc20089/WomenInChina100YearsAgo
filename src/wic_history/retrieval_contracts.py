"""Retrieval models and compatibility serializers for evidence contracts."""

from collections.abc import Callable
from enum import StrEnum
from typing import Final, Literal, Required, TypeAlias, TypedDict
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, model_serializer, model_validator
from typing_extensions import TypeAliasType

from .source_provenance import SCHEMA_VERSION, SourcePointer, StrictModel

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)
SerializedValue = TypeAliasType(
    "SerializedValue",
    JsonScalar | UUID | list["SerializedValue"] | dict[str, "SerializedValue"],
)


class LegacySourcePayload(TypedDict, total=False):
    """Source fields consumed by schema 1.0 compatibility serialization."""

    page_id: UUID | str | None
    image_uri: str | None
    text_version_id: UUID | str | None
    text_selection_id: UUID | str | None


class LegacyHitPayload(TypedDict, total=False):
    """Retrieval-hit fields consumed by schema 1.0 compatibility serialization."""

    target_kind: str
    target_id: UUID | str | None
    coherent_unit_id: UUID | str | None
    sources: list[JsonValue]
    source: Required[LegacySourcePayload]


class _RetrievalResponsePayload(TypedDict):
    schema_version: Literal["1.0", "1.1"]
    hits: list[LegacyHitPayload]


_RetrievalResponseSerializer: TypeAlias = Callable[
    ["RetrievalResponse"],
    _RetrievalResponsePayload,
]


class _RetrievalValidationError(ValueError):
    pass


_DOCUMENT_OFFSETS: Final = (
    "document_end must be greater than or equal to document_start"
)
_CITATION_IDENTITY: Final = "citation_id must match deterministic provenance identity"
_REGION_SOURCE: Final = "region hits require a singular source"
_REGION_TARGET: Final = "region target_id must match source.region_id"
_REGION_COHERENT_ID: Final = "region hits cannot have a coherent_unit_id"
_REGION_SPAN: Final = "region hits require one canonical source span"
_COHERENT_TARGET: Final = (
    "reviewed coherent-unit hits require target_id and coherent_unit_id"
)
_COHERENT_SOURCE: Final = "reviewed coherent-unit hits must not have a singular source"
_COHERENT_SPANS: Final = "reviewed coherent-unit hits require nonempty ordered sources"
_COHERENT_DOCUMENT: Final = (
    "all coherent-unit sources must identify the target document"
)
_COHERENT_NONEMPTY: Final = "coherent-unit sources must have nonempty spans"
_COHERENT_UNIQUE: Final = "coherent-unit citation_id values must be unique"
_COHERENT_ORDER: Final = "coherent-unit sources must be ordered and nonoverlapping"
_RESPONSE_SCHEMA: Final = "reviewed coherent-unit hits require response schema 1.1"


class RetrievalMode(StrEnum):
    """Available retrieval strategies."""

    HYBRID = "hybrid"
    LEXICAL = "lexical"
    DENSE = "dense"
    GRAPH = "graph"


class RetrievalSourceSpan(StrictModel):
    """Ordered source span with deterministic citation identity."""

    citation_id: str = ""
    document_id: UUID | None
    sequence_number: int = Field(ge=0)
    document_start: int = Field(ge=0)
    document_end: int = Field(ge=0)
    role: str = Field(min_length=1)
    source: SourcePointer

    @model_validator(mode="after")
    def validate_document_offsets(self) -> "RetrievalSourceSpan":
        """Derive citation identity after validating document offsets."""
        if self.document_end < self.document_start:
            raise _RetrievalValidationError(_DOCUMENT_OFFSETS)
        citation_id = _retrieval_citation_id(
            self.document_id,
            self.sequence_number,
            self.source,
        )
        if self.citation_id and self.citation_id != citation_id:
            raise _RetrievalValidationError(_CITATION_IDENTITY)
        self.citation_id = citation_id
        return self


def _retrieval_citation_id(
    document_id: UUID | None,
    sequence_number: int,
    source: SourcePointer,
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
        ),
    )
    return f"citation:{uuid5(NAMESPACE_URL, identity)}"


class RetrievalHit(StrictModel):
    """Ranked region or reviewed coherent-unit retrieval result."""

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
    explanation: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_target_provenance(self) -> "RetrievalHit":
        """Canonicalize and validate provenance for the selected target kind."""
        if self.target_kind == "region":
            self._validate_region()
        else:
            self._validate_reviewed_coherent_unit()
        return self

    def _validate_region(self) -> None:
        if self.source is None:
            raise _RetrievalValidationError(_REGION_SOURCE)
        if self.target_id is None and self.source.region_id is not None:
            self.target_id = self.source.region_id
        if (
            self.source.region_id is not None
            and self.target_id != self.source.region_id
        ):
            raise _RetrievalValidationError(_REGION_TARGET)
        if self.coherent_unit_id is not None:
            raise _RetrievalValidationError(_REGION_COHERENT_ID)
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
            raise _RetrievalValidationError(_REGION_SPAN)

    def _validate_reviewed_coherent_unit(self) -> None:
        if self.target_id is None or self.coherent_unit_id is None:
            raise _RetrievalValidationError(_COHERENT_TARGET)
        if self.source is not None:
            raise _RetrievalValidationError(_COHERENT_SOURCE)
        if not self.sources:
            raise _RetrievalValidationError(_COHERENT_SPANS)
        if any(span.document_id != self.target_id for span in self.sources):
            raise _RetrievalValidationError(_COHERENT_DOCUMENT)
        if any(span.document_end == span.document_start for span in self.sources):
            raise _RetrievalValidationError(_COHERENT_NONEMPTY)
        citation_ids = [span.citation_id for span in self.sources]
        if len(set(citation_ids)) != len(citation_ids):
            raise _RetrievalValidationError(_COHERENT_UNIQUE)
        sequence_numbers = [span.sequence_number for span in self.sources]
        if sequence_numbers != list(range(len(self.sources))):
            raise _RetrievalValidationError(_COHERENT_ORDER)
        if any(
            current.document_start < previous.document_end
            for previous, current in zip(
                self.sources,
                self.sources[1:],
                strict=False,
            )
        ):
            raise _RetrievalValidationError(_COHERENT_ORDER)


def serialize_legacy_sources(
    sources: list[LegacySourcePayload],
) -> list[LegacySourcePayload]:
    """Remove source fields unavailable in schema 1.0 payloads."""
    for source in sources:
        if "page_id" in source:
            del source["page_id"]
        if "image_uri" in source:
            del source["image_uri"]
        if "text_version_id" in source:
            del source["text_version_id"]
        if "text_selection_id" in source:
            del source["text_selection_id"]
    return sources


def serialize_legacy_hits(hits: list[LegacyHitPayload]) -> list[LegacyHitPayload]:
    """Remove retrieval fields unavailable in schema 1.0 payloads."""
    for hit in hits:
        if "target_kind" in hit:
            del hit["target_kind"]
        if "target_id" in hit:
            del hit["target_id"]
        if "coherent_unit_id" in hit:
            del hit["coherent_unit_id"]
        if "sources" in hit:
            del hit["sources"]
        hit["source"] = serialize_legacy_sources([hit["source"]])[0]
    return hits


class RetrievalResponse(StrictModel):
    """Versioned response containing ranked retrieval hits."""

    schema_version: Literal["1.0", "1.1"] = SCHEMA_VERSION
    query: str
    mode: RetrievalMode
    hits: list[RetrievalHit]
    generated_answer: str | None = None
    answer_model: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_schema_features(self) -> "RetrievalResponse":
        """Reject coherent-unit features from schema 1.0 responses."""
        if self.schema_version == "1.0" and any(
            hit.target_kind == "reviewed_coherent_unit" for hit in self.hits
        ):
            raise _RetrievalValidationError(_RESPONSE_SCHEMA)
        return self

    @model_serializer(mode="wrap", return_type=dict[str, SerializedValue])
    def serialize_versioned(
        self,
        handler: _RetrievalResponseSerializer,
    ) -> _RetrievalResponsePayload:
        """Serialize schema 1.0 without fields introduced by schema 1.1."""
        payload = handler(self)
        if self.schema_version == "1.1":
            return payload
        payload["hits"] = serialize_legacy_hits(payload["hits"])
        return payload
