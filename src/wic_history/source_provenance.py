"""Source provenance models shared by evidence and retrieval contracts."""

from enum import StrEnum
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


class _SourcePointerErrorMessage(StrEnum):
    OFFSETS_PAIRED = "text_start and text_end must be provided together"
    OFFSETS_ORDERED = "text_end must be greater than or equal to text_start"


class _SourcePointerValidationError(ValueError):
    pass


class StrictModel(BaseModel):
    """Pydantic contract that rejects undeclared input fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class Point(StrictModel):
    """Non-negative image coordinate."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)


class Polygon(StrictModel):
    """Image polygon containing at least three points."""

    points: list[Point] = Field(min_length=3)


class SourcePointer(StrictModel):
    """Versioned pointer from derived evidence to its source location."""

    source_uri: str
    source_sha256: str | None = None
    page_id: UUID | None = None
    derivative_id: UUID | None = None
    image_uri: str | None = None
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_tier: (
        Literal[
            "screening_derivative",
            "unreviewed_input",
            "non_gold_lossless_pilot",
            "historian_selected_gold",
        ]
        | None
    ) = None
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
        """Require optional text offsets to be paired and ordered."""
        if (self.text_start is None) != (self.text_end is None):
            raise _SourcePointerValidationError(
                _SourcePointerErrorMessage.OFFSETS_PAIRED,
            )
        if (
            self.text_start is not None
            and self.text_end is not None
            and self.text_end < self.text_start
        ):
            raise _SourcePointerValidationError(
                _SourcePointerErrorMessage.OFFSETS_ORDERED,
            )
        return self
