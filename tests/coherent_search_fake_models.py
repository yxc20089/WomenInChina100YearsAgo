from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class BulkIndex(StrictModel):
    document_id: str = Field(alias="_id")
    index_name: str = Field(alias="_index")


class BulkAction(StrictModel):
    index: BulkIndex


class AliasTarget(StrictModel):
    index: str
    alias: str
    must_exist: bool | None = None


class AliasAction(StrictModel):
    remove: AliasTarget | None = None
    add: AliasTarget | None = None


class AliasEnvelope(StrictModel):
    actions: tuple[AliasAction, ...]


class StoredSource(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)


class StoredDocument(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)

    revision_id: str
    year_min: int
    year_max: int
    sources: tuple[StoredSource, ...]
