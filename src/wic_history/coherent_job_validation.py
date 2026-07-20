"""Validate coherent-job completion receipts against their leased inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, final, override
from uuid import UUID

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .coherent_search_contracts import JsonValue


class CoherentValidationFailure(StrEnum):
    """Stable coherent receipt validation failures."""

    EMBEDDING_BOOLEAN_RECEIPT = (
        "Coherent embedding requires an explicit stale_noop receipt"
    )
    EMBEDDING_LEASED_REVISION = (
        "Coherent embedding revision differs from its leased job"
    )
    EMBEDDING_LEASED_FINGERPRINT = (
        "Coherent embedding fingerprint differs from its leased job"
    )
    EMBEDDING_PLANNED_CONTENT = (
        "Coherent embedding receipt differs from its planned content"
    )
    EMBEDDING_PLANNED_CONFIGURATION = (
        "Coherent embedding receipt differs from its planned configuration"
    )
    EMBEDDING_STALE_RECEIPT = "Stale coherent embedding receipt is inconsistent"
    EMBEDDING_RECONCILIATION = "Stale coherent embedding requires reconciliation"
    EMBEDDING_CURRENT_PLAN = "Stale coherent embedding requires a current plan"
    EMBEDDING_ACTIVE_ARTIFACT = (
        "Active coherent embedding requires exactly one artifact"
    )
    PROJECTION_INDEX_PREFIX = "Coherent projection requires its dedicated index prefix"
    PROJECTION_BUILD_IDENTITY = "Coherent projection build and index identities differ"
    PROJECTION_SOURCE_SNAPSHOT = "Coherent projection requires source_snapshot_sha256"
    PROJECTION_PLANNED_SNAPSHOT = (
        "Coherent projection receipt differs from its planned snapshot"
    )
    PROJECTION_PLANNED_COVERAGE = (
        "Coherent projection requires exact positive planned coverage"
    )
    PROJECTION_PUBLICATION = "Coherent projection receipt must confirm publication"


@final
@dataclass(frozen=True, slots=True)
class CoherentResultValidationError(ValueError):
    """Report a stable coherent receipt invariant failure."""

    failure: CoherentValidationFailure

    @override
    def __str__(self) -> str:
        return self.failure.value


class RequiredFieldKind(StrEnum):
    """Supported required-field validation kinds."""

    UUID = "UUID"
    NONNEGATIVE_INTEGER = "nonnegative integer"


@final
@dataclass(frozen=True, slots=True)
class RequiredStageResultFieldError(ValueError):
    """Report a missing or invalid typed stage-result field."""

    kind: RequiredFieldKind
    field: str

    @override
    def __str__(self) -> str:
        return f"Stage result requires {self.kind.value} field {self.field}"


@dataclass(frozen=True, slots=True)
class CoherentEmbeddingResultValidation:
    """Inputs that bind an embedding receipt to its leased job."""

    configuration: Mapping[str, JsonValue]
    result: Mapping[str, JsonValue]
    input_fingerprint: str | None
    revision_id: UUID | None


@dataclass(frozen=True, slots=True)
class CoherentProjectionResultValidation:
    """Inputs that bind a projection receipt to its leased job."""

    configuration: Mapping[str, JsonValue]
    result: Mapping[str, JsonValue]
    input_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class _EmbeddingReceiptState:
    stale_noop: bool
    active: bool
    total: int


def _required_uuid(result: Mapping[str, JsonValue], field: str) -> UUID:
    try:
        return UUID(str(result[field]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RequiredStageResultFieldError(RequiredFieldKind.UUID, field) from exc


def _required_count(result: Mapping[str, JsonValue], field: str) -> int:
    value = result.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RequiredStageResultFieldError(
            RequiredFieldKind.NONNEGATIVE_INTEGER,
            field,
        )
    return value


def _validate_embedding_plan(
    validation: CoherentEmbeddingResultValidation,
    state: _EmbeddingReceiptState,
) -> None:
    result = validation.result
    configuration = validation.configuration
    if validation.revision_id is not None and result["revision_id"] != str(
        validation.revision_id,
    ):
        raise CoherentResultValidationError(
            CoherentValidationFailure.EMBEDDING_LEASED_REVISION,
        )
    if (
        validation.input_fingerprint is not None
        and result.get("input_fingerprint") != validation.input_fingerprint
    ):
        raise CoherentResultValidationError(
            CoherentValidationFailure.EMBEDDING_LEASED_FINGERPRINT,
        )
    if result.get("planned_input_sha256") != configuration.get(
        "planned_input_sha256",
    ) or result.get("planned_content_sha256") != configuration.get(
        "planned_content_sha256",
    ):
        raise CoherentResultValidationError(
            CoherentValidationFailure.EMBEDDING_PLANNED_CONTENT,
        )
    if result.get("embedding_configuration_sha256") != configuration.get(
        "embedding_configuration_sha256",
    ):
        raise CoherentResultValidationError(
            CoherentValidationFailure.EMBEDDING_PLANNED_CONFIGURATION,
        )
    _validate_embedding_state(validation, state)


def _validate_embedding_state(
    validation: CoherentEmbeddingResultValidation,
    state: _EmbeddingReceiptState,
) -> None:
    result = validation.result
    if state.stale_noop:
        if (
            state.active
            or state.total != 0
            or result.get("reconciliation_scheduled") is not True
        ):
            raise CoherentResultValidationError(
                CoherentValidationFailure.EMBEDDING_STALE_RECEIPT,
            )
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            str(result.get("reconciliation_plan_key", "")),
        ):
            raise CoherentResultValidationError(
                CoherentValidationFailure.EMBEDDING_RECONCILIATION,
            )
        if result.get("reconciliation_plan_key") == validation.input_fingerprint:
            raise CoherentResultValidationError(
                CoherentValidationFailure.EMBEDDING_CURRENT_PLAN,
            )
    elif not state.active or state.total != 1:
        raise CoherentResultValidationError(
            CoherentValidationFailure.EMBEDDING_ACTIVE_ARTIFACT,
        )


def validate_coherent_unit_embedding_result(
    validation: CoherentEmbeddingResultValidation,
) -> None:
    """Validate a coherent embedding receipt."""
    result = validation.result
    _ = _required_uuid(result, "revision_id")
    embeddings_inserted = _required_count(result, "embeddings_inserted")
    embeddings_reused = _required_count(result, "embeddings_reused")
    stale_noop = result.get("stale_noop")
    active = result.get("active")
    if type(stale_noop) is not bool or type(active) is not bool:
        raise CoherentResultValidationError(
            CoherentValidationFailure.EMBEDDING_BOOLEAN_RECEIPT,
        )
    _validate_embedding_plan(
        validation,
        _EmbeddingReceiptState(
            stale_noop,
            active,
            embeddings_inserted + embeddings_reused,
        ),
    )


def validate_coherent_unit_search_projection_result(
    validation: CoherentProjectionResultValidation,
) -> None:
    """Validate a coherent search-projection receipt."""
    configuration = validation.configuration
    result = validation.result
    input_fingerprint = validation.input_fingerprint
    build_id = _required_uuid(result, "projection_build_id")
    documents_indexed = _required_count(result, "documents_indexed")
    if not str(result.get("index_name", "")).startswith(
        "wic-coherent-units-build-",
    ):
        raise CoherentResultValidationError(
            CoherentValidationFailure.PROJECTION_INDEX_PREFIX,
        )
    if result["index_name"] != f"wic-coherent-units-build-{build_id.hex}":
        raise CoherentResultValidationError(
            CoherentValidationFailure.PROJECTION_BUILD_IDENTITY,
        )
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        str(result.get("source_snapshot_sha256", "")),
    ):
        raise CoherentResultValidationError(
            CoherentValidationFailure.PROJECTION_SOURCE_SNAPSHOT,
        )
    planned_snapshot = configuration.get("planned_snapshot_sha256")
    planned_count = configuration.get("planned_revision_count")
    if (
        input_fingerprint is not None
        and result["source_snapshot_sha256"] != input_fingerprint
    ) or result.get("planned_snapshot_sha256") != planned_snapshot:
        raise CoherentResultValidationError(
            CoherentValidationFailure.PROJECTION_PLANNED_SNAPSHOT,
        )
    if (
        not isinstance(planned_count, int)
        or isinstance(planned_count, bool)
        or planned_count < 1
        or documents_indexed != planned_count
        or result.get("planned_revision_count") != planned_count
    ):
        raise CoherentResultValidationError(
            CoherentValidationFailure.PROJECTION_PLANNED_COVERAGE,
        )
    if result.get("published") is not True:
        raise CoherentResultValidationError(
            CoherentValidationFailure.PROJECTION_PUBLICATION,
        )
