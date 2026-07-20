"""Bridge coherent-unit jobs into the ingestion worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from .coherent_job_embedding import execute_coherent_embedding
from .coherent_job_projection_execution import execute_coherent_projection
from .coherent_jobs import (
    COHERENT_STAGES,
    ActiveRevision,
    CoherentExecution,
    CoherentJobContext,
    CoherentJobError,
    coherent_plan_key,
    load_coherent_job_context,
)

if TYPE_CHECKING:
    from .ingestion_jobs import JobLease


@final
class UnsupportedCoherentStageError(ValueError):
    """Report a coherent stage without a worker executor."""

    stage: str

    def __init__(self, stage: str) -> None:
        """Record the unsupported coherent stage."""
        self.stage = stage
        super().__init__(f"No coherent worker exists for stage {stage}")


def load_coherent_execution_context(
    database_url: str,
    lease: JobLease,
) -> CoherentJobContext | None:
    """Load context only for a coherent-stage lease."""
    if lease.stage not in COHERENT_STAGES:
        return None
    return load_coherent_job_context(database_url, lease.job_id)


def ensure_coherent_snapshot(
    planned_fingerprint: str,
    revisions: tuple[ActiveRevision, ...],
) -> None:
    """Reject an empty or stale coherent projection snapshot."""
    if not revisions:
        message = "Coherent projection active snapshot is empty"
        error = CoherentJobError(message)
        raise error
    if coherent_plan_key(revisions) != planned_fingerprint:
        message = "Coherent projection active snapshot is stale"
        error = CoherentJobError(message)
        raise error


def execute_coherent_stage(
    database_url: str,
    context: CoherentJobContext,
    *,
    opensearch_url: str,
) -> CoherentExecution:
    """Execute the worker selected by a coherent job's stage."""
    if context.stage == "coherent_unit_embedding":
        return execute_coherent_embedding(database_url, context)
    if context.stage == "coherent_unit_search_projection":
        return execute_coherent_projection(
            database_url,
            context,
            opensearch_url=opensearch_url,
        )
    raise UnsupportedCoherentStageError(context.stage)
