from __future__ import annotations

from uuid import UUID
from unittest.mock import patch

import pytest

from wic_history.article_embedding import (
    ArticleEmbeddingSummary,
    ReviewedArticleUnavailableError,
)
from wic_history.coherent_jobs import (
    COHERENT_CONFIGURATION,
    ActiveRevision,
    CoherentJobError,
    CoherentJobContext,
    coherent_plan_key,
    execute_coherent_embedding,
)
from wic_history.coherent_jobs import CoherentPlanResult


def _embedding_context(revision_id: UUID) -> tuple[CoherentJobContext, ActiveRevision]:
    from dataclasses import asdict
    from wic_history.coherent_jobs import _sha256

    planned = ActiveRevision(revision_id, "a" * 64, "b" * 64)
    fingerprint = _sha256(
        {"revision": asdict(planned), "configuration": COHERENT_CONFIGURATION}
    )
    configuration = {
        **COHERENT_CONFIGURATION,
        "planned_revision_id": str(revision_id),
        "planned_input_sha256": planned.input_sha256,
        "planned_content_sha256": planned.content_sha256,
        "planned_embedding_fingerprint": fingerprint,
    }
    return (
        CoherentJobContext(
            "job", "batch", "coherent_unit_embedding", fingerprint, configuration, revision_id
        ),
        planned,
    )
from wic_history.ingestion_jobs import ALL_STAGES, PAGE_STAGES, build_parser
from wic_history.ingestion_worker import ensure_coherent_snapshot


def test_plan_key_is_content_addressed_over_sorted_active_snapshot() -> None:
    # Given: the same active articles discovered in opposite database orders.
    first = ActiveRevision(UUID(int=2), "b" * 64)
    second = ActiveRevision(UUID(int=1), "a" * 64)

    # When: both snapshots are planned.
    left = coherent_plan_key((first, second))
    right = coherent_plan_key((second, first))

    # Then: ordering is canonical while reviewed-text input identity remains material.
    assert left == right
    assert left != coherent_plan_key(
        (ActiveRevision(first.revision_id, "c" * 64), second)
    )


def test_coherent_stages_are_not_page_stages_and_configuration_is_pinned() -> None:
    # Given: the coherent lifecycle configuration and scheduler stage sets.
    configuration = COHERENT_CONFIGURATION

    # When: their immutable identities are inspected.
    values = (
        configuration["model"],
        configuration["revision"],
        configuration["dimension"],
        configuration["window_policy"],
        configuration["alias"],
        configuration["index_prefix"],
        configuration["projection_kind"],
        configuration["mapping_sha256"],
    )

    # Then: the article lifecycle is pinned and does not alter the page DAG.
    assert values == (
        "BAAI/bge-m3",
        "5617a9f61b028005a4858fdac845db406aefb181",
        1024,
        "windowed_mean_v1",
        "wic-coherent-units-current",
        "wic-coherent-units-build-",
        "opensearch_coherent_unit",
        configuration["mapping_sha256"],
    )
    mapping_sha256 = configuration["mapping_sha256"]
    assert isinstance(mapping_sha256, str)
    assert len(mapping_sha256) == 64
    assert "coherent_unit_embedding" in ALL_STAGES
    assert "coherent_unit_search_projection" in ALL_STAGES
    assert "coherent_unit_embedding" not in PAGE_STAGES
    assert "coherent_unit_search_projection" not in PAGE_STAGES


def test_backfill_cli_has_an_explicit_safety_bound() -> None:
    # Given: the existing batch CLI.
    parser = build_parser()

    # When: an operator requests coherent article backfill.
    args = parser.parse_args(
        ["coherent-backfill", "--created-by", "operator", "--max-revisions", "25"]
    )

    # Then: the bound and actor survive boundary parsing.
    assert (args.command, args.created_by, args.max_revisions) == (
        "coherent-backfill",
        "operator",
        25,
    )


def test_embedding_worker_targets_exact_revision_without_global_discovery() -> None:
    # Given: one leased immutable revision job.
    revision_id = UUID(int=7)
    context, planned = _embedding_context(revision_id)
    summary = ArticleEmbeddingSummary(1, 1, 0, ("run",), "model", "revision")

    # When: the article embedding adapter completes it.
    with (
        patch("wic_history.coherent_jobs._active_revisions_from_database", return_value=(planned,)),
        patch("wic_history.coherent_jobs.embed_reviewed_articles", return_value=summary) as embedder,
    ):
        execution = execute_coherent_embedding("postgresql://unused", context)

    # Then: discovery is pinned to the leased revision and success is auditable.
    assert embedder.call_args.args[0].revision_id == revision_id
    assert execution.result["revision_id"] == str(revision_id)
    assert execution.result["embeddings_inserted"] == 1
    assert execution.result["embeddings_reused"] == 0
    assert execution.result["stale_noop"] is False
    assert execution.result["active"] is True


def test_missing_embedding_input_is_noop_only_after_supersession() -> None:
    # Given: an exact revision that the article loader can no longer materialize.
    revision_id = UUID(int=8)
    context, planned = _embedding_context(revision_id)
    missing = ReviewedArticleUnavailableError(revision_id)

    # When: the revision is confirmed superseded.
    with (
        patch("wic_history.coherent_jobs.embed_reviewed_articles", side_effect=missing),
        patch("wic_history.coherent_jobs._revision_superseded", return_value=True),
        patch("wic_history.coherent_jobs._active_revisions_from_database", return_value=(planned,)),
        patch(
            "wic_history.coherent_jobs._schedule_current_reconcile",
            return_value=CoherentPlanResult("batch", "c" * 64, 1, 2, True),
        ),
    ):
        execution = execute_coherent_embedding("postgresql://unused", context)

    # Then: supersession is recorded as a completed stale no-op.
    assert execution.result["stale_noop"] is True

    # When: the same load failure belongs to an active or missing revision.
    with (
        patch("wic_history.coherent_jobs.embed_reviewed_articles", side_effect=missing),
        patch("wic_history.coherent_jobs._revision_superseded", return_value=False),
        patch("wic_history.coherent_jobs._active_revisions_from_database", return_value=(planned,)),
        pytest.raises(ReviewedArticleUnavailableError),
    ):
        _ = execute_coherent_embedding("postgresql://unused", context)


def test_projection_refuses_stale_or_empty_global_snapshot() -> None:
    # Given: a projection job planned from one active reviewed revision.
    revision = ActiveRevision(UUID(int=9), "d" * 64)
    planned = coherent_plan_key((revision,))

    # When: the exact snapshot is rechecked immediately before publication.
    ensure_coherent_snapshot(planned, (revision,))

    # Then: empty or changed global state is rejected before the projector runs.
    with pytest.raises(CoherentJobError, match="empty"):
        ensure_coherent_snapshot(planned, ())
    with pytest.raises(CoherentJobError, match="stale"):
        ensure_coherent_snapshot(
            planned, (ActiveRevision(revision.revision_id, "e" * 64),)
        )
