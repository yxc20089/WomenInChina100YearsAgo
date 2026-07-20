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
)
from wic_history.coherent_jobs import CoherentPlanResult
from wic_history.coherent_job_embedding import execute_coherent_embedding
from wic_history.coherent_job_projection_execution import execute_coherent_projection
from wic_history.coherent_job_projection import build_coherent_manifest
from wic_history.coherent_search import ProjectionResult
from wic_history.coherent_search_documents import validated_documents
from wic_history.ingestion_jobs import (
    ALL_STAGES,
    PAGE_STAGES,
    build_parser,
    validate_stage_result,
)
from wic_history.ingestion_worker import ensure_coherent_snapshot
from tests.coherent_search_support import manifest


def _embedding_context(revision_id: UUID) -> tuple[CoherentJobContext, ActiveRevision]:
    from dataclasses import asdict
    from wic_history.coherent_job_hashing import coherent_sha256

    planned = ActiveRevision(revision_id, "a" * 64, "b" * 64)
    fingerprint = coherent_sha256(
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
            "job",
            "batch",
            "coherent_unit_embedding",
            fingerprint,
            configuration,
            revision_id,
        ),
        planned,
    )


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
        patch(
            "wic_history.coherent_job_embedding._active_revisions_from_database",
            return_value=(planned,),
        ),
        patch(
            "wic_history.coherent_job_embedding.embed_reviewed_articles",
            return_value=summary,
        ) as embedder,
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
        patch(
            "wic_history.coherent_job_embedding.embed_reviewed_articles",
            side_effect=missing,
        ),
        patch(
            "wic_history.coherent_job_embedding._revision_superseded", return_value=True
        ),
        patch(
            "wic_history.coherent_job_embedding._active_revisions_from_database",
            return_value=(planned,),
        ),
        patch(
            "wic_history.coherent_job_embedding._schedule_current_reconcile",
            return_value=CoherentPlanResult("batch", "c" * 64, 1, 2, True),
        ),
    ):
        execution = execute_coherent_embedding("postgresql://unused", context)

    # Then: supersession is recorded as a completed stale no-op.
    assert execution.result["stale_noop"] is True

    # When: the same load failure belongs to an active or missing revision.
    with (
        patch(
            "wic_history.coherent_job_embedding.embed_reviewed_articles",
            side_effect=missing,
        ),
        patch(
            "wic_history.coherent_job_embedding._revision_superseded",
            return_value=False,
        ),
        patch(
            "wic_history.coherent_job_embedding._active_revisions_from_database",
            return_value=(planned,),
        ),
        pytest.raises(ReviewedArticleUnavailableError),
    ):
        _ = execute_coherent_embedding("postgresql://unused", context)


def test_reselected_embedding_job_noops_and_schedules_current_snapshot() -> None:
    revision_id = UUID(int=15)
    context, planned = _embedding_context(revision_id)
    current = ActiveRevision(revision_id, "c" * 64, "d" * 64)
    reconciliation = CoherentPlanResult("new-batch", "e" * 64, 1, 2, True)

    with (
        patch(
            "wic_history.coherent_job_embedding._active_revisions_from_database",
            return_value=(current,),
        ),
        patch(
            "wic_history.coherent_job_embedding._schedule_current_reconcile",
            return_value=reconciliation,
        ) as reconcile,
        patch("wic_history.coherent_job_embedding.embed_reviewed_articles") as embedder,
    ):
        execution = execute_coherent_embedding("postgresql://unused", context)

    assert execution.result["stale_noop"] is True
    assert execution.result["active"] is False
    assert execution.result["reconciliation_plan_key"] == reconciliation.plan_key
    reconcile.assert_called_once()
    embedder.assert_not_called()


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


def test_coherent_receipts_are_bound_to_the_leased_identity() -> None:
    revision_id = UUID(int=10)
    context, planned = _embedding_context(revision_id)
    valid = {
        "revision_id": str(revision_id),
        "planned_input_sha256": planned.input_sha256,
        "planned_content_sha256": planned.content_sha256,
        "input_fingerprint": context.input_fingerprint,
        "embedding_configuration_sha256": COHERENT_CONFIGURATION[
            "embedding_configuration_sha256"
        ],
        "embeddings_inserted": 1,
        "embeddings_reused": 0,
        "stale_noop": False,
        "active": True,
    }
    validate_stage_result(
        context.stage,
        dict(context.configuration),
        valid,
        context.input_fingerprint,
        revision_id,
    )
    for changed in (
        {**valid, "revision_id": str(UUID(int=11))},
        {**valid, "stale_noop": 1},
        {**valid, "embeddings_inserted": 0},
        {**valid, "input_fingerprint": "f" * 64},
    ):
        with pytest.raises(ValueError):
            validate_stage_result(
                context.stage,
                dict(context.configuration),
                changed,
                context.input_fingerprint,
                revision_id,
            )


def test_projection_receipt_requires_exact_positive_published_snapshot() -> None:
    snapshot = "d" * 64
    configuration = {
        **COHERENT_CONFIGURATION,
        "planned_snapshot_sha256": snapshot,
        "planned_revision_count": 2,
    }
    valid = {
        "projection_build_id": str(UUID(int=12)),
        "index_name": f"wic-coherent-units-build-{UUID(int=12).hex}",
        "documents_indexed": 2,
        "source_snapshot_sha256": snapshot,
        "planned_snapshot_sha256": snapshot,
        "planned_revision_count": 2,
        "published": True,
    }
    validate_stage_result(
        "coherent_unit_search_projection", configuration, valid, snapshot, None
    )
    for changed in (
        {**valid, "documents_indexed": 0},
        {**valid, "source_snapshot_sha256": "e" * 64},
        {**valid, "published": False},
        {**valid, "projection_build_id": "not-a-uuid"},
    ):
        with pytest.raises(ValueError):
            validate_stage_result(
                "coherent_unit_search_projection",
                configuration,
                changed,
                snapshot,
                None,
            )


def test_projection_commit_failure_compensates_published_alias() -> None:
    revision = ActiveRevision(UUID(int=13), "a" * 64, "b" * 64)
    snapshot = coherent_plan_key((revision,))
    configuration = {
        **COHERENT_CONFIGURATION,
        "planned_snapshot_sha256": snapshot,
        "planned_revision_count": 1,
    }
    context = CoherentJobContext(
        "job",
        "batch",
        "coherent_unit_search_projection",
        snapshot,
        configuration,
        None,
    )
    projected = ProjectionResult(
        str(UUID(int=14)), "wic-coherent-units-build-new", 1, "c" * 64, ("old",)
    )

    class CommitFailure(RuntimeError):
        pass

    class Result:
        def fetchone(self):
            return None

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            raise CommitFailure("commit failed")

        def execute(self, *_args, **_kwargs):
            return Result()

    class Psycopg:
        Error = CommitFailure

        @staticmethod
        def connect(*_args, **_kwargs):
            return Connection()

    with (
        patch(
            "wic_history.coherent_job_projection_execution.database_clients",
            return_value=(Psycopg, object()),
        ),
        patch(
            "wic_history.coherent_job_projection_execution.load_active_documents",
            return_value=[
                (
                    {
                        "id": str(revision.revision_id),
                        "metadata": {
                            "input_sha256": revision.input_sha256,
                            "content_sha256": revision.content_sha256,
                        },
                    },
                    [],
                )
            ],
        ),
        patch(
            "wic_history.coherent_job_projection_execution.build_coherent_manifest"
        ) as manifest_builder,
        patch(
            "wic_history.coherent_job_projection_execution.project_coherent_units",
            return_value=projected,
        ),
        patch(
            "wic_history.coherent_job_projection_execution.restore_coherent_alias"
        ) as restore,
        pytest.raises(CommitFailure, match="commit failed"),
    ):
        manifest_builder.return_value.snapshot_sha256 = projected.source_snapshot_sha256
        _ = execute_coherent_projection(
            "postgresql://unused", context, opensearch_url="http://unused"
        )

    restore.assert_called_once_with("http://unused", projected)


def test_manifest_accepts_uuid_rows_and_ignores_historical_embedding() -> None:
    frozen = manifest()
    article = frozen.articles[0]
    embedding = frozen.embeddings[0]
    documents = [
        (
            {
                "id": str(article.bundle.coherent_unit_revision_id),
                "title": article.title,
                "text": article.bundle.content,
                "metadata": {
                    "coherent_unit_id": str(article.coherent_unit_id),
                    "input_sha256": article.bundle.input_sha256,
                    "content_sha256": article.bundle.content_sha256,
                },
            },
            [
                {
                    "sequence_number": segment.sequence_number,
                    "region_id": str(segment.region_id),
                    "selected_text_version_id": str(segment.text_version_id),
                    "text_selection_id": str(segment.selection_id),
                    "region_text_start": segment.text_start,
                    "region_text_end": segment.text_end,
                    "start_char": segment.composite_start,
                    "end_char": segment.composite_end,
                    "exported_text": segment.text,
                    "role": segment.role,
                    "polygon": source.source.polygon,
                    "source_uri": source.source.source_uri,
                    "source_sha256": source.source.source_sha256,
                    "derivative_id": str(source.source.derivative_id),
                    "source_image_uri": source.source.image_uri,
                    "source_image_sha256": source.source.image_sha256,
                    "evidence_tier": source.source.evidence_tier,
                    "volume_number": source.source.volume_number,
                    "publication_year": source.source.publication_year,
                    "page_number": source.source.page_number,
                }
                for segment, source in zip(
                    article.bundle.segments, article.sources, strict=True
                )
            ],
        )
    ]
    page_rows = [
        {"region_id": segment.region_id, "page_id": segment.page_id}
        for segment in article.bundle.segments
    ]
    exact_row = {
        "target_id": embedding.revision_id,
        "model_name": embedding.model_name,
        "model_revision": embedding.model_revision,
        "input_sha256": embedding.input_sha256,
        "content_sha256": embedding.content_sha256,
        "configuration_sha256": embedding.configuration_sha256,
        "vector": "[" + ",".join(str(value) for value in embedding.vector) + "]",
    }
    stale_row = {**exact_row, "input_sha256": "f" * 64}

    class Rows:
        def __init__(self, rows):
            self.rows = rows

        def fetchall(self):
            return self.rows

    class Connection:
        def execute(self, query, _params=None):
            return Rows(
                page_rows if "evidence.ocr_region" in query else [stale_row, exact_row]
            )

    built = build_coherent_manifest(Connection(), documents)

    assert built.articles[0].bundle.coherent_unit_revision_id == embedding.revision_id
    assert built.articles[0].bundle.content == article.bundle.content
    assert built.embeddings == frozen.embeddings
    assert len(validated_documents(built)) == 1

    page_rows[0] = {
        "region_id": object(),
        "page_id": article.bundle.segments[0].page_id,
    }
    with pytest.raises(CoherentJobError, match="UUID region id"):
        _ = build_coherent_manifest(Connection(), documents)
