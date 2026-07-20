from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from wic_history.article_embedding import (
    ArticleIdentity,
    ArticleSelection,
    InconsistentArticleEmbeddingRunError,
    PostgresArticleStore,
    StaleReviewedArticleError,
)
from wic_history.article_embedding_contracts import WindowConfiguration

DATABASE_URL = os.environ.get("DATABASE_URL")


@dataclass(frozen=True, slots=True)
class SeededArticle:
    revision_id: UUID
    region_id: UUID
    selection_id: UUID
    text_version_id: UUID
    model_name: str

    @property
    def selection(self) -> tuple[ArticleSelection, ...]:
        return (
            ArticleSelection(
                0, self.region_id, self.selection_id, self.text_version_id
            ),
        )


@contextmanager
def seeded_article(database_url: str, *, incomplete: bool = False):
    import psycopg

    seed = SeededArticle(uuid4(), uuid4(), uuid4(), uuid4(), f"qa-{uuid4()}")
    approval_id, page_id, segmentation_id, review_id, unit_id = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    with psycopg.connect(database_url) as connection:
        _ = connection.execute("SET session_replication_role = replica")
        _ = connection.execute(
            "INSERT INTO evidence.page_article_segmentation_selection (selection_id, page_id, run_id, review_id, selection_basis, selected_by) VALUES (%s, %s, %s, %s, 'historian_approved', 'qa')",
            (approval_id, page_id, segmentation_id, review_id),
        )
        _ = connection.execute(
            "INSERT INTO evidence.coherent_unit (unit_id) VALUES (%s)", (unit_id,)
        )
        _ = connection.execute(
            "INSERT INTO evidence.coherent_unit_revision (revision_id, unit_id, revision_number, unit_kind, approval_selection_id, content_sha256, approved_by) VALUES (%s, %s, 1, 'article', %s, %s, 'qa')",
            (seed.revision_id, unit_id, approval_id, "b" * 64),
        )
        regions = [(seed.region_id, 0)]
        if incomplete:
            regions.append((uuid4(), 1))
        for region_id, sequence_number in regions:
            _ = connection.execute(
                "INSERT INTO evidence.coherent_unit_span (revision_id, region_id, sequence_number, text_start, text_end) VALUES (%s, %s, %s, 0, 1)",
                (seed.revision_id, region_id, sequence_number),
            )
        _ = connection.execute(
            "INSERT INTO evidence.text_version (text_version_id, region_id, variant, text_content, text_sha256, review_status) VALUES (%s, %s, 'corrected_transcription', '文', %s, 'reviewed')",
            (seed.text_version_id, seed.region_id, "d" * 64),
        )
        _ = connection.execute(
            "INSERT INTO evidence.region_text_selection (selection_id, region_id, text_version_id, review_id, selection_basis, selected_by) VALUES (%s, %s, %s, %s, 'historian_approved', 'qa')",
            (seed.selection_id, seed.region_id, seed.text_version_id, uuid4()),
        )
        _ = connection.execute("SET session_replication_role = origin")
    try:
        yield seed
    finally:
        with psycopg.connect(database_url) as connection:
            _ = connection.execute("SET session_replication_role = replica")
            _ = connection.execute(
                "DELETE FROM retrieval.embedding WHERE target_id = %s",
                (seed.revision_id,),
            )
            _ = connection.execute(
                "DELETE FROM evidence.processing_run WHERE model_name = %s",
                (seed.model_name,),
            )
            for table in (
                "region_text_selection",
                "text_version",
                "coherent_unit_span",
            ):
                column = "revision_id" if table == "coherent_unit_span" else "region_id"
                value = (
                    seed.revision_id
                    if table == "coherent_unit_span"
                    else seed.region_id
                )
                _ = connection.execute(
                    f"DELETE FROM evidence.{table} WHERE {column} = %s", (value,)
                )
            _ = connection.execute(
                "DELETE FROM evidence.coherent_unit_revision WHERE revision_id = %s",
                (seed.revision_id,),
            )
            _ = connection.execute(
                "DELETE FROM evidence.coherent_unit WHERE unit_id = %s", (unit_id,)
            )
            _ = connection.execute(
                """DELETE FROM evidence.page_article_segmentation_selection
                WHERE selection_id = %s""",
                (approval_id,),
            )
            _ = connection.execute("SET session_replication_role = origin")


def identity(seed: SeededArticle) -> ArticleIdentity:
    configuration = WindowConfiguration("windowed_mean_v1", 6, 6, 6, 1, 1024)
    return ArticleIdentity(
        uuid4(),
        seed.revision_id,
        "coherent_unit_revision",
        seed.model_name,
        "revision",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        configuration,
    )


pytestmark = pytest.mark.skipif(
    DATABASE_URL is None, reason="requires migrated PostgreSQL"
)


def test_live_store_fresh_insert_and_exact_retry() -> None:
    # Given: a complete active reviewed article in migrated PostgreSQL.
    assert DATABASE_URL is not None
    with seeded_article(DATABASE_URL) as seed:
        store = PostgresArticleStore(DATABASE_URL)
        expected = identity(seed)
        vector = (1.0,) + (0.0,) * 1023

        # When: the exact completed embedding is persisted twice.
        first = store.persist_completed(expected, seed.selection, vector)
        retry = store.persist_completed(expected, seed.selection, vector)

        # Then: the first inserts and the exact retry reuses one atomic result.
        assert (first, retry, store.completed_run(expected)) == (True, False, True)


def test_live_store_rejects_stale_selection_without_partial_run() -> None:
    # Given: a complete revision but stale expected selection identity.
    assert DATABASE_URL is not None
    import psycopg

    with seeded_article(DATABASE_URL) as seed:
        store = PostgresArticleStore(DATABASE_URL)
        expected = identity(seed)
        stale = (ArticleSelection(0, seed.region_id, uuid4(), seed.text_version_id),)

        # When/Then: persistence rejects it and leaves no processing run behind.
        with pytest.raises(StaleReviewedArticleError):
            store.persist_completed(expected, stale, (1.0,) + (0.0,) * 1023)
        with psycopg.connect(DATABASE_URL) as connection:
            count = connection.execute(
                "SELECT count(*) FROM evidence.processing_run WHERE run_id = %s",
                (expected.run_id,),
            ).fetchone()
        assert count == (0,)


def test_live_store_detects_exact_identity_collision() -> None:
    # Given: an exact versioned identity already owned by another completed run.
    assert DATABASE_URL is not None
    import psycopg
    from psycopg.types.json import Jsonb

    with seeded_article(DATABASE_URL) as seed:
        store = PostgresArticleStore(DATABASE_URL)
        expected = identity(seed)
        other_run_id = uuid4()
        with psycopg.connect(DATABASE_URL) as connection:
            _ = connection.execute(
                """INSERT INTO evidence.processing_run (
                    run_id, kind, engine, model_name, model_revision,
                    configuration, status, started_at, completed_at
                ) VALUES (%s, 'embedding', 'sentence-transformers', %s, %s,
                          %s, 'completed', now(), now())""",
                (
                    other_run_id,
                    expected.model_name,
                    expected.model_revision,
                    Jsonb(asdict(expected.configuration)),
                ),
            )
            _ = connection.execute(
                """INSERT INTO retrieval.embedding (
                    target_kind, target_id, run_id, model_name, model_revision,
                    embedding, input_sha256, content_sha256, configuration_sha256
                ) VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, %s)""",
                (
                    expected.target_kind,
                    expected.revision_id,
                    other_run_id,
                    expected.model_name,
                    expected.model_revision,
                    "[1," + ",".join("0" for _ in range(1023)) + "]",
                    expected.input_sha256,
                    expected.content_sha256,
                    expected.configuration_sha256,
                ),
            )

        # When/Then: deterministic retry rejects the inconsistent owner.
        with pytest.raises(InconsistentArticleEmbeddingRunError, match="another run"):
            store.completed_run(expected)


def test_live_store_rolls_back_run_when_vector_insert_fails() -> None:
    # Given: a valid selection but a vector rejected by the 1024-dimension column.
    assert DATABASE_URL is not None
    import psycopg

    with seeded_article(DATABASE_URL) as seed:
        store = PostgresArticleStore(DATABASE_URL)
        expected = identity(seed)

        # When: vector insertion fails after the processing-run insert.
        with pytest.raises(psycopg.Error):
            store.persist_completed(expected, seed.selection, (1.0, 0.0, 0.0))

        # Then: the transaction rolls the completed run back too.
        with psycopg.connect(DATABASE_URL) as connection:
            count = connection.execute(
                "SELECT count(*) FROM evidence.processing_run WHERE run_id = %s",
                (expected.run_id,),
            ).fetchone()
        assert count == (0,)


def test_live_discovery_and_real_cli_exclude_incomplete_review() -> None:
    # Given: an active article with one span lacking an active reviewed selection.
    assert DATABASE_URL is not None
    with seeded_article(DATABASE_URL, incomplete=True) as seed:
        store = PostgresArticleStore(DATABASE_URL)

        # When: discovery and the real reviewed-unit CLI run without model fakes.
        discovered = store.discover(seed.revision_id)
        completed = subprocess.run(
            [
                "uv",
                "run",
                "wic-embed",
                "--database-url",
                DATABASE_URL,
                "--unit",
                "reviewed_coherent_unit",
            ],
            cwd=Path(__file__).parent.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        explicit = subprocess.run(
            [
                "uv",
                "run",
                "wic-embed",
                "--database-url",
                DATABASE_URL,
                "--unit",
                "reviewed_coherent_unit",
                "--revision-id",
                str(seed.revision_id),
            ],
            cwd=Path(__file__).parent.parent,
            check=False,
            capture_output=True,
            text=True,
        )

        # Then: batch reports zero while explicit targeting fails loudly.
        assert discovered == ()
        assert json.loads(completed.stdout)["revisions_discovered"] == 0
        assert explicit.returncode != 0
        assert str(seed.revision_id) in explicit.stderr
