from __future__ import annotations

import ast
import inspect
import os
import textwrap
from collections.abc import Callable
from uuid import uuid4

import pytest

from wic_history.embedding_pipeline import EmbeddingResult, embed_regions
from wic_history.identity_models import (
    IdentityEmbeddingResult,
    persist_identity_profile_embeddings,
)


DATABASE_URL = os.environ.get("DATABASE_URL")


Writer = Callable[..., EmbeddingResult | IdentityEmbeddingResult]


def _embedding_insert_sql(function: Writer) -> str:
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    statements = (
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "INSERT INTO retrieval.embedding" in node.value
    )
    return next(statements)


def _assert_legacy_writer_is_idempotent(sql: str) -> None:
    import psycopg

    assert DATABASE_URL is not None
    run_id = uuid4()
    target_id = uuid4()
    vector = "[" + ",".join("0" for _ in range(1024)) + "]"
    with psycopg.connect(DATABASE_URL) as connection:
        _ = connection.execute(
            """
            INSERT INTO evidence.processing_run (
                run_id, kind, engine, model_name, model_revision, started_at
            ) VALUES (%s, 'embedding', 'qa', 'qa', '1', now())
            """,
            (run_id,),
        )
        query = sql.encode("utf-8")
        first = connection.execute(
            query, (target_id, run_id, "qa-model", "1", vector)
        )
        second = connection.execute(
            query, (target_id, run_id, "qa-model", "1", vector)
        )
        assert first.rowcount == 1
        assert second.rowcount == 0
        connection.rollback()


@pytest.mark.skipif(DATABASE_URL is None, reason="requires migrated PostgreSQL")
@pytest.mark.parametrize(
    "writer",
    (embed_regions, persist_identity_profile_embeddings),
)
def test_legacy_embedding_writer_is_idempotent_after_partial_index_migration(
    writer: Writer,
) -> None:
    # Given: a migrated PostgreSQL database and the exact INSERT used by a legacy writer.
    sql = _embedding_insert_sql(writer)

    # When: the writer INSERT is executed twice for the same legacy identity.
    _assert_legacy_writer_is_idempotent(sql)

    # Then: the first insert succeeds and the second follows the partial-index conflict path.
