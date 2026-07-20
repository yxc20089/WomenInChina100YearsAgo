from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Final, final
from uuid import UUID

from .article_embedding_contracts import (
    ArticleIdentity,
    ArticleSelection,
    InconsistentArticleEmbeddingRunError,
    StaleReviewedArticleError,
)

DISCOVER_SQL: Final = """
SELECT revision.revision_id
FROM evidence.coherent_unit_revision revision
JOIN evidence.page_article_segmentation_selection approval
  ON approval.selection_id = revision.approval_selection_id
 AND approval.superseded_at IS NULL
WHERE revision.superseded_at IS NULL
  AND revision.unit_kind = 'article'
  AND (%s::uuid IS NULL OR revision.revision_id = %s)
  AND EXISTS (
      SELECT 1 FROM evidence.coherent_unit_span present
      WHERE present.revision_id = revision.revision_id
  )
  AND NOT EXISTS (
      SELECT 1
      FROM evidence.coherent_unit_span required
      LEFT JOIN evidence.region_text_selection selection
        ON selection.region_id = required.region_id
       AND selection.superseded_at IS NULL
      LEFT JOIN evidence.text_version version
        ON version.text_version_id = selection.text_version_id
       AND version.region_id = required.region_id
       AND version.review_status = 'reviewed'
      WHERE required.revision_id = revision.revision_id
        AND version.text_version_id IS NULL
  )
ORDER BY revision.revision_id
"""
SELECTION_SQL: Final = """
SELECT span.sequence_number, span.region_id,
       selection.selection_id, selection.text_version_id
FROM evidence.coherent_unit_revision revision
JOIN evidence.page_article_segmentation_selection approval
  ON approval.selection_id = revision.approval_selection_id
 AND approval.superseded_at IS NULL
JOIN evidence.coherent_unit_span span
  ON span.revision_id = revision.revision_id
JOIN evidence.region_text_selection selection
  ON selection.region_id = span.region_id
 AND selection.superseded_at IS NULL
JOIN evidence.text_version version
  ON version.text_version_id = selection.text_version_id
 AND version.region_id = span.region_id
 AND version.review_status = 'reviewed'
WHERE revision.revision_id = %s
  AND revision.superseded_at IS NULL
  AND revision.unit_kind = 'article'
ORDER BY span.sequence_number
FOR UPDATE OF revision, approval, span, selection, version
"""
RUN_SQL: Final = """
SELECT run.status, run.engine, run.model_name, run.model_revision,
       run.configuration, count(embedding.embedding_id) AS embedding_count,
       count(embedding.embedding_id) FILTER (
           WHERE embedding.target_kind = %s
             AND embedding.target_id = %s
             AND embedding.model_name = %s
             AND embedding.model_revision = %s
             AND embedding.input_sha256 = %s
             AND embedding.content_sha256 = %s
             AND embedding.configuration_sha256 = %s
       ) AS exact_count
FROM evidence.processing_run run
LEFT JOIN retrieval.embedding embedding ON embedding.run_id = run.run_id
WHERE run.run_id = %s
GROUP BY run.run_id
"""
COLLISION_SQL: Final = """
SELECT count(*) AS exact_collision_count
FROM retrieval.embedding
WHERE target_kind = %s AND target_id = %s
  AND model_name = %s AND model_revision = %s
  AND input_sha256 = %s AND content_sha256 = %s
  AND configuration_sha256 = %s AND run_id <> %s
"""


@dataclass(frozen=True, slots=True)
class _RevisionRow:
    revision_id: UUID


@dataclass(frozen=True, slots=True)
class _RunRow:
    status: str
    engine: str
    model_name: str
    model_revision: str
    configuration: Mapping[str, int | str]
    embedding_count: int
    exact_count: int


@dataclass(frozen=True, slots=True)
class _CountRow:
    exact_collision_count: int


@final
class PostgresArticleStore:
    def __init__(self, database_url: str):
        self.database_url: str = database_url

    def discover(self, revision_id: UUID | None) -> tuple[UUID, ...]:
        import psycopg
        from psycopg.rows import class_row

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor(row_factory=class_row(_RevisionRow)) as cursor:
                rows = cursor.execute(
                    DISCOVER_SQL, (revision_id, revision_id)
                ).fetchall()
        return tuple(row.revision_id for row in rows)

    @staticmethod
    def _run_parameters(identity: ArticleIdentity) -> tuple[str | UUID, ...]:
        return (
            identity.target_kind,
            identity.revision_id,
            identity.model_name,
            identity.model_revision,
            identity.input_sha256,
            identity.content_sha256,
            identity.configuration_sha256,
            identity.run_id,
        )

    @staticmethod
    def _completed(
        identity: ArticleIdentity,
        row: _RunRow | None,
    ) -> bool:
        if row is None:
            return False
        expected = asdict(identity.configuration)
        if (
            row.status == "completed"
            and row.engine == "sentence-transformers"
            and row.model_name == identity.model_name
            and row.model_revision == identity.model_revision
            and row.configuration == expected
            and row.embedding_count == 1
            and row.exact_count == 1
        ):
            return True
        raise InconsistentArticleEmbeddingRunError(identity.run_id)

    def completed_run(self, identity: ArticleIdentity) -> bool:
        import psycopg
        from psycopg.rows import class_row

        parameters = self._run_parameters(identity)
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor(row_factory=class_row(_RunRow)) as cursor:
                row = cursor.execute(RUN_SQL, parameters).fetchone()
            with connection.cursor(row_factory=class_row(_CountRow)) as cursor:
                collision = cursor.execute(COLLISION_SQL, parameters).fetchone()
        if collision is None:
            raise InconsistentArticleEmbeddingRunError(
                identity.run_id, "exact embedding count query returned no row"
            )
        if collision.exact_collision_count:
            raise InconsistentArticleEmbeddingRunError(
                identity.run_id, "exact embedding identity belongs to another run"
            )
        return self._completed(identity, row)

    def persist_completed(
        self,
        identity: ArticleIdentity,
        selection: tuple[ArticleSelection, ...],
        vector: tuple[float, ...],
    ) -> bool:
        import psycopg
        from psycopg.rows import class_row
        from psycopg.types.json import Jsonb

        with psycopg.connect(self.database_url) as connection:
            _ = connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (str(identity.run_id),),
            )
            with connection.cursor(row_factory=class_row(_RunRow)) as cursor:
                row = cursor.execute(RUN_SQL, self._run_parameters(identity)).fetchone()
            if row is not None:
                _ = self._completed(identity, row)
                return False
            with connection.cursor(row_factory=class_row(ArticleSelection)) as cursor:
                current = tuple(
                    cursor.execute(SELECTION_SQL, (identity.revision_id,)).fetchall()
                )
            if current != selection:
                raise StaleReviewedArticleError(identity.revision_id)
            _ = connection.execute(
                """INSERT INTO evidence.processing_run (
                    run_id, kind, engine, model_name, model_revision,
                    software_version, configuration, status, started_at, completed_at
                ) VALUES (
                    %s, 'embedding', 'sentence-transformers', %s, %s,
                    'sentence-transformers-5.x', %s, 'completed', now(), now()
                )""",
                (
                    identity.run_id,
                    identity.model_name,
                    identity.model_revision,
                    Jsonb(asdict(identity.configuration)),
                ),
            )
            literal = "[" + ",".join(format(value, ".9g") for value in vector) + "]"
            _ = connection.execute(
                """INSERT INTO retrieval.embedding (
                    target_kind, target_id, run_id, model_name, model_revision,
                    embedding, input_sha256, content_sha256, configuration_sha256
                ) VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, %s)""",
                (
                    identity.target_kind,
                    identity.revision_id,
                    identity.run_id,
                    identity.model_name,
                    identity.model_revision,
                    literal,
                    identity.input_sha256,
                    identity.content_sha256,
                    identity.configuration_sha256,
                ),
            )
        return True
