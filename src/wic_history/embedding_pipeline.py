"""Generate versioned dense embeddings and persist them in pgvector."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Sequence, assert_never
from uuid import UUID, uuid4

from .model_config import load_pipeline_model_configuration

_PIPELINE_MODELS = load_pipeline_model_configuration()
DEFAULT_MODEL = _PIPELINE_MODELS.retrieval.passage_embedding.model_name
DEFAULT_REVISION = _PIPELINE_MODELS.retrieval.passage_embedding.model_revision
EMBEDDING_DIMENSION = _PIPELINE_MODELS.retrieval.passage_embedding.dimension


class EmbeddingUnit(StrEnum):
    REGION = "region"
    REVIEWED_COHERENT_UNIT = "reviewed_coherent_unit"


class BGEEmbedder:
    def __init__(
        self, model_name: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - minimal installations
            raise RuntimeError("Install the NER extra: uv sync --extra ner") from exc
        self.model_name = model_name
        self.revision = revision
        self.model = SentenceTransformer(model_name, revision=revision, device="cpu")

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        values = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )
        vectors = values.tolist()
        if any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional BGE-M3 vectors"
            )
        return vectors

    def encode_query(self, query: str) -> list[float]:
        return self.encode([query], batch_size=1)[0]


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(format(value, ".9g") for value in vector) + "]"


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    run_id: str
    regions_processed: int
    embeddings_inserted: int
    model_name: str
    model_revision: str


def embed_regions(
    database_url: str,
    model_name: str = DEFAULT_MODEL,
    model_revision: str = DEFAULT_REVISION,
    batch_size: int = 16,
    source_ocr_run_id: str | None = None,
) -> EmbeddingResult:
    try:
        import psycopg
        from psycopg.types.json import Jsonb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc

    with psycopg.connect(database_url) as connection:
        parameters: tuple[Any, ...] = ()
        where = ""
        if source_ocr_run_id:
            where = "WHERE r.run_id = %s"
            parameters = (source_ocr_run_id,)
        rows = connection.execute(
            f"""
            SELECT r.region_id, COALESCE(r.normalized_text, r.raw_text) AS text
            FROM evidence.ocr_region r
            {where}
            ORDER BY r.region_id
            """,
            parameters,
        ).fetchall()
    if not rows:
        raise ValueError("No OCR regions matched the embedding request")

    started_at = datetime.now(timezone.utc)
    embedder = BGEEmbedder(model_name, model_revision)
    vectors = embedder.encode([row[1] for row in rows], batch_size)
    completed_at = datetime.now(timezone.utc)
    run_id = uuid4()
    inserted = 0
    with psycopg.connect(database_url) as connection:
        connection.execute(
            """
            INSERT INTO evidence.processing_run (
                run_id, kind, engine, model_name, model_revision, software_version,
                configuration, status, started_at, completed_at
            ) VALUES (%s, 'embedding', 'sentence-transformers', %s, %s, %s, %s,
                      'completed', %s, %s)
            """,
            (
                run_id,
                model_name,
                model_revision,
                "sentence-transformers-5.x",
                Jsonb(
                    {
                        "normalized": True,
                        "dimension": EMBEDDING_DIMENSION,
                        "batch_size": batch_size,
                    }
                ),
                started_at,
                completed_at,
            ),
        )
        for (region_id, _), vector in zip(rows, vectors, strict=True):
            cursor = connection.execute(
                """
                INSERT INTO retrieval.embedding (
                    target_kind, target_id, run_id, model_name, model_revision, embedding
                ) VALUES ('region', %s, %s, %s, %s, %s::vector)
                ON CONFLICT (target_kind, target_id, model_name, model_revision)
                WHERE input_sha256 IS NULL
                  AND content_sha256 IS NULL
                  AND configuration_sha256 IS NULL
                DO NOTHING
                """,
                (
                    region_id,
                    run_id,
                    model_name,
                    model_revision,
                    _vector_literal(vector),
                ),
            )
            inserted += cursor.rowcount
    return EmbeddingResult(str(run_id), len(rows), inserted, model_name, model_revision)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--model-config",
        help="Complete model configuration; individual model overrides are not accepted",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--source-ocr-run-id")
    parser.add_argument(
        "--unit",
        type=EmbeddingUnit,
        choices=tuple(EmbeddingUnit),
        default=EmbeddingUnit.REGION,
        help="Embedding unit (default: region)",
    )
    parser.add_argument("--revision-id", type=UUID)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    configuration = load_pipeline_model_configuration(args.model_config)
    embedding = configuration.retrieval.passage_embedding
    match args.unit:
        case EmbeddingUnit.REGION:
            result = embed_regions(
                args.database_url,
                embedding.model_name,
                embedding.model_revision,
                args.batch_size,
                args.source_ocr_run_id,
            )
        case EmbeddingUnit.REVIEWED_COHERENT_UNIT:
            from .article_embedding import (
                ArticleEmbeddingRequest,
                embed_reviewed_articles,
            )

            result = embed_reviewed_articles(
                ArticleEmbeddingRequest(
                    args.database_url,
                    embedding.model_name,
                    embedding.model_revision,
                    args.batch_size,
                    args.revision_id,
                )
            )
        case unreachable:
            assert_never(unreachable)
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
