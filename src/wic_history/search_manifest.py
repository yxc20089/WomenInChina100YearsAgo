from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import TypeAdapter

from .coherent_search import (
    CoherentEmbedding,
    CoherentProjectionError,
    CoherentSource,
    FrozenProjectionManifest,
    ProjectionArticle,
)
from .evidence import Polygon, SourcePointer
from .search_manifest_sql import ACTIVE_ARTICLES_SQL, EMBEDDING_SQL, SOURCES_SQL
from .semantic_repository import CoherentTextBundle, load_reviewed_coherent_text


EvidenceTier = Literal[
    "screening_derivative",
    "unreviewed_input",
    "non_gold_lossless_pilot",
    "historian_selected_gold",
]


@dataclass(frozen=True, slots=True)
class CoherentProjectionPins:
    model_name: str
    model_revision: str
    configuration_sha256: str
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class _ArticleRow:
    revision_id: UUID
    unit_id: UUID
    title: str | None


@dataclass(frozen=True, slots=True)
class _EmbeddingRow:
    vector: str


@dataclass(frozen=True, slots=True)
class _SourceRow:
    region_id: UUID
    source_uri: str
    source_sha256: str
    page_id: UUID
    derivative_id: UUID
    image_uri: str
    image_sha256: str
    evidence_tier: EvidenceTier
    volume_number: int
    publication_year: int
    page_number: int
    polygon: Polygon
    warnings: list[str] | None


def _article_sources(
    rows: tuple[_SourceRow, ...],
    bundle: CoherentTextBundle,
) -> tuple[CoherentSource, ...]:
    by_region = {row.region_id: row for row in rows}
    if len(by_region) != len(rows) or set(by_region) != {
        segment.region_id for segment in bundle.segments
    }:
        raise CoherentProjectionError(
            "coherent projection source provenance is incomplete or ambiguous"
        )
    sources: list[CoherentSource] = []
    for segment in bundle.segments:
        row = by_region[segment.region_id]
        warnings = tuple(row.warnings or ())
        sources.append(
            CoherentSource(
                segment.sequence_number,
                SourcePointer(
                    source_uri=row.source_uri,
                    source_sha256=row.source_sha256,
                    page_id=row.page_id,
                    derivative_id=row.derivative_id,
                    image_uri=row.image_uri,
                    image_sha256=row.image_sha256,
                    evidence_tier=row.evidence_tier,
                    volume_number=row.volume_number,
                    publication_year=row.publication_year,
                    page_number=row.page_number,
                    region_id=segment.region_id,
                    text_version_id=segment.text_version_id,
                    text_selection_id=segment.selection_id,
                    polygon=row.polygon,
                    text_start=segment.text_start,
                    text_end=segment.text_end,
                ),
                warnings,
            )
        )
    return tuple(sources)


def _embedding(
    rows: tuple[_EmbeddingRow, ...],
    bundle: CoherentTextBundle,
    pins: CoherentProjectionPins,
) -> CoherentEmbedding:
    if len(rows) != 1:
        raise CoherentProjectionError(
            "active reviewed article lacks one exact completed embedding"
        )
    vector = TypeAdapter(list[float]).validate_python(json.loads(rows[0].vector))
    return CoherentEmbedding(
        bundle.coherent_unit_revision_id,
        "coherent_unit_revision",
        pins.model_name,
        pins.model_revision,
        bundle.input_sha256,
        bundle.content_sha256,
        pins.configuration_sha256,
        tuple(float(value) for value in vector),
    )


def load_coherent_projection_manifest(
    database_url: str,
    pins: CoherentProjectionPins,
) -> FrozenProjectionManifest:
    """Freeze every active reviewed article with one exact completed embedding."""
    if not pins.model_name or not pins.model_revision:
        raise CoherentProjectionError("coherent projection model identity is incomplete")
    for label, value in (
        ("configuration_sha256", pins.configuration_sha256),
        ("snapshot_sha256", pins.snapshot_sha256),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise CoherentProjectionError(
                f"coherent projection {label} must be a lowercase SHA-256 hash"
            )
    try:
        import psycopg
        from psycopg.rows import class_row
    except ImportError as exc:
        raise CoherentProjectionError(
            "Install the data extra: uv sync --extra data"
        ) from exc
    with psycopg.connect(database_url) as connection:
        with connection.cursor(row_factory=class_row(_ArticleRow)) as cursor:
            revisions = tuple(cursor.execute(ACTIVE_ARTICLES_SQL).fetchall())
        if not revisions:
            raise CoherentProjectionError(
                "coherent projection requires active reviewed articles"
            )
        articles: list[ProjectionArticle] = []
        embeddings: list[CoherentEmbedding] = []
        bundles: list[CoherentTextBundle] = []
        for row in revisions:
            bundle = load_reviewed_coherent_text(database_url, row.revision_id)
            bundles.append(bundle)
            with connection.cursor(row_factory=class_row(_SourceRow)) as cursor:
                source_rows = tuple(
                    cursor.execute(
                        SOURCES_SQL,
                        ([segment.region_id for segment in bundle.segments],),
                    ).fetchall()
                )
            with connection.cursor(row_factory=class_row(_EmbeddingRow)) as cursor:
                embedding_rows = tuple(
                    cursor.execute(
                        EMBEDDING_SQL,
                        (
                            bundle.coherent_unit_revision_id,
                            pins.model_name,
                            pins.model_revision,
                            bundle.input_sha256,
                            bundle.content_sha256,
                            pins.configuration_sha256,
                        ),
                    ).fetchall()
                )
            articles.append(
                ProjectionArticle(
                    row.unit_id,
                    row.title or "",
                    bundle,
                    _article_sources(source_rows, bundle),
                )
            )
            embeddings.append(_embedding(embedding_rows, bundle, pins))
        with connection.cursor(row_factory=class_row(_ArticleRow)) as cursor:
            current = tuple(cursor.execute(ACTIVE_ARTICLES_SQL).fetchall())
    if current != revisions:
        raise CoherentProjectionError(
            "active reviewed article snapshot changed during projection assembly"
        )
    refreshed = tuple(
        load_reviewed_coherent_text(database_url, row.revision_id)
        for row in current
    )
    identities = tuple(
        (
            bundle.input_sha256,
            bundle.content_sha256,
            tuple(
                (segment.selection_id, segment.text_version_id)
                for segment in bundle.segments
            ),
        )
        for bundle in bundles
    )
    refreshed_identities = tuple(
        (
            bundle.input_sha256,
            bundle.content_sha256,
            tuple(
                (segment.selection_id, segment.text_version_id)
                for segment in bundle.segments
            ),
        )
        for bundle in refreshed
    )
    if identities != refreshed_identities:
        raise CoherentProjectionError(
            "reviewed text selection changed during projection assembly"
        )
    manifest = FrozenProjectionManifest.freeze(tuple(articles), tuple(embeddings))
    if manifest.snapshot_sha256 != pins.snapshot_sha256:
        raise CoherentProjectionError("coherent projection snapshot hash is stale")
    return manifest
