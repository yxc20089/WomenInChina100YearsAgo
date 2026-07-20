from __future__ import annotations

import hashlib

from .coherent_search_contracts import (
    EMBEDDING_DIMENSION,
    TARGET_KIND,
    CoherentEmbedding,
    CoherentProjectionError,
    CoherentSource,
    FrozenProjectionManifest,
    JsonValue,
    ProjectionArticle,
    require_sha256,
    snapshot_sha256,
)
from .semantic_repository import CoherentTextSegment


def _validate_segments(article: ProjectionArticle) -> tuple[CoherentTextSegment, ...]:
    ordered = tuple(
        sorted(article.bundle.segments, key=lambda value: value.sequence_number)
    )
    if not ordered or tuple(value.sequence_number for value in ordered) != tuple(
        range(len(ordered))
    ):
        raise CoherentProjectionError(
            "canonical segments require contiguous ordered sequence numbers"
        )
    position = 0
    for sequence, segment in enumerate(ordered):
        if sequence:
            position += 1
        expected_end = position + len(segment.text)
        if (segment.composite_start, segment.composite_end) != (position, expected_end):
            raise CoherentProjectionError(
                "canonical segment has an invalid composite interval"
            )
        if (
            not segment.text
            or article.bundle.content[position:expected_end] != segment.text
        ):
            raise CoherentProjectionError(
                "canonical content differs from its segment text"
            )
        position = expected_end
    reconstructed = "\n".join(segment.text for segment in ordered)
    if reconstructed != article.bundle.content:
        raise CoherentProjectionError(
            "canonical content differs from its ordered segments"
        )
    return ordered


def _source_document(
    segment: CoherentTextSegment, provenance: CoherentSource
) -> dict[str, JsonValue]:
    source = provenance.source
    required = (
        source.source_sha256,
        source.page_id,
        source.derivative_id,
        source.image_uri,
        source.image_sha256,
        source.evidence_tier,
        source.volume_number,
        source.publication_year,
        source.region_id,
        source.text_version_id,
        source.text_selection_id,
        source.polygon,
        source.text_start,
        source.text_end,
    )
    if any(value is None for value in required):
        raise CoherentProjectionError(
            f"source span {segment.sequence_number} has incomplete atomic provenance"
        )
    if provenance.sequence_number != segment.sequence_number:
        raise CoherentProjectionError(
            "source span sequence differs from its canonical segment"
        )
    if (
        source.page_id,
        source.region_id,
        source.text_version_id,
        source.text_selection_id,
    ) != (
        segment.page_id,
        segment.region_id,
        segment.text_version_id,
        segment.selection_id,
    ):
        raise CoherentProjectionError(
            "source span identity differs from its canonical segment"
        )
    if (source.text_start, source.text_end) != (segment.text_start, segment.text_end):
        raise CoherentProjectionError(
            "source span offsets differ from its canonical segment"
        )
    if segment.text_end - segment.text_start != len(segment.text):
        raise CoherentProjectionError(
            "canonical segment selected interval length differs from its text"
        )
    if (
        source.source_sha256 is None
        or source.image_sha256 is None
        or source.polygon is None
    ):
        raise CoherentProjectionError("source span has incomplete atomic provenance")
    require_sha256(source.source_sha256, "source_sha256")
    require_sha256(source.image_sha256, "image_sha256")
    return {
        "sequence_number": segment.sequence_number,
        "document_start": segment.composite_start,
        "document_end": segment.composite_end,
        "role": segment.role,
        "source_uri": source.source_uri,
        "source_sha256": source.source_sha256,
        "page_id": str(source.page_id),
        "derivative_id": str(source.derivative_id),
        "image_uri": source.image_uri,
        "image_sha256": source.image_sha256,
        "evidence_tier": source.evidence_tier,
        "volume_number": source.volume_number,
        "publication_year": source.publication_year,
        "page_number": source.page_number,
        "region_id": str(source.region_id),
        "text_version_id": str(source.text_version_id),
        "text_selection_id": str(source.text_selection_id),
        "text_start": source.text_start,
        "text_end": source.text_end,
        "polygon": source.polygon.model_dump(mode="json"),
        "warnings": list(provenance.warnings),
    }


def article_document(
    article: ProjectionArticle, embedding: CoherentEmbedding
) -> dict[str, JsonValue]:
    bundle = article.bundle
    if not article.active or not article.reviewed or article.unit_kind != "article":
        raise CoherentProjectionError(
            "projection requires active reviewed article revisions"
        )
    if not article.title.strip() or not bundle.content.strip():
        raise CoherentProjectionError("article projection input is incomplete")
    ordered = _validate_segments(article)
    require_sha256(bundle.input_sha256, "article input_sha256")
    require_sha256(bundle.content_sha256, "article content_sha256")
    if hashlib.sha256(bundle.content.encode()).hexdigest() != bundle.content_sha256:
        raise CoherentProjectionError("article content differs from content_sha256")
    if (
        embedding.target_kind != TARGET_KIND
        or embedding.revision_id != bundle.coherent_unit_revision_id
    ):
        raise CoherentProjectionError("article lacks a matching embedding target")
    for label, value in (
        ("embedding input_sha256", embedding.input_sha256),
        ("embedding content_sha256", embedding.content_sha256),
        ("embedding configuration_sha256", embedding.configuration_sha256),
    ):
        require_sha256(value, label)
    if (embedding.input_sha256, embedding.content_sha256) != (
        bundle.input_sha256,
        bundle.content_sha256,
    ):
        raise CoherentProjectionError(
            "embedding hashes do not match canonical reviewed text"
        )
    if len(embedding.vector) != EMBEDDING_DIMENSION:
        raise CoherentProjectionError(
            f"embedding must have {EMBEDDING_DIMENSION} dimensions"
        )
    provenance = {value.sequence_number: value for value in article.sources}
    if set(provenance) != set(range(len(ordered))) or len(provenance) != len(
        article.sources
    ):
        raise CoherentProjectionError(
            "every canonical segment requires matching source provenance"
        )
    sources = [
        _source_document(segment, provenance[segment.sequence_number])
        for segment in ordered
    ]
    years = [source.source.publication_year for source in article.sources]
    if any(year is None for year in years):
        raise CoherentProjectionError("source publication year is incomplete")
    complete_years = [year for year in years if year is not None]
    return {
        "revision_id": str(bundle.coherent_unit_revision_id),
        "coherent_unit_id": str(article.coherent_unit_id),
        "title": article.title,
        "content": bundle.content,
        "input_sha256": bundle.input_sha256,
        "content_sha256": bundle.content_sha256,
        "embedding_model": embedding.model_name,
        "embedding_model_revision": embedding.model_revision,
        "embedding_configuration_sha256": embedding.configuration_sha256,
        "year_min": min(complete_years),
        "year_max": max(complete_years),
        "sources": sources,
        "embedding": list(embedding.vector),
    }


def validated_documents(
    manifest: FrozenProjectionManifest,
) -> list[tuple[str, dict[str, JsonValue]]]:
    if not manifest.articles:
        raise CoherentProjectionError("projection manifest must not be empty")
    revisions = [
        article.bundle.coherent_unit_revision_id for article in manifest.articles
    ]
    embeddings = {value.revision_id: value for value in manifest.embeddings}
    if len(embeddings) != len(manifest.embeddings) or set(embeddings) != set(revisions):
        raise CoherentProjectionError(
            "every article requires exactly one matching embedding"
        )
    require_sha256(manifest.configuration_sha256, "manifest configuration_sha256")
    require_sha256(manifest.snapshot_sha256, "manifest snapshot_sha256")
    if not manifest.model_name or not manifest.model_revision:
        raise CoherentProjectionError(
            "manifest pinned embedding model identity is incomplete"
        )
    if manifest.snapshot_sha256 != snapshot_sha256(manifest):
        raise CoherentProjectionError("projection manifest snapshot hash is stale")
    unit_ids = [article.coherent_unit_id for article in manifest.articles]
    if len(set(revisions)) != len(revisions):
        raise CoherentProjectionError(
            "projection manifest contains duplicate article revisions"
        )
    if len(set(unit_ids)) != len(unit_ids):
        raise CoherentProjectionError(
            "projection manifest contains duplicate coherent_unit_id"
        )
    pinned = (
        manifest.model_name,
        manifest.model_revision,
        manifest.configuration_sha256,
    )
    if any(
        (value.model_name, value.model_revision, value.configuration_sha256) != pinned
        for value in manifest.embeddings
    ):
        raise CoherentProjectionError(
            "embedding differs from the manifest pinned embedding model"
        )
    return [
        (
            str(article.bundle.coherent_unit_revision_id),
            article_document(
                article, embeddings[article.bundle.coherent_unit_revision_id]
            ),
        )
        for article in manifest.articles
    ]
