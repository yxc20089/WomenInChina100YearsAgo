from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, final, override
from uuid import UUID

from .evidence import SourcePointer
from .semantic_inputs import CoherentTextBundle, CoherentTextSegment


COHERENT_ALIAS: Final = "wic-coherent-units-current"
COHERENT_INDEX_PREFIX: Final = "wic-coherent-units-build-"
EMBEDDING_DIMENSION: Final = 1024
TARGET_KIND: Final = "coherent_unit_revision"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


@final
class CoherentProjectionError(RuntimeError):
    reason: str

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class CoherentSource:
    sequence_number: int
    source: SourcePointer
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectionArticle:
    coherent_unit_id: UUID
    title: str
    bundle: CoherentTextBundle
    sources: tuple[CoherentSource, ...]
    unit_kind: Literal["article"] = "article"
    active: bool = True
    reviewed: bool = True


@dataclass(frozen=True, slots=True)
class CoherentEmbedding:
    revision_id: UUID
    target_kind: Literal["coherent_unit_revision"]
    model_name: str
    model_revision: str
    input_sha256: str
    content_sha256: str
    configuration_sha256: str
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FrozenProjectionManifest:
    articles: tuple[ProjectionArticle, ...]
    embeddings: tuple[CoherentEmbedding, ...]
    model_name: str
    model_revision: str
    configuration_sha256: str
    snapshot_sha256: str

    @classmethod
    def freeze(
        cls,
        articles: tuple[ProjectionArticle, ...],
        embeddings: tuple[CoherentEmbedding, ...],
    ) -> FrozenProjectionManifest:
        first = embeddings[0] if embeddings else None
        provisional = cls(
            articles,
            embeddings,
            first.model_name if first else "",
            first.model_revision if first else "",
            first.configuration_sha256 if first else "",
            "",
        )
        return dataclass_replace_snapshot(provisional, snapshot_sha256(provisional))


def dataclass_replace_snapshot(
    manifest: FrozenProjectionManifest, snapshot_sha256: str
) -> FrozenProjectionManifest:
    return FrozenProjectionManifest(
        manifest.articles,
        manifest.embeddings,
        manifest.model_name,
        manifest.model_revision,
        manifest.configuration_sha256,
        snapshot_sha256,
    )


def _segment_identity(segment: CoherentTextSegment) -> dict[str, JsonValue]:
    return {
        "sequence_number": segment.sequence_number,
        "region_id": str(segment.region_id),
        "page_id": str(segment.page_id),
        "text_version_id": str(segment.text_version_id),
        "selection_id": str(segment.selection_id),
        "text_start": segment.text_start,
        "text_end": segment.text_end,
        "composite_start": segment.composite_start,
        "composite_end": segment.composite_end,
        "text": segment.text,
        "role": segment.role,
    }


def snapshot_sha256(manifest: FrozenProjectionManifest) -> str:
    articles: list[JsonValue] = [
        {
            "coherent_unit_id": str(article.coherent_unit_id),
            "title": article.title,
            "revision_id": str(article.bundle.coherent_unit_revision_id),
            "content": article.bundle.content,
            "input_sha256": article.bundle.input_sha256,
            "content_sha256": article.bundle.content_sha256,
            "unit_kind": article.unit_kind,
            "active": article.active,
            "reviewed": article.reviewed,
            "segments": [_segment_identity(value) for value in article.bundle.segments],
            "sources": [
                {
                    "sequence_number": source.sequence_number,
                    "source": source.source.model_dump(mode="json"),
                    "warnings": list(source.warnings),
                }
                for source in article.sources
            ],
        }
        for article in sorted(
            manifest.articles,
            key=lambda value: str(value.bundle.coherent_unit_revision_id),
        )
    ]
    embeddings: list[JsonValue] = [
        {
            "revision_id": str(value.revision_id),
            "target_kind": value.target_kind,
            "model_name": value.model_name,
            "model_revision": value.model_revision,
            "input_sha256": value.input_sha256,
            "content_sha256": value.content_sha256,
            "configuration_sha256": value.configuration_sha256,
            "vector": list(value.vector),
        }
        for value in sorted(
            manifest.embeddings, key=lambda value: str(value.revision_id)
        )
    ]
    identity: JsonValue = {
        "model_name": manifest.model_name,
        "model_revision": manifest.model_revision,
        "configuration_sha256": manifest.configuration_sha256,
        "articles": articles,
        "embeddings": embeddings,
    }
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    build_id: str
    index_name: str
    documents_indexed: int
    source_snapshot_sha256: str
    previous_index_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchSpec:
    query: str
    limit: int = 10
    year_min: int | None = None
    year_max: int | None = None
    candidate_limit: int = 50
    rrf_k: int = 60
    model_name: str | None = None
    model_revision: str | None = None
    configuration_sha256: str | None = None


class QueryEmbedder(Protocol):
    model_name: str
    model_revision: str
    configuration_sha256: str

    def encode_query(self, query: str) -> list[float]: ...


def require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CoherentProjectionError(f"{label} must be a lowercase SHA-256 hash")


def coherent_index_body() -> Mapping[str, JsonValue]:
    source_properties: dict[str, JsonValue] = {
        "sequence_number": {"type": "integer"},
        "document_start": {"type": "integer"},
        "document_end": {"type": "integer"},
        "role": {"type": "keyword"},
        "source_uri": {"type": "keyword"},
        "source_sha256": {"type": "keyword"},
        "page_id": {"type": "keyword"},
        "derivative_id": {"type": "keyword"},
        "image_uri": {"type": "keyword"},
        "image_sha256": {"type": "keyword"},
        "evidence_tier": {"type": "keyword"},
        "volume_number": {"type": "integer"},
        "publication_year": {"type": "integer"},
        "page_number": {"type": "integer"},
        "region_id": {"type": "keyword"},
        "text_version_id": {"type": "keyword"},
        "text_selection_id": {"type": "keyword"},
        "text_start": {"type": "integer"},
        "text_end": {"type": "integer"},
        "polygon": {"type": "object", "enabled": False},
        "warnings": {"type": "keyword", "ignore_above": 4096},
    }
    properties: dict[str, JsonValue] = {
        "revision_id": {"type": "keyword"},
        "coherent_unit_id": {"type": "keyword"},
        "title": {"type": "text", "analyzer": "cjk"},
        "content": {"type": "text", "analyzer": "cjk"},
        "input_sha256": {"type": "keyword"},
        "content_sha256": {"type": "keyword"},
        "embedding_model": {"type": "keyword"},
        "embedding_model_revision": {"type": "keyword"},
        "embedding_configuration_sha256": {"type": "keyword"},
        "year_min": {"type": "integer"},
        "year_max": {"type": "integer"},
        "sources": {
            "type": "nested",
            "dynamic": "strict",
            "properties": source_properties,
        },
        "embedding": {
            "type": "knn_vector",
            "dimension": EMBEDDING_DIMENSION,
            "method": {
                "name": "hnsw",
                "space_type": "cosinesimil",
                "engine": "lucene",
            },
        },
    }
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn": True,
            }
        },
        "mappings": {"dynamic": "strict", "properties": properties},
    }
