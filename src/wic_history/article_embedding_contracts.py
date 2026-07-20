from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, override
from uuid import UUID


@dataclass(frozen=True, slots=True)
class InconsistentArticleEmbeddingRunError(RuntimeError):
    run_id: UUID
    reason: str = "deterministic embedding run is partial or inconsistent"

    @override
    def __str__(self) -> str:
        return f"{self.reason}: {self.run_id}"


@dataclass(frozen=True, slots=True)
class StaleReviewedArticleError(RuntimeError):
    revision_id: UUID

    @override
    def __str__(self) -> str:
        return (
            f"reviewed article selection changed during embedding: {self.revision_id}"
        )


@dataclass(frozen=True, slots=True)
class ReviewedArticleUnavailableError(ValueError):
    revision_id: UUID

    @override
    def __str__(self) -> str:
        return f"reviewed article revision is missing or incomplete: {self.revision_id}"


@dataclass(frozen=True, slots=True)
class MissingArticleEmbeddingDependencyError(RuntimeError):
    dependency: str

    @override
    def __str__(self) -> str:
        return f"article embedding dependency is unavailable: {self.dependency}"


@dataclass(frozen=True, slots=True)
class ArticleSelection:
    sequence_number: int
    region_id: UUID
    selection_id: UUID
    text_version_id: UUID


@dataclass(frozen=True, slots=True)
class ReviewedArticle:
    revision_id: UUID
    content: str
    input_sha256: str
    content_sha256: str
    selection: tuple[ArticleSelection, ...]


@dataclass(frozen=True, slots=True)
class WindowConfiguration:
    policy: str
    tokenizer_limit: int
    model_limit: int
    effective_limit: int
    overlap_tokens: int
    dimension: int


@dataclass(frozen=True, slots=True)
class ArticleIdentity:
    run_id: UUID
    revision_id: UUID
    target_kind: str
    model_name: str
    model_revision: str
    input_sha256: str
    content_sha256: str
    configuration_sha256: str
    configuration: WindowConfiguration


@dataclass(frozen=True, slots=True)
class ArticleEmbeddingRequest:
    database_url: str
    model_name: str
    model_revision: str
    batch_size: int
    revision_id: UUID | None = None
    expected_configuration_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ArticleEmbeddingSummary:
    revisions_discovered: int
    embeddings_inserted: int
    embeddings_reused: int
    run_ids: tuple[str, ...]
    model_name: str
    model_revision: str


class ArticleEncoder(Protocol):
    tokenizer_limit: int
    model_limit: int

    def tokenize(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def encode(self, texts: list[str], batch_size: int) -> list[list[float]]: ...


class ArticleStore(Protocol):
    def discover(self, revision_id: UUID | None) -> tuple[UUID, ...]: ...
    def completed_run(self, identity: ArticleIdentity) -> bool: ...
    def persist_completed(
        self,
        identity: ArticleIdentity,
        selection: tuple[ArticleSelection, ...],
        vector: tuple[float, ...],
    ) -> bool: ...


class ReviewedLoader(Protocol):
    def __call__(self, database_url: str, revision_id: UUID, /) -> ReviewedArticle: ...


class EncoderFactory(Protocol):
    def __call__(self, model_name: str, model_revision: str, /) -> ArticleEncoder: ...


@dataclass(frozen=True, slots=True)
class ArticleEmbeddingRuntime:
    store: ArticleStore
    load_reviewed: ReviewedLoader
    encoder_factory: EncoderFactory
