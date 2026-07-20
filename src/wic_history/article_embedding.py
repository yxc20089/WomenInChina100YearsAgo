from __future__ import annotations

from uuid import UUID

from .article_embedding_contracts import (
    ArticleEmbeddingRequest,
    ArticleEmbeddingRuntime,
    ArticleEmbeddingSummary,
    ArticleIdentity,
    ArticleSelection,
    InconsistentArticleEmbeddingRunError,
    ReviewedArticle,
    ReviewedArticleUnavailableError,
    StaleReviewedArticleError,
)
from .article_embedding_store import PostgresArticleStore
from .article_embedding_vectors import (
    BGEArticleEncoder,
    article_identity,
    encode_windowed_mean,
    window_configuration,
)
from .semantic_repository import CoherentTextBundle, load_reviewed_coherent_text

__all__ = (
    "ArticleEmbeddingRequest",
    "ArticleEmbeddingRuntime",
    "ArticleEmbeddingSummary",
    "ArticleIdentity",
    "ArticleSelection",
    "InconsistentArticleEmbeddingRunError",
    "PostgresArticleStore",
    "ReviewedArticle",
    "ReviewedArticleUnavailableError",
    "StaleReviewedArticleError",
    "embed_reviewed_articles",
    "encode_windowed_mean",
)


def _from_bundle(bundle: CoherentTextBundle) -> ReviewedArticle:
    return ReviewedArticle(
        bundle.coherent_unit_revision_id,
        bundle.content,
        bundle.input_sha256,
        bundle.content_sha256,
        tuple(
            ArticleSelection(
                segment.sequence_number,
                segment.region_id,
                segment.selection_id,
                segment.text_version_id,
            )
            for segment in bundle.segments
        ),
    )


def _default_loader(database_url: str, revision_id: UUID) -> ReviewedArticle:
    return _from_bundle(load_reviewed_coherent_text(database_url, revision_id))


def embed_reviewed_articles(
    request: ArticleEmbeddingRequest,
    runtime: ArticleEmbeddingRuntime | None = None,
) -> ArticleEmbeddingSummary:
    dependencies = runtime or ArticleEmbeddingRuntime(
        PostgresArticleStore(request.database_url), _default_loader, BGEArticleEncoder
    )
    revision_ids = dependencies.store.discover(request.revision_id)
    if not revision_ids:
        if request.revision_id is not None:
            raise ReviewedArticleUnavailableError(request.revision_id)
        return ArticleEmbeddingSummary(
            0, 0, 0, (), request.model_name, request.model_revision
        )
    articles = tuple(
        dependencies.load_reviewed(request.database_url, revision_id)
        for revision_id in revision_ids
    )
    encoder = dependencies.encoder_factory(request.model_name, request.model_revision)
    configuration = window_configuration(encoder)
    if request.expected_configuration_sha256 is not None:
        probe = ReviewedArticle(UUID(int=0), "probe", "0" * 64, "0" * 64, ())
        actual_configuration_sha256 = article_identity(
            probe, request, configuration
        ).configuration_sha256
        if actual_configuration_sha256 != request.expected_configuration_sha256:
            raise InconsistentArticleEmbeddingRunError(
                UUID(int=0), "worker embedding configuration differs from its plan"
            )
    inserted = reused = 0
    run_ids: list[str] = []
    for article in articles:
        identity = article_identity(article, request, configuration)
        run_ids.append(str(identity.run_id))
        if dependencies.store.completed_run(identity):
            reused += 1
            continue
        vector = encode_windowed_mean(article.content, encoder, request.batch_size)
        if dependencies.store.persist_completed(identity, article.selection, vector):
            inserted += 1
        else:
            reused += 1
    return ArticleEmbeddingSummary(
        len(articles),
        inserted,
        reused,
        tuple(run_ids),
        request.model_name,
        request.model_revision,
    )
