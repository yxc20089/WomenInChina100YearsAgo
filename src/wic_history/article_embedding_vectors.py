from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Final, Protocol, final, runtime_checkable
from uuid import NAMESPACE_URL, UUID, uuid5

from .article_embedding_contracts import (
    ArticleEmbeddingRequest,
    ArticleEncoder,
    ArticleIdentity,
    InconsistentArticleEmbeddingRunError,
    MissingArticleEmbeddingDependencyError,
    ReviewedArticle,
    WindowConfiguration,
)

DIMENSION: Final = 1024
POLICY: Final = "windowed_mean_v1"
OVERLAP_RATIO: Final = 0.125
TARGET_KIND: Final = "coherent_unit_revision"


class _FloatMatrix(Protocol):
    def tolist(self) -> list[list[float]]: ...


@runtime_checkable
class _ArticleTokenizer(Protocol):
    model_max_length: int

    def num_special_tokens_to_add(self, *, pair: bool) -> int: ...
    def encode(
        self, text: str, *, add_special_tokens: bool, truncation: bool
    ) -> list[int]: ...
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


class _SentenceModel(Protocol):
    tokenizer: _ArticleTokenizer
    max_seq_length: int

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> _FloatMatrix: ...


class _SentenceTransformerFactory(Protocol):
    def __call__(
        self, model_name: str, *, revision: str, device: str
    ) -> _SentenceModel: ...


if TYPE_CHECKING:
    _construct_sentence_transformer: _SentenceTransformerFactory
else:

    def _construct_sentence_transformer(
        model_name: str, *, revision: str, device: str
    ) -> _SentenceModel:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise MissingArticleEmbeddingDependencyError(
                "sentence-transformers"
            ) from exc
        return SentenceTransformer(model_name, revision=revision, device=device)


def _load_sentence_model(model_name: str, model_revision: str) -> _SentenceModel:
    return _construct_sentence_transformer(
        model_name, revision=model_revision, device="cpu"
    )


@final
class BGEArticleEncoder:
    def __init__(self, model_name: str, model_revision: str):
        self._model: _SentenceModel = _load_sentence_model(model_name, model_revision)
        special_tokens = self._model.tokenizer.num_special_tokens_to_add(pair=False)
        self.tokenizer_limit: int = (
            self._model.tokenizer.model_max_length - special_tokens
        )
        self.model_limit: int = self._model.max_seq_length - special_tokens

    def tokenize(self, text: str) -> list[int]:
        return self._model.tokenizer.encode(
            text, add_special_tokens=False, truncation=False
        )

    def decode(self, token_ids: list[int]) -> str:
        return self._model.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        values = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )
        return values.tolist()


def window_configuration(encoder: ArticleEncoder) -> WindowConfiguration:
    effective = min(encoder.tokenizer_limit, encoder.model_limit)
    if effective <= 0:
        raise InconsistentArticleEmbeddingRunError(
            UUID(int=0), "tokenizer and model limits must be positive"
        )
    return WindowConfiguration(
        POLICY,
        encoder.tokenizer_limit,
        encoder.model_limit,
        effective,
        min(effective - 1, max(1, int(effective * OVERLAP_RATIO))),
        DIMENSION,
    )


def _normalize(vector: list[float]) -> tuple[float, ...]:
    if len(vector) != DIMENSION or any(not math.isfinite(value) for value in vector):
        raise InconsistentArticleEmbeddingRunError(
            UUID(int=0), "encoder returned a wrong-dimension or nonfinite vector"
        )
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise InconsistentArticleEmbeddingRunError(
            UUID(int=0), "encoder returned a zero or nonfinite vector norm"
        )
    return tuple(value / norm for value in vector)


def encode_windowed_mean(
    text: str, encoder: ArticleEncoder, batch_size: int
) -> tuple[float, ...]:
    configuration = window_configuration(encoder)
    token_ids = encoder.tokenize(text)
    if len(token_ids) <= configuration.effective_limit:
        texts = [text]
    else:
        step = configuration.effective_limit - configuration.overlap_tokens
        windows: list[list[int]] = []
        start = 0
        while start < len(token_ids):
            window = token_ids[start : start + configuration.effective_limit]
            windows.append(window)
            if start + configuration.effective_limit >= len(token_ids):
                break
            start += step
        texts = [encoder.decode(window) for window in windows]
    vectors = encoder.encode(texts, batch_size)
    if len(vectors) != len(texts):
        raise InconsistentArticleEmbeddingRunError(
            UUID(int=0), "encoder returned the wrong vector count"
        )
    normalized = [_normalize(vector) for vector in vectors]
    mean = [
        sum(vector[index] for vector in normalized) / len(normalized)
        for index in range(DIMENSION)
    ]
    return _normalize(mean)


def article_identity(
    article: ReviewedArticle,
    request: ArticleEmbeddingRequest,
    configuration: WindowConfiguration,
) -> ArticleIdentity:
    encoded = json.dumps(asdict(configuration), sort_keys=True, separators=(",", ":"))
    configuration_sha256 = hashlib.sha256(encoded.encode()).hexdigest()
    name = ":".join(
        (
            str(article.revision_id),
            request.model_name,
            request.model_revision,
            article.input_sha256,
            article.content_sha256,
            configuration_sha256,
        )
    )
    return ArticleIdentity(
        uuid5(NAMESPACE_URL, f"wic-reviewed-article-embedding:{name}"),
        article.revision_id,
        TARGET_KIND,
        request.model_name,
        request.model_revision,
        article.input_sha256,
        article.content_sha256,
        configuration_sha256,
        configuration,
    )
