from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from types import ModuleType
from uuid import UUID

import pytest

from wic_history.article_embedding import (
    ArticleEmbeddingRequest,
    ArticleEmbeddingRuntime,
    ArticleIdentity,
    ArticleSelection,
    InconsistentArticleEmbeddingRunError,
    ReviewedArticle,
    StaleReviewedArticleError,
    embed_reviewed_articles,
    encode_windowed_mean,
)
from wic_history.article_embedding_vectors import BGEArticleEncoder
from wic_history.article_embedding_vectors import window_configuration


class FakeEncoder:
    tokenizer_limit: int = 6
    model_limit: int = 6

    def __init__(self) -> None:
        self.encoded: list[list[str]] = []

    def tokenize(self, text: str) -> list[int]:
        return [int(token) for token in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token) for token in token_ids)

    def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        del batch_size
        self.encoded.append(texts)
        return [
            [float(sum(int(token) for token in text.split())), 1.0] + [0.0] * 1022
            for text in texts
        ]


def test_bge_encoder_adapts_typed_model_tokenizer_and_matrix_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a SentenceTransformer-compatible model with explicit boundary values.
    class Tokenizer:
        model_max_length = 10

        def num_special_tokens_to_add(self, *, pair: bool) -> int:
            assert pair is False
            return 2

        def encode(
            self, text: str, *, add_special_tokens: bool, truncation: bool
        ) -> list[int]:
            assert (text, add_special_tokens, truncation) == ("article", False, False)
            return [1, 2]

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            assert (skip_special_tokens, clean_up_tokenization_spaces) == (True, False)
            return " ".join(str(token_id) for token_id in token_ids)

    class Matrix:
        def tolist(self) -> list[list[float]]:
            return [[1.0, 2.0]]

    class Model:
        tokenizer = Tokenizer()
        max_seq_length = 12

        def __init__(self, model_name: str, *, revision: str, device: str) -> None:
            assert (model_name, revision, device) == ("model", "revision", "cpu")

        def encode(
            self,
            texts: list[str],
            *,
            batch_size: int,
            normalize_embeddings: bool,
            show_progress_bar: bool,
        ) -> Matrix:
            assert (texts, batch_size, normalize_embeddings, show_progress_bar) == (
                ["article"],
                4,
                True,
                False,
            )
            return Matrix()

    class SentenceTransformerModule(ModuleType):
        SentenceTransformer: type[Model]

    module = SentenceTransformerModule("sentence_transformers")
    module.SentenceTransformer = Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    # When: the production adapter exercises every third-party boundary.
    encoder = BGEArticleEncoder("model", "revision")

    # Then: limits, tokenizer values, and matrix conversion remain concrete.
    assert (encoder.tokenizer_limit, encoder.model_limit) == (8, 10)
    assert encoder.tokenize("article") == [1, 2]
    assert encoder.decode([1, 2]) == "1 2"
    assert encoder.encode(["article"], 4) == [[1.0, 2.0]]


@dataclass
class FakeStore:
    revisions: tuple[UUID, ...]
    completed: set[UUID]
    stale: bool = False
    partial: bool = False
    persisted: list[tuple[ArticleIdentity, tuple[float, ...]]] = field(
        default_factory=list, init=False
    )

    def discover(self, revision_id: UUID | None) -> tuple[UUID, ...]:
        return (
            self.revisions
            if revision_id is None
            else tuple(item for item in self.revisions if item == revision_id)
        )

    def completed_run(self, identity: ArticleIdentity) -> bool:
        if self.partial:
            raise InconsistentArticleEmbeddingRunError(identity.run_id)
        return identity.run_id in self.completed

    def persist_completed(
        self,
        identity: ArticleIdentity,
        selection: tuple[ArticleSelection, ...],
        vector: tuple[float, ...],
    ) -> bool:
        del selection
        if self.stale:
            raise StaleReviewedArticleError(identity.revision_id)
        if identity.run_id in self.completed:
            return False
        self.completed.add(identity.run_id)
        self.persisted.append((identity, vector))
        return True


def reviewed(revision_id: UUID, content: str = "1 2") -> ReviewedArticle:
    return ReviewedArticle(
        revision_id=revision_id,
        content=content,
        input_sha256="a" * 64,
        content_sha256="b" * 64,
        selection=(ArticleSelection(0, UUID(int=2), UUID(int=3), UUID(int=4)),),
    )


def runtime(store: FakeStore, encoder: FakeEncoder, article: ReviewedArticle):
    return ArticleEmbeddingRuntime(
        store=store,
        load_reviewed=lambda _database_url, _revision_id: article,
        encoder_factory=lambda _model, _revision: encoder,
    )


def request(revision_id: UUID | None = None) -> ArticleEmbeddingRequest:
    return ArticleEmbeddingRequest(
        "postgresql://unused", "model", "revision", 8, revision_id
    )


def test_one_reviewed_text_produces_one_versioned_vector() -> None:
    # Given: one active reviewed article revision.
    revision_id = UUID(int=1)
    store = FakeStore((revision_id,), set())
    encoder = FakeEncoder()

    # When: reviewed article embedding runs.
    summary = embed_reviewed_articles(
        request(), runtime(store, encoder, reviewed(revision_id))
    )

    # Then: exactly one vector is persisted with all three exact hashes.
    assert (summary.revisions_discovered, summary.embeddings_inserted) == (1, 1)
    identity = store.persisted[0][0]
    assert (identity.target_kind, identity.input_sha256, identity.content_sha256) == (
        "coherent_unit_revision",
        "a" * 64,
        "b" * 64,
    )
    assert len(identity.configuration_sha256) == 64
    assert (
        identity.configuration.policy,
        identity.configuration.tokenizer_limit,
        identity.configuration.model_limit,
    ) == ("windowed_mean_v1", 6, 6)


def test_long_text_uses_overlapping_normalized_window_mean() -> None:
    # Given: seven tokens and an effective six-token model window.
    encoder = FakeEncoder()

    # When: the deterministic window policy embeds the text.
    vector = encode_windowed_mean("1 2 3 4 5 6 7", encoder, batch_size=4)

    # Then: windows overlap by one token and their normalized mean is normalized again.
    assert encoder.encoded == [["1 2 3 4 5 6", "6 7"]]
    first = (21.0 / math.sqrt(442), 1.0 / math.sqrt(442))
    second = (13.0 / math.sqrt(170), 1.0 / math.sqrt(170))
    mean = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
    mean_norm = math.sqrt(sum(value * value for value in mean))
    assert vector[:2] == pytest.approx(tuple(value / mean_norm for value in mean))
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


def test_window_generation_stops_after_first_window_reaches_text_end() -> None:
    # Given: token length where the second full window reaches the end exactly.
    encoder = FakeEncoder()

    # When: the overlapping window policy embeds the text.
    _ = encode_windowed_mean("1 2 3 4 5 6 7 8 9 10 11", encoder, batch_size=4)

    # Then: it does not emit a third overlap-only tail window.
    assert encoder.encoded == [["1 2 3 4 5 6", "6 7 8 9 10 11"]]


@pytest.mark.parametrize(
    ("limit", "expected_windows"),
    [(1, ["1", "2"]), (2, ["1 2"])],
)
def test_tiny_window_limits_terminate_with_valid_overlap(
    limit: int, expected_windows: list[str]
) -> None:
    # Given: two tokens and a model whose effective limit is one or two.
    encoder = FakeEncoder()
    encoder.tokenizer_limit = limit
    encoder.model_limit = limit

    # When: configuration is validated before the text is embedded.
    configuration = window_configuration(encoder)

    # Then: overlap leaves a positive step and embedding terminates finitely.
    assert 0 <= configuration.overlap_tokens < configuration.effective_limit
    _ = encode_windowed_mean("1 2", encoder, batch_size=4)
    assert encoder.encoded == [expected_windows]


def test_normal_window_overlap_remains_twelve_and_a_half_percent() -> None:
    # Given: an effective model limit whose exact 12.5 percent overlap is two tokens.
    encoder = FakeEncoder()
    encoder.tokenizer_limit = 16
    encoder.model_limit = 16

    # When: the deterministic window configuration is calculated.
    configuration = window_configuration(encoder)

    # Then: the established policy still uses a two-token overlap.
    assert configuration.overlap_tokens == 2


def test_identical_retry_reuses_completed_run() -> None:
    # Given: a completed deterministic run for the exact article identity.
    revision_id = UUID(int=1)
    store = FakeStore((revision_id,), set())
    encoder = FakeEncoder()
    first = embed_reviewed_articles(
        request(), runtime(store, encoder, reviewed(revision_id))
    )

    # When: the exact request is retried.
    second = embed_reviewed_articles(
        request(), runtime(store, encoder, reviewed(revision_id))
    )

    # Then: the completed row is reused without a duplicate vector.
    assert first.run_ids == second.run_ids
    assert (second.embeddings_inserted, second.embeddings_reused) == (0, 1)
    assert len(store.persisted) == 1


def test_batch_size_is_excluded_from_deterministic_configuration_identity() -> None:
    # Given: one exact reviewed article embedded with one operational batch size.
    revision_id = UUID(int=1)
    store = FakeStore((revision_id,), set())
    encoder = FakeEncoder()
    first = embed_reviewed_articles(
        request(), runtime(store, encoder, reviewed(revision_id))
    )

    # When: the exact content is retried with a different batch size.
    changed_batch = ArticleEmbeddingRequest(
        "postgresql://unused", "model", "revision", 99
    )
    second = embed_reviewed_articles(
        changed_batch, runtime(store, encoder, reviewed(revision_id))
    )

    # Then: operational batching does not create a false content identity.
    assert first.run_ids == second.run_ids
    assert second.embeddings_reused == 1


def test_reselection_history_creates_distinct_run_identity() -> None:
    # Given: the same revision after a reviewed selection changes its input identity.
    revision_id = UUID(int=1)
    store = FakeStore((revision_id,), set())
    encoder = FakeEncoder()
    first = reviewed(revision_id)
    second = ReviewedArticle(
        revision_id,
        first.content,
        "c" * 64,
        first.content_sha256,
        (ArticleSelection(0, UUID(int=2), UUID(int=5), UUID(int=6)),),
    )

    # When: both immutable histories are embedded.
    first_summary = embed_reviewed_articles(request(), runtime(store, encoder, first))
    second_summary = embed_reviewed_articles(request(), runtime(store, encoder, second))

    # Then: both exact identities remain addressable.
    assert first_summary.run_ids != second_summary.run_ids
    assert len(store.persisted) == 2


def test_stale_race_rejects_persistence_after_encoding() -> None:
    # Given: selection identity changes after the model encoded the reviewed text.
    revision_id = UUID(int=1)
    store = FakeStore((revision_id,), set(), stale=True)

    # When/Then: the atomic persistence seam rejects the stale result.
    with pytest.raises(StaleReviewedArticleError):
        embed_reviewed_articles(
            request(), runtime(store, FakeEncoder(), reviewed(revision_id))
        )


@pytest.mark.parametrize("mode", ["count", "dimension"])
def test_encoder_rejects_wrong_vector_count_or_dimension(mode: str) -> None:
    # Given: an encoder returning an invalid model output contract.
    class BrokenEncoder(FakeEncoder):
        def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
            values = super().encode(texts, batch_size)
            return values[:-1] if mode == "count" else [[0.0] * 3 for _ in values]

    # When/Then: persistence cannot receive a misleading vector.
    with pytest.raises(InconsistentArticleEmbeddingRunError):
        encode_windowed_mean("1 2 3 4 5 6 7", BrokenEncoder(), batch_size=4)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_encoder_rejects_nonfinite_vector_elements(invalid: float) -> None:
    # Given: one model vector containing a nonfinite element.
    class NonfiniteEncoder(FakeEncoder):
        def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
            del batch_size
            return [[invalid, 1.0] + [0.0] * 1022 for _ in texts]

    # When/Then: normalization rejects the invalid embedding contract.
    with pytest.raises(InconsistentArticleEmbeddingRunError):
        encode_windowed_mean("1 2", NonfiniteEncoder(), batch_size=4)


def test_encoder_rejects_zero_vector() -> None:
    # Given: a model output with zero norm.
    class ZeroEncoder(FakeEncoder):
        def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
            del batch_size
            return [[0.0] * 1024 for _ in texts]

    # When/Then: normalization rejects it before persistence.
    with pytest.raises(InconsistentArticleEmbeddingRunError):
        encode_windowed_mean("1 2", ZeroEncoder(), batch_size=4)


def test_no_reviewed_units_returns_zero_without_loading_model() -> None:
    # Given: no active reviewed article revision.
    loaded = False

    def fail_if_loaded(_model: str, _revision: str) -> FakeEncoder:
        nonlocal loaded
        loaded = True
        return FakeEncoder()

    store = FakeStore((), set())
    dependencies = ArticleEmbeddingRuntime(
        store, lambda _url, _id: reviewed(UUID(int=1)), fail_if_loaded
    )

    # When: reviewed article embedding runs.
    summary = embed_reviewed_articles(request(), dependencies)

    # Then: the result is an explicit zero summary and no model is loaded.
    assert (summary.revisions_discovered, summary.embeddings_inserted) == (0, 0)
    assert loaded is False


def test_explicit_revision_materialization_failure_is_not_silent() -> None:
    # Given: one explicitly targeted revision whose canonical reviewed text fails.
    revision_id = UUID(int=1)
    store = FakeStore((revision_id,), set())
    dependencies = ArticleEmbeddingRuntime(
        store,
        lambda _url, _id: (_ for _ in ()).throw(ValueError("incomplete review")),
        lambda _model, _revision: FakeEncoder(),
    )

    # When/Then: the explicit target fails instead of returning a zero success summary.
    with pytest.raises(ValueError, match="incomplete review"):
        embed_reviewed_articles(request(revision_id), dependencies)


def test_explicit_revision_without_eligible_review_is_not_zero_success() -> None:
    # Given: an explicitly requested revision excluded by reviewed eligibility.
    revision_id = UUID(int=1)
    dependencies = ArticleEmbeddingRuntime(
        FakeStore((), set()),
        lambda _url, _id: reviewed(revision_id),
        lambda _model, _revision: FakeEncoder(),
    )

    # When/Then: the missing eligible target is reported as an error.
    with pytest.raises(ValueError, match=str(revision_id)):
        embed_reviewed_articles(request(revision_id), dependencies)


def test_partial_deterministic_run_is_an_error() -> None:
    # Given: a deterministic run ID exists without its one completed vector.
    revision_id = UUID(int=1)
    store = FakeStore((revision_id,), set(), partial=True)

    # When/Then: retry refuses to misreport the partial run as success.
    with pytest.raises(InconsistentArticleEmbeddingRunError):
        embed_reviewed_articles(
            request(), runtime(store, FakeEncoder(), reviewed(revision_id))
        )
