"""Dispatch reviewed coherent-unit searches for the HTTP API boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    TypedDict,
    final,
    override,
)

from .coherent_search import (
    QueryEmbedder,
    SearchSpec,
    coherent_dense_search,
    coherent_hybrid_search,
    coherent_lexical_search,
)
from .search_runtime import PinnedQueryEmbedder, pinned_coherent_query_identity

if TYPE_CHECKING:
    from collections.abc import Callable

    from .evidence import RetrievalResponse


_PINNED_IDENTITY: Final = pinned_coherent_query_identity()
PINNED_IDENTITY_DETAIL: Final = (
    "Coherent dense search requires pinned embedding model, revision, and configuration"
)


class CoherentSearchRequest(Protocol):
    """Values accepted from the validated API search request."""

    query: str
    mode: Literal["lexical", "dense", "hybrid"]
    limit: int
    year_start: int | None
    year_end: int | None


@final
@dataclass(frozen=True, slots=True)
class IncompleteCoherentEmbeddingIdentityError(RuntimeError):
    """Indicate that dense retrieval lacks a complete pinned identity."""

    @override
    def __str__(self) -> str:
        return PINNED_IDENTITY_DETAIL


@dataclass(frozen=True, slots=True)
class CoherentSearchResult:
    """Return retrieval output together with reusable embedder state."""

    response: RetrievalResponse
    embedder: QueryEmbedder | None


class _SearchDispatch(TypedDict):
    lexical: Callable[[], CoherentSearchResult]
    dense: Callable[[], CoherentSearchResult]
    hybrid: Callable[[], CoherentSearchResult]


def _query_embedder(
    spec: SearchSpec,
    cached_embedder: QueryEmbedder | None,
) -> QueryEmbedder:
    if not spec.model_name or not spec.model_revision or not spec.configuration_sha256:
        raise IncompleteCoherentEmbeddingIdentityError
    if cached_embedder is not None:
        return cached_embedder
    return PinnedQueryEmbedder(
        spec.model_name,
        spec.model_revision,
        spec.configuration_sha256,
    )


def run_coherent_api_search(
    opensearch_url: str,
    request: CoherentSearchRequest,
    cached_embedder: QueryEmbedder | None,
) -> CoherentSearchResult:
    """Dispatch one validated coherent-corpus search request."""
    spec = SearchSpec(
        request.query,
        limit=request.limit,
        year_min=request.year_start,
        year_max=request.year_end,
        # identity comes only from config/pipeline-models.toml, never from
        # the environment (sole-source-of-model-identity invariant)
        model_name=_PINNED_IDENTITY.model_name,
        model_revision=_PINNED_IDENTITY.model_revision,
        configuration_sha256=_PINNED_IDENTITY.configuration_sha256,
    )

    def lexical() -> CoherentSearchResult:
        return CoherentSearchResult(
            coherent_lexical_search(opensearch_url, spec),
            cached_embedder,
        )

    def dense() -> CoherentSearchResult:
        embedder = _query_embedder(spec, cached_embedder)
        return CoherentSearchResult(
            coherent_dense_search(opensearch_url, spec, embedder),
            embedder,
        )

    def hybrid() -> CoherentSearchResult:
        embedder = _query_embedder(spec, cached_embedder)
        return CoherentSearchResult(
            coherent_hybrid_search(opensearch_url, spec, embedder),
            embedder,
        )

    dispatch = _SearchDispatch(lexical=lexical, dense=dense, hybrid=hybrid)
    return dispatch[request.mode]()
