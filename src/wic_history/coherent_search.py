from .coherent_search_contracts import (
    COHERENT_ALIAS,
    COHERENT_INDEX_PREFIX,
    EMBEDDING_DIMENSION,
    CoherentEmbedding,
    CoherentProjectionError,
    CoherentSource,
    FrozenProjectionManifest,
    ProjectionArticle,
    ProjectionResult,
    QueryEmbedder,
    SearchSpec,
    coherent_index_body,
)
from .coherent_search_projection import project_coherent_units, restore_coherent_alias
from .coherent_search_retrieval import (
    coherent_dense_search,
    coherent_hybrid_search,
    coherent_lexical_search,
)


__all__ = [
    "COHERENT_ALIAS",
    "COHERENT_INDEX_PREFIX",
    "EMBEDDING_DIMENSION",
    "CoherentEmbedding",
    "CoherentProjectionError",
    "CoherentSource",
    "FrozenProjectionManifest",
    "ProjectionArticle",
    "ProjectionResult",
    "QueryEmbedder",
    "SearchSpec",
    "coherent_dense_search",
    "coherent_hybrid_search",
    "coherent_index_body",
    "coherent_lexical_search",
    "project_coherent_units",
    "restore_coherent_alias",
]
