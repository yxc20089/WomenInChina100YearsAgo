from __future__ import annotations

from uuid import UUID

from .coherent_search_contracts import (
    COHERENT_ALIAS,
    EMBEDDING_DIMENSION,
    CoherentProjectionError,
    JsonValue,
    QueryEmbedder,
    SearchSpec,
    require_sha256,
)
from .coherent_search_results import ParsedHits, parse_hits
from .coherent_search_wire import opensearch_client
from .evidence import (
    RetrievalHit,
    RetrievalMode,
    RetrievalResponse,
)


def _validate_spec(spec: SearchSpec, requires_embedding: bool) -> None:
    if not spec.query.strip():
        raise CoherentProjectionError("search query must not be blank")
    if spec.limit < 1 or spec.candidate_limit < spec.limit or spec.rrf_k < 1:
        raise CoherentProjectionError(
            "search limits must be positive and candidate_limit must cover limit"
        )
    if (
        spec.year_min is not None
        and spec.year_max is not None
        and spec.year_min > spec.year_max
    ):
        raise CoherentProjectionError("year_min must not exceed year_max")
    if requires_embedding:
        _ = _required_embedding_identity(spec)


def _required_embedding_identity(spec: SearchSpec) -> tuple[str, str, str]:
    if not spec.model_name or not spec.model_revision or not spec.configuration_sha256:
        raise CoherentProjectionError("dense query requires pinned embedding identity")
    require_sha256(spec.configuration_sha256, "query configuration_sha256")
    return spec.model_name, spec.model_revision, spec.configuration_sha256


def _year_filters(spec: SearchSpec) -> list[JsonValue]:
    filters: list[JsonValue] = []
    if spec.year_max is not None:
        filters.append({"range": {"year_min": {"lte": spec.year_max}}})
    if spec.year_min is not None:
        filters.append({"range": {"year_max": {"gte": spec.year_min}}})
    return filters


def _response(
    spec: SearchSpec, mode: RetrievalMode, parsed: ParsedHits
) -> RetrievalResponse:
    return RetrievalResponse(
        schema_version="1.1",
        query=spec.query,
        mode=mode,
        hits=parsed.hits,
        warnings=sorted(parsed.warnings),
    )


def coherent_lexical_search(opensearch_url: str, spec: SearchSpec) -> RetrievalResponse:
    return _coherent_lexical_search(opensearch_url, spec, None)


def _coherent_lexical_search(
    opensearch_url: str,
    spec: SearchSpec,
    expected_embedding_identity: tuple[str, str, str] | None,
) -> RetrievalResponse:
    _validate_spec(spec, False)
    search = opensearch_client(opensearch_url)
    try:
        payload: JsonValue = search.search(
            index=COHERENT_ALIAS,
            body={
                "size": spec.limit,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": spec.query,
                                    "fields": ["title^3", "content"],
                                    "type": "best_fields",
                                }
                            }
                        ],
                        "filter": _year_filters(spec),
                    }
                },
                "_source": {"excludes": ["embedding"]},
            },
        )
        return _response(
            spec,
            RetrievalMode.LEXICAL,
            parse_hits(
                payload,
                "OpenSearch CJK coherent-unit lexical",
                expected_embedding_identity,
            ),
        )
    finally:
        search.close()


def coherent_dense_search(
    opensearch_url: str, spec: SearchSpec, embedder: QueryEmbedder
) -> RetrievalResponse:
    _validate_spec(spec, True)
    expected = _required_embedding_identity(spec)
    actual = (
        embedder.model_name,
        embedder.model_revision,
        embedder.configuration_sha256,
    )
    if actual != expected:
        raise CoherentProjectionError(
            "query embedding identity differs from SearchSpec"
        )
    vector = embedder.encode_query(spec.query)
    if len(vector) != EMBEDDING_DIMENSION:
        raise CoherentProjectionError(
            f"query embedding must have {EMBEDDING_DIMENSION} dimensions"
        )
    knn: dict[str, JsonValue] = {"vector": vector, "k": spec.limit}
    filters = [
        *_year_filters(spec),
        {"term": {"embedding_model": spec.model_name}},
        {"term": {"embedding_model_revision": spec.model_revision}},
        {"term": {"embedding_configuration_sha256": spec.configuration_sha256}},
    ]
    if filters:
        knn["filter"] = {"bool": {"filter": filters}}
    search = opensearch_client(opensearch_url)
    try:
        payload: JsonValue = search.search(
            index=COHERENT_ALIAS,
            body={
                "size": spec.limit,
                "query": {"knn": {"embedding": knn}},
                "_source": {"excludes": ["embedding"]},
            },
        )
        return _response(
            spec,
            RetrievalMode.DENSE,
            parse_hits(
                payload,
                "OpenSearch BGE-M3 coherent-unit dense",
                expected,
            ),
        )
    finally:
        search.close()


def coherent_hybrid_search(
    opensearch_url: str, spec: SearchSpec, embedder: QueryEmbedder
) -> RetrievalResponse:
    _validate_spec(spec, True)
    candidates = SearchSpec(
        spec.query,
        spec.candidate_limit,
        spec.year_min,
        spec.year_max,
        spec.candidate_limit,
        spec.rrf_k,
        spec.model_name,
        spec.model_revision,
        spec.configuration_sha256,
    )
    lexical = _coherent_lexical_search(
        opensearch_url,
        candidates,
        _required_embedding_identity(spec),
    )
    dense = coherent_dense_search(opensearch_url, candidates, embedder)
    fused: dict[tuple[UUID, str], tuple[RetrievalHit, float, dict[str, int]]] = {}
    for retriever, response in (("lexical", lexical), ("dense", dense)):
        for hit in response.hits:
            if hit.target_id is None:
                raise CoherentProjectionError(
                    "coherent-unit result lacks a target revision"
                )
            key = (hit.target_id, hit.target_kind)
            current = fused.get(key)
            score = (current[1] if current else 0.0) + 1.0 / (spec.rrf_k + hit.rank)
            ranks = dict(current[2]) if current else {}
            ranks[retriever] = hit.rank
            fused[key] = (current[0] if current else hit, score, ranks)
    ordered = sorted(fused.values(), key=lambda value: value[1], reverse=True)[
        : spec.limit
    ]
    hits = [
        hit.model_copy(
            update={
                "rank": rank,
                "score": score,
                "explanation": {
                    **hit.explanation,
                    "retriever": "RRF(CJK lexical+BGE-M3 coherent unit)",
                    "component_ranks": ranks,
                },
            }
        )
        for rank, (hit, score, ranks) in enumerate(ordered, 1)
    ]
    return RetrievalResponse(
        schema_version="1.1",
        query=spec.query,
        mode=RetrievalMode.HYBRID,
        hits=hits,
        warnings=sorted(set(lexical.warnings) | set(dense.warnings)),
    )
