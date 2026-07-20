from __future__ import annotations

import json
from uuid import UUID

import pytest

from tests.coherent_search_support import (
    OpenSearchFake,
    QueryEmbedder,
    manifest,
)
from wic_history.coherent_search import (
    CoherentProjectionError,
    SearchSpec,
    coherent_dense_search,
    coherent_hybrid_search,
    coherent_lexical_search,
    project_coherent_units,
)


def _search_spec() -> SearchSpec:
    return SearchSpec(
        "女子教育",
        limit=5,
        year_min=1925,
        year_max=1926,
        model_name="BAAI/bge-m3",
        model_revision="model-revision",
        configuration_sha256="f" * 64,
    )


def test_search_modes_return_one_multispan_article(search_server: str) -> None:
    _ = project_coherent_units(search_server, manifest())
    spec = _search_spec()
    responses = (
        coherent_lexical_search(search_server, spec),
        coherent_dense_search(search_server, spec, QueryEmbedder()),
        coherent_hybrid_search(search_server, spec, QueryEmbedder()),
    )
    assert [response.hits[0].target_kind for response in responses] == [
        "reviewed_coherent_unit"
    ] * 3
    assert [len(response.hits[0].sources) for response in responses] == [2, 2, 2]
    assert all(response.hits[0].source is None for response in responses)
    assert len(responses[2].hits) == 1
    search_bodies = [
        body
        for method, path, body in OpenSearchFake.requests
        if method == "POST" and path.endswith("/_search")
    ]
    assert any(
        "year_min" in json.dumps(body) and "year_max" in json.dumps(body)
        for body in search_bodies
    )


def test_search_refuses_malformed_external_json(search_server: str) -> None:
    OpenSearchFake.documents["malformed"] = json.dumps(
        {"revision_id": str(UUID(int=0))}
    )
    with pytest.raises(CoherentProjectionError, match="OpenSearch response"):
        _ = coherent_lexical_search(search_server, SearchSpec("女子教育"))


def test_dense_rejects_embedder_differing_from_spec_identity(
    search_server: str,
) -> None:
    spec = SearchSpec(
        "女子教育",
        model_name="expected-model",
        model_revision="expected-revision",
        configuration_sha256="1" * 64,
    )
    with pytest.raises(CoherentProjectionError, match="query embedding identity"):
        _ = coherent_dense_search(search_server, spec, QueryEmbedder())


def test_dense_wire_query_pins_document_embedding_identity(
    search_server: str,
) -> None:
    _ = project_coherent_units(search_server, manifest())
    _ = coherent_dense_search(search_server, _search_spec(), QueryEmbedder())
    body = next(
        body
        for method, path, body in reversed(OpenSearchFake.requests)
        if method == "POST" and path.endswith("/_search")
    )
    assert isinstance(body, str)
    assert '"embedding_model":"BAAI/bge-m3"' in body
    assert '"embedding_model_revision":"model-revision"' in body
    assert f'"embedding_configuration_sha256":"{"f" * 64}"' in body


@pytest.mark.parametrize(
    ("field", "expected", "replacement"),
    [
        ("embedding_model", "BAAI/bge-m3", "other-model"),
        ("embedding_model_revision", "model-revision", "other-revision"),
        ("embedding_configuration_sha256", "f" * 64, "1" * 64),
    ],
)
def test_dense_rejects_returned_document_with_incompatible_embedding_identity(
    search_server: str,
    field: str,
    expected: str,
    replacement: str,
) -> None:
    _ = project_coherent_units(search_server, manifest())
    revision_id, document = next(iter(OpenSearchFake.documents.items()))
    original = f'"{field}": "{expected}"'
    OpenSearchFake.documents[revision_id] = document.replace(
        original, f'"{field}": "{replacement}"'
    )
    assert OpenSearchFake.documents[revision_id] != document
    assert coherent_lexical_search(search_server, SearchSpec("女子教育")).hits
    with pytest.raises(CoherentProjectionError, match="embedding identity"):
        _ = coherent_dense_search(search_server, _search_spec(), QueryEmbedder())


def test_hybrid_rejects_incompatible_dense_result(search_server: str) -> None:
    _ = project_coherent_units(search_server, manifest())
    revision_id, document = next(iter(OpenSearchFake.documents.items()))
    OpenSearchFake.documents[revision_id] = document.replace(
        '"embedding_model": "BAAI/bge-m3"',
        '"embedding_model": "other-model"',
    )
    with pytest.raises(CoherentProjectionError, match="embedding identity"):
        _ = coherent_hybrid_search(search_server, _search_spec(), QueryEmbedder())


def test_hybrid_rejects_incompatible_lexical_only_result(
    search_server: str,
) -> None:
    OpenSearchFake.filter_dense_identity = True
    _ = project_coherent_units(search_server, manifest())
    revision_id, document = next(iter(OpenSearchFake.documents.items()))
    OpenSearchFake.documents[revision_id] = document.replace(
        '"embedding_model": "BAAI/bge-m3"',
        '"embedding_model": "other-model"',
    )
    assert coherent_lexical_search(search_server, SearchSpec("女子教育")).hits
    with pytest.raises(CoherentProjectionError, match="embedding identity"):
        _ = coherent_hybrid_search(search_server, _search_spec(), QueryEmbedder())


def test_hybrid_validates_original_spec_before_component_limits(
    search_server: str,
) -> None:
    spec = SearchSpec(
        "女子教育",
        limit=20,
        candidate_limit=10,
        model_name="BAAI/bge-m3",
        model_revision="model-revision",
        configuration_sha256="f" * 64,
    )
    with pytest.raises(CoherentProjectionError, match="candidate_limit"):
        _ = coherent_hybrid_search(search_server, spec, QueryEmbedder())
