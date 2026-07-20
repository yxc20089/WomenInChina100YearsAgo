from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeGuard

from .coherent_search_contracts import CoherentProjectionError, JsonValue


class OpenSearchIndices(Protocol):
    def create(self, *, index: str, body: Mapping[str, JsonValue]) -> JsonValue: ...

    def refresh(self, *, index: str) -> JsonValue: ...

    def get_alias(self, *, name: str, ignore: list[int]) -> JsonValue: ...

    def update_aliases(self, *, body: Mapping[str, JsonValue]) -> JsonValue: ...


class OpenSearchClient(Protocol):
    @property
    def indices(self) -> OpenSearchIndices: ...

    def bulk(self, *, body: str, refresh: bool) -> JsonValue: ...

    def count(self, *, index: str) -> JsonValue: ...

    def search(self, *, index: str, body: Mapping[str, JsonValue]) -> JsonValue: ...

    def close(self) -> None: ...


class OpenSearchClientCandidate(Protocol):
    pass


def _implements_client(
    value: OpenSearchClient | OpenSearchClientCandidate,
) -> TypeGuard[OpenSearchClient]:
    indices = getattr(value, "indices", None)
    return all(
        callable(member)
        for member in (
            getattr(value, "bulk", None),
            getattr(value, "count", None),
            getattr(value, "search", None),
            getattr(value, "close", None),
            getattr(indices, "create", None),
            getattr(indices, "refresh", None),
            getattr(indices, "get_alias", None),
            getattr(indices, "update_aliases", None),
        )
    )


def opensearch_client(opensearch_url: str) -> OpenSearchClient:
    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:
        raise CoherentProjectionError(
            "Install the data extra: uv sync --extra data"
        ) from exc
    client = OpenSearch(hosts=[opensearch_url], http_compress=True)
    if _implements_client(client):
        return client
    raise CoherentProjectionError(
        "OpenSearch client does not implement its required protocol"
    )


def require_mapping(value: JsonValue, label: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise CoherentProjectionError(f"malformed OpenSearch {label} envelope")
    return value


def require_list(value: JsonValue, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise CoherentProjectionError(f"malformed OpenSearch {label} envelope")
    return value


def require_str(value: JsonValue, label: str) -> str:
    if not isinstance(value, str):
        raise CoherentProjectionError(f"malformed OpenSearch {label} envelope")
    return value


def require_int(value: JsonValue, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CoherentProjectionError(f"malformed OpenSearch {label} envelope")
    return value


def require_float(value: JsonValue, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise CoherentProjectionError(f"malformed OpenSearch {label} envelope")
    return float(value)


def parse_bulk_response(payload: JsonValue, expected_count: int) -> None:
    envelope = require_mapping(payload, "bulk")
    errors = envelope.get("errors")
    if not isinstance(errors, bool):
        raise CoherentProjectionError("malformed OpenSearch bulk envelope")
    if errors:
        raise CoherentProjectionError("OpenSearch bulk projection reported item errors")
    items = require_list(envelope.get("items"), "bulk")
    if len(items) != expected_count:
        raise CoherentProjectionError(
            "OpenSearch bulk projection returned a misleading item count"
        )
    for item in items:
        operation = require_mapping(
            require_mapping(item, "bulk item").get("index"), "bulk item"
        )
        status = require_int(operation.get("status"), "bulk item")
        if status < 200 or status >= 300:
            raise CoherentProjectionError(
                "OpenSearch bulk projection reported failed item status"
            )


def parse_count_response(payload: JsonValue) -> int:
    envelope = require_mapping(payload, "count")
    return require_int(envelope.get("count"), "count")


def parse_refresh_response(payload: JsonValue) -> None:
    envelope = require_mapping(payload, "refresh")
    shards = require_mapping(envelope.get("_shards"), "refresh")
    total = require_int(shards.get("total"), "refresh")
    successful = require_int(shards.get("successful"), "refresh")
    failed = require_int(shards.get("failed"), "refresh")
    if failed != 0 or successful != total:
        raise CoherentProjectionError("OpenSearch refresh did not complete all shards")


def parse_alias_response(payload: JsonValue) -> tuple[str, ...]:
    envelope = require_mapping(payload, "alias lookup")
    if "error" in envelope:
        return ()
    indices: list[str] = []
    for index_name, value in envelope.items():
        index = require_mapping(value, "alias lookup")
        _ = require_mapping(index.get("aliases"), "alias lookup")
        indices.append(index_name)
    return tuple(indices)


def require_acknowledged(payload: JsonValue, operation: str) -> None:
    envelope = require_mapping(payload, operation)
    if envelope.get("acknowledged") is not True:
        raise CoherentProjectionError(f"OpenSearch {operation} was not acknowledged")
