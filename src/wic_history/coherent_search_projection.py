from __future__ import annotations

import json
from uuid import uuid4

from .coherent_search_contracts import (
    COHERENT_ALIAS,
    COHERENT_INDEX_PREFIX,
    CoherentProjectionError,
    FrozenProjectionManifest,
    JsonValue,
    ProjectionResult,
    coherent_index_body,
)
from .coherent_search_documents import validated_documents
from .coherent_search_wire import (
    opensearch_client,
    parse_alias_response,
    parse_bulk_response,
    parse_count_response,
    parse_refresh_response,
    require_acknowledged,
)


def project_coherent_units(
    opensearch_url: str,
    manifest: FrozenProjectionManifest,
) -> ProjectionResult:
    documents = validated_documents(manifest)
    build_id = uuid4()
    index_name = f"{COHERENT_INDEX_PREFIX}{build_id.hex}"
    search = opensearch_client(opensearch_url)
    try:
        create_response: JsonValue = search.indices.create(
            index=index_name, body=coherent_index_body()
        )
        require_acknowledged(create_response, "index creation")
        operations: list[str] = []
        for revision_id, document in documents:
            operations.extend(
                (
                    json.dumps({"index": {"_index": index_name, "_id": revision_id}}),
                    json.dumps(document, ensure_ascii=False),
                )
            )
        bulk_response: JsonValue = search.bulk(
            body="\n".join(operations) + "\n", refresh=False
        )
        parse_bulk_response(bulk_response, len(documents))
        refresh_response = search.indices.refresh(index=index_name)
        parse_refresh_response(refresh_response)
        count_response: JsonValue = search.count(index=index_name)
        count = parse_count_response(count_response)
        if count != len(documents):
            raise CoherentProjectionError(
                "OpenSearch projected document count failed validation"
            )
        alias_response: JsonValue = search.indices.get_alias(
            name=COHERENT_ALIAS, ignore=[404]
        )
        old_indices = parse_alias_response(alias_response)
        actions: list[dict[str, dict[str, str]]] = [
            {"remove": {"index": old_index, "alias": COHERENT_ALIAS}}
            for old_index in old_indices
            if old_index != index_name
        ]
        actions.append({"add": {"index": index_name, "alias": COHERENT_ALIAS}})
        update_response: JsonValue = search.indices.update_aliases(
            body={"actions": actions}
        )
        require_acknowledged(update_response, "alias update")
        return ProjectionResult(
            str(build_id),
            index_name,
            count,
            manifest.snapshot_sha256,
            tuple(sorted(old_indices)),
        )
    finally:
        search.close()


def restore_coherent_alias(opensearch_url: str, projection: ProjectionResult) -> None:
    search = opensearch_client(opensearch_url)
    try:
        current_response: JsonValue = search.indices.get_alias(
            name=COHERENT_ALIAS, ignore=[404]
        )
        current_indices = parse_alias_response(current_response)
        if current_indices != (projection.index_name,):
            raise CoherentProjectionError(
                "coherent alias changed after publication; compensation refused"
            )
        actions: list[dict[str, dict[str, str | bool]]] = [
            {
                "remove": {
                    "index": projection.index_name,
                    "alias": COHERENT_ALIAS,
                    "must_exist": True,
                }
            }
        ]
        actions.extend(
            {"add": {"index": name, "alias": COHERENT_ALIAS}}
            for name in projection.previous_index_names
        )
        response: JsonValue = search.indices.update_aliases(body={"actions": actions})
        require_acknowledged(response, "alias compensation")
    finally:
        search.close()
