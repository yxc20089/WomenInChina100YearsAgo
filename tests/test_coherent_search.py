from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wic_history.coherent_search import (
    COHERENT_ALIAS,
    COHERENT_INDEX_PREFIX,
    coherent_index_body,
)
from wic_history.search import DEFAULT_ALIAS


def test_mapping_is_dedicated_strict_cjk_and_cosine() -> None:
    mappings = coherent_index_body()["mappings"]
    assert isinstance(mappings, Mapping)
    properties = mappings["properties"]
    assert isinstance(properties, Mapping)
    title, content, embedding = (
        properties["title"],
        properties["content"],
        properties["embedding"],
    )
    assert isinstance(title, Mapping)
    assert isinstance(content, Mapping)
    assert isinstance(embedding, Mapping)
    method = embedding["method"]
    assert isinstance(method, Mapping)
    assert COHERENT_ALIAS == "wic-coherent-units-current"
    assert COHERENT_INDEX_PREFIX == "wic-coherent-units-build-"
    assert mappings["dynamic"] == "strict"
    assert title["analyzer"] == content["analyzer"] == "cjk"
    assert embedding["dimension"] == 1024
    assert method["space_type"] == "cosinesimil"
    assert DEFAULT_ALIAS == "wic-regions-current"


def test_coherent_search_modules_stay_within_strict_size_limit() -> None:
    oversized: list[tuple[str, int]] = []
    for module in Path("src/wic_history").glob("coherent_search*.py"):
        pure_lines = [
            line
            for line in module.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(pure_lines) > 250:
            oversized.append((module.name, len(pure_lines)))
    assert oversized == []
