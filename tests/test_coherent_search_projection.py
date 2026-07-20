from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from tests.coherent_search_fake_models import StoredDocument
from tests.coherent_search_support import OpenSearchFake, manifest
from wic_history.coherent_search import (
    COHERENT_ALIAS,
    COHERENT_INDEX_PREFIX,
    CoherentProjectionError,
    FrozenProjectionManifest,
    project_coherent_units,
    restore_coherent_alias,
)
from wic_history.search import DEFAULT_ALIAS


def test_projection_publishes_validated_revision_document(search_server: str) -> None:
    result = project_coherent_units(search_server, manifest())
    assert result.index_name.startswith(COHERENT_INDEX_PREFIX)
    assert result.documents_indexed == 1
    assert OpenSearchFake.aliases[DEFAULT_ALIAS] == "wic-regions-v2"
    revision_id, raw_document = next(iter(OpenSearchFake.documents.items()))
    document = StoredDocument.model_validate_json(raw_document)
    assert document.revision_id == revision_id
    assert document.year_min == 1925
    assert document.year_max == 1926
    assert len(document.sources) == 2
    assert "entity_ids" not in raw_document and "claim_ids" not in raw_document
    assert result.previous_index_names == ("wic-coherent-units-build-old",)


def test_projection_compensation_restores_previous_alias(search_server: str) -> None:
    projected = project_coherent_units(search_server, manifest())
    assert OpenSearchFake.aliases[COHERENT_ALIAS] == projected.index_name

    restore_coherent_alias(search_server, projected)

    assert OpenSearchFake.aliases[COHERENT_ALIAS] == "wic-coherent-units-build-old"


def test_projection_compensation_refuses_to_overwrite_newer_publish(
    search_server: str,
) -> None:
    projected = project_coherent_units(search_server, manifest())
    OpenSearchFake.aliases[COHERENT_ALIAS] = "wic-coherent-units-build-newer"

    with pytest.raises(CoherentProjectionError, match="changed"):
        restore_coherent_alias(search_server, projected)

    assert OpenSearchFake.aliases[COHERENT_ALIAS] == "wic-coherent-units-build-newer"


def test_projection_rejects_stale_manifest_before_alias_move(
    search_server: str,
) -> None:
    frozen = manifest()
    stale = replace(frozen, snapshot_sha256="0" * 64)
    with pytest.raises(CoherentProjectionError, match="snapshot"):
        _ = project_coherent_units(search_server, stale)
    assert OpenSearchFake.aliases[COHERENT_ALIAS] == "wic-coherent-units-build-old"
    assert not any(path == "/_aliases" for _, path, _ in OpenSearchFake.requests)


def test_projection_refuses_empty_incomplete_and_wrong_dimension(
    search_server: str,
) -> None:
    with pytest.raises(CoherentProjectionError, match="empty"):
        _ = project_coherent_units(
            search_server, FrozenProjectionManifest.freeze((), ())
        )
    frozen = manifest()
    with pytest.raises(CoherentProjectionError, match="matching embedding"):
        _ = project_coherent_units(
            search_server, FrozenProjectionManifest.freeze(frozen.articles, ())
        )
    short = replace(frozen.embeddings[0], vector=(0.1,))
    with pytest.raises(CoherentProjectionError, match="1024"):
        _ = project_coherent_units(
            search_server, FrozenProjectionManifest.freeze(frozen.articles, (short,))
        )


def test_projection_rejects_content_that_differs_from_exact_hash(
    search_server: str,
) -> None:
    frozen = manifest()
    article = frozen.articles[0]
    changed = FrozenProjectionManifest.freeze(
        (
            replace(
                article,
                bundle=replace(article.bundle, content=article.bundle.content + "篡改"),
            ),
        ),
        frozen.embeddings,
    )
    with pytest.raises(
        CoherentProjectionError, match="canonical content|content_sha256"
    ):
        _ = project_coherent_units(search_server, changed)


def test_projection_reconstructs_content_from_exact_segment_text(
    search_server: str,
) -> None:
    frozen = manifest()
    article = frozen.articles[0]
    changed_segment = replace(article.bundle.segments[0], text="女子學校改立")
    changed_bundle = replace(
        article.bundle, segments=(changed_segment, *article.bundle.segments[1:])
    )
    changed = FrozenProjectionManifest.freeze(
        (replace(article, bundle=changed_bundle),), frozen.embeddings
    )
    with pytest.raises(CoherentProjectionError, match="canonical content"):
        _ = project_coherent_units(search_server, changed)


def test_projection_validates_each_segment_composite_interval(
    search_server: str,
) -> None:
    frozen = manifest()
    article = frozen.articles[0]
    changed_segment = replace(article.bundle.segments[1], composite_start=0)
    changed_bundle = replace(
        article.bundle, segments=(article.bundle.segments[0], changed_segment)
    )
    changed = FrozenProjectionManifest.freeze(
        (replace(article, bundle=changed_bundle),), frozen.embeddings
    )
    with pytest.raises(CoherentProjectionError, match="composite interval"):
        _ = project_coherent_units(search_server, changed)


def test_projection_rejects_selected_interval_longer_than_segment_text(
    search_server: str,
) -> None:
    frozen = manifest()
    article = frozen.articles[0]
    original_segment = article.bundle.segments[0]
    first_segment = replace(original_segment, text_end=original_segment.text_end + 1)
    first_source = article.sources[0]
    assert first_source.source.text_end is not None
    changed_source = replace(
        first_source,
        source=first_source.source.model_copy(
            update={"text_end": first_source.source.text_end + 1}
        ),
    )
    changed_article = replace(
        article,
        bundle=replace(
            article.bundle,
            segments=(first_segment, *article.bundle.segments[1:]),
        ),
        sources=(changed_source, *article.sources[1:]),
    )
    changed = FrozenProjectionManifest.freeze((changed_article,), frozen.embeddings)
    with pytest.raises(CoherentProjectionError, match="selected interval"):
        _ = project_coherent_units(search_server, changed)


def test_manifest_rejects_embeddings_outside_pinned_model(search_server: str) -> None:
    frozen = manifest()
    article = frozen.articles[0]
    second_revision = uuid4()
    second_article = replace(
        article,
        coherent_unit_id=uuid4(),
        bundle=replace(article.bundle, coherent_unit_revision_id=second_revision),
    )
    second_embedding = replace(
        frozen.embeddings[0], revision_id=second_revision, model_name="other-model"
    )
    changed = FrozenProjectionManifest.freeze(
        (article, second_article), (frozen.embeddings[0], second_embedding)
    )
    with pytest.raises(CoherentProjectionError, match="pinned embedding model"):
        _ = project_coherent_units(search_server, changed)


def test_projection_rejects_two_active_revisions_for_one_unit(
    search_server: str,
) -> None:
    frozen = manifest()
    article = frozen.articles[0]
    second_revision = uuid4()
    second_article = replace(
        article,
        bundle=replace(article.bundle, coherent_unit_revision_id=second_revision),
    )
    second_embedding = replace(frozen.embeddings[0], revision_id=second_revision)
    changed = FrozenProjectionManifest.freeze(
        (article, second_article), (frozen.embeddings[0], second_embedding)
    )
    with pytest.raises(CoherentProjectionError, match="coherent_unit_id"):
        _ = project_coherent_units(search_server, changed)


@pytest.mark.parametrize("field", ["bundle_input", "source", "embedding_config"])
def test_projection_rejects_noncanonical_sha_fields(
    search_server: str, field: str
) -> None:
    frozen = manifest()
    article, embedding = frozen.articles[0], frozen.embeddings[0]
    if field == "bundle_input":
        article = replace(
            article, bundle=replace(article.bundle, input_sha256="C" * 64)
        )
        embedding = replace(embedding, input_sha256="C" * 64)
    elif field == "source":
        first = article.sources[0]
        first = replace(
            first,
            source=first.source.model_copy(update={"source_sha256": "A" * 64}),
        )
        article = replace(article, sources=(first, *article.sources[1:]))
    else:
        embedding = replace(embedding, configuration_sha256="F" * 64)
    changed = FrozenProjectionManifest.freeze((article,), (embedding,))
    with pytest.raises(CoherentProjectionError, match="lowercase SHA-256"):
        _ = project_coherent_units(search_server, changed)
