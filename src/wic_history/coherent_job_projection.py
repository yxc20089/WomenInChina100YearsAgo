from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from .coherent_jobs import (
    COHERENT_CONFIGURATION,
    CoherentJobError,
)
from .coherent_search import (
    CoherentEmbedding,
    CoherentSource,
    FrozenProjectionManifest,
    ProjectionArticle,
)
from .coherent_search_contracts import JsonValue
from .evidence import Polygon, SourcePointer
from .semantic_repository import CoherentTextBundle, CoherentTextSegment

Document = tuple[Mapping[str, JsonValue], Sequence[Mapping[str, JsonValue]]]
_OBJECT_MAPPING = TypeAdapter(dict[str, object])


class _Rows(Protocol):
    def fetchall(self) -> Sequence[Mapping[str, object]]: ...


class _Connection(Protocol):
    def execute(self, query: str, params: object = None) -> _Rows: ...


def _mapping(value: object, label: str) -> Mapping[str, object]:
    try:
        return _OBJECT_MAPPING.validate_python(value)
    except ValidationError as exc:
        raise CoherentJobError(f"Coherent projection requires mapping {label}") from exc


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise CoherentJobError(f"Coherent projection requires string {label}")
    return value


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CoherentJobError(f"Coherent projection requires integer {label}")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    return None if value is None else _integer(value, label)


def _evidence_tier(
    value: object,
) -> Literal[
    "screening_derivative",
    "unreviewed_input",
    "non_gold_lossless_pilot",
    "historian_selected_gold",
]:
    if value == "screening_derivative":
        return "screening_derivative"
    if value == "unreviewed_input":
        return "unreviewed_input"
    if value == "non_gold_lossless_pilot":
        return "non_gold_lossless_pilot"
    if value == "historian_selected_gold":
        return "historian_selected_gold"
    raise CoherentJobError("Coherent projection requires a valid evidence tier")


def _uuid(value: object, label: str) -> UUID:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError as exc:
            raise CoherentJobError(
                f"Coherent projection requires UUID {label}"
            ) from exc
    raise CoherentJobError(f"Coherent projection requires UUID {label}")


def build_coherent_manifest(
    connection: _Connection, documents: Sequence[Document]
) -> FrozenProjectionManifest:
    revision_ids = [
        UUID(_string(document["id"], "revision id")) for document, _ in documents
    ]
    expected = {
        UUID(_string(document["id"], "revision id")): (
            _string(
                _mapping(document["metadata"], "metadata")["input_sha256"], "input hash"
            ),
            _string(
                _mapping(document["metadata"], "metadata")["content_sha256"],
                "content hash",
            ),
        )
        for document, _ in documents
    }
    region_ids = [
        UUID(_string(citation["region_id"], "region id"))
        for _, citations in documents
        for citation in citations
    ]
    page_rows = connection.execute(
        "SELECT region_id, page_id FROM evidence.ocr_region WHERE region_id = ANY(%s::uuid[])",
        (region_ids,),
    ).fetchall()
    page_ids = {
        _uuid(row["region_id"], "region id"): _uuid(row["page_id"], "page id")
        for row in page_rows
    }
    rows = connection.execute(
        """SELECT embedding.target_id, embedding.model_name,
                  embedding.model_revision, embedding.input_sha256,
                  embedding.content_sha256, embedding.configuration_sha256,
                  embedding.embedding::text AS vector
           FROM retrieval.embedding embedding
           JOIN evidence.processing_run run USING (run_id)
           WHERE embedding.target_kind = 'coherent_unit_revision'
             AND embedding.target_id = ANY(%s::uuid[])
             AND embedding.model_name = %s AND embedding.model_revision = %s
             AND run.status = 'completed'
             AND embedding.configuration_sha256 = %s""",
        (
            revision_ids,
            COHERENT_CONFIGURATION["model"],
            COHERENT_CONFIGURATION["revision"],
            COHERENT_CONFIGURATION["embedding_configuration_sha256"],
        ),
    ).fetchall()
    exact_rows = [
        row
        for row in rows
        if expected.get(_uuid(row["target_id"], "revision id"))
        == (
            _string(row["input_sha256"], "input hash"),
            _string(row["content_sha256"], "content hash"),
        )
    ]
    if len(exact_rows) != len(revision_ids):
        raise CoherentJobError(
            "Coherent projection requires exact complete embedding coverage"
        )
    embeddings = {
        _uuid(row["target_id"], "revision id"): CoherentEmbedding(
            _uuid(row["target_id"], "revision id"),
            "coherent_unit_revision",
            _string(row["model_name"], "model name"),
            _string(row["model_revision"], "model revision"),
            _string(row["input_sha256"], "input hash"),
            _string(row["content_sha256"], "content hash"),
            _string(row["configuration_sha256"], "configuration hash"),
            tuple(
                float(value)
                for value in _string(row["vector"], "vector").strip("[]").split(",")
            ),
        )
        for row in exact_rows
    }
    if len(embeddings) != len(exact_rows):
        raise CoherentJobError("Coherent projection found duplicate exact embeddings")
    articles: list[ProjectionArticle] = []
    for document, citations in documents:
        metadata = _mapping(document["metadata"], "metadata")
        revision_id = UUID(_string(document["id"], "revision id"))
        segments = tuple(
            CoherentTextSegment(
                _integer(citation["sequence_number"], "sequence"),
                UUID(_string(citation["region_id"], "region id")),
                page_ids[UUID(_string(citation["region_id"], "region id"))],
                UUID(_string(citation["selected_text_version_id"], "text version id")),
                UUID(_string(citation["text_selection_id"], "selection id")),
                _integer(citation["region_text_start"], "text start"),
                _integer(citation["region_text_end"], "text end"),
                _integer(citation["start_char"], "document start"),
                _integer(citation["end_char"], "document end"),
                _string(citation["exported_text"], "text"),
                _string(citation["role"], "role"),
                citation["polygon"],
            )
            for citation in citations
        )
        bundle = CoherentTextBundle(
            revision_id,
            _string(document["text"], "article text"),
            _string(metadata["input_sha256"], "input hash"),
            segments,
            (),
            _string(metadata["content_sha256"], "content hash"),
            "",
        )
        sources = tuple(
            CoherentSource(
                _integer(citation["sequence_number"], "sequence"),
                SourcePointer(
                    source_uri=_string(citation["source_uri"], "source uri"),
                    source_sha256=_string(citation["source_sha256"], "source hash"),
                    page_id=page_ids[UUID(_string(citation["region_id"], "region id"))],
                    derivative_id=UUID(
                        _string(citation["derivative_id"], "derivative id")
                    ),
                    image_uri=_string(citation["source_image_uri"], "image uri"),
                    image_sha256=_string(citation["source_image_sha256"], "image hash"),
                    evidence_tier=_evidence_tier(citation["evidence_tier"]),
                    volume_number=_integer(citation["volume_number"], "volume"),
                    publication_year=_optional_integer(
                        citation["publication_year"], "year"
                    ),
                    page_number=_integer(citation["page_number"], "page"),
                    region_id=UUID(_string(citation["region_id"], "region id")),
                    text_version_id=UUID(
                        _string(citation["selected_text_version_id"], "text version id")
                    ),
                    text_selection_id=UUID(
                        _string(citation["text_selection_id"], "selection id")
                    ),
                    polygon=Polygon.model_validate(citation["polygon"]),
                    text_start=_integer(citation["region_text_start"], "text start"),
                    text_end=_integer(citation["region_text_end"], "text end"),
                ),
            )
            for citation in citations
        )
        articles.append(
            ProjectionArticle(
                UUID(_string(metadata["coherent_unit_id"], "unit id")),
                _string(document["title"], "title"),
                bundle,
                sources,
            )
        )
    return FrozenProjectionManifest.freeze(
        tuple(articles), tuple(embeddings[item] for item in revision_ids)
    )
