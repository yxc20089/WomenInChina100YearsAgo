from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from .coherent_search_contracts import CoherentProjectionError, JsonValue
from .coherent_search_wire import (
    require_float,
    require_int,
    require_list,
    require_mapping,
    require_str,
)
from .evidence import RetrievalHit, RetrievalSourceSpan, SourcePointer


@dataclass(frozen=True, slots=True)
class ParsedHits:
    hits: list[RetrievalHit]
    warnings: set[str]


def _source_pointer(raw: dict[str, JsonValue]) -> SourcePointer:
    return SourcePointer.model_validate(
        {
            "source_uri": raw.get("source_uri"),
            "source_sha256": raw.get("source_sha256"),
            "page_id": raw.get("page_id"),
            "derivative_id": raw.get("derivative_id"),
            "image_uri": raw.get("image_uri"),
            "image_sha256": raw.get("image_sha256"),
            "evidence_tier": raw.get("evidence_tier"),
            "volume_number": raw.get("volume_number"),
            "publication_year": raw.get("publication_year"),
            "page_number": raw.get("page_number"),
            "region_id": raw.get("region_id"),
            "text_version_id": raw.get("text_version_id"),
            "text_selection_id": raw.get("text_selection_id"),
            "polygon": raw.get("polygon"),
            "text_start": raw.get("text_start"),
            "text_end": raw.get("text_end"),
        }
    )


def parse_hits(
    payload: JsonValue,
    retriever: str,
    expected_embedding_identity: tuple[str, str, str] | None = None,
) -> ParsedHits:
    incompatible_identity = False
    try:
        envelope = require_mapping(payload, "search response")
        hits_envelope = require_mapping(envelope.get("hits"), "search response")
        raw_hits = require_list(hits_envelope.get("hits"), "search response")
        hits: list[RetrievalHit] = []
        all_warnings: set[str] = set()
        for rank, value in enumerate(raw_hits, 1):
            item = dict(require_mapping(value, "search hit"))
            document = dict(require_mapping(item.get("_source"), "search hit source"))
            revision_id = UUID(require_str(document.get("revision_id"), "search hit"))
            embedding_identity = (
                require_str(document.get("embedding_model"), "search hit"),
                require_str(document.get("embedding_model_revision"), "search hit"),
                require_str(
                    document.get("embedding_configuration_sha256"), "search hit"
                ),
            )
            if (
                expected_embedding_identity is not None
                and embedding_identity != expected_embedding_identity
            ):
                incompatible_identity = True
            spans: list[RetrievalSourceSpan] = []
            source_warnings: dict[str, list[str]] = {}
            for source_value in require_list(document.get("sources"), "search sources"):
                raw = dict(require_mapping(source_value, "search source"))
                span = RetrievalSourceSpan(
                    document_id=revision_id,
                    sequence_number=require_int(
                        raw.get("sequence_number"), "search source"
                    ),
                    document_start=require_int(
                        raw.get("document_start"), "search source"
                    ),
                    document_end=require_int(raw.get("document_end"), "search source"),
                    role=require_str(raw.get("role"), "search source"),
                    source=_source_pointer(raw),
                )
                warnings = [
                    require_str(warning, "search source warning")
                    for warning in require_list(raw.get("warnings"), "search source")
                ]
                spans.append(span)
                source_warnings[span.citation_id] = warnings
                all_warnings.update(warnings)
            hits.append(
                RetrievalHit(
                    rank=rank,
                    score=require_float(item.get("_score"), "search hit"),
                    target_kind="reviewed_coherent_unit",
                    target_id=revision_id,
                    coherent_unit_id=UUID(
                        require_str(document.get("coherent_unit_id"), "search hit")
                    ),
                    sources=spans,
                    text=require_str(document.get("content"), "search hit"),
                    explanation={
                        "retriever": retriever,
                        "index": require_str(item.get("_index"), "search hit"),
                        "source_warnings": source_warnings,
                        "embedding_model": embedding_identity[0],
                        "embedding_model_revision": embedding_identity[1],
                        "input_sha256": require_str(
                            document.get("input_sha256"), "search hit"
                        ),
                        "content_sha256": require_str(
                            document.get("content_sha256"), "search hit"
                        ),
                        "embedding_configuration_sha256": embedding_identity[2],
                    },
                )
            )
        parsed = ParsedHits(hits, all_warnings)
    except (CoherentProjectionError, ValueError) as exc:
        raise CoherentProjectionError(
            "malformed OpenSearch response for coherent-unit retrieval"
        ) from exc
    if incompatible_identity:
        raise CoherentProjectionError(
            "OpenSearch result embedding identity differs from SearchSpec"
        )
    return parsed
