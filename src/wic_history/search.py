"""Rebuildable OpenSearch projection and evidence-citing retrieval."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID, uuid4

from .evidence import (
    Point,
    Polygon,
    RetrievalHit,
    RetrievalMode,
    RetrievalResponse,
    SourcePointer,
)
from .embedding_pipeline import BGEEmbedder, DEFAULT_MODEL, DEFAULT_REVISION


DEFAULT_INDEX = "wic-regions-v2"
DEFAULT_ALIAS = "wic-regions-current"


def region_index_body() -> dict[str, Any]:
    """Return the versioned region mapping; changing it requires a new index."""
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn": True,
            }
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "region_id": {"type": "keyword"},
                "page_id": {"type": "keyword"},
                "run_id": {"type": "keyword"},
                "source_uri": {"type": "keyword"},
                "source_sha256": {"type": "keyword"},
                "derivative_id": {"type": "keyword"},
                "source_image_uri": {"type": "keyword"},
                "source_image_sha256": {"type": "keyword"},
                "evidence_tier": {"type": "keyword"},
                "ocr_selection_basis": {"type": "keyword"},
                "volume_number": {"type": "integer"},
                "publication_year": {"type": "integer"},
                "page_number": {"type": "integer"},
                "reading_order": {"type": "integer"},
                "region_kind": {"type": "keyword"},
                "raw_text": {
                    "type": "text",
                    "analyzer": "cjk",
                    "fields": {"exact": {"type": "keyword", "ignore_above": 2048}},
                },
                "normalized_text": {"type": "text", "analyzer": "cjk"},
                "confidence": {"type": "half_float"},
                "language": {"type": "keyword"},
                "direction": {"type": "keyword"},
                "polygon": {"type": "object", "enabled": False},
                "page_warnings": {"type": "keyword", "ignore_above": 4096},
                "ocr_model": {"type": "keyword"},
                "ocr_model_revision": {"type": "keyword"},
                "embedding_model": {"type": "keyword"},
                "embedding_model_revision": {"type": "keyword"},
                "entity_ids": {"type": "keyword"},
                "claim_ids": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
                "indexed_at": {"type": "date"},
            },
        },
    }


def _clients(database_url: str, opensearch_url: str) -> tuple[Any, Any, Any]:
    try:
        import psycopg
        from opensearchpy import OpenSearch
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    database = psycopg.connect(database_url, row_factory=dict_row)
    search = OpenSearch(hosts=[opensearch_url], http_compress=True)
    return database, search, psycopg


def _as_polygon(value: dict[str, Any]) -> Polygon:
    return Polygon(points=[Point.model_validate(point) for point in value["points"]])


def region_document(row: dict[str, Any], indexed_at: str) -> dict[str, Any]:
    document = {
        "region_id": str(row["region_id"]),
        "page_id": str(row["page_id"]),
        "run_id": str(row["run_id"]),
        "source_uri": row["source_uri"],
        "source_sha256": row["source_sha256"],
        "derivative_id": str(row["derivative_id"]),
        "source_image_uri": row["source_image_uri"],
        "source_image_sha256": row["source_image_sha256"],
        "evidence_tier": row["evidence_tier"],
        "ocr_selection_basis": row["ocr_selection_basis"],
        "volume_number": row["volume_number"],
        "publication_year": row["publication_year"],
        "page_number": row["page_number"],
        "reading_order": row["reading_order"],
        "region_kind": row["region_kind"],
        "raw_text": row["raw_text"],
        "normalized_text": row["normalized_text"],
        "confidence": row["confidence"],
        "language": row["language"],
        "direction": row["direction"],
        "polygon": row["polygon"],
        "page_warnings": row["page_warnings"] or [],
        "ocr_model": row["ocr_model"],
        "ocr_model_revision": row["ocr_model_revision"],
        "embedding_model": row.get("embedding_model"),
        "embedding_model_revision": row.get("embedding_model_revision"),
        "entity_ids": [str(value) for value in (row["entity_ids"] or [])],
        "claim_ids": [str(value) for value in (row["claim_ids"] or [])],
        "indexed_at": indexed_at,
    }
    if row.get("embedding_text"):
        document["embedding"] = json.loads(row["embedding_text"])
    return document


REGION_PROJECTION_SQL = """
    SELECT r.region_id, r.page_id, r.run_id, r.region_kind, r.reading_order,
           r.polygon, r.raw_text, r.normalized_text, r.confidence, r.language,
           r.direction, p.page_number, derivative.derivative_id,
           derivative.image_uri AS source_image_uri,
           derivative.image_sha256 AS source_image_sha256,
           derivative.evidence_tier,
           derivative.metadata->'warnings' AS page_warnings,
           selection.selection_basis AS ocr_selection_basis,
           v.volume_number, v.publication_year, s.source_uri,
           s.sha256 AS source_sha256,
           pr.model_name AS ocr_model, pr.model_revision AS ocr_model_revision,
           em.embedding_text, em.embedding_model, em.embedding_model_revision,
           ARRAY(
               SELECT DISTINCT m.entity_id FROM evidence.entity_mention m
               WHERE m.region_id = r.region_id AND m.entity_id IS NOT NULL
           ) AS entity_ids,
           ARRAY(
               SELECT DISTINCT ce.claim_id
               FROM evidence.claim_evidence ce
               JOIN evidence.claim c USING (claim_id)
               WHERE ce.region_id = r.region_id AND c.claim_status = 'reviewed'
           ) AS claim_ids
    FROM evidence.ocr_region r
    JOIN archive.page p USING (page_id)
    JOIN archive.volume v USING (volume_id)
    JOIN archive.source_object s USING (source_object_id)
    JOIN evidence.processing_run pr USING (run_id)
    JOIN evidence.page_ocr_selection selection
      ON selection.page_id = r.page_id
     AND selection.run_id = r.run_id
     AND selection.superseded_at IS NULL
    JOIN evidence.ocr_run_input input
      ON input.run_id = selection.run_id
     AND input.page_id = selection.page_id
     AND input.derivative_id = selection.derivative_id
    JOIN archive.page_derivative derivative
      ON derivative.derivative_id = input.derivative_id
     AND derivative.page_id = input.page_id
    LEFT JOIN LATERAL (
        SELECT e.embedding::text AS embedding_text,
               e.model_name AS embedding_model,
               e.model_revision AS embedding_model_revision
        FROM retrieval.embedding e
        WHERE e.target_kind = 'region' AND e.target_id = r.region_id
        ORDER BY e.created_at DESC
        LIMIT 1
    ) em ON true
    ORDER BY v.volume_number, p.page_number, r.reading_order
"""


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    build_id: str
    index_name: str
    documents_indexed: int


def project_regions(
    database_url: str,
    opensearch_url: str,
    index_name: str = DEFAULT_INDEX,
    alias: str = DEFAULT_ALIAS,
    recreate: bool = False,
    batch_size: int = 500,
) -> ProjectionResult:
    """Build an OpenSearch index from PostgreSQL and atomically move its alias."""
    if not index_name.startswith("wic-regions-"):
        raise ValueError("Refusing to manage an index outside the wic-regions-* namespace")
    database, search, _ = _clients(database_url, opensearch_url)
    build_id = uuid4()
    count = 0
    try:
        database.execute(
            """
            INSERT INTO retrieval.projection_build (
                build_id, projection_kind, source_schema_version, configuration
            ) VALUES (%s, 'opensearch', '2.0', %s::jsonb)
            """,
            (
                build_id,
                json.dumps(
                    {
                        "index_name": index_name,
                        "alias": alias,
                        "ocr_run_policy": "active_page_selection_only",
                    }
                ),
            ),
        )
        database.commit()
        if search.indices.exists(index=index_name):
            if not recreate:
                raise ValueError(f"Index {index_name} already exists; use --recreate or a new version")
            search.indices.delete(index=index_name)
        search.indices.create(index=index_name, body=region_index_body())

        from opensearchpy.helpers import bulk

        indexed_at = datetime.now(timezone.utc).isoformat()
        with database.cursor(name="region_projection") as cursor:
            cursor.execute(REGION_PROJECTION_SQL)
            while rows := cursor.fetchmany(batch_size):
                actions = (
                    {
                        "_op_type": "index",
                        "_index": index_name,
                        "_id": str(row["region_id"]),
                        "_source": region_document(row, indexed_at),
                    }
                    for row in rows
                )
                indexed, errors = bulk(
                    search,
                    actions,
                    chunk_size=batch_size,
                    raise_on_error=False,
                    refresh=False,
                )
                if errors:
                    raise RuntimeError(f"OpenSearch bulk projection failed: {errors[:3]}")
                count += indexed
        search.indices.refresh(index=index_name)
        alias_actions: list[dict[str, Any]] = []
        if search.indices.exists_alias(name=alias):
            aliases = search.indices.get_alias(name=alias)
            alias_actions.extend(
                {"remove": {"index": old_index, "alias": alias}}
                for old_index in aliases
                if old_index != index_name
            )
        alias_actions.append({"add": {"index": index_name, "alias": alias}})
        search.indices.update_aliases(body={"actions": alias_actions})
        database.execute(
            """
            UPDATE retrieval.projection_build
            SET status = 'completed', completed_at = now(), artifact_uri = %s
            WHERE build_id = %s
            """,
            (f"opensearch://{index_name}", build_id),
        )
        database.commit()
        return ProjectionResult(str(build_id), index_name, count)
    except Exception as exc:
        database.rollback()
        database.execute(
            """
            UPDATE retrieval.projection_build
            SET status = 'failed', completed_at = now(), error_details = %s::jsonb
            WHERE build_id = %s
            """,
            (json.dumps({"type": type(exc).__name__, "message": str(exc)}), build_id),
        )
        database.commit()
        raise
    finally:
        database.close()


def lexical_search(
    opensearch_url: str,
    query: str,
    index: str = DEFAULT_ALIAS,
    limit: int = 10,
    year_start: int | None = None,
    year_end: int | None = None,
) -> RetrievalResponse:
    """Search raw and normalized OCR while retaining exact evidence pointers."""
    if not query.strip():
        raise ValueError("query must not be blank")
    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    search = OpenSearch(hosts=[opensearch_url], http_compress=True)
    filters: list[dict[str, Any]] = []
    if year_start is not None or year_end is not None:
        bounds = {key: value for key, value in (("gte", year_start), ("lte", year_end)) if value}
        filters.append({"range": {"publication_year": bounds}})
    body = {
        "size": limit,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["raw_text^2", "normalized_text"],
                            "type": "best_fields",
                        }
                    }
                ],
                "filter": filters,
            }
        },
        "_source": {"excludes": ["embedding"]},
    }
    response = search.search(index=index, body=body)
    hits = []
    for rank, item in enumerate(response["hits"]["hits"], 1):
        source = item["_source"]
        hits.append(
            RetrievalHit(
                rank=rank,
                score=float(item["_score"]),
                source=SourcePointer(
                    source_uri=source["source_uri"],
                    source_sha256=source["source_sha256"],
                    derivative_id=UUID(source["derivative_id"]),
                    image_sha256=source["source_image_sha256"],
                    evidence_tier=source["evidence_tier"],
                    volume_number=source["volume_number"],
                    publication_year=source["publication_year"],
                    page_number=source["page_number"],
                    region_id=UUID(source["region_id"]),
                    polygon=_as_polygon(source["polygon"]),
                ),
                text=source["raw_text"],
                normalized_text=source["normalized_text"],
                entity_ids=[UUID(value) for value in source["entity_ids"]],
                claim_ids=[UUID(value) for value in source["claim_ids"]],
                explanation={
                    "retriever": "OpenSearch CJK lexical",
                    "index": item["_index"],
                    "page_warnings": source["page_warnings"],
                    "derivative_id": source["derivative_id"],
                    "evidence_tier": source["evidence_tier"],
                    "ocr_selection_basis": source["ocr_selection_basis"],
                },
            )
        )
    return RetrievalResponse(
        query=query,
        mode=RetrievalMode.LEXICAL,
        hits=hits,
        warnings=_artifact_warnings(hits),
    )


def dense_search(
    opensearch_url: str,
    query: str,
    embedder: BGEEmbedder,
    index: str = DEFAULT_ALIAS,
    limit: int = 10,
    year_start: int | None = None,
    year_end: int | None = None,
) -> RetrievalResponse:
    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    search = OpenSearch(hosts=[opensearch_url], http_compress=True)
    vector = embedder.encode_query(query)
    knn: dict[str, Any] = {"vector": vector, "k": limit}
    if year_start is not None or year_end is not None:
        bounds = {
            key: value
            for key, value in (("gte", year_start), ("lte", year_end))
            if value is not None
        }
        knn["filter"] = {"range": {"publication_year": bounds}}
    response = search.search(
        index=index,
        body={
            "size": limit,
            "query": {"knn": {"embedding": knn}},
            "_source": {"excludes": ["embedding"]},
        },
    )
    hits = _retrieval_hits(response)
    warnings = _artifact_warnings(hits)
    return RetrievalResponse(query=query, mode=RetrievalMode.DENSE, hits=hits, warnings=warnings)


def hybrid_search(
    opensearch_url: str,
    query: str,
    embedder: BGEEmbedder,
    index: str = DEFAULT_ALIAS,
    limit: int = 10,
    candidate_limit: int = 50,
    rrf_k: int = 60,
    year_start: int | None = None,
    year_end: int | None = None,
) -> RetrievalResponse:
    lexical = lexical_search(
        opensearch_url, query, index, candidate_limit, year_start, year_end
    )
    dense = dense_search(
        opensearch_url, query, embedder, index, candidate_limit, year_start, year_end
    )
    fused: dict[UUID, tuple[RetrievalHit, float, dict[str, int]]] = {}
    for retriever, response in (("lexical", lexical), ("dense", dense)):
        for hit in response.hits:
            region_id = hit.source.region_id
            if region_id is None:
                continue
            existing = fused.get(region_id)
            score = (existing[1] if existing else 0.0) + 1.0 / (rrf_k + hit.rank)
            ranks = dict(existing[2]) if existing else {}
            ranks[retriever] = hit.rank
            fused[region_id] = (existing[0] if existing else hit, score, ranks)
    ordered = sorted(fused.values(), key=lambda item: item[1], reverse=True)[:limit]
    hits = []
    for rank, (hit, score, ranks) in enumerate(ordered, 1):
        hits.append(
            hit.model_copy(
                update={
                    "rank": rank,
                    "score": score,
                    "explanation": {**hit.explanation, "retriever": "RRF(CJK lexical+BGE-M3)", "component_ranks": ranks},
                }
            )
        )
    return RetrievalResponse(
        query=query,
        mode=RetrievalMode.HYBRID,
        hits=hits,
        warnings=_artifact_warnings(hits),
    )


def _retrieval_hits(response: dict[str, Any]) -> list[RetrievalHit]:
    hits = []
    for rank, item in enumerate(response["hits"]["hits"], 1):
        source = item["_source"]
        hits.append(
            RetrievalHit(
                rank=rank,
                score=float(item["_score"]),
                source=SourcePointer(
                    source_uri=source["source_uri"],
                    source_sha256=source["source_sha256"],
                    derivative_id=UUID(source["derivative_id"]),
                    image_sha256=source["source_image_sha256"],
                    evidence_tier=source["evidence_tier"],
                    volume_number=source["volume_number"],
                    publication_year=source["publication_year"],
                    page_number=source["page_number"],
                    region_id=UUID(source["region_id"]),
                    polygon=_as_polygon(source["polygon"]),
                ),
                text=source["raw_text"],
                normalized_text=source["normalized_text"],
                entity_ids=[UUID(value) for value in source["entity_ids"]],
                claim_ids=[UUID(value) for value in source["claim_ids"]],
                explanation={
                    "retriever": "OpenSearch BGE-M3 dense",
                    "index": item["_index"],
                    "page_warnings": source["page_warnings"],
                    "derivative_id": source["derivative_id"],
                    "evidence_tier": source["evidence_tier"],
                    "ocr_selection_basis": source["ocr_selection_basis"],
                },
            )
        )
    return hits


def _artifact_warnings(hits: list[RetrievalHit]) -> list[str]:
    return (
        ["One or more hits come from OCR that is not historian-selected gold."]
        if any(
            hit.explanation.get("page_warnings")
            or hit.explanation.get("evidence_tier") != "historian_selected_gold"
            for hit in hits
        )
        else []
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opensearch-url", default=os.environ.get("OPENSEARCH_URL", "http://127.0.0.1:9200"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    project = subparsers.add_parser("project")
    project.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    project.add_argument("--index", default=DEFAULT_INDEX)
    project.add_argument("--alias", default=DEFAULT_ALIAS)
    project.add_argument("--recreate", action="store_true")
    query = subparsers.add_parser("query")
    query.add_argument("query")
    query.add_argument("--index", default=DEFAULT_ALIAS)
    query.add_argument("--limit", type=int, default=10)
    query.add_argument("--year-start", type=int)
    query.add_argument("--year-end", type=int)
    query.add_argument("--mode", choices=("lexical", "dense", "hybrid"), default="lexical")
    query.add_argument("--model", default=DEFAULT_MODEL)
    query.add_argument("--revision", default=DEFAULT_REVISION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "project":
        if not args.database_url:
            raise SystemExit("DATABASE_URL or --database-url is required")
        result = project_regions(
            args.database_url,
            args.opensearch_url,
            args.index,
            args.alias,
            args.recreate,
        )
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 0
    if args.mode == "lexical":
        response = lexical_search(
            args.opensearch_url,
            args.query,
            args.index,
            args.limit,
            args.year_start,
            args.year_end,
        )
    else:
        embedder = BGEEmbedder(args.model, args.revision)
        response = (
            dense_search(
                args.opensearch_url,
                args.query,
                embedder,
                args.index,
                args.limit,
                args.year_start,
                args.year_end,
            )
            if args.mode == "dense"
            else hybrid_search(
                args.opensearch_url,
                args.query,
                embedder,
                args.index,
                args.limit,
                year_start=args.year_start,
                year_end=args.year_end,
            )
        )
    print(response.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
