"""Rebuild the reviewed-only Neo4j projection from PostgreSQL evidence."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Sequence
from uuid import uuid4


REVIEWED_ENTITIES_SQL = """
    SELECT entity_id, entity_type, canonical_name, normalized_name,
           authority_uri, attributes
    FROM evidence.entity
    WHERE entity_status = 'reviewed'
    ORDER BY entity_id
"""

REVIEWED_CLAIMS_SQL = """
    SELECT c.claim_id, c.subject_entity_id, c.predicate, c.object_entity_id,
           c.object_literal, c.event_date_start, c.event_date_end,
           c.claim_status, c.confidence, c.supporting_quote
    FROM evidence.claim c
    JOIN evidence.entity subject ON subject.entity_id = c.subject_entity_id
    LEFT JOIN evidence.entity object ON object.entity_id = c.object_entity_id
    WHERE c.claim_status = 'reviewed'
      AND subject.entity_status = 'reviewed'
      AND (c.object_entity_id IS NULL OR object.entity_status = 'reviewed')
    ORDER BY c.claim_id
"""

REVIEWED_MENTIONS_SQL = """
    SELECT m.mention_id, m.entity_id, m.region_id, m.entity_type,
           m.mention_text, m.text_start, m.text_end, m.confidence, m.polygon,
           r.raw_text, p.page_id, p.page_number, p.source_image_uri,
           v.volume_number, v.publication_year, s.source_uri
    FROM evidence.entity_mention m
    JOIN evidence.entity e ON e.entity_id = m.entity_id
    JOIN evidence.ocr_region r USING (region_id)
    JOIN archive.page p USING (page_id)
    JOIN archive.volume v USING (volume_id)
    JOIN archive.source_object s USING (source_object_id)
    WHERE m.mention_status = 'reviewed' AND e.entity_status = 'reviewed'
    ORDER BY m.mention_id
"""

REVIEWED_CLAIM_EVIDENCE_SQL = """
    SELECT ce.claim_id, ce.region_id, ce.text_start, ce.text_end,
           ce.evidence_quote, ce.polygon, r.raw_text,
           p.page_id, p.page_number, p.source_image_uri,
           v.volume_number, v.publication_year, s.source_uri
    FROM evidence.claim_evidence ce
    JOIN evidence.claim c USING (claim_id)
    JOIN evidence.ocr_region r USING (region_id)
    JOIN archive.page p USING (page_id)
    JOIN archive.volume v USING (volume_id)
    JOIN archive.source_object s USING (source_object_id)
    WHERE c.claim_status = 'reviewed'
    ORDER BY ce.claim_id, ce.region_id
"""


def _json(value: Any) -> str | None:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else None


def _date(value: date | None) -> str | None:
    return value.isoformat() if value else None


def entity_payload(row: dict[str, Any], build_id: str) -> dict[str, Any]:
    return {
        "entity_id": str(row["entity_id"]),
        "entity_type": row["entity_type"],
        "canonical_name": row["canonical_name"],
        "normalized_name": row["normalized_name"],
        "authority_uri": row["authority_uri"],
        "attributes_json": _json(row["attributes"]),
        "projection_build_id": build_id,
    }


def claim_payload(row: dict[str, Any], build_id: str) -> dict[str, Any]:
    return {
        "claim_id": str(row["claim_id"]),
        "subject_entity_id": str(row["subject_entity_id"]),
        "predicate": row["predicate"],
        "object_entity_id": str(row["object_entity_id"]) if row["object_entity_id"] else None,
        "object_literal_json": _json(row["object_literal"]),
        "event_date_start": _date(row["event_date_start"]),
        "event_date_end": _date(row["event_date_end"]),
        "claim_status": row["claim_status"],
        "confidence": row["confidence"],
        "supporting_quote": row["supporting_quote"],
        "projection_build_id": build_id,
    }


def evidence_payload(row: dict[str, Any], build_id: str) -> dict[str, Any]:
    return {
        "claim_id": str(row["claim_id"]) if row.get("claim_id") else None,
        "mention_id": str(row["mention_id"]) if row.get("mention_id") else None,
        "entity_id": str(row["entity_id"]) if row.get("entity_id") else None,
        "region_id": str(row["region_id"]),
        "page_id": str(row["page_id"]),
        "source_uri": row["source_uri"],
        "source_image_uri": row["source_image_uri"],
        "volume_number": row["volume_number"],
        "publication_year": row["publication_year"],
        "page_number": row["page_number"],
        "raw_text": row["raw_text"],
        "mention_text": row.get("mention_text"),
        "entity_type": row.get("entity_type"),
        "text_start": row.get("text_start"),
        "text_end": row.get("text_end"),
        "confidence": row.get("confidence"),
        "evidence_quote": row.get("evidence_quote"),
        "polygon_json": _json(row.get("polygon")),
        "projection_build_id": build_id,
    }


@dataclass(frozen=True, slots=True)
class GraphProjectionResult:
    build_id: str
    entities: int
    claims: int
    mentions: int
    claim_evidence: int


def project_reviewed_graph(
    database_url: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> GraphProjectionResult:
    try:
        import psycopg
        from neo4j import GraphDatabase
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc

    build_id = uuid4()
    with psycopg.connect(database_url, row_factory=dict_row) as database:
        database.execute(
            """
            INSERT INTO retrieval.projection_build (
                build_id, projection_kind, source_schema_version, configuration
            ) VALUES (%s, 'neo4j', '1.0', %s::jsonb)
            """,
            (build_id, json.dumps({"reviewed_only": True, "neo4j_uri": neo4j_uri})),
        )
        entities = [entity_payload(row, str(build_id)) for row in database.execute(REVIEWED_ENTITIES_SQL)]
        claims = [claim_payload(row, str(build_id)) for row in database.execute(REVIEWED_CLAIMS_SQL)]
        mentions = [evidence_payload(row, str(build_id)) for row in database.execute(REVIEWED_MENTIONS_SQL)]
        claim_evidence = [
            evidence_payload(row, str(build_id))
            for row in database.execute(REVIEWED_CLAIM_EVIDENCE_SQL)
        ]

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            for statement in (
                "CREATE CONSTRAINT wic_entity_id IF NOT EXISTS FOR (n:WICEntity) REQUIRE n.entity_id IS UNIQUE",
                "CREATE CONSTRAINT wic_claim_id IF NOT EXISTS FOR (n:WICClaim) REQUIRE n.claim_id IS UNIQUE",
                "CREATE CONSTRAINT wic_region_id IF NOT EXISTS FOR (n:WICRegion) REQUIRE n.region_id IS UNIQUE",
                "CREATE CONSTRAINT wic_page_id IF NOT EXISTS FOR (n:WICPage) REQUIRE n.page_id IS UNIQUE",
            ):
                session.run(statement).consume()
            session.run("MATCH (n:WICProjection) DETACH DELETE n").consume()
            session.run(
                """
                UNWIND $rows AS row
                CREATE (e:WICProjection:WICEntity:Entity)
                SET e = row
                """,
                rows=entities,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MATCH (subject:WICEntity {entity_id: row.subject_entity_id})
                CREATE (claim:WICProjection:WICClaim:Claim)
                SET claim = row
                CREATE (claim)-[:SUBJECT]->(subject)
                FOREACH (_ IN CASE WHEN row.object_entity_id IS NULL THEN [] ELSE [1] END |
                    MERGE (object:WICEntity {entity_id: row.object_entity_id})
                    CREATE (claim)-[:OBJECT]->(object)
                    CREATE (subject)-[:RELATED_TO {
                        claim_id: row.claim_id,
                        predicate: row.predicate,
                        status: row.claim_status
                    }]->(object)
                )
                """,
                rows=claims,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (page:WICProjection:WICPage:Page {page_id: row.page_id})
                SET page.source_uri = row.source_uri,
                    page.source_image_uri = row.source_image_uri,
                    page.volume_number = row.volume_number,
                    page.publication_year = row.publication_year,
                    page.page_number = row.page_number,
                    page.projection_build_id = row.projection_build_id
                MERGE (region:WICProjection:WICRegion:Region {region_id: row.region_id})
                SET region.raw_text = row.raw_text,
                    region.polygon_json = row.polygon_json,
                    region.projection_build_id = row.projection_build_id
                MERGE (region)-[:ON_PAGE]->(page)
                WITH row, region
                MATCH (entity:WICEntity {entity_id: row.entity_id})
                CREATE (entity)-[:MENTIONED_AS {
                    mention_id: row.mention_id,
                    text: row.mention_text,
                    text_start: row.text_start,
                    text_end: row.text_end,
                    confidence: row.confidence
                }]->(region)
                """,
                rows=mentions,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (page:WICProjection:WICPage:Page {page_id: row.page_id})
                SET page.source_uri = row.source_uri,
                    page.source_image_uri = row.source_image_uri,
                    page.volume_number = row.volume_number,
                    page.publication_year = row.publication_year,
                    page.page_number = row.page_number,
                    page.projection_build_id = row.projection_build_id
                MERGE (region:WICProjection:WICRegion:Region {region_id: row.region_id})
                SET region.raw_text = row.raw_text,
                    region.polygon_json = row.polygon_json,
                    region.projection_build_id = row.projection_build_id
                MERGE (region)-[:ON_PAGE]->(page)
                WITH row, region
                MATCH (claim:WICClaim {claim_id: row.claim_id})
                CREATE (claim)-[:EVIDENCED_BY {
                    quote: row.evidence_quote,
                    text_start: row.text_start,
                    text_end: row.text_end,
                    polygon_json: row.polygon_json
                }]->(region)
                """,
                rows=claim_evidence,
            ).consume()
        with psycopg.connect(database_url) as database:
            database.execute(
                """
                UPDATE retrieval.projection_build
                SET status = 'completed', completed_at = now(), artifact_uri = %s
                WHERE build_id = %s
                """,
                (f"neo4j://reviewed/{build_id}", build_id),
            )
    except Exception as exc:
        with psycopg.connect(database_url) as database:
            database.execute(
                """
                UPDATE retrieval.projection_build
                SET status = 'failed', completed_at = now(), error_details = %s::jsonb
                WHERE build_id = %s
                """,
                (json.dumps({"type": type(exc).__name__, "message": str(exc)}), build_id),
            )
        raise
    finally:
        driver.close()
    return GraphProjectionResult(
        str(build_id), len(entities), len(claims), len(mentions), len(claim_evidence)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url or not args.neo4j_password:
        raise SystemExit("DATABASE_URL and NEO4J_PASSWORD (or CLI equivalents) are required")
    result = project_reviewed_graph(
        args.database_url, args.neo4j_uri, args.neo4j_user, args.neo4j_password
    )
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
