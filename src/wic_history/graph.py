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
    SELECT c.claim_id,
           COALESCE(subject_redirect.canonical_entity_id, c.subject_entity_id)
               AS subject_entity_id,
           c.predicate,
           COALESCE(object_redirect.canonical_entity_id, c.object_entity_id)
               AS object_entity_id,
           c.object_literal, c.event_date_start, c.event_date_end,
           c.claim_status, c.confidence, c.supporting_quote
    FROM evidence.claim c
    LEFT JOIN evidence.entity_redirect subject_redirect
      ON subject_redirect.superseded_entity_id = c.subject_entity_id
     AND subject_redirect.reversed_at IS NULL
    JOIN evidence.entity subject
      ON subject.entity_id = COALESCE(
          subject_redirect.canonical_entity_id, c.subject_entity_id
      )
    LEFT JOIN evidence.entity_redirect object_redirect
      ON object_redirect.superseded_entity_id = c.object_entity_id
     AND object_redirect.reversed_at IS NULL
    LEFT JOIN evidence.entity object
      ON object.entity_id = COALESCE(
          object_redirect.canonical_entity_id, c.object_entity_id
      )
    WHERE c.claim_status = 'reviewed'
      AND subject.entity_status = 'reviewed'
      AND (c.object_entity_id IS NULL OR object.entity_status = 'reviewed')
    ORDER BY c.claim_id
"""

REVIEWED_MENTIONS_SQL = """
    SELECT m.mention_id,
           COALESCE(redirect.canonical_entity_id, resolution.proposed_entity_id)
               AS entity_id,
           m.region_id, m.entity_type,
           m.mention_text, m.text_start, m.text_end, m.confidence, m.polygon,
           span.evidence_span_id, span.text_version_id, span.surface_text,
           version.text_content AS authoritative_text,
           r.raw_text, p.page_id, p.page_number, p.source_image_uri,
           v.volume_number, v.publication_year,
           s.source_object_id, s.source_uri
    FROM evidence.entity_mention m
    JOIN evidence.mention_resolution resolution
      ON resolution.mention_id = m.mention_id
     AND resolution.review_status = 'reviewed'
     AND resolution.superseded_at IS NULL
     AND NOT resolution.is_nil
    LEFT JOIN evidence.entity_redirect redirect
      ON redirect.superseded_entity_id = resolution.proposed_entity_id
     AND redirect.reversed_at IS NULL
    JOIN evidence.entity e
      ON e.entity_id = COALESCE(
          redirect.canonical_entity_id, resolution.proposed_entity_id
      )
    JOIN evidence.evidence_span span
      ON span.evidence_span_id = m.evidence_span_id
    JOIN evidence.text_version version
      ON version.text_version_id = span.text_version_id
     AND version.region_id = m.region_id
    JOIN evidence.ocr_region r ON r.region_id = m.region_id
    JOIN archive.page p USING (page_id)
    JOIN archive.volume v USING (volume_id)
    JOIN archive.source_object s USING (source_object_id)
    WHERE m.mention_status = 'reviewed' AND e.entity_status = 'reviewed'
    ORDER BY m.mention_id
"""

REVIEWED_LOCAL_CLUSTERS_SQL = """
    SELECT cluster.local_cluster_id,
           run.coherent_unit_revision_id,
           cluster.review_status
    FROM evidence.local_coreference_cluster cluster
    JOIN evidence.local_coreference_run run USING (local_coreference_run_id)
    JOIN evidence.coherent_unit_revision revision
      ON revision.revision_id = run.coherent_unit_revision_id
     AND revision.superseded_at IS NULL
    WHERE cluster.review_status = 'reviewed'
    ORDER BY cluster.local_cluster_id
"""

REVIEWED_LOCAL_CLUSTER_MEMBERS_SQL = """
    SELECT member.local_cluster_id, mention.mention_id,
           mention.region_id, mention.entity_type,
           mention.mention_text, mention.text_start, mention.text_end,
           mention.confidence, mention.polygon,
           span.evidence_span_id, span.text_version_id, span.surface_text,
           version.text_content AS authoritative_text,
           region.raw_text, page.page_id, page.page_number,
           derivative.image_uri AS source_image_uri,
           volume.volume_number, volume.publication_year,
           source.source_object_id, source.source_uri
    FROM evidence.local_coreference_member member
    JOIN evidence.local_coreference_cluster cluster USING (local_cluster_id)
    JOIN evidence.local_coreference_run run
      ON run.local_coreference_run_id = member.local_coreference_run_id
    JOIN evidence.coherent_unit_revision revision
      ON revision.revision_id = run.coherent_unit_revision_id
     AND revision.superseded_at IS NULL
    JOIN evidence.entity_mention mention USING (mention_id)
    JOIN evidence.evidence_span span
      ON span.evidence_span_id = mention.evidence_span_id
    JOIN evidence.text_version version USING (text_version_id)
    JOIN evidence.ocr_region region ON region.region_id = version.region_id
    JOIN archive.page page USING (page_id)
    JOIN archive.page_derivative derivative
      ON derivative.derivative_id = page.preferred_derivative_id
    JOIN archive.volume volume USING (volume_id)
    JOIN archive.source_object source USING (source_object_id)
    WHERE cluster.review_status = 'reviewed'
      AND mention.mention_status = 'reviewed'
    ORDER BY member.local_cluster_id, mention.mention_id
"""

REVIEWED_CLAIM_EVIDENCE_SQL = """
    SELECT ce.claim_id, ce.claim_evidence_id, ce.region_id,
           span.text_start, span.text_end, ce.evidence_quote,
           COALESCE(span.polygon, ce.polygon) AS polygon,
           ce.support_role, span.evidence_span_id, span.text_version_id,
           span.surface_text, version.text_content AS authoritative_text,
           r.raw_text,
           p.page_id, p.page_number, p.source_image_uri,
           v.volume_number, v.publication_year,
           s.source_object_id, s.source_uri
    FROM evidence.claim_evidence ce
    JOIN evidence.claim c USING (claim_id)
    JOIN evidence.evidence_span span
      ON span.evidence_span_id = ce.evidence_span_id
    JOIN evidence.text_version version
      ON version.text_version_id = span.text_version_id
     AND version.region_id = ce.region_id
    JOIN evidence.ocr_region r ON r.region_id = ce.region_id
    JOIN archive.page p USING (page_id)
    JOIN archive.volume v USING (volume_id)
    JOIN archive.source_object s USING (source_object_id)
    WHERE c.claim_status = 'reviewed'
    ORDER BY ce.claim_id, ce.region_id
"""

REVIEWED_EVENTS_SQL = """
    SELECT event.event_id, event.event_type, event.trigger_evidence_span_id,
           event.date_start, event.date_end, event.date_precision,
           event.date_uncertainty,
           COALESCE(location_redirect.canonical_entity_id, event.location_entity_id)
               AS location_entity_id,
           event.location_literal, event.aspect, event.event_status,
           event.confidence, event.attributes
    FROM evidence.event event
    LEFT JOIN evidence.entity_redirect location_redirect
      ON location_redirect.superseded_entity_id = event.location_entity_id
     AND location_redirect.reversed_at IS NULL
    LEFT JOIN evidence.entity location
      ON location.entity_id = COALESCE(
          location_redirect.canonical_entity_id, event.location_entity_id
      )
    WHERE event.event_status = 'reviewed'
      AND (event.location_entity_id IS NULL OR location.entity_status = 'reviewed')
    ORDER BY event.event_id
"""

REVIEWED_EVENT_PARTICIPANTS_SQL = """
    SELECT participant.event_participant_id, participant.event_id,
           COALESCE(
               redirect.canonical_entity_id,
               participant.entity_id,
               resolution.proposed_entity_id
           ) AS entity_id,
           local_cluster.local_cluster_id,
           participant.mention_id, participant.participant_role,
           participant.evidence_span_id
    FROM evidence.event_participant participant
    LEFT JOIN evidence.mention_resolution resolution
      ON resolution.mention_id = participant.mention_id
     AND resolution.review_status = 'reviewed'
     AND resolution.superseded_at IS NULL
     AND NOT resolution.is_nil
    LEFT JOIN evidence.entity_redirect redirect
      ON redirect.superseded_entity_id = COALESCE(
          participant.entity_id, resolution.proposed_entity_id
      )
     AND redirect.reversed_at IS NULL
    LEFT JOIN evidence.local_coreference_member local_member
      ON local_member.mention_id = participant.mention_id
    LEFT JOIN evidence.local_coreference_cluster local_cluster
      ON local_cluster.local_cluster_id = local_member.local_cluster_id
     AND local_cluster.review_status = 'reviewed'
    JOIN evidence.event event USING (event_id)
    LEFT JOIN evidence.entity entity
      ON entity.entity_id = COALESCE(
          redirect.canonical_entity_id,
          participant.entity_id,
          resolution.proposed_entity_id
      )
    WHERE participant.review_status = 'reviewed'
      AND event.event_status = 'reviewed'
      AND (
          entity.entity_status = 'reviewed'
          OR local_cluster.local_cluster_id IS NOT NULL
      )
    ORDER BY participant.event_id, participant.event_participant_id
"""

REVIEWED_EVENT_EVIDENCE_SQL = """
    SELECT event_evidence.event_evidence_id, event_evidence.event_id,
           event_evidence.support_role,
           span.evidence_span_id, span.text_version_id, span.surface_text,
           span.text_start, span.text_end,
           version.text_content AS authoritative_text,
           region.region_id, region.raw_text,
           COALESCE(span.polygon, region.polygon) AS polygon,
           page.page_id, page.page_number, page.source_image_uri,
           volume.volume_number, volume.publication_year,
           source.source_object_id, source.source_uri
    FROM evidence.event_evidence event_evidence
    JOIN evidence.event event USING (event_id)
    JOIN evidence.evidence_span span USING (evidence_span_id)
    JOIN evidence.text_version version USING (text_version_id)
    JOIN evidence.ocr_region region ON region.region_id = version.region_id
    JOIN archive.page page USING (page_id)
    JOIN archive.volume volume USING (volume_id)
    JOIN archive.source_object source USING (source_object_id)
    WHERE event.event_status = 'reviewed'
      AND event_evidence.review_status = 'reviewed'
    ORDER BY event_evidence.event_id, event_evidence.event_evidence_id
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
        "source_object_id": str(row["source_object_id"]),
        "claim_id": str(row["claim_id"]) if row.get("claim_id") else None,
        "claim_evidence_id": (
            str(row["claim_evidence_id"]) if row.get("claim_evidence_id") else None
        ),
        "event_id": str(row["event_id"]) if row.get("event_id") else None,
        "event_evidence_id": (
            str(row["event_evidence_id"]) if row.get("event_evidence_id") else None
        ),
        "mention_id": str(row["mention_id"]) if row.get("mention_id") else None,
        "local_cluster_id": (
            str(row["local_cluster_id"]) if row.get("local_cluster_id") else None
        ),
        "entity_id": str(row["entity_id"]) if row.get("entity_id") else None,
        "evidence_span_id": str(row["evidence_span_id"]),
        "text_version_id": str(row["text_version_id"]),
        "region_id": str(row["region_id"]),
        "page_id": str(row["page_id"]),
        "source_uri": row["source_uri"],
        "source_image_uri": row["source_image_uri"],
        "volume_number": row["volume_number"],
        "publication_year": row["publication_year"],
        "page_number": row["page_number"],
        "raw_text": row["raw_text"],
        "authoritative_text": row["authoritative_text"],
        "surface_text": row["surface_text"],
        "mention_text": row.get("mention_text"),
        "entity_type": row.get("entity_type"),
        "text_start": row.get("text_start"),
        "text_end": row.get("text_end"),
        "confidence": row.get("confidence"),
        "evidence_quote": row.get("evidence_quote"),
        "support_role": row.get("support_role"),
        "polygon_json": _json(row.get("polygon")),
        "projection_build_id": build_id,
    }


def event_payload(row: dict[str, Any], build_id: str) -> dict[str, Any]:
    return {
        "event_id": str(row["event_id"]),
        "event_type": row["event_type"],
        "trigger_evidence_span_id": (
            str(row["trigger_evidence_span_id"])
            if row["trigger_evidence_span_id"]
            else None
        ),
        "date_start": _date(row["date_start"]),
        "date_end": _date(row["date_end"]),
        "date_precision": row["date_precision"],
        "date_uncertainty": row["date_uncertainty"],
        "location_entity_id": (
            str(row["location_entity_id"]) if row["location_entity_id"] else None
        ),
        "location_literal": row["location_literal"],
        "aspect": row["aspect"],
        "event_status": row["event_status"],
        "confidence": row["confidence"],
        "attributes_json": _json(row["attributes"]),
        "projection_build_id": build_id,
    }


def participant_payload(row: dict[str, Any], build_id: str) -> dict[str, Any]:
    return {
        "event_participant_id": str(row["event_participant_id"]),
        "event_id": str(row["event_id"]),
        "entity_id": str(row["entity_id"]) if row["entity_id"] else None,
        "local_cluster_id": (
            str(row["local_cluster_id"]) if row["local_cluster_id"] else None
        ),
        "mention_id": str(row["mention_id"]) if row["mention_id"] else None,
        "participant_role": row["participant_role"],
        "evidence_span_id": (
            str(row["evidence_span_id"]) if row["evidence_span_id"] else None
        ),
        "projection_build_id": build_id,
    }


def local_cluster_payload(row: dict[str, Any], build_id: str) -> dict[str, Any]:
    return {
        "local_cluster_id": str(row["local_cluster_id"]),
        "coherent_unit_revision_id": str(row["coherent_unit_revision_id"]),
        "review_status": row["review_status"],
        "projection_build_id": build_id,
    }


@dataclass(frozen=True, slots=True)
class GraphProjectionResult:
    build_id: str
    entities: int
    claims: int
    mentions: int
    claim_evidence: int
    events: int
    event_participants: int
    event_evidence: int
    local_identity_clusters: int
    local_cluster_members: int


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
        events = [
            event_payload(row, str(build_id))
            for row in database.execute(REVIEWED_EVENTS_SQL)
        ]
        event_participants = [
            participant_payload(row, str(build_id))
            for row in database.execute(REVIEWED_EVENT_PARTICIPANTS_SQL)
        ]
        event_evidence = [
            evidence_payload(row, str(build_id))
            for row in database.execute(REVIEWED_EVENT_EVIDENCE_SQL)
        ]
        local_identity_clusters = [
            local_cluster_payload(row, str(build_id))
            for row in database.execute(REVIEWED_LOCAL_CLUSTERS_SQL)
        ]
        local_cluster_members = [
            evidence_payload(row, str(build_id))
            for row in database.execute(REVIEWED_LOCAL_CLUSTER_MEMBERS_SQL)
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
                "CREATE CONSTRAINT wic_material_id IF NOT EXISTS FOR (n:WICMaterial) REQUIRE n.source_object_id IS UNIQUE",
                "CREATE CONSTRAINT wic_evidence_span_id IF NOT EXISTS FOR (n:WICEvidenceSpan) REQUIRE n.evidence_span_id IS UNIQUE",
                "CREATE CONSTRAINT wic_mention_id IF NOT EXISTS FOR (n:WICMention) REQUIRE n.mention_id IS UNIQUE",
                "CREATE CONSTRAINT wic_event_id IF NOT EXISTS FOR (n:WICEvent) REQUIRE n.event_id IS UNIQUE",
                "CREATE CONSTRAINT wic_local_cluster_id IF NOT EXISTS FOR (n:WICLocalIdentityCluster) REQUIRE n.local_cluster_id IS UNIQUE",
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
                MERGE (material:WICProjection:WICMaterial:Material {
                    source_object_id: row.source_object_id
                })
                SET material.source_uri = row.source_uri,
                    material.volume_number = row.volume_number,
                    material.publication_year = row.publication_year,
                    material.projection_build_id = row.projection_build_id
                MERGE (page:WICProjection:WICPage:Page {page_id: row.page_id})
                SET page.source_uri = row.source_uri,
                    page.source_image_uri = row.source_image_uri,
                    page.volume_number = row.volume_number,
                    page.publication_year = row.publication_year,
                    page.page_number = row.page_number,
                    page.projection_build_id = row.projection_build_id
                MERGE (page)-[:PART_OF]->(material)
                MERGE (region:WICProjection:WICRegion:Region {region_id: row.region_id})
                SET region.raw_text = row.raw_text,
                    region.polygon_json = row.polygon_json,
                    region.projection_build_id = row.projection_build_id
                MERGE (region)-[:ON_PAGE]->(page)
                MERGE (span:WICProjection:WICEvidenceSpan:EvidenceSpan {
                    evidence_span_id: row.evidence_span_id
                })
                SET span.text_version_id = row.text_version_id,
                    span.surface_text = row.surface_text,
                    span.authoritative_text = row.authoritative_text,
                    span.text_start = row.text_start,
                    span.text_end = row.text_end,
                    span.polygon_json = row.polygon_json,
                    span.projection_build_id = row.projection_build_id
                MERGE (span)-[:IN_REGION]->(region)
                MERGE (mention:WICProjection:WICMention:Mention {
                    mention_id: row.mention_id
                })
                SET mention.text = row.mention_text,
                    mention.entity_type = row.entity_type,
                    mention.confidence = row.confidence,
                    mention.projection_build_id = row.projection_build_id
                MERGE (mention)-[:ANCHORED_AT]->(span)
                WITH row, mention
                MATCH (entity:WICEntity {entity_id: row.entity_id})
                MERGE (mention)-[:REFERS_TO]->(entity)
                """,
                rows=mentions,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (material:WICProjection:WICMaterial:Material {
                    source_object_id: row.source_object_id
                })
                SET material.source_uri = row.source_uri,
                    material.volume_number = row.volume_number,
                    material.publication_year = row.publication_year,
                    material.projection_build_id = row.projection_build_id
                MERGE (page:WICProjection:WICPage:Page {page_id: row.page_id})
                SET page.source_uri = row.source_uri,
                    page.source_image_uri = row.source_image_uri,
                    page.volume_number = row.volume_number,
                    page.publication_year = row.publication_year,
                    page.page_number = row.page_number,
                    page.projection_build_id = row.projection_build_id
                MERGE (page)-[:PART_OF]->(material)
                MERGE (region:WICProjection:WICRegion:Region {region_id: row.region_id})
                SET region.raw_text = row.raw_text,
                    region.polygon_json = row.polygon_json,
                    region.projection_build_id = row.projection_build_id
                MERGE (region)-[:ON_PAGE]->(page)
                MERGE (span:WICProjection:WICEvidenceSpan:EvidenceSpan {
                    evidence_span_id: row.evidence_span_id
                })
                SET span.text_version_id = row.text_version_id,
                    span.surface_text = row.surface_text,
                    span.authoritative_text = row.authoritative_text,
                    span.text_start = row.text_start,
                    span.text_end = row.text_end,
                    span.polygon_json = row.polygon_json,
                    span.projection_build_id = row.projection_build_id
                MERGE (span)-[:IN_REGION]->(region)
                WITH row, span
                MATCH (claim:WICClaim {claim_id: row.claim_id})
                MERGE (claim)-[:EVIDENCED_BY {
                    claim_evidence_id: row.claim_evidence_id,
                    support_role: row.support_role,
                    quote: row.evidence_quote,
                    text_start: row.text_start,
                    text_end: row.text_end
                }]->(span)
                """,
                rows=claim_evidence,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                CREATE (event:WICProjection:WICEvent:Event)
                SET event = row
                FOREACH (_ IN CASE WHEN row.location_entity_id IS NULL THEN [] ELSE [1] END |
                    MERGE (location:WICEntity {entity_id: row.location_entity_id})
                    MERGE (event)-[:LOCATED_AT]->(location)
                )
                """,
                rows=events,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MATCH (event:WICEvent {event_id: row.event_id})
                FOREACH (_ IN CASE WHEN row.entity_id IS NULL THEN [] ELSE [1] END |
                    MERGE (entity:WICEntity {entity_id: row.entity_id})
                    CREATE (event)-[:HAS_PARTICIPANT {
                        event_participant_id: row.event_participant_id,
                        participant_role: row.participant_role,
                        mention_id: row.mention_id,
                        evidence_span_id: row.evidence_span_id
                    }]->(entity)
                )
                FOREACH (_ IN CASE WHEN row.local_cluster_id IS NULL THEN [] ELSE [1] END |
                    MERGE (cluster:WICLocalIdentityCluster {
                        local_cluster_id: row.local_cluster_id
                    })
                    CREATE (event)-[:HAS_LOCAL_PARTICIPANT {
                        event_participant_id: row.event_participant_id,
                        participant_role: row.participant_role,
                        mention_id: row.mention_id,
                        evidence_span_id: row.evidence_span_id
                    }]->(cluster)
                )
                """,
                rows=event_participants,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (material:WICProjection:WICMaterial:Material {
                    source_object_id: row.source_object_id
                })
                SET material.source_uri = row.source_uri,
                    material.volume_number = row.volume_number,
                    material.publication_year = row.publication_year,
                    material.projection_build_id = row.projection_build_id
                MERGE (page:WICProjection:WICPage:Page {page_id: row.page_id})
                SET page.source_uri = row.source_uri,
                    page.source_image_uri = row.source_image_uri,
                    page.volume_number = row.volume_number,
                    page.publication_year = row.publication_year,
                    page.page_number = row.page_number,
                    page.projection_build_id = row.projection_build_id
                MERGE (page)-[:PART_OF]->(material)
                MERGE (region:WICProjection:WICRegion:Region {region_id: row.region_id})
                SET region.raw_text = row.raw_text,
                    region.polygon_json = row.polygon_json,
                    region.projection_build_id = row.projection_build_id
                MERGE (region)-[:ON_PAGE]->(page)
                MERGE (span:WICProjection:WICEvidenceSpan:EvidenceSpan {
                    evidence_span_id: row.evidence_span_id
                })
                SET span.text_version_id = row.text_version_id,
                    span.surface_text = row.surface_text,
                    span.authoritative_text = row.authoritative_text,
                    span.polygon_json = row.polygon_json,
                    span.projection_build_id = row.projection_build_id
                MERGE (span)-[:IN_REGION]->(region)
                WITH row, span
                MATCH (event:WICEvent {event_id: row.event_id})
                CREATE (event)-[:EVIDENCED_BY {
                    event_evidence_id: row.event_evidence_id,
                    support_role: row.support_role
                }]->(span)
                """,
                rows=event_evidence,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                CREATE (cluster:WICProjection:WICLocalIdentityCluster:LocalIdentityCluster)
                SET cluster = row
                """,
                rows=local_identity_clusters,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (material:WICProjection:WICMaterial:Material {
                    source_object_id: row.source_object_id
                })
                SET material.source_uri = row.source_uri,
                    material.volume_number = row.volume_number,
                    material.publication_year = row.publication_year,
                    material.projection_build_id = row.projection_build_id
                MERGE (page:WICProjection:WICPage:Page {page_id: row.page_id})
                SET page.source_uri = row.source_uri,
                    page.source_image_uri = row.source_image_uri,
                    page.volume_number = row.volume_number,
                    page.publication_year = row.publication_year,
                    page.page_number = row.page_number,
                    page.projection_build_id = row.projection_build_id
                MERGE (page)-[:PART_OF]->(material)
                MERGE (region:WICProjection:WICRegion:Region {region_id: row.region_id})
                SET region.raw_text = row.raw_text,
                    region.polygon_json = row.polygon_json,
                    region.projection_build_id = row.projection_build_id
                MERGE (region)-[:ON_PAGE]->(page)
                MERGE (span:WICProjection:WICEvidenceSpan:EvidenceSpan {
                    evidence_span_id: row.evidence_span_id
                })
                SET span.text_version_id = row.text_version_id,
                    span.surface_text = row.surface_text,
                    span.authoritative_text = row.authoritative_text,
                    span.text_start = row.text_start,
                    span.text_end = row.text_end,
                    span.polygon_json = row.polygon_json,
                    span.projection_build_id = row.projection_build_id
                MERGE (span)-[:IN_REGION]->(region)
                MERGE (mention:WICProjection:WICMention:Mention {
                    mention_id: row.mention_id
                })
                SET mention.text = row.mention_text,
                    mention.entity_type = row.entity_type,
                    mention.confidence = row.confidence,
                    mention.projection_build_id = row.projection_build_id
                MERGE (mention)-[:ANCHORED_AT]->(span)
                WITH row, mention
                MATCH (cluster:WICLocalIdentityCluster {
                    local_cluster_id: row.local_cluster_id
                })
                MERGE (mention)-[:MEMBER_OF]->(cluster)
                """,
                rows=local_cluster_members,
            ).consume()
            session.run(
                """
                MATCH (event:WICEvent), (span:WICEvidenceSpan)
                WHERE event.trigger_evidence_span_id = span.evidence_span_id
                MERGE (event)-[:TRIGGERED_AT]->(span)
                """
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
        str(build_id),
        len(entities),
        len(claims),
        len(mentions),
        len(claim_evidence),
        len(events),
        len(event_participants),
        len(event_evidence),
        len(local_identity_clusters),
        len(local_cluster_members),
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
