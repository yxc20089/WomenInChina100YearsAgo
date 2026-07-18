"""Reviewed-only analytical signals from PostgreSQL and the Neo4j projection."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field

from .evidence import StrictModel


class InsightKind(StrEnum):
    NETWORK_BRIDGE = "network_bridge"
    LONGITUDINAL_PRESENCE = "longitudinal_presence"
    MULTI_SOURCE_CLAIM = "multi_source_claim"


class EvidenceCounts(StrictModel):
    reviewed_entities: int = 0
    reviewed_mentions: int = 0
    reviewed_claims: int = 0
    reviewed_claim_evidence: int = 0


class GraphProjectionStatus(StrictModel):
    latest_build_id: UUID | None = None
    latest_completed_at: datetime | None = None
    latest_reviewed_at: datetime | None = None
    stale: bool = False
    reason: str


class InsightItem(StrictModel):
    kind: InsightKind
    title: str
    summary: str
    entity_ids: list[UUID] = Field(default_factory=list)
    claim_ids: list[UUID] = Field(default_factory=list)
    supporting_page_ids: list[UUID] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    epistemic_label: str = "analytical_signal_not_historical_claim"


class InsightReport(StrictModel):
    generated_at: datetime
    evidence_counts: EvidenceCounts
    graph_projection: GraphProjectionStatus
    items: list[InsightItem]
    warnings: list[str] = Field(default_factory=list)


EVIDENCE_COUNTS_SQL = """
    SELECT
      (SELECT count(*) FROM evidence.entity WHERE entity_status = 'reviewed') AS reviewed_entities,
      (SELECT count(*) FROM evidence.entity_mention WHERE mention_status = 'reviewed') AS reviewed_mentions,
      (SELECT count(*) FROM evidence.claim WHERE claim_status = 'reviewed') AS reviewed_claims,
      (SELECT count(*) FROM evidence.claim_evidence ce
         JOIN evidence.claim c USING (claim_id)
        WHERE c.claim_status = 'reviewed') AS reviewed_claim_evidence
"""

PROJECTION_STATUS_SQL = """
    SELECT
      (SELECT build_id FROM retrieval.projection_build
        WHERE projection_kind = 'neo4j' AND status = 'completed'
        ORDER BY completed_at DESC NULLS LAST, started_at DESC LIMIT 1) AS latest_build_id,
      (SELECT completed_at FROM retrieval.projection_build
        WHERE projection_kind = 'neo4j' AND status = 'completed'
        ORDER BY completed_at DESC NULLS LAST, started_at DESC LIMIT 1) AS latest_completed_at,
      (SELECT max(changed_at) FROM (
         SELECT reviewed_at AS changed_at
           FROM evidence.review_decision
          WHERE (target_kind = 'entity_link' AND new_value->>'entity_id' IS NOT NULL)
             OR (target_kind = 'claim' AND new_value->>'claim_status' = 'reviewed')
         UNION ALL
         SELECT greatest(created_at, updated_at) FROM evidence.entity
          WHERE entity_status = 'reviewed'
         UNION ALL
         SELECT created_at FROM evidence.entity_mention
          WHERE mention_status = 'reviewed' AND entity_id IS NOT NULL
         UNION ALL
         SELECT updated_at FROM evidence.claim WHERE claim_status = 'reviewed'
       ) graph_changes) AS latest_reviewed_at
"""

GRAPH_HUBS_CYPHER = """
    MATCH (entity:WICProjection:WICEntity)
    OPTIONAL MATCH (entity)-[relationship:RELATED_TO]-(other:WICEntity)
    WITH entity, count(DISTINCT other) AS degree,
         [value IN collect(DISTINCT relationship.predicate) WHERE value IS NOT NULL] AS predicates
    WHERE degree >= 2
    RETURN entity.entity_id AS entity_id,
           entity.canonical_name AS canonical_name,
           entity.entity_type AS entity_type,
           degree, predicates
    ORDER BY degree DESC, canonical_name
    LIMIT 20
"""

ENTITY_TIMELINES_CYPHER = """
    MATCH (entity:WICProjection:WICEntity)-[:MENTIONED_AS]->
          (:WICProjection:WICRegion)-[:ON_PAGE]->(page:WICProjection:WICPage)
    WITH entity, count(*) AS mention_count,
         collect(DISTINCT page.publication_year) AS years,
         collect(DISTINCT page.page_id) AS page_ids
    WHERE size(years) >= 2
    RETURN entity.entity_id AS entity_id,
           entity.canonical_name AS canonical_name,
           entity.entity_type AS entity_type,
           mention_count, years, page_ids
    ORDER BY size(years) DESC, mention_count DESC, canonical_name
    LIMIT 20
"""

MULTI_SOURCE_CLAIMS_CYPHER = """
    MATCH (claim:WICProjection:WICClaim)-[:EVIDENCED_BY]->
          (:WICProjection:WICRegion)-[:ON_PAGE]->(page:WICProjection:WICPage)
    WITH claim, count(DISTINCT page) AS page_count,
         collect(DISTINCT page.page_id) AS page_ids,
         collect(DISTINCT page.publication_year) AS years
    WHERE page_count >= 2
    RETURN claim.claim_id AS claim_id, claim.predicate AS predicate,
           claim.subject_entity_id AS subject_entity_id,
           page_count, page_ids, years
    ORDER BY page_count DESC, claim_id
    LIMIT 20
"""


def _uuid_values(values: list[str] | None) -> list[UUID]:
    return [UUID(value) for value in (values or [])]


def graph_rows_to_items(
    hubs: list[dict[str, Any]],
    timelines: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> list[InsightItem]:
    items = []
    for row in hubs:
        items.append(
            InsightItem(
                kind=InsightKind.NETWORK_BRIDGE,
                title=f"Potential network bridge: {row['canonical_name']}",
                summary=(
                    f"This reviewed entity connects to {row['degree']} other reviewed entities "
                    "in the derived claim graph. Inspect the cited claims before interpretation."
                ),
                entity_ids=[UUID(row["entity_id"])],
                metrics={"degree": row["degree"], "predicates": row["predicates"]},
            )
        )
    for row in timelines:
        years = sorted(value for value in row["years"] if value is not None)
        items.append(
            InsightItem(
                kind=InsightKind.LONGITUDINAL_PRESENCE,
                title=f"Repeated presence across years: {row['canonical_name']}",
                summary=(
                    f"This reviewed entity has {row['mention_count']} reviewed mentions across "
                    f"{len(years)} publication years. This is a lead for longitudinal research."
                ),
                entity_ids=[UUID(row["entity_id"])],
                supporting_page_ids=_uuid_values(row["page_ids"]),
                metrics={"mention_count": row["mention_count"], "years": years},
            )
        )
    for row in claims:
        items.append(
            InsightItem(
                kind=InsightKind.MULTI_SOURCE_CLAIM,
                title=f"Reviewed claim supported on {row['page_count']} pages",
                summary=(
                    f"The reviewed predicate “{row['predicate']}” has evidence on multiple pages. "
                    "Compare those passages for corroboration or disagreement."
                ),
                entity_ids=[UUID(row["subject_entity_id"])],
                claim_ids=[UUID(row["claim_id"])],
                supporting_page_ids=_uuid_values(row["page_ids"]),
                metrics={"page_count": row["page_count"], "years": row["years"]},
            )
        )
    return items


def projection_status(
    row: dict[str, Any], counts: EvidenceCounts
) -> GraphProjectionStatus:
    latest_build_id = row["latest_build_id"]
    latest_completed_at = row["latest_completed_at"]
    latest_reviewed_at = row["latest_reviewed_at"]
    if latest_completed_at is None:
        has_reviewed_graph_data = bool(
            counts.reviewed_entities or counts.reviewed_mentions or counts.reviewed_claims
        )
        return GraphProjectionStatus(
            latest_build_id=latest_build_id,
            latest_completed_at=latest_completed_at,
            latest_reviewed_at=latest_reviewed_at,
            stale=has_reviewed_graph_data,
            reason=(
                "No completed Neo4j projection exists for the reviewed evidence."
                if has_reviewed_graph_data
                else "No completed Neo4j projection is required while reviewed graph data is empty."
            ),
        )
    stale = bool(latest_reviewed_at and latest_reviewed_at > latest_completed_at)
    return GraphProjectionStatus(
        latest_build_id=latest_build_id,
        latest_completed_at=latest_completed_at,
        latest_reviewed_at=latest_reviewed_at,
        stale=stale,
        reason=(
            "Reviewed evidence changes are newer than the projection; run wic-graph before graph analysis."
            if stale
            else "The latest completed Neo4j projection is at least as new as reviewed evidence changes."
        ),
    )


def build_insight_report(
    database_url: str,
    *,
    neo4j_uri: str | None = None,
    neo4j_user: str = "neo4j",
    neo4j_password: str | None = None,
) -> InsightReport:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    with psycopg.connect(database_url, row_factory=dict_row) as database:
        counts = EvidenceCounts.model_validate(database.execute(EVIDENCE_COUNTS_SQL).fetchone())
        graph_projection = projection_status(
            database.execute(PROJECTION_STATUS_SQL).fetchone(), counts
        )

    warnings = [
        "Insights are analytical signals over reviewed data, not new historical facts."
    ]
    items: list[InsightItem] = []
    if graph_projection.stale:
        warnings.append(graph_projection.reason)
    elif not neo4j_uri or not neo4j_password:
        warnings.append("Neo4j insight queries are unavailable because graph credentials are not configured.")
    else:
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            try:
                driver.verify_connectivity()
                with driver.session() as session:
                    hubs = (
                        session.run(GRAPH_HUBS_CYPHER).data()
                        if counts.reviewed_claims
                        else []
                    )
                    timelines = (
                        session.run(ENTITY_TIMELINES_CYPHER).data()
                        if counts.reviewed_mentions
                        else []
                    )
                    claims = (
                        session.run(MULTI_SOURCE_CLAIMS_CYPHER).data()
                        if counts.reviewed_claims
                        else []
                    )
                items = graph_rows_to_items(hubs, timelines, claims)
            finally:
                driver.close()
        except Exception as exc:
            warnings.append(f"Neo4j insight queries failed: {type(exc).__name__}: {exc}")
    if counts.reviewed_entities == 0 and counts.reviewed_claims == 0:
        warnings.append(
            "No reviewed entities or claims exist yet; the insight report correctly contains no signals."
        )
    elif not items:
        warnings.append("Reviewed data exists, but no current graph pattern passes the insight thresholds.")
    return InsightReport(
        generated_at=datetime.now(timezone.utc),
        evidence_counts=counts,
        graph_projection=graph_projection,
        items=items,
        warnings=warnings,
    )
