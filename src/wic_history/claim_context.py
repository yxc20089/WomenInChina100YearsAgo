"""Resolve reviewed claims into deterministic, source-citing model context."""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Iterable
from uuid import UUID

from .evidence import Polygon, ScenarioEvidenceItem, SourcePointer


REVIEWED_CLAIM_CONTEXT_SQL = """
    SELECT c.claim_id, c.predicate, c.object_literal,
           subject.canonical_name AS subject_name,
           object.canonical_name AS object_name,
           ce.region_id, ce.text_start, ce.text_end, ce.polygon,
           s.source_uri, s.sha256 AS source_sha256,
           input.derivative_id, derivative.image_sha256,
           derivative.evidence_tier,
           v.volume_number, v.publication_year, p.page_number
    FROM evidence.claim c
    JOIN evidence.entity subject ON subject.entity_id = c.subject_entity_id
    LEFT JOIN evidence.entity object ON object.entity_id = c.object_entity_id
    JOIN evidence.claim_evidence ce USING (claim_id)
    JOIN evidence.ocr_region r USING (region_id)
    LEFT JOIN evidence.ocr_run_input input
      ON input.run_id = r.run_id AND input.page_id = r.page_id
    LEFT JOIN archive.page_derivative derivative
      ON derivative.derivative_id = input.derivative_id
     AND derivative.page_id = input.page_id
    JOIN archive.page p ON p.page_id = r.page_id
    JOIN archive.volume v ON v.volume_id = p.volume_id
    JOIN archive.source_object s ON s.source_object_id = v.source_object_id
    WHERE c.claim_id = ANY(%s)
      AND c.claim_status = 'reviewed'
      AND subject.entity_status = 'reviewed'
      AND (object.entity_id IS NULL OR object.entity_status = 'reviewed')
    ORDER BY c.claim_id, v.volume_number, p.page_number, ce.region_id
"""


def _object_label(row: dict[str, Any]) -> str:
    if row["object_name"]:
        return row["object_name"]
    return json.dumps(row["object_literal"], ensure_ascii=False, sort_keys=True)


def claim_items_from_rows(rows: Iterable[dict[str, Any]]) -> list[ScenarioEvidenceItem]:
    grouped: OrderedDict[UUID, dict[str, Any]] = OrderedDict()
    for row in rows:
        claim_id = row["claim_id"]
        item = grouped.setdefault(
            claim_id,
            {
                "statement": f"{row['subject_name']} — {row['predicate']} — {_object_label(row)}",
                "sources": [],
            },
        )
        item["sources"].append(
            SourcePointer(
                source_uri=row["source_uri"],
                source_sha256=row.get("source_sha256"),
                derivative_id=row.get("derivative_id"),
                image_sha256=row.get("image_sha256"),
                evidence_tier=row.get("evidence_tier"),
                volume_number=row["volume_number"],
                publication_year=row["publication_year"],
                page_number=row["page_number"],
                region_id=row["region_id"],
                polygon=Polygon.model_validate(row["polygon"]) if row["polygon"] else None,
                text_start=row["text_start"],
                text_end=row["text_end"],
            )
        )
    return [
        ScenarioEvidenceItem(
            statement=value["statement"],
            epistemic_label="directly_evidenced",
            sources=value["sources"],
            claim_ids=[claim_id],
        )
        for claim_id, value in grouped.items()
    ]


def load_reviewed_claim_items(
    database_url: str, claim_ids: set[UUID]
) -> list[ScenarioEvidenceItem]:
    if not claim_ids:
        return []
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(REVIEWED_CLAIM_CONTEXT_SQL, (list(claim_ids),)).fetchall()
    return claim_items_from_rows(rows)
