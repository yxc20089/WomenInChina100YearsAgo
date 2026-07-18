from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

from wic_history.insights import (
    ENTITY_TIMELINES_CYPHER,
    GRAPH_HUBS_CYPHER,
    MULTI_SOURCE_CLAIMS_CYPHER,
    EvidenceCounts,
    InsightKind,
    graph_rows_to_items,
    projection_status,
)


class InsightTests(unittest.TestCase):
    def test_graph_queries_are_reviewed_projection_only(self):
        for query in (GRAPH_HUBS_CYPHER, ENTITY_TIMELINES_CYPHER, MULTI_SOURCE_CLAIMS_CYPHER):
            self.assertIn("WICProjection", query)
            self.assertNotIn("candidate", query.lower())

    def test_graph_hub_is_labeled_as_analytical_signal(self):
        rows = [
            {
                "entity_id": "00000000-0000-0000-0000-000000000001",
                "canonical_name": "王女士",
                "entity_type": "person",
                "degree": 3,
                "predicates": ["attended_school"],
            }
        ]
        items = graph_rows_to_items(rows, [], [])
        self.assertEqual(items[0].kind, InsightKind.NETWORK_BRIDGE)
        self.assertEqual(items[0].epistemic_label, "analytical_signal_not_historical_claim")

    def test_projection_is_stale_when_review_is_newer(self):
        completed = datetime(2026, 7, 18, tzinfo=timezone.utc)
        result = projection_status(
            {
                "latest_build_id": UUID("00000000-0000-0000-0000-000000000001"),
                "latest_completed_at": completed,
                "latest_reviewed_at": completed + timedelta(seconds=1),
            },
            EvidenceCounts(reviewed_entities=1),
        )
        self.assertTrue(result.stale)

    def test_empty_evidence_does_not_require_projection(self):
        result = projection_status(
            {
                "latest_build_id": None,
                "latest_completed_at": None,
                "latest_reviewed_at": None,
            },
            EvidenceCounts(),
        )
        self.assertFalse(result.stale)
