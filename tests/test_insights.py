from __future__ import annotations

import unittest

from wic_history.insights import (
    ENTITY_TIMELINES_CYPHER,
    GRAPH_HUBS_CYPHER,
    MULTI_SOURCE_CLAIMS_CYPHER,
    InsightKind,
    graph_rows_to_items,
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
