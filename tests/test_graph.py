from __future__ import annotations

import unittest
from datetime import date
from uuid import uuid4

from wic_history.graph import (
    REVIEWED_CLAIMS_SQL,
    REVIEWED_EVENT_EVIDENCE_SQL,
    REVIEWED_EVENTS_SQL,
    REVIEWED_LOCAL_CLUSTERS_SQL,
    REVIEWED_LOCAL_CLUSTER_MEMBERS_SQL,
    REVIEWED_MENTIONS_SQL,
    claim_payload,
    entity_payload,
)


class GraphProjectionTests(unittest.TestCase):
    def test_projection_query_is_reviewed_only(self):
        self.assertIn("c.claim_status = 'reviewed'", REVIEWED_CLAIMS_SQL)
        self.assertIn("subject.entity_status = 'reviewed'", REVIEWED_CLAIMS_SQL)
        self.assertIn("resolution.review_status = 'reviewed'", REVIEWED_MENTIONS_SQL)
        self.assertIn("resolution.superseded_at IS NULL", REVIEWED_MENTIONS_SQL)
        self.assertIn("evidence.entity_redirect", REVIEWED_MENTIONS_SQL)
        self.assertIn("event.event_status = 'reviewed'", REVIEWED_EVENTS_SQL)
        self.assertIn("event_evidence.review_status = 'reviewed'", REVIEWED_EVENT_EVIDENCE_SQL)
        self.assertIn("cluster.review_status = 'reviewed'", REVIEWED_LOCAL_CLUSTERS_SQL)
        self.assertIn("revision.superseded_at IS NULL", REVIEWED_LOCAL_CLUSTER_MEMBERS_SQL)
        self.assertIn("mention.mention_status = 'reviewed'", REVIEWED_LOCAL_CLUSTER_MEMBERS_SQL)

    def test_payloads_keep_reified_claim_and_provenance(self):
        subject_id = uuid4()
        entity = entity_payload(
            {
                "entity_id": subject_id,
                "entity_type": "person",
                "canonical_name": "某女士",
                "normalized_name": "某女士",
                "authority_uri": None,
                "attributes": {"reviewed": True},
            },
            "build-1",
        )
        claim = claim_payload(
            {
                "claim_id": uuid4(),
                "subject_entity_id": subject_id,
                "predicate": "attended",
                "object_entity_id": None,
                "object_literal": {"school": "某校"},
                "event_date_start": date(1925, 1, 1),
                "event_date_end": None,
                "claim_status": "reviewed",
                "confidence": 0.8,
                "supporting_quote": "某女士入學",
            },
            "build-1",
        )
        self.assertEqual(entity["projection_build_id"], "build-1")
        self.assertEqual(claim["subject_entity_id"], str(subject_id))
        self.assertEqual(claim["event_date_start"], "1925-01-01")
        self.assertEqual(claim["supporting_quote"], "某女士入學")


if __name__ == "__main__":
    unittest.main()
