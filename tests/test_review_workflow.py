from __future__ import annotations

import unittest
from uuid import UUID

from wic_history.review_workflow import (
    CLAIM_EVIDENCE_VALIDATION_SQL,
    CLAIM_QUEUE_SQL,
    MENTION_QUEUE_SQL,
    ClaimReviewRequest,
    EntityResolutionRequest,
)


class ReviewWorkflowTests(unittest.TestCase):
    def test_create_new_requires_canonical_name(self):
        with self.assertRaises(ValueError):
            EntityResolutionRequest(
                selected_link_candidate_id=UUID(
                    "00000000-0000-0000-0000-000000000001"
                ),
                action="create_new",
                reviewer="historian-a",
            )

    def test_existing_link_rejects_new_entity_fields(self):
        with self.assertRaises(ValueError):
            EntityResolutionRequest(
                selected_link_candidate_id=UUID(
                    "00000000-0000-0000-0000-000000000001"
                ),
                action="link_existing",
                reviewer="historian-a",
                canonical_name="王氏",
            )

    def test_queue_joins_source_and_processing_provenance(self):
        self.assertIn("JOIN evidence.ocr_region", MENTION_QUEUE_SQL)
        self.assertIn("JOIN evidence.processing_run", MENTION_QUEUE_SQL)
        self.assertIn("m.mention_status", MENTION_QUEUE_SQL)

    def test_claim_queue_carries_entities_model_and_evidence_join_key(self):
        self.assertIn("subject.canonical_name", CLAIM_QUEUE_SQL)
        self.assertIn("JOIN evidence.processing_run", CLAIM_QUEUE_SQL)
        self.assertIn("c.claim_status", CLAIM_QUEUE_SQL)

    def test_claim_review_decision_is_constrained(self):
        request = ClaimReviewRequest(decision="dispute", reviewer="historian-a")
        self.assertEqual(request.decision, "dispute")
        with self.assertRaises(ValueError):
            ClaimReviewRequest(decision="publish", reviewer="historian-a")

    def test_claim_acceptance_revalidates_exact_ocr_offsets(self):
        self.assertIn("substring", CLAIM_EVIDENCE_VALIDATION_SQL)
        self.assertIn("supporting_quote", CLAIM_EVIDENCE_VALIDATION_SQL)
        self.assertIn("evidence_quote = ''", CLAIM_EVIDENCE_VALIDATION_SQL)
