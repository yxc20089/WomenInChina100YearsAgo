from __future__ import annotations

import unittest
from uuid import UUID

from wic_history.review_workflow import EntityResolutionRequest, MENTION_QUEUE_SQL


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
