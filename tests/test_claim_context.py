from __future__ import annotations

import unittest
from uuid import UUID

from wic_history.claim_context import claim_items_from_rows


class ClaimContextTests(unittest.TestCase):
    def test_claim_rows_become_one_deterministic_statement_with_sources(self):
        claim_id = UUID("00000000-0000-0000-0000-000000000001")
        region_id = UUID("00000000-0000-0000-0000-000000000002")
        rows = [
            {
                "claim_id": claim_id,
                "predicate": "attended_school",
                "object_literal": None,
                "subject_name": "王女士",
                "object_name": "務本女塾",
                "region_id": region_id,
                "text_start": 0,
                "text_end": 4,
                "polygon": {"points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]},
                "source_uri": "s3://example/volume.pdf",
                "volume_number": 219,
                "publication_year": 1925,
                "page_number": 308,
            }
        ]

        items = claim_items_from_rows(rows)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].statement, "王女士 — attended_school — 務本女塾")
        self.assertEqual(items[0].claim_ids, [claim_id])
        self.assertEqual(items[0].sources[0].region_id, region_id)
