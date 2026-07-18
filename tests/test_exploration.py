from __future__ import annotations

import unittest
from pathlib import Path
from uuid import UUID

from wic_history.exploration import (
    ACTIVE_OCR_CTE,
    COUNTS_SQL,
    NER_AGREEMENT_SQL,
    THEMES,
    theme_rows_to_leads,
)


class ExplorationTests(unittest.TestCase):
    def test_theme_rows_keep_exact_derivative_and_region_evidence(self):
        row = {
            "theme_id": "elite_women_publics",
            "matched_regions": 2,
            "matched_pages": 1,
            "year_start": 1925,
            "year_end": 1925,
            "region_id": UUID("00000000-0000-0000-0000-000000000001"),
            "raw_text": "曰富紳淑女固不貪便宜",
            "normalized_text": "曰富紳淑女固不貪便宜",
            "confidence": 0.91,
            "polygon": {
                "points": [
                    {"x": 1, "y": 1},
                    {"x": 2, "y": 1},
                    {"x": 2, "y": 2},
                    {"x": 1, "y": 2},
                ]
            },
            "source_uri": "s3://bucket/volume.pdf",
            "source_sha256": "a" * 64,
            "derivative_id": UUID("00000000-0000-0000-0000-000000000002"),
            "image_sha256": "b" * 64,
            "evidence_tier": "unreviewed_input",
            "volume_number": 219,
            "publication_year": 1925,
            "page_number": 308,
        }
        leads = theme_rows_to_leads([row])
        lead = next(item for item in leads if item.theme_id == "elite_women_publics")
        self.assertEqual(lead.matched_regions, 2)
        self.assertEqual(lead.examples[0].source.region_id, row["region_id"])
        self.assertEqual(lead.examples[0].source.derivative_id, row["derivative_id"])
        self.assertIn("machine_observation", lead.epistemic_label)
        self.assertEqual(len(leads), len(THEMES))

    def test_unmatched_themes_remain_visible_with_zero_counts(self):
        leads = theme_rows_to_leads([])
        self.assertTrue(leads)
        self.assertTrue(all(item.matched_regions == 0 for item in leads))

    def test_exploration_queries_use_only_active_ocr_selection(self):
        self.assertIn("selection.superseded_at IS NULL", ACTIVE_OCR_CTE)
        self.assertIn("candidate_ner_runs", COUNTS_SQL)
        self.assertIn("exact_agreements", NER_AGREEMENT_SQL)
        self.assertIn("JOIN active_ocr USING (region_id)", NER_AGREEMENT_SQL)

    def test_researcher_ui_labels_exploration_as_unreviewed(self):
        root = Path(__file__).parents[1]
        html = (root / "src/wic_history/static/index.html").read_text()
        javascript = (root / "src/wic_history/static/app.js").read_text()
        self.assertIn('id="exploration-button"', html)
        self.assertIn("Machine observations · unreviewed", html)
        self.assertIn("They are not historical findings", html)
        self.assertIn("fetch('/api/exploration')", javascript)


if __name__ == "__main__":
    unittest.main()
