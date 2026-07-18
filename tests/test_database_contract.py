from __future__ import annotations

import unittest
from pathlib import Path

from wic_history.migrate import migration_files


class DatabaseContractTests(unittest.TestCase):
    def test_evidence_migration_contains_required_layers_and_constraints(self):
        sql = Path("db/migrations/001_evidence_schema.sql").read_text(encoding="utf-8")
        for fragment in [
            "CREATE SCHEMA IF NOT EXISTS archive",
            "CREATE SCHEMA IF NOT EXISTS evidence",
            "CREATE SCHEMA IF NOT EXISTS retrieval",
            "CREATE TABLE IF NOT EXISTS evidence.ocr_region",
            "CREATE TABLE IF NOT EXISTS evidence.entity_mention",
            "CREATE TABLE IF NOT EXISTS evidence.claim_evidence",
            "num_nonnulls(object_entity_id, object_literal) = 1",
            "embedding vector(1024)",
        ]:
            self.assertIn(fragment, sql)

    def test_migrations_are_ordered(self):
        files = migration_files(Path("db/migrations"))
        self.assertEqual(
            [path.name for path in files],
            [
                "001_evidence_schema.sql",
                "002_review_workflow_indexes.sql",
                "003_claim_review_index.sql",
                "004_page_derivatives.sql",
            ],
        )

    def test_page_derivatives_preserve_multiple_image_tiers(self):
        sql = Path("db/migrations/004_page_derivatives.sql").read_text(encoding="utf-8")
        self.assertIn("CREATE TABLE archive.page_derivative", sql)
        self.assertIn("historian_selected_gold", sql)
        self.assertIn("preferred_derivative_id", sql)
        self.assertIn("UNIQUE (page_id, image_sha256)", sql)


if __name__ == "__main__":
    unittest.main()
