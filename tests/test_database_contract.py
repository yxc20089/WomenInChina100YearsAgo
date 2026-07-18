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
                "005_ocr_run_selection.sql",
                "006_ner_run_input.sql",
                "007_ingestion_jobs.sql",
                "008_batch_terminal_states.sql",
                "009_job_replay_events.sql",
            ],
        )

    def test_page_derivatives_preserve_multiple_image_tiers(self):
        sql = Path("db/migrations/004_page_derivatives.sql").read_text(encoding="utf-8")
        self.assertIn("CREATE TABLE archive.page_derivative", sql)
        self.assertIn("historian_selected_gold", sql)
        self.assertIn("preferred_derivative_id", sql)
        self.assertIn("UNIQUE (page_id, image_sha256)", sql)

    def test_ocr_projection_requires_an_explicit_active_run(self):
        sql = Path("db/migrations/005_ocr_run_selection.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("CREATE TABLE evidence.ocr_run_input", sql)
        self.assertIn("CREATE TABLE evidence.page_ocr_selection", sql)
        self.assertIn("WHERE superseded_at IS NULL", sql)
        self.assertIn("historian_approved", sql)

    def test_ner_run_input_persists_benchmark_identity(self):
        sql = Path("db/migrations/006_ner_run_input.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("CREATE TABLE evidence.ner_run_input", sql)
        self.assertIn("source_ocr_run_id", sql)
        self.assertIn("input_sha256", sql)
        self.assertIn("ontology_version", sql)

    def test_ingestion_jobs_are_leased_and_dependency_gated(self):
        sql = Path("db/migrations/007_ingestion_jobs.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("CREATE TABLE pipeline.ingestion_job", sql)
        self.assertIn("CREATE TABLE pipeline.ingestion_job_dependency", sql)
        self.assertIn("lease_expires_at", sql)
        self.assertIn("input_fingerprint", sql)
        self.assertIn("CREATE TABLE pipeline.ingestion_job_event", sql)

    def test_ingestion_batches_have_terminal_failure_state(self):
        sql = Path("db/migrations/008_batch_terminal_states.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("'failed'", sql)
        self.assertIn("ingestion_job_dead_letter_idx", sql)

    def test_dead_letter_replay_is_auditable(self):
        sql = Path("db/migrations/009_job_replay_events.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("'reopened'", sql)


if __name__ == "__main__":
    unittest.main()
