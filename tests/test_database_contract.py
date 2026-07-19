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
                "010_article_segmentation_review.sql",
                "011_segmentation_span_splits.sql",
                "012_coherent_unit_provenance_guards.sql",
                "013_segmentation_action_idempotency.sql",
                "014_ingestion_entity_link_stage.sql",
                "015_e2e_evidence_identity_events.sql",
                "016_layout_ingestion_stage.sql",
                "017_reviewed_text_semantic_runs.sql",
                "018_semantic_discovery_task.sql",
                "019_batch_identity_resolution.sql",
                "020_identity_candidate_model_runs.sql",
                "021_authoritative_visual_outputs_local_identity.sql",
            ],
        )

    def test_ingestion_stage_constraint_is_append_only_widened_for_entity_link(self):
        sql = Path("db/migrations/014_ingestion_entity_link_stage.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("DROP CONSTRAINT ingestion_job_stage_check", sql)
        self.assertIn("'entity_link'", sql)
        self.assertIn("VALIDATE CONSTRAINT ingestion_job_stage_check", sql)

    def test_e2e_schema_preserves_occurrences_events_and_reversible_identity(self):
        sql = Path(
            "db/migrations/015_e2e_evidence_identity_events.sql"
        ).read_text(encoding="utf-8")
        for fragment in (
            "CREATE TABLE evidence.layout_region",
            "CREATE TABLE evidence.text_version",
            "CREATE TABLE evidence.evidence_span",
            "evidence_span_surface_trigger",
            "CREATE TABLE evidence.mention_resolution",
            "mention_resolution_one_active_reviewed_idx",
            "CREATE TABLE evidence.entity_redirect",
            "entity_redirect_cycle_trigger",
            "CREATE TABLE evidence.event",
            "CREATE TABLE evidence.event_participant",
            "CREATE TABLE evidence.event_evidence",
            "claim_evidence_id",
        ):
            self.assertIn(fragment, sql)

    def test_batch_identity_resolution_is_frozen_and_reviewable(self):
        sql = Path("db/migrations/019_batch_identity_resolution.sql").read_text(
            encoding="utf-8"
        )
        for fragment in (
            "CREATE TABLE evidence.identity_resolution_cohort",
            "CREATE TABLE evidence.identity_profile",
            "CREATE TABLE evidence.identity_pair_candidate",
            "CREATE TABLE evidence.identity_pair_decision",
            "'SAME', 'DIFFERENT', 'INSUFFICIENT'",
        ):
            self.assertIn(fragment, sql)

        run_sql = Path(
            "db/migrations/020_identity_candidate_model_runs.sql"
        ).read_text(encoding="utf-8")
        self.assertIn("embedding_run_id", run_sql)
        self.assertIn("reranker_run_id", run_sql)

    def test_first_build_visual_outputs_and_identity_are_immutable_and_local(self):
        sql = Path(
            "db/migrations/021_authoritative_visual_outputs_local_identity.sql"
        ).read_text(encoding="utf-8")
        for fragment in (
            "CREATE TABLE evidence.confidence_calibration",
            "CREATE TABLE evidence.visual_model_output",
            "'spotting', 'layout', 'recognition'",
            "CREATE TABLE evidence.visual_model_evidence_path",
            "visual_model_output_sha256_trigger",
            "visual_model_evidence_path_trigger",
            "semantic_run_output_sha256_trigger",
            "semantic_run_input_immutable_trigger",
            "confidence_status",
            "calibration_id",
            "local_coreference_cluster_run_scope_fk",
            "local_coreference_member_cluster_scope_fk",
            "local_coreference_run_active_revision_trigger",
            "local_coreference_member_scope_trigger",
            "local_coreference_cluster_review_transition_trigger",
            "'mention_resolution', 'local_coreference_cluster'",
            "mention.evidence_span_id IS NOT NULL",
        ):
            self.assertIn(fragment, sql)
        self.assertNotIn("INSERT INTO evidence.entity_redirect", sql)
        self.assertNotIn("UPDATE evidence.entity\n", sql)

    def test_reviewed_segmentation_is_distinct_from_machine_proposals(self):
        sql = Path("db/migrations/010_article_segmentation_review.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("source_ocr_selection_id", sql)
        self.assertIn("evidence.coherent_unit_revision", sql)
        self.assertIn("evidence.coherent_unit_span", sql)
        self.assertIn("archive.page_issue_assignment", sql)
        self.assertIn("review_decision IS DISTINCT FROM 'accept'", sql)
        self.assertIn("reject_proposal_mutation", sql)

    def test_approved_spans_are_bound_back_to_the_reviewed_proposal(self):
        sql = Path("db/migrations/012_coherent_unit_provenance_guards.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("coherent_revision_proposal_selection_trigger", sql)
        self.assertIn("coherent_span_proposal_membership_trigger", sql)

    def test_one_review_cannot_be_activated_twice(self):
        sql = Path("db/migrations/013_segmentation_action_idempotency.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("UNIQUE (review_id)", sql)

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
