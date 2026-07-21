from __future__ import annotations

import unittest

from wic_history.ingestion_jobs import (
    AGGREGATE_STAGES,
    DEFAULT_CONFIGURATION,
    PAGE_STAGES,
    build_parser,
    canonical_sha256,
    normalize_stages,
    normalize_aggregate_stages,
    resolve_stage_configuration,
    validate_stage_result,
)


class IngestionJobTests(unittest.TestCase):
    def test_canonical_hash_is_order_independent_and_unicode_safe(self):
        left = canonical_sha256({"page": 308, "label": "申報"})
        right = canonical_sha256({"label": "申報", "page": 308})
        self.assertEqual(left, right)
        self.assertEqual(len(left), 64)

    def test_stage_order_and_dependencies_are_explicit(self):
        self.assertEqual(
            normalize_stages(["embedding", "render_lossless", "ocr", "layout"]),
            ("render_lossless", "layout", "ocr", "embedding"),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported page stages: ner"):
            normalize_stages(["ner"])
        with self.assertRaisesRegex(ValueError, "Unsupported page stages: entity_link"):
            normalize_stages(
                ["render_lossless", "layout", "ocr", "entity_link"]
            )

    def test_aggregate_stages_require_their_page_inputs(self):
        self.assertEqual(
            normalize_aggregate_stages(
                ["graph_projection", "search_projection", "rag_export"],
                PAGE_STAGES,
            ),
            AGGREGATE_STAGES,
        )
        with self.assertRaisesRegex(ValueError, "requires page stage embedding"):
            normalize_aggregate_stages(["search_projection"], ("ocr",))

    def test_cli_defaults_to_the_full_page_dag(self):
        args = build_parser().parse_args(
            ["plan", "--name", "pilot", "--created-by", "researcher"]
        )
        self.assertEqual(args.stages, ",".join(PAGE_STAGES))
        self.assertEqual(args.max_pages, 1000)
        self.assertEqual(args.aggregate_stages, "")

    def test_legacy_ner_stage_refuses_under_remote_semantic_provider(self):
        # The pinned semantic provider is a remote frontier model with no
        # verifiable local runtime, so the legacy in-ingestion NER stage has
        # no profile and planning it refuses explicitly (article-level
        # frontier extraction replaces it).
        self.assertIsNone(DEFAULT_CONFIGURATION["ner"])
        with self.assertRaisesRegex(ValueError, "local Ollama semantic"):
            resolve_stage_configuration("ner")
        with self.assertRaisesRegex(ValueError, "local Ollama semantic"):
            resolve_stage_configuration("ner", {"adapter": "rules+gliner"})

    def test_terminal_control_commands_require_explicit_scope(self):
        failures = build_parser().parse_args(
            [
                "failures",
                "--batch-id",
                "00000000-0000-0000-0000-000000000001",
            ]
        )
        self.assertEqual(failures.command, "failures")
        cancelled = build_parser().parse_args(
            [
                "cancel",
                "--batch-id",
                "00000000-0000-0000-0000-000000000001",
                "--cancelled-by",
                "operator",
                "--reason",
                "cost guard",
            ]
        )
        self.assertEqual(cancelled.cancelled_by, "operator")
        replay = build_parser().parse_args(
            [
                "replay",
                "--job-id",
                "00000000-0000-0000-0000-000000000001",
                "--requested-by",
                "operator",
                "--reason",
                "transient dependency fixed",
            ]
        )
        self.assertEqual(replay.command, "replay")

    def test_bounded_ner_result_must_match_the_plan(self):
        result = {
            "ner_run_id": "00000000-0000-0000-0000-000000000001",
            "mentions": 82,
            "regions_attempted": 50,
            "regions_succeeded": 48,
            "regions_abstained": 2,
            "invalid_outputs": 2,
            "candidate_only": True,
            "bounded_regions": 50,
        }
        validate_stage_result("ner", {"max_regions": 50}, result)
        with self.assertRaisesRegex(ValueError, "bounded_regions"):
            validate_stage_result("ner", {"max_regions": None}, result)

    def test_stage_results_require_typed_provenance(self):
        with self.assertRaisesRegex(ValueError, "ocr_run_id"):
            validate_stage_result("ocr", {}, {"regions": 10})
        with self.assertRaisesRegex(ValueError, "render_sha256"):
            validate_stage_result("render_lossless", {}, {})
        with self.assertRaisesRegex(ValueError, "source_object_sha256"):
            validate_stage_result(
                "render_lossless", {}, {"render_sha256": "a" * 64}
            )

    def test_entity_link_result_requires_full_nil_coverage_and_no_mutation(self):
        result = {
            "entity_link_run_id": "00000000-0000-0000-0000-000000000001",
            "source_ner_run_id": "00000000-0000-0000-0000-000000000002",
            "links": 7,
            "mentions": 3,
            "nil_links": 3,
            "candidate_only": True,
            "identity_mutations": 0,
            "authority_catalog_sha256": "a" * 64,
        }
        validate_stage_result("entity_link", {"top_k": 5}, result)
        for update, message in (
            ({"nil_links": 2}, "exactly one NIL"),
            ({"identity_mutations": 1}, "cannot mutate"),
            ({"links": 19}, "candidate count"),
            ({"candidate_only": False}, "candidate_only"),
        ):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, message):
                validate_stage_result(
                    "entity_link", {"top_k": 5}, {**result, **update}
                )

    def test_aggregate_results_have_projection_contracts(self):
        validate_stage_result(
            "search_projection",
            {},
            {
                "projection_build_id": "00000000-0000-0000-0000-000000000001",
                "documents_indexed": 10,
                "index_name": "wic-regions-batch-abc",
            },
        )
        validate_stage_result(
            "rag_export",
            {},
            {"documents": 1, "exported_regions": 10, "manifest_sha256": "a" * 64},
        )
        with self.assertRaisesRegex(ValueError, "reviewed_only"):
            validate_stage_result(
                "graph_projection",
                {},
                {
                    "projection_build_id": "00000000-0000-0000-0000-000000000001",
                    "entities": 0,
                    "claims": 0,
                    "mentions": 0,
                    "claim_evidence": 0,
                    "events": 0,
                    "event_participants": 0,
                    "event_evidence": 0,
                    "local_identity_clusters": 0,
                    "local_cluster_members": 0,
                    "reviewed_only": False,
                },
            )


if __name__ == "__main__":
    unittest.main()
