from __future__ import annotations

import unittest

from wic_history.ingestion_jobs import (
    PAGE_STAGES,
    build_parser,
    canonical_sha256,
    normalize_stages,
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
            normalize_stages(["ner", "render_lossless", "ocr"]),
            ("render_lossless", "ocr", "ner"),
        )
        with self.assertRaisesRegex(ValueError, "requires ocr"):
            normalize_stages(["ner"])

    def test_cli_defaults_to_the_full_page_dag(self):
        args = build_parser().parse_args(
            ["plan", "--name", "pilot", "--created-by", "researcher"]
        )
        self.assertEqual(args.stages, ",".join(PAGE_STAGES))
        self.assertEqual(args.max_pages, 1000)

    def test_bounded_ner_result_must_match_the_plan(self):
        result = {
            "ner_run_id": "00000000-0000-0000-0000-000000000001",
            "mentions": 82,
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


if __name__ == "__main__":
    unittest.main()
