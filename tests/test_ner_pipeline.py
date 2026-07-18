from __future__ import annotations

import json
import unittest
from pathlib import Path

from wic_history.evidence import EntityType
from wic_history.ner_pipeline import (
    MODEL_LABELS,
    RulePredictor,
    SpanCandidate,
    build_parser,
    merge_candidates,
)


class NERPipelineTests(unittest.TestCase):
    def test_cli_accepts_fixed_corpus_language(self):
        args = build_parser().parse_args(
            [
                "--ocr-artifact",
                "ocr.json",
                "--output",
                "ner.json",
                "--word-splitter-language",
                "zh-hant",
                "--max-regions",
                "50",
            ]
        )
        self.assertEqual(args.word_splitter_language, "zh-hant")
        self.assertEqual(args.max_regions, 50)

    def test_rules_return_exact_offsets(self):
        text = "上海女子學校女學生讀申報"
        spans = RulePredictor().predict([text])[0]
        for span in spans:
            self.assertEqual(text[span.start : span.end], span.text)
        self.assertTrue(any(span.entity_type == EntityType.SCHOOL for span in spans))
        self.assertTrue(any(span.text == "女學生" for span in spans))
        self.assertTrue(any(span.text == "申報" for span in spans))

    def test_merge_retains_highest_score_for_same_span_type(self):
        lower = SpanCandidate(0, 2, "申報", EntityType.PUBLICATION, 0.5, "one")
        higher = SpanCandidate(0, 2, "申報", EntityType.PUBLICATION, 0.9, "two")
        merged = merge_candidates([[lower]], [[higher]])
        self.assertEqual(merged[0][0].score, 0.9)
        self.assertEqual(merged[0][0].extractor, "two")
        self.assertEqual(merged[0][0].supports, (("one", 0.5), ("two", 0.9)))

    def test_model_labels_cover_the_full_project_ontology(self):
        self.assertEqual(set(MODEL_LABELS.values()), set(EntityType))

    def test_research_registry_pins_the_supervised_and_open_type_tournaments(self):
        root = Path(__file__).parents[1]
        registry = json.loads(
            (root / "experiments/ner/candidates.json").read_text(encoding="utf-8")
        )
        candidates = {item["id"]: item for item in registry["candidates"]}
        self.assertTrue(
            {
                "macbert-w2ner",
                "mmbert-w2ner",
                "guji-roberta-w2ner",
                "sikubert-w2ner",
                "otter-ce-mmbert",
                "gliner-x-large",
                "nuextract3",
                "qwen3.6-27b",
            }.issubset(candidates)
        )
        self.assertTrue(
            all(len(candidate["revision"]) == 40 for candidate in candidates.values())
        )
        for candidate in candidates.values():
            if candidate["license"] is None:
                self.assertIn("license-gated", candidate["role"])

    def test_benchmark_spec_has_no_claimed_results_and_covers_registry(self):
        root = Path(__file__).parents[1]
        registry = json.loads(
            (root / "experiments/ner/candidates.json").read_text(encoding="utf-8")
        )
        specification = json.loads(
            (root / "experiments/ner/benchmark-spec.json").read_text(encoding="utf-8")
        )
        represented = {
            arm["candidate_id"]
            for arm in specification["arms"]
            if arm["candidate_id"] is not None
        }
        self.assertEqual(
            represented,
            {candidate["id"] for candidate in registry["candidates"]},
        )
        self.assertEqual(specification["benchmark_results"], [])
        self.assertEqual(
            specification["frozen_metrics"]["primary"],
            "exact_span_and_type_micro_f1",
        )


if __name__ == "__main__":
    unittest.main()
