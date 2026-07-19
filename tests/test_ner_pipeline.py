from __future__ import annotations

import json
import hashlib
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
from wic_history.ner_structured import (
    STRUCTURED_NER_PROMPT_SCHEMA_SHA256,
    STRUCTURED_NER_RESPONSE_FORMAT_SHA256,
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
                "chinese-modernbert-large-wwm-w2ner",
                "guji-roberta-w2ner",
                "sikubert-w2ner",
                "otter-ce-mmbert",
                "gliner-x-large",
                "gliner-x-large-v0.5",
                "nuextract3",
                "qwen3.5-0.8b-structured-ner",
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
        modernbert = next(
            arm
            for arm in specification["arms"]
            if arm["id"] == "chinese-modernbert-large-wwm-w2ner"
        )
        self.assertIn("tokenizer_raw_offset_round_trip_test", modernbert["blockers"])
        self.assertEqual(
            specification["training_augmentation_protocol"]["external_evidence"][
                "implementation_policy"
            ],
            "do_not_copy_unlicensed_code_clean_room_implementation_only",
        )
        self.assertEqual(
            specification["training_augmentation_protocol"]["status"],
            "exporter_implemented_eligible_training_run_pending",
        )
        head_selection = specification["supervised_head_selection"]
        self.assertEqual(
            head_selection["status"], "no_winner_controlled_comparison_required"
        )
        self.assertIsNone(head_selection["selection"])
        self.assertEqual(
            {item["id"] for item in head_selection["candidate_heads"]},
            {"w2ner", "global_pointer", "bio_crf"},
        )

    def test_qwen_and_gliner_local_challengers_are_content_addressed(self):
        root = Path(__file__).parents[1]
        registry = json.loads(
            (root / "experiments/ner/candidates.json").read_text(encoding="utf-8")
        )
        specification = json.loads(
            (root / "experiments/ner/benchmark-spec.json").read_text(encoding="utf-8")
        )
        candidates = {item["id"]: item for item in registry["candidates"]}
        arms = {item["id"]: item for item in specification["arms"]}

        qwen = candidates["qwen3.5-0.8b-structured-ner"]
        qwen_arm = arms["qwen3.5-0.8b-structured-ner"]
        ollama = qwen["local_runtime_conditions"][0]
        self.assertEqual(ollama["id"], "ollama-q8-canonical")
        self.assertEqual(
            qwen_arm["canonical_runtime"]["required_manifest_digest"],
            ollama["manifest_digest"],
        )
        self.assertEqual(qwen_arm["decoding_contract"]["temperature"], 0)
        self.assertFalse(qwen_arm["decoding_contract"]["thinking"])
        self.assertEqual(
            qwen_arm["status"],
            "selected_first_pass_build_default_live_canary_passed_target_quality_unmeasured",
        )
        self.assertEqual(
            qwen_arm["implementation"]["prompt_schema_sha256"],
            STRUCTURED_NER_PROMPT_SCHEMA_SHA256,
        )
        self.assertEqual(
            qwen_arm["implementation"]["response_format_sha256"],
            STRUCTURED_NER_RESPONSE_FORMAT_SHA256,
        )
        self.assertNotIn(
            "backend_neutral_openai_compatible_adapter", qwen_arm["blockers"]
        )

        gliner = candidates["gliner-x-large"]
        controls = {item["model"]: item for item in gliner["family_size_controls"]}
        self.assertEqual(
            controls["knowledgator/gliner-x-large"]["revision"], gliner["revision"]
        )
        self.assertGreater(
            controls["knowledgator/gliner-x-large"]["official_zh_pud_f1"],
            controls["knowledgator/gliner-x-base"]["official_zh_pud_f1"],
        )
        self.assertGreater(
            candidates["gliner-x-large-v0.5"]["official_zh_pud_f1"],
            controls["knowledgator/gliner-x-large"]["official_zh_pud_f1"],
        )

    def test_mmbert_registry_pins_the_passing_offset_qualification(self):
        root = Path(__file__).parents[1]
        registry = json.loads(
            (root / "experiments/ner/candidates.json").read_text(encoding="utf-8")
        )
        specification = json.loads(
            (root / "experiments/ner/benchmark-spec.json").read_text(encoding="utf-8")
        )
        candidate = next(
            item for item in registry["candidates"] if item["id"] == "mmbert-w2ner"
        )
        arm = next(
            item for item in specification["arms"] if item["id"] == "mmbert-w2ner"
        )
        qualification = candidate["tokenizer_qualification"]
        artifact_path = root / qualification["path"]
        artifact_bytes = artifact_path.read_bytes()
        artifact = json.loads(artifact_bytes)

        self.assertEqual(
            hashlib.sha256(artifact_bytes).hexdigest(), qualification["sha256"]
        )
        self.assertEqual(arm["tokenizer_qualification"], qualification)
        self.assertTrue(artifact["passed"])
        self.assertEqual(artifact["code_revision"], qualification["code_revision"])
        self.assertEqual(
            artifact["tokenizer_file_manifest_sha256"],
            qualification["tokenizer_file_manifest_sha256"],
        )
        self.assertNotIn("tokenizer_offset_round_trip_test", arm["blockers"])


if __name__ == "__main__":
    unittest.main()
