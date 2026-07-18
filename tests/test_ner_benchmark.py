from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from wic_history.evidence import EntityType, SourcePointer
from wic_history.ner_adapters.base import (
    AdapterIdentity,
    IssueSplitManifest,
    NERBenchmarkDataset,
    SnippetSplitAssignment,
)
from wic_history.ner_benchmark import (
    PredictorBenchmarkAdapter,
    execute_benchmark,
    main as benchmark_main,
    prepare_benchmark_dataset,
)
from wic_history.ner_benchmark_compare import (
    compare_score_reports,
    main as compare_main,
)
from wic_history.ner_gold import (
    GoldAdjudication,
    GoldEntitySpan,
    GoldSnippet,
    NERGoldSet,
    ReviewerAnnotation,
    main as score_main,
    score_ner_artifact,
)
from wic_history.ner_pipeline import RulePredictor, SpanCandidate


NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)
GOLD_SHA256 = "1" * 64
MANIFEST_SHA256 = "2" * 64
CODE_REVISION = "a" * 40


def _uuid(number: int) -> UUID:
    return UUID(int=number)


def _snippet(number: int, text: str = "上海女子學校女學生") -> GoldSnippet:
    entity = GoldEntitySpan(
        entity_type=EntityType.SCHOOL,
        corrected_start=0,
        corrected_end=6,
        corrected_text="上海女子學校",
        raw_start=0,
        raw_end=6,
        raw_text="上海女子學校",
    )
    annotation = {
        "corrected_text": text,
        "entities": [entity],
        "annotated_at": NOW,
    }
    return GoldSnippet(
        snippet_id=f"snippet-{number}",
        gold_region_id=_uuid(100 + number),
        source_ocr_run_id=_uuid(200 + number),
        source_ocr_region_id=_uuid(300 + number),
        source=SourcePointer(
            source_uri=f"s3://example/v{number}.pdf",
            volume_number=number,
            publication_year=1910 + (number * 10),
            page_number=1,
            region_id=_uuid(300 + number),
        ),
        raw_ocr_text=text,
        page_genre="news_editorial",
        layout="vertical",
        scan_quality="moderate",
        reviews=[
            ReviewerAnnotation(reviewer="reviewer-a", **annotation),
            ReviewerAnnotation(reviewer="reviewer-b", **annotation),
        ],
        adjudication=GoldAdjudication(
            adjudicator="adjudicator-c",
            corrected_text=text,
            entities=[entity],
            adjudicated_at=NOW,
        ),
    )


def _gold() -> NERGoldSet:
    return NERGoldSet(
        schema_version="1.1",
        dataset_id="gold-v1",
        created_at=NOW,
        ontology_version="women-history-zh-v1",
        snippets=[_snippet(1), _snippet(2), _snippet(3), _snippet(4)],
    )


def _manifest() -> IssueSplitManifest:
    return IssueSplitManifest(
        dataset_id="gold-v1",
        created_at=NOW,
        assigned_by="historian",
        assignments=[
            SnippetSplitAssignment(
                snippet_id="snippet-1", issue_id="issue-development", split="development"
            ),
            SnippetSplitAssignment(
                snippet_id="snippet-2", issue_id="issue-development", split="development"
            ),
            SnippetSplitAssignment(
                snippet_id="snippet-3", issue_id="issue-train", split="train"
            ),
            SnippetSplitAssignment(
                snippet_id="snippet-4", issue_id="issue-test", split="test"
            ),
        ],
    )


def _identity(**updates: object) -> AdapterIdentity:
    values = {
        "adapter_id": "rules-v1",
        "family": "rules",
        "model_name": "historical-women-rules",
        "model_revision": "rules-v1",
        "license": "project-code",
        "modalities": ["text"],
        "runtime": "python-re",
        "code_revision": CODE_REVISION,
        "device": "cpu",
        "dtype": "deterministic",
        "ontology_version": "women-history-zh-v1",
    }
    values.update(updates)
    return AdapterIdentity.model_validate(values)


class InvalidPredictor:
    def predict(self, texts: list[str], threshold: float):
        return [
            [
                SpanCandidate(0, 2, "錯字", EntityType.PERSON, 0.9, "invalid"),
                SpanCandidate(0, len(text) + 1, text, EntityType.PLACE, 0.8, "invalid"),
            ]
            for text in texts
        ]


class NERBenchmarkTests(unittest.TestCase):
    def _dataset(self):
        return prepare_benchmark_dataset(
            _gold(),
            GOLD_SHA256,
            _manifest(),
            MANIFEST_SHA256,
            dataset_id="benchmark-v1",
            input_variants=["raw_ocr", "corrected_text"],
            generated_at=NOW,
        )

    def test_issue_split_manifest_rejects_cross_split_leakage(self):
        data = _manifest().model_dump(mode="json")
        data["assignments"][3]["issue_id"] = "issue-development"
        with self.assertRaisesRegex(ValueError, "cannot cross benchmark splits"):
            IssueSplitManifest.model_validate(data)

    def test_prepare_requires_exact_manifest_coverage_and_marks_small_set_ineligible(self):
        manifest = _manifest().model_copy(
            update={"assignments": _manifest().assignments[:-1]}
        )
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            prepare_benchmark_dataset(
                _gold(),
                GOLD_SHA256,
                manifest,
                MANIFEST_SHA256,
                dataset_id="benchmark-v1",
                input_variants=["raw_ocr"],
            )

        dataset = self._dataset()
        self.assertFalse(dataset.benchmark_eligible)
        self.assertEqual(len(dataset.inputs), 8)
        self.assertTrue(any("at least 500" in failure for failure in dataset.eligibility_failures))

        tampered = dataset.model_dump(mode="json")
        tampered["benchmark_eligible"] = True
        tampered["eligibility_failures"] = []
        with self.assertRaisesRegex(ValueError, "failures disagree"):
            NERBenchmarkDataset.model_validate(tampered)

    def test_ineligible_data_is_blocked_unless_run_is_explicitly_technical(self):
        adapter = PredictorBenchmarkAdapter(
            _identity(), [RulePredictor()], threshold=0.0, batch_size=2
        )
        with self.assertRaisesRegex(ValueError, "ineligible"):
            execute_benchmark(
                self._dataset(),
                adapter,
                split="development",
                input_variant="raw_ocr",
            )

    def test_multi_run_artifact_scores_only_its_frozen_split_inputs(self):
        dataset = self._dataset()
        adapter = PredictorBenchmarkAdapter(
            _identity(), [RulePredictor()], threshold=0.0, batch_size=2
        )
        artifact = execute_benchmark(
            dataset,
            adapter,
            split="development",
            input_variant="raw_ocr",
            allow_ineligible_technical_run=True,
        )

        self.assertEqual(len(artifact.source_ocr_run_ids), 2)
        self.assertEqual(len(artifact.results), 2)
        self.assertTrue(all(result.input_text_sha256 for result in artifact.results))
        self.assertTrue(all(mention.attributes["candidate_only"] for mention in artifact.mentions))
        report = score_ner_artifact(
            _gold(),
            artifact,
            "raw_ocr",
            gold_sha256=GOLD_SHA256,
            benchmark_dataset=dataset,
        )
        self.assertEqual(report["snippets"], 2)
        self.assertEqual(report["dataset_split"], "development")
        self.assertTrue(report["source_gold_sha256_verified"])
        self.assertTrue(report["benchmark_dataset_sha256_verified"])
        self.assertEqual(report["exact"]["true_positive"], 2)
        self.assertEqual(set(report["by_issue"]), {"issue-development"})

        comparison = compare_score_reports(
            report, report, bootstrap_samples=1_000, seed=7
        )
        self.assertEqual(comparison["exact_f1_delta"], 0.0)
        self.assertEqual(
            comparison["paired_issue_cluster_bootstrap"]["lower_95"], 0.0
        )

        mismatched = dict(report)
        mismatched["input_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "input_sha256"):
            compare_score_reports(report, mismatched, bootstrap_samples=1_000)

    def test_invalid_adapter_spans_are_counted_but_never_serialized_as_mentions(self):
        adapter = PredictorBenchmarkAdapter(
            _identity(), [InvalidPredictor()], threshold=0.0, batch_size=2
        )
        artifact = execute_benchmark(
            self._dataset(),
            adapter,
            split="test",
            input_variant="raw_ocr",
            allow_ineligible_technical_run=True,
        )
        self.assertEqual(len(artifact.mentions), 0)
        self.assertEqual(artifact.results[0].invalid_outputs, 2)
        self.assertEqual(artifact.run.configuration["invalid_outputs"], 2)

    def test_adapter_identity_rejects_moving_or_incomplete_provenance(self):
        with self.assertRaisesRegex(ValueError, "moving label"):
            _identity(model_revision="refs/heads/main")
        with self.assertRaisesRegex(ValueError, "head-code"):
            _identity(
                family="w2ner",
                model_revision="b" * 40,
            )
        with self.assertRaisesRegex(ValueError, "prompt/schema"):
            _identity(
                family="structured_generation",
                model_revision="b" * 40,
            )

    def test_cli_keeps_test_split_locked_without_explicit_confirmation(self):
        with self.assertRaisesRegex(SystemExit, "test split is locked"):
            benchmark_main(
                [
                    "run",
                    "--dataset",
                    "does-not-need-to-exist.json",
                    "--output",
                    "unused.json",
                    "--split",
                    "test",
                    "--input-variant",
                    "raw_ocr",
                    "--adapter",
                    "rules",
                    "--code-revision",
                    CODE_REVISION,
                ]
            )

    def test_scoring_rejects_gold_file_or_input_drift(self):
        adapter = PredictorBenchmarkAdapter(
            _identity(), [RulePredictor()], threshold=0.0, batch_size=2
        )
        artifact = execute_benchmark(
            self._dataset(),
            adapter,
            split="test",
            input_variant="raw_ocr",
            allow_ineligible_technical_run=True,
        )
        with self.assertRaisesRegex(ValueError, "file hash"):
            score_ner_artifact(_gold(), artifact, "raw_ocr", gold_sha256="f" * 64)

        changed_gold = _gold().model_dump(mode="json")
        snippet = changed_gold["snippets"][3]
        snippet["raw_ocr_text"] += "改"
        changed = NERGoldSet.model_validate(changed_gold)
        with self.assertRaisesRegex(ValueError, "input text hash"):
            score_ner_artifact(changed, artifact, "raw_ocr")

    def test_prepare_run_score_compare_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.json"
            manifest_path = root / "splits.json"
            dataset_path = root / "dataset.json"
            predictions_path = root / "predictions.json"
            score_path = root / "score.json"
            comparison_path = root / "comparison.json"
            gold_path.write_text(_gold().model_dump_json(indent=2), encoding="utf-8")
            manifest_path.write_text(
                _manifest().model_dump_json(indent=2), encoding="utf-8"
            )

            self.assertEqual(
                benchmark_main(
                    [
                        "prepare",
                        "--gold",
                        str(gold_path),
                        "--split-manifest",
                        str(manifest_path),
                        "--dataset-id",
                        "benchmark-v1",
                        "--input-variant",
                        "raw_ocr",
                        "--input-variant",
                        "corrected_text",
                        "--output",
                        str(dataset_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                benchmark_main(
                    [
                        "run",
                        "--dataset",
                        str(dataset_path),
                        "--output",
                        str(predictions_path),
                        "--split",
                        "development",
                        "--input-variant",
                        "raw_ocr",
                        "--adapter",
                        "rules",
                        "--code-revision",
                        CODE_REVISION,
                        "--allow-ineligible-technical-run",
                    ]
                ),
                0,
            )
            self.assertEqual(
                score_main(
                    [
                        "--gold",
                        str(gold_path),
                        "--predictions",
                        str(predictions_path),
                        "--benchmark-dataset",
                        str(dataset_path),
                        "--input-text",
                        "raw_ocr",
                        "--output",
                        str(score_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                compare_main(
                    [
                        "--baseline-score",
                        str(score_path),
                        "--challenger-score",
                        str(score_path),
                        "--bootstrap-samples",
                        "1000",
                        "--output",
                        str(comparison_path),
                    ]
                ),
                0,
            )
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            self.assertEqual(comparison["exact_f1_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
