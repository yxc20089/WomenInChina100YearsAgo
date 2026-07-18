from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from wic_history.evidence import (
    Point,
    Polygon,
    RetrievalHit,
    ScenarioContextBundle,
    ScenarioEvidenceItem,
    SourcePointer,
)
from wic_history.evaluation import QuestionCategory
from wic_history.generation import GenerationStatus, GenerationTask
from wic_history.generation_benchmark import (
    GenerationBenchmarkCase,
    GenerationBenchmarkDataset,
    HumanGenerationAdjudication,
    HumanGenerationGradeSet,
    HumanGenerationReport,
    HumanGenerationReview,
    blinded_generation_packet_sha256,
    build_blinded_generation_packet,
    build_generation_benchmark_dataset,
    compare_human_generation_reports,
    execute_generation_benchmark,
    score_generation_benchmark,
    score_human_generation_grades,
)


NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)
REGION_ID = UUID("00000000-0000-0000-0000-000000000001")
SOURCE = SourcePointer(
    source_uri="s3://example/volume.pdf",
    source_sha256="1" * 64,
    derivative_id="00000000-0000-0000-0000-000000000002",
    image_sha256="2" * 64,
    evidence_tier="historian_selected_gold",
    volume_number=219,
    publication_year=1925,
    page_number=308,
    region_id=REGION_ID,
    polygon=Polygon(
        points=[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10)]
    ),
)


class PinnedFakeGenerator:
    model_identity = "fixture-model@fixture-revision-2026-07-18"
    model_revision = "fixture-revision-2026-07-18"
    provider_kind = "controlled_fixture"
    generation_configuration_sha256 = "a" * 64

    def complete(self, messages):
        return (
            "The cited OCR lead is relevant and remains subject to review "
            "[region:00000000-0000-0000-0000-000000000001]."
        )


def context(reviewed: bool = False) -> ScenarioContextBundle:
    evidence_items = []
    if reviewed:
        evidence_items = [
            ScenarioEvidenceItem(
                statement="王女士 — attended_school — 女塾",
                epistemic_label="directly_evidenced",
                sources=[SOURCE],
                claim_ids=["00000000-0000-0000-0000-000000000003"],
            )
        ]
    return ScenarioContextBundle(
        research_query="女學生",
        evidence_items=evidence_items,
        retrieved_context=[
            RetrievalHit(
                rank=1,
                score=1,
                source=SOURCE,
                text="女學生",
                explanation={"retriever": "fixture"},
            )
        ],
    )


def benchmark_case() -> GenerationBenchmarkCase:
    return GenerationBenchmarkCase(
        case_id="case-001-secret-to-graders",
        category="exact_lookup",
        task=GenerationTask.RESEARCH_BRIEF,
        context=context(),
        answerable=True,
        expected_support_region_ids=[REGION_ID],
        author="technical-smoke",
    )


class GenerationBenchmarkTests(unittest.TestCase):
    def test_machine_readable_spec_claims_no_results(self):
        root = Path(__file__).parents[1]
        specification = json.loads(
            (root / "experiments/generation/benchmark-spec.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(specification["benchmark_results"], [])
        self.assertEqual(specification["decision_status"], "protocol_frozen_no_live_model_no_winner")

    def dataset(self) -> GenerationBenchmarkDataset:
        return build_generation_benchmark_dataset(
            "generation-fixture-v1",
            "fixture-builder",
            [benchmark_case()],
            generated_at=NOW,
        )

    def test_case_contract_rejects_answerability_or_scene_evidence_drift(self):
        data = benchmark_case().model_dump(mode="json")
        data["category"] = "unanswerable"
        with self.assertRaisesRegex(ValueError, "answerability"):
            GenerationBenchmarkCase.model_validate(data)

        data = benchmark_case().model_dump(mode="json")
        data["task"] = "reconstructed_scene"
        with self.assertRaisesRegex(ValueError, "allowed context"):
            GenerationBenchmarkCase.model_validate(data)

    def test_eligibility_is_derived_and_cannot_be_flipped(self):
        dataset = self.dataset()
        self.assertFalse(dataset.benchmark_eligible)
        self.assertTrue(any("at least 30" in failure for failure in dataset.eligibility_failures))
        tampered = dataset.model_dump(mode="json")
        tampered["benchmark_eligible"] = True
        tampered["eligibility_failures"] = []
        with self.assertRaisesRegex(ValueError, "eligibility disagrees"):
            GenerationBenchmarkDataset.model_validate(tampered)

    def test_representative_historian_authored_case_mix_can_become_eligible(self):
        answerable_categories = [
            category
            for category in QuestionCategory
            if category != QuestionCategory.UNANSWERABLE
        ]
        cases = []
        for index in range(30):
            unanswerable = index < 5
            task = list(GenerationTask)[index % len(GenerationTask)]
            category = (
                QuestionCategory.UNANSWERABLE
                if unanswerable
                else answerable_categories[(index - 5) % len(answerable_categories)]
            )
            case_context = context(
                reviewed=task == GenerationTask.RECONSTRUCTED_SCENE
            ).model_copy(update={"research_query": f"女學生 {index}"})
            cases.append(
                GenerationBenchmarkCase(
                    case_id=f"historian-case-{index:03d}",
                    category=category,
                    task=task,
                    context=case_context,
                    answerable=not unanswerable,
                    expected_support_region_ids=[] if unanswerable else [REGION_ID],
                    author="historian-a" if index % 2 else "historian-b",
                )
            )
        dataset = build_generation_benchmark_dataset(
            "eligible-v1", "historian-team", cases, generated_at=NOW
        )
        self.assertTrue(dataset.benchmark_eligible)
        self.assertEqual(dataset.eligibility_failures, [])

        duplicate_data = cases[-1].model_dump(mode="json")
        duplicate_data["case_id"] = "duplicate-input"
        duplicate = GenerationBenchmarkCase.model_validate(duplicate_data)
        duplicated_dataset = build_generation_benchmark_dataset(
            "duplicate-v1", "historian-team", [*cases, duplicate], generated_at=NOW
        )
        self.assertFalse(duplicated_dataset.benchmark_eligible)
        self.assertTrue(
            any("duplicate task/context/history" in item for item in duplicated_dataset.eligibility_failures)
        )

    def test_execution_requires_eligible_data_or_an_explicit_technical_flag(self):
        with self.assertRaisesRegex(ValueError, "ineligible"):
            execute_generation_benchmark(
                self.dataset(), PinnedFakeGenerator(), code_revision="b" * 40
            )

        class MovingGenerator(PinnedFakeGenerator):
            model_revision = "latest"

        with self.assertRaisesRegex(ValueError, "immutable"):
            execute_generation_benchmark(
                self.dataset(),
                MovingGenerator(),
                code_revision="b" * 40,
                allow_ineligible_technical_run=True,
            )

    def test_execution_scoring_and_blinding_preserve_provenance_without_model_leak(self):
        dataset = self.dataset()
        artifact = execute_generation_benchmark(
            dataset,
            PinnedFakeGenerator(),
            code_revision="b" * 40,
            allow_ineligible_technical_run=True,
        )
        report = score_generation_benchmark(dataset, artifact)
        packet = build_blinded_generation_packet(dataset, artifact)

        self.assertEqual(artifact.results[0].response.status, GenerationStatus.COMPLETED)
        self.assertEqual(report.answerable_completion_rate, 1.0)
        self.assertEqual(report.micro_expected_support_recall, 1.0)
        self.assertEqual(report.micro_citation_precision_against_expected, 1.0)
        self.assertEqual(report.token_usage_complete_rate, 0.0)
        self.assertIsNone(report.estimated_cost_usd)
        serialized_packet = packet.model_dump_json()
        self.assertNotIn("fixture-model", serialized_packet)
        self.assertNotIn("case-001-secret-to-graders", serialized_packet)
        self.assertNotIn('"answerable"', serialized_packet)
        self.assertNotIn('"category"', serialized_packet)

    def test_two_reviewer_adjudication_contract_and_human_report(self):
        dataset = self.dataset()
        artifact = execute_generation_benchmark(
            dataset,
            PinnedFakeGenerator(),
            code_revision="b" * 40,
            allow_ineligible_technical_run=True,
        )
        packet = build_blinded_generation_packet(dataset, artifact)
        blind_id = packet.cases[0].blind_id

        def review(reviewer: str, decision: str, score: int):
            return HumanGenerationReview(
                blind_id=blind_id,
                reviewer=reviewer,
                reviewed_at=NOW,
                evidence_entailment=score,
                completeness=score,
                epistemic_separation=score,
                historical_safety=score,
                researcher_usefulness=score,
                unsupported_claim_count=0,
                decision=decision,
                notes="Inspected the cited context independently.",
            )

        grades = HumanGenerationGradeSet(
            dataset_sha256=packet.dataset_sha256,
            prediction_artifact_sha256=packet.prediction_artifact_sha256,
            blinded_packet_sha256=blinded_generation_packet_sha256(packet),
            reviews=[review("reviewer-a", "pass", 5), review("reviewer-b", "fail", 3)],
            adjudications=[
                HumanGenerationAdjudication(
                    blind_id=blind_id,
                    adjudicator="adjudicator-c",
                    adjudicated_at=NOW,
                    decision="pass",
                    evidence_entailment=4,
                    completeness=4,
                    epistemic_separation=4,
                    historical_safety=4,
                    researcher_usefulness=4,
                    unsupported_claim_count=0,
                    notes="Resolved the disagreement against the cited scan context.",
                )
            ],
        )
        report = score_human_generation_grades(packet, grades)
        self.assertEqual(report.pass_rate, 1.0)
        self.assertEqual(report.review_decision_agreement_rate, 0.0)
        self.assertEqual(report.mean_rating_absolute_difference, 2.0)
        self.assertEqual(report.case_results[0].pairing_id, packet.cases[0].pairing_id)

        changed = grades.model_dump(mode="json")
        changed["blinded_packet_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "packet hash"):
            score_human_generation_grades(
                packet, HumanGenerationGradeSet.model_validate(changed)
            )

        stronger_data = report.model_dump(mode="json")
        stronger_data["prediction_artifact_sha256"] = "e" * 64
        stronger_data["mean_evidence_entailment"] = 5
        stronger_data["mean_completeness"] = 5
        stronger_data["mean_epistemic_separation"] = 5
        stronger_data["mean_historical_safety"] = 5
        stronger_data["mean_researcher_usefulness"] = 5
        for field in (
            "evidence_entailment",
            "completeness",
            "epistemic_separation",
            "historical_safety",
            "researcher_usefulness",
        ):
            stronger_data["case_results"][0][field] = 5
        stronger = HumanGenerationReport.model_validate(stronger_data)
        comparison = compare_human_generation_reports(
            report,
            stronger,
            label_a="candidate-a",
            label_b="candidate-b",
            bootstrap_resamples=100,
        )
        self.assertEqual(comparison.mean_quality_difference_b_minus_a, 1.0)
        self.assertEqual(comparison.mean_quality_difference_ci95, (1.0, 1.0))
        self.assertEqual(comparison.b_wins, 1)

        mismatched = stronger.model_dump(mode="json")
        mismatched["case_results"][0]["pairing_id"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "identical generation inputs"):
            compare_human_generation_reports(
                report,
                HumanGenerationReport.model_validate(mismatched),
                label_a="candidate-a",
                label_b="candidate-b",
                bootstrap_resamples=100,
            )

    def test_grade_set_rejects_nonindependent_review_or_adjudication(self):
        dataset = self.dataset()
        artifact = execute_generation_benchmark(
            dataset,
            PinnedFakeGenerator(),
            code_revision="b" * 40,
            allow_ineligible_technical_run=True,
        )
        packet = build_blinded_generation_packet(dataset, artifact)
        blind_id = packet.cases[0].blind_id
        review = {
            "blind_id": blind_id,
            "reviewed_at": NOW,
            "evidence_entailment": 4,
            "completeness": 4,
            "epistemic_separation": 4,
            "historical_safety": 4,
            "researcher_usefulness": 4,
            "unsupported_claim_count": 0,
            "decision": "pass",
            "notes": "Reviewed.",
        }
        adjudication = {
            "blind_id": blind_id,
            "adjudicator": "same-person",
            "adjudicated_at": NOW,
            "decision": "pass",
            "evidence_entailment": 4,
            "completeness": 4,
            "epistemic_separation": 4,
            "historical_safety": 4,
            "researcher_usefulness": 4,
            "unsupported_claim_count": 0,
            "notes": "Adjudicated.",
        }
        with self.assertRaisesRegex(ValueError, "two distinct reviewers"):
            HumanGenerationGradeSet.model_validate(
                {
                    "dataset_sha256": packet.dataset_sha256,
                    "prediction_artifact_sha256": packet.prediction_artifact_sha256,
                    "blinded_packet_sha256": blinded_generation_packet_sha256(packet),
                    "reviews": [
                        {**review, "reviewer": "same-person"},
                        {**review, "reviewer": "same-person"},
                    ],
                    "adjudications": [adjudication],
                }
            )


if __name__ == "__main__":
    unittest.main()
