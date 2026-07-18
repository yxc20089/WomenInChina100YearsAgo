from __future__ import annotations

import hashlib
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from wic_history.evidence import EntityType, Point, Polygon, SourcePointer
from wic_history.ner_gold import (
    GoldAdjudication,
    GoldEntitySpan,
    GoldSnippet,
    NERGoldSet,
    ReviewerAnnotation,
)
from wic_history.relation_benchmark import (
    RelationAdjudication,
    RelationBenchmarkDataset,
    RelationBenchmarkUnit,
    RelationGoldEdge,
    RelationGoldMention,
    RelationNERTextMapping,
    RelationPredicateDefinition,
    RelationPredictionArtifact,
    RelationReviewerAnnotation,
    build_relation_benchmark_dataset,
    compare_relation_reports,
    execute_relation_rule_baseline,
    score_relation_benchmark,
    verify_relation_dataset_ner_gold,
)


NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)


PREDICATES = [
    RelationPredicateDefinition(
        predicate="attended_school",
        label_zh="就讀學校",
        description="A person explicitly attended or graduated from a school.",
        subject_entity_types=[EntityType.PERSON],
        object_entity_types=[EntityType.SCHOOL],
    ),
    RelationPredicateDefinition(
        predicate="affiliated_with",
        label_zh="任職機構",
        description="A person is explicitly affiliated with an organization.",
        subject_entity_types=[EntityType.PERSON],
        object_entity_types=[EntityType.ORGANIZATION],
    ),
    RelationPredicateDefinition(
        predicate="resided_in",
        label_zh="居住地",
        description="A person is explicitly described as residing in a place.",
        subject_entity_types=[EntityType.PERSON],
        object_entity_types=[EntityType.PLACE],
    ),
]


def source_pointer(year: int = 1925) -> SourcePointer:
    return SourcePointer(
        source_uri="s3://example/volume.pdf",
        source_sha256="1" * 64,
        derivative_id=uuid4(),
        image_sha256="2" * 64,
        evidence_tier="historian_selected_gold",
        volume_number=219,
        publication_year=year,
        page_number=308,
        region_id=uuid4(),
        polygon=Polygon(
            points=[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10)]
        ),
    )


def relation_unit(
    index: int = 0,
    *,
    predicate: str = "attended_school",
    positive: bool = True,
    issue_id: str = "issue-technical",
    split: str = "test",
    selected_by: str = "technical-smoke",
    year: int = 1925,
) -> RelationBenchmarkUnit:
    if predicate == "attended_school":
        subject_text, cue, object_text, object_type = (
            f"宋{index}女士",
            "入學",
            f"上海{index}女子學校",
            EntityType.SCHOOL,
        )
    elif predicate == "affiliated_with":
        subject_text, cue, object_text, object_type = (
            f"宋{index}女士",
            "任職",
            f"上海{index}婦女會",
            EntityType.ORGANIZATION,
        )
    else:
        subject_text, cue, object_text, object_type = (
            f"宋{index}女士",
            "居於",
            f"上海{index}",
            EntityType.PLACE,
        )
    connector = cue if positive else "與"
    text = f"{subject_text}{connector}{object_text}記事{index}"
    subject_start = text.index(subject_text)
    object_start = text.index(object_text)
    subject_id, object_id = uuid4(), uuid4()
    mentions = [
        RelationGoldMention(
            mention_id=subject_id,
            entity_type=EntityType.PERSON,
            corrected_start=subject_start,
            corrected_end=subject_start + len(subject_text),
            corrected_text=subject_text,
            raw_start=subject_start,
            raw_end=subject_start + len(subject_text),
            raw_text=subject_text,
        ),
        RelationGoldMention(
            mention_id=object_id,
            entity_type=object_type,
            corrected_start=object_start,
            corrected_end=object_start + len(object_text),
            corrected_text=object_text,
            raw_start=object_start,
            raw_end=object_start + len(object_text),
            raw_text=object_text,
        ),
    ]
    relations = []
    if positive:
        relations = [
            RelationGoldEdge(
                subject_mention_id=subject_id,
                predicate=predicate,
                object_mention_id=object_id,
                corrected_evidence_start=0,
                corrected_evidence_end=object_start + len(object_text),
                corrected_evidence_text=text[: object_start + len(object_text)],
                raw_evidence_start=0,
                raw_evidence_end=object_start + len(object_text),
                raw_evidence_text=text[: object_start + len(object_text)],
            )
        ]
    pointer = source_pointer(year)
    return RelationBenchmarkUnit(
        unit_id=f"unit-{issue_id}-{index}",
        gold_unit_id=uuid4(),
        coherent_unit_revision_id=uuid4(),
        source_ner_mappings=[
            RelationNERTextMapping(
                source_ner_snippet_id=f"ner-{issue_id}-{index}",
                source_region_id=pointer.region_id,
                unit_corrected_start=0,
                unit_corrected_end=len(text),
                snippet_corrected_start=0,
                snippet_corrected_end=len(text),
                unit_raw_start=0,
                unit_raw_end=len(text),
                snippet_raw_start=0,
                snippet_raw_end=len(text),
            )
        ],
        issue_id=issue_id,
        split=split,
        selected_by=selected_by,
        source_regions=[pointer],
        corrected_text=text,
        raw_ocr_text=text,
        mentions=mentions,
        page_genre="news_editorial",
        layout="vertical",
        scan_quality="moderate",
        reviews=[
            RelationReviewerAnnotation(
                reviewer="historian-reviewer-a",
                annotated_at=NOW,
                relations=relations,
                notes="Reviewed the relation independently.",
            ),
            RelationReviewerAnnotation(
                reviewer="historian-reviewer-b",
                annotated_at=NOW,
                relations=relations,
                notes="Reviewed the relation independently.",
            ),
        ],
        adjudication=RelationAdjudication(
            adjudicator="historian-adjudicator-c",
            adjudicated_at=NOW,
            relations=relations,
            notes="Adjudicated against the exact source span.",
        ),
    )


def ner_gold_for_unit(unit: RelationBenchmarkUnit) -> tuple[NERGoldSet, bytes]:
    entities = [
        GoldEntitySpan(
            entity_type=mention.entity_type,
            corrected_start=mention.corrected_start,
            corrected_end=mention.corrected_end,
            corrected_text=mention.corrected_text,
            raw_start=mention.raw_start,
            raw_end=mention.raw_end,
            raw_text=mention.raw_text,
        )
        for mention in unit.mentions
    ]
    reviews = [
        ReviewerAnnotation(
            reviewer="ner-reviewer-a",
            corrected_text=unit.corrected_text,
            entities=entities,
            annotated_at=NOW,
        ),
        ReviewerAnnotation(
            reviewer="ner-reviewer-b",
            corrected_text=unit.corrected_text,
            entities=entities,
            annotated_at=NOW,
        ),
    ]
    gold = NERGoldSet(
        dataset_id="ner-gold-fixture-v1",
        created_at=NOW,
        ontology_version="women-history-entities-v1",
        snippets=[
            GoldSnippet(
                snippet_id=unit.source_ner_mappings[0].source_ner_snippet_id,
                source=unit.source_regions[0],
                raw_ocr_text=unit.raw_ocr_text,
                page_genre=unit.page_genre,
                layout=unit.layout,
                scan_quality=unit.scan_quality,
                reviews=reviews,
                adjudication=GoldAdjudication(
                    adjudicator="ner-adjudicator-c",
                    corrected_text=unit.corrected_text,
                    entities=entities,
                    adjudicated_at=NOW,
                ),
            )
        ],
    )
    data = gold.model_dump_json().encode("utf-8")
    return gold, data


def dataset_for_unit(unit: RelationBenchmarkUnit) -> tuple[RelationBenchmarkDataset, bytes]:
    gold, gold_bytes = ner_gold_for_unit(unit)
    dataset = build_relation_benchmark_dataset(
        "relation-fixture-v1",
        "women-history-relations-v1",
        gold.dataset_id,
        hashlib.sha256(gold_bytes).hexdigest(),
        PREDICATES,
        [unit],
        created_at=NOW,
    )
    return dataset, gold_bytes


class RelationBenchmarkTests(unittest.TestCase):
    def test_machine_spec_claims_no_results_or_winner(self):
        root = Path(__file__).parents[1]
        specification = json.loads(
            (root / "experiments/relation/benchmark-spec.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(specification["benchmark_results"], [])
        self.assertEqual(
            specification["decision_status"],
            "protocol_frozen_no_historian_gold_no_winner",
        )

    def test_rule_execution_and_scoring_preserve_exact_evidence(self):
        dataset, gold_bytes = dataset_for_unit(relation_unit())
        self.assertFalse(dataset.benchmark_eligible)
        verify_relation_dataset_ner_gold(dataset, gold_bytes)
        artifact = execute_relation_rule_baseline(
            dataset,
            split="test",
            input_variant="raw_ocr",
            code_revision="a" * 40,
            source_ner_gold_bytes=gold_bytes,
            allow_ineligible_technical_run=True,
        )
        report = score_relation_benchmark(dataset, artifact)
        self.assertEqual(report["exact_relation"]["f1"], 1.0)
        self.assertEqual(report["exact_relation_and_evidence"]["f1"], 1.0)
        self.assertEqual(report["invalid_evidence_predictions"], 0)
        self.assertEqual(report["raw_relation_recoverability"], 1.0)
        self.assertTrue(report["source_ner_gold_verified"])
        self.assertIsNone(report["token_usage_complete_rate"])

        weaker_data = artifact.model_dump(mode="json")
        weaker_data["artifact_id"] = str(uuid4())
        weaker_data["results"][0]["relations"] = []
        weaker = RelationPredictionArtifact.model_validate(weaker_data)
        weaker_report = score_relation_benchmark(dataset, weaker)
        comparison = compare_relation_reports(
            report,
            weaker_report,
            label_a="rules-with-cue",
            label_b="empty-control",
            bootstrap_resamples=100,
        )
        self.assertEqual(comparison["exact_f1_difference_b_minus_a"], -1.0)
        self.assertEqual(comparison["exact_f1_difference_ci95"], [-1.0, -1.0])

    def test_rule_adapter_restores_offsets_from_exact_ner_mapping(self):
        base = relation_unit()
        changed = base.model_dump(mode="json")
        prefix = "前文"
        shift = len(prefix)
        original_length = len(base.corrected_text)
        changed["corrected_text"] = prefix + base.corrected_text
        changed["raw_ocr_text"] = prefix + base.raw_ocr_text
        for mention in changed["mentions"]:
            mention["corrected_start"] += shift
            mention["corrected_end"] += shift
            mention["raw_start"] += shift
            mention["raw_end"] += shift
        for annotation in [*changed["reviews"], changed["adjudication"]]:
            for relation in annotation["relations"]:
                relation["corrected_evidence_start"] += shift
                relation["corrected_evidence_end"] += shift
                relation["raw_evidence_start"] += shift
                relation["raw_evidence_end"] += shift
        mapping = changed["source_ner_mappings"][0]
        mapping.update(
            {
                "unit_corrected_start": shift,
                "unit_corrected_end": shift + original_length,
                "snippet_corrected_start": shift,
                "snippet_corrected_end": shift + original_length,
                "unit_raw_start": shift,
                "unit_raw_end": shift + original_length,
                "snippet_raw_start": shift,
                "snippet_raw_end": shift + original_length,
            }
        )
        unit = RelationBenchmarkUnit.model_validate(changed)
        dataset, gold_bytes = dataset_for_unit(unit)
        artifact = execute_relation_rule_baseline(
            dataset,
            split="test",
            input_variant="raw_ocr",
            code_revision="a" * 40,
            source_ner_gold_bytes=gold_bytes,
            allow_ineligible_technical_run=True,
        )
        prediction = artifact.results[0].relations[0]
        self.assertEqual(prediction.evidence_start, shift)
        self.assertEqual(prediction.subject.text_start, shift)
        self.assertEqual(prediction.evidence_text, base.corrected_text[:-3])

    def test_invalid_argument_surface_is_counted_not_silently_matched(self):
        dataset, _ = dataset_for_unit(relation_unit())
        artifact = execute_relation_rule_baseline(
            dataset,
            split="test",
            input_variant="corrected_text",
            code_revision="a" * 40,
            allow_ineligible_technical_run=True,
        )
        changed = artifact.model_dump(mode="json")
        changed["results"][0]["relations"][0]["subject"]["text"] = "錯字"
        changed_artifact = RelationPredictionArtifact.model_validate(changed)
        report = score_relation_benchmark(dataset, changed_artifact)
        self.assertEqual(report["exact_relation"]["f1"], 0.0)
        self.assertEqual(report["invalid_evidence_predictions"], 1)

        changed = artifact.model_dump(mode="json")
        changed["artifact_id"] = str(uuid4())
        changed["results"][0]["relations"][0]["predicate"] = "resided_in"
        changed_artifact = RelationPredictionArtifact.model_validate(changed)
        report = score_relation_benchmark(dataset, changed_artifact)
        self.assertEqual(report["invalid_evidence_predictions"], 1)
        self.assertEqual(
            report["invalid_prediction_examples"][0]["reason"],
            "argument_types_violate_predicate_ontology",
        )

    def test_raw_unrecoverable_positive_is_not_mislabeled_as_negative(self):
        changed = relation_unit().model_dump(mode="json")
        for mention in changed["mentions"]:
            mention.update({"raw_start": None, "raw_end": None, "raw_text": None})
        for annotation in [*changed["reviews"], changed["adjudication"]]:
            for relation in annotation["relations"]:
                relation.update(
                    {
                        "raw_evidence_start": None,
                        "raw_evidence_end": None,
                        "raw_evidence_text": None,
                    }
                )
        mapping = changed["source_ner_mappings"][0]
        mapping.update(
            {
                "unit_raw_start": None,
                "unit_raw_end": None,
                "snippet_raw_start": None,
                "snippet_raw_end": None,
            }
        )
        unit = RelationBenchmarkUnit.model_validate(changed)
        dataset, gold_bytes = dataset_for_unit(unit)
        artifact = execute_relation_rule_baseline(
            dataset,
            split="test",
            input_variant="raw_ocr",
            code_revision="a" * 40,
            source_ner_gold_bytes=gold_bytes,
            allow_ineligible_technical_run=True,
        )
        report = score_relation_benchmark(dataset, artifact)
        self.assertEqual(report["negative_units"], 0)
        self.assertEqual(report["raw_relation_recoverability"], 0.0)
        self.assertEqual(report["end_to_end_raw_relation"]["recall"], 0.0)

    def test_source_ner_gold_hash_and_mapping_are_verified(self):
        dataset, gold_bytes = dataset_for_unit(relation_unit())
        with self.assertRaisesRegex(ValueError, "file hash"):
            verify_relation_dataset_ner_gold(dataset, gold_bytes + b" ")

        gold = NERGoldSet.model_validate_json(gold_bytes)
        changed = dataset.model_dump(mode="json")
        changed["units"][0]["source_ner_mappings"][0][
            "source_ner_snippet_id"
        ] = "missing-snippet"
        changed_dataset = RelationBenchmarkDataset.model_validate(changed)
        with self.assertRaisesRegex(ValueError, "unknown NER gold snippet"):
            verify_relation_dataset_ner_gold(
                changed_dataset, gold.model_dump_json().encode("utf-8")
            )

        changed = dataset.model_dump(mode="json")
        changed["units"][0]["mentions"][0].update(
            {
                "corrected_end": 2,
                "corrected_text": changed["units"][0]["corrected_text"][:2],
                "raw_end": 2,
                "raw_text": changed["units"][0]["raw_ocr_text"][:2],
            }
        )
        changed_dataset = RelationBenchmarkDataset.model_validate(changed)
        with self.assertRaisesRegex(ValueError, "not an exact copy"):
            verify_relation_dataset_ner_gold(changed_dataset, gold_bytes)

    def test_eligibility_is_derived_and_cannot_be_flipped(self):
        dataset, _ = dataset_for_unit(relation_unit())
        changed = dataset.model_dump(mode="json")
        changed["benchmark_eligible"] = True
        changed["eligibility_failures"] = []
        with self.assertRaisesRegex(ValueError, "eligibility disagrees"):
            RelationBenchmarkDataset.model_validate(changed)

    def test_representative_issue_split_dataset_can_become_eligible(self):
        units = []
        predicates = [definition.predicate for definition in PREDICATES]
        for issue_index in range(30):
            split = (
                "train"
                if issue_index < 18
                else "development"
                if issue_index < 24
                else "test"
            )
            for local_index in range(12):
                positive = local_index >= 2
                predicate = predicates[(issue_index * 10 + local_index - 2) % 3]
                units.append(
                    relation_unit(
                        issue_index * 100 + local_index,
                        predicate=predicate,
                        positive=positive,
                        issue_id=f"issue-{issue_index:02d}",
                        split=split,
                        selected_by=(
                            "historian-selector-a"
                            if (issue_index + local_index) % 2
                            else "historian-selector-b"
                        ),
                        year=1925 if issue_index % 2 else 1935,
                    )
                )
        dataset = build_relation_benchmark_dataset(
            "eligible-relation-v1",
            "women-history-relations-v1",
            "ner-gold-v1",
            "f" * 64,
            PREDICATES,
            units,
            created_at=NOW,
        )
        self.assertTrue(dataset.benchmark_eligible, dataset.eligibility_failures)
        self.assertEqual(dataset.eligibility_failures, [])
        with self.assertRaisesRegex(ValueError, "verified source NER gold bytes"):
            execute_relation_rule_baseline(
                dataset,
                split="test",
                input_variant="raw_ocr",
                code_revision="a" * 40,
            )

    def test_moving_model_revision_is_rejected(self):
        dataset, _ = dataset_for_unit(relation_unit())
        artifact = execute_relation_rule_baseline(
            dataset,
            split="test",
            input_variant="corrected_text",
            code_revision="a" * 40,
            allow_ineligible_technical_run=True,
        )
        changed = artifact.model_dump(mode="json")
        changed["model_revision"] = "latest"
        with self.assertRaisesRegex(ValueError, "immutable"):
            RelationPredictionArtifact.model_validate(changed)


if __name__ == "__main__":
    unittest.main()
