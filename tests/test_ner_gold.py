from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from wic_history.evidence import (
    EntityMentionCandidate,
    EntityType,
    NERArtifact,
    ProcessingRun,
    RunKind,
    SourcePointer,
)
from wic_history.ner_gold import (
    GoldAdjudication,
    GoldEntitySpan,
    GoldSnippet,
    NERGoldSet,
    ReviewerAnnotation,
    character_error_distance,
    score_ner_artifact,
)


REGION_ID = UUID("00000000-0000-0000-0000-000000000001")


def entity_spans() -> list[GoldEntitySpan]:
    return [
        GoldEntitySpan(
            entity_type="person",
            corrected_start=0,
            corrected_end=2,
            corrected_text="王氏",
            raw_start=0,
            raw_end=2,
            raw_text="王民",
        ),
        GoldEntitySpan(
            entity_type="school",
            corrected_start=4,
            corrected_end=6,
            corrected_text="女塾",
        ),
    ]


def gold_set() -> NERGoldSet:
    annotated_at = datetime(2026, 7, 18, tzinfo=timezone.utc)
    return NERGoldSet(
        dataset_id="unit-gold",
        created_at=annotated_at,
        ontology_version="1.0",
        snippets=[
            GoldSnippet(
                snippet_id="v1-p1-r1",
                source=SourcePointer(
                    source_uri="s3://example/v1.pdf",
                    volume_number=1,
                    publication_year=1925,
                    page_number=1,
                    region_id=REGION_ID,
                ),
                raw_ocr_text="王民入學",
                page_genre="news_editorial",
                layout="vertical",
                scan_quality="poor",
                reviews=[
                    ReviewerAnnotation(
                        reviewer="reviewer-a",
                        corrected_text="王氏入學女塾",
                        entities=entity_spans(),
                        annotated_at=annotated_at,
                    ),
                    ReviewerAnnotation(
                        reviewer="reviewer-b",
                        corrected_text="王氏入學女塾",
                        entities=entity_spans(),
                        annotated_at=annotated_at,
                    ),
                ],
                adjudication=GoldAdjudication(
                    adjudicator="adjudicator-c",
                    corrected_text="王氏入學女塾",
                    entities=entity_spans(),
                    adjudicated_at=annotated_at,
                ),
            )
        ],
    )


def predictions(input_text: str) -> NERArtifact:
    started_at = datetime(2026, 7, 18, tzinfo=timezone.utc)
    run_id = uuid4()
    run = ProcessingRun(
        run_id=run_id,
        kind=RunKind.NER,
        engine="unit",
        model_name="unit-model",
        model_revision="unit-revision",
        started_at=started_at,
        completed_at=started_at + timedelta(seconds=2),
    )
    if input_text == "corrected":
        spans = [
            (EntityType.PERSON, "王氏", 0, 2),
            (EntityType.SCHOOL, "女", 4, 5),
            (EntityType.ORGANIZATION, "錯", 0, 1),
        ]
    else:
        spans = [
            (EntityType.PERSON, "王民", 0, 2),
            (EntityType.ORGANIZATION, "錯", 0, 1),
        ]
    mentions = [
        EntityMentionCandidate(
            entity_type=entity_type,
            text=text,
            source=SourcePointer(
                source_uri="s3://example/v1.pdf",
                volume_number=1,
                publication_year=1925,
                page_number=1,
                region_id=REGION_ID,
                text_start=start,
                text_end=end,
            ),
            confidence=0.9,
            run_id=run_id,
        )
        for entity_type, text, start, end in spans
    ]
    return NERArtifact(source_ocr_run_id=uuid4(), run=run, mentions=mentions)


class NERGoldTests(unittest.TestCase):
    def test_character_error_distance(self):
        self.assertEqual(character_error_distance("王氏入學女塾", "王民入學"), 3)

    def test_corrected_score_penalizes_invalid_evidence_and_supports_relaxed_overlap(self):
        report = score_ner_artifact(gold_set(), predictions("corrected"), "corrected")
        self.assertEqual(report["exact"]["true_positive"], 1)
        self.assertAlmostEqual(report["exact"]["precision"], 1 / 3)
        self.assertEqual(report["relaxed_overlap"]["true_positive"], 2)
        self.assertAlmostEqual(report["invalid_evidence_rate"], 1 / 3)
        self.assertEqual(report["by_layout"]["vertical"]["false_negative"], 1)
        self.assertEqual(report["mentions_per_second"], 1.5)

    def test_raw_score_separates_recoverability_from_end_to_end_recall(self):
        report = score_ner_artifact(gold_set(), predictions("raw_ocr"), "raw_ocr")
        self.assertEqual(report["gold_entities"], 1)
        self.assertEqual(report["total_adjudicated_entities"], 2)
        self.assertEqual(report["raw_recoverability"], 0.5)
        self.assertEqual(report["exact"]["recall"], 1.0)
        self.assertEqual(report["end_to_end_exact_recall"], 0.5)
        self.assertEqual(report["ocr_cer"], 0.5)

    def test_gold_requires_two_distinct_reviewers(self):
        data = gold_set().model_dump(mode="json")
        data["snippets"][0]["reviews"][1]["reviewer"] = "reviewer-a"
        with self.assertRaises(ValueError):
            NERGoldSet.model_validate(data)

    def test_gold_rejects_offsets_beyond_the_text(self):
        data = gold_set().model_dump(mode="json")
        entity = data["snippets"][0]["adjudication"]["entities"][1]
        entity["corrected_end"] = 100
        with self.assertRaises(ValueError):
            NERGoldSet.model_validate(data)


if __name__ == "__main__":
    unittest.main()
