from __future__ import annotations

import unittest
from uuid import uuid4

from pydantic import ValidationError

from wic_history.evidence import (
    ClaimCandidate,
    EntityType,
    Point,
    Polygon,
    ProcessingRun,
    NERArtifact,
    RunKind,
    SourcePointer,
)


class EvidenceContractTests(unittest.TestCase):
    def setUp(self):
        self.polygon = Polygon(
            points=[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=20)]
        )

    def test_source_offsets_are_paired(self):
        with self.assertRaisesRegex(ValidationError, "provided together"):
            SourcePointer(source_uri="s3://bucket/key", page_number=1, text_start=3)

    def test_claim_requires_grounded_single_object(self):
        run = ProcessingRun(
            kind=RunKind.RELATION,
            engine="test",
            model_name="fixture",
            model_revision="1",
        )
        source = SourcePointer(
            source_uri="s3://bucket/key",
            page_number=1,
            polygon=self.polygon,
            text_start=0,
            text_end=4,
        )
        claim = ClaimCandidate(
            subject_entity_id=uuid4(),
            predicate="attended",
            object_literal={"name": "school"},
            evidence=[source],
            supporting_quote="女学生入学",
            run_id=run.run_id,
        )
        self.assertEqual(claim.status, "candidate")

        with self.assertRaisesRegex(ValidationError, "exactly one"):
            ClaimCandidate(
                subject_entity_id=uuid4(),
                predicate="attended",
                object_literal={"name": "school"},
                object_entity_id=uuid4(),
                evidence=[source],
                supporting_quote="女学生入学",
                run_id=run.run_id,
            )

    def test_entity_enum_includes_women_history_types(self):
        self.assertEqual(EntityType.KINSHIP_TERM.value, "kinship_term")
        self.assertEqual(EntityType.SCHOOL.value, "school")

    def test_ner_artifact_rejects_non_ner_run(self):
        run = ProcessingRun(
            kind=RunKind.OCR,
            engine="test",
            model_name="fixture",
            model_revision="1",
        )
        with self.assertRaisesRegex(ValidationError, "kind=ner"):
            NERArtifact(source_ocr_run_id=uuid4(), run=run, mentions=[])


if __name__ == "__main__":
    unittest.main()
