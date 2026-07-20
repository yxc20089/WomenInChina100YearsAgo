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
    RetrievalHit,
    RetrievalMode,
    RetrievalResponse,
    RetrievalSourceSpan,
    RunKind,
    SourcePointer,
)


class EvidenceContractTests(unittest.TestCase):
    def test_region_response_serializes_exact_schema_1_0_keys(self):
        source = SourcePointer(
            source_uri="s3://bucket/key",
            page_id=uuid4(),
            derivative_id=uuid4(),
            image_uri="artifacts/page.png",
            page_number=1,
            region_id=uuid4(),
            text_version_id=uuid4(),
            text_selection_id=uuid4(),
        )
        response = RetrievalResponse(
            query="女子",
            mode=RetrievalMode.LEXICAL,
            hits=[RetrievalHit(rank=1, score=1, source=source, text="女子")],
        )

        payload = response.model_dump(mode="json")

        self.assertEqual(
            set(payload["hits"][0]),
            {
                "rank",
                "score",
                "source",
                "text",
                "normalized_text",
                "entity_ids",
                "claim_ids",
                "explanation",
            },
        )
        self.assertEqual(
            set(payload["hits"][0]["source"]),
            {
                "source_uri",
                "source_sha256",
                "derivative_id",
                "image_sha256",
                "evidence_tier",
                "volume_number",
                "publication_year",
                "page_number",
                "region_id",
                "polygon",
                "text_start",
                "text_end",
            },
        )

    def setUp(self):
        self.polygon = Polygon(
            points=[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=20)]
        )

    def test_source_offsets_are_paired(self):
        with self.assertRaisesRegex(ValidationError, "provided together"):
            SourcePointer(source_uri="s3://bucket/key", page_number=1, text_start=3)

    def test_source_pointer_validates_derivative_hash(self):
        with self.assertRaises(ValidationError):
            SourcePointer(
                source_uri="s3://bucket/key",
                page_number=1,
                image_sha256="not-a-sha256",
            )

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

    def test_ner_schema_1_1_requires_exact_input_identity(self):
        run = ProcessingRun(
            kind=RunKind.NER,
            engine="test",
            model_name="fixture",
            model_revision="1",
        )
        with self.assertRaisesRegex(ValidationError, "input identity"):
            NERArtifact(
                schema_version="1.1",
                source_ocr_run_id=uuid4(),
                run=run,
                mentions=[],
            )

    def test_legacy_region_hit_canonicalizes_target_and_source_span(self):
        region_id = uuid4()
        source = SourcePointer(
            source_uri="s3://bucket/key",
            page_number=1,
            region_id=region_id,
            text_start=2,
            text_end=6,
        )

        hit = RetrievalHit(rank=1, score=1, source=source, text="女子")

        self.assertEqual(hit.target_kind, "region")
        self.assertEqual(hit.target_id, region_id)
        self.assertEqual(hit.source, source)
        self.assertEqual(len(hit.sources), 1)
        self.assertEqual(hit.sources[0].source, source)
        self.assertEqual(hit.sources[0].document_start, 0)
        self.assertEqual(hit.sources[0].document_end, 2)

        response = RetrievalResponse(
            query="女子", mode=RetrievalMode.LEXICAL, hits=[hit]
        )
        parsed = RetrievalResponse.model_validate_json(response.model_dump_json())
        self.assertEqual(parsed.schema_version, "1.0")
        self.assertEqual(parsed.hits[0].source, source)

    def test_legacy_empty_region_text_remains_constructible(self):
        source = SourcePointer(
            source_uri="s3://bucket/key", page_number=1, region_id=uuid4()
        )

        hit = RetrievalHit(rank=1, score=0, source=source, text="")

        self.assertEqual(hit.sources[0].document_start, 0)
        self.assertEqual(hit.sources[0].document_end, 0)

    def test_legacy_region_hit_without_region_id_remains_constructible(self):
        source = SourcePointer(source_uri="s3://bucket/key", page_number=1)

        hit = RetrievalHit(rank=1, score=1, source=source, text="女子")
        parsed = RetrievalHit.model_validate_json(hit.model_dump_json())

        self.assertIsNone(parsed.target_id)
        self.assertEqual(parsed.source, source)
        self.assertEqual(parsed.sources[0].source, source)

    def test_region_hit_rejects_caller_controlled_inconsistent_source_span(self):
        region_id = uuid4()
        source = SourcePointer(
            source_uri="s3://bucket/key", page_number=1, region_id=region_id
        )
        unrelated_document_id = uuid4()
        inconsistent = RetrievalSourceSpan(
            document_id=unrelated_document_id,
            sequence_number=7,
            document_start=4,
            document_end=6,
            role="body",
            source=source,
        )

        with self.assertRaisesRegex(ValidationError, "canonical source span"):
            RetrievalHit(
                rank=1,
                score=1,
                source=source,
                sources=[inconsistent],
                text="女子",
            )

    def test_source_span_derives_and_validates_deterministic_citation_id(self):
        document_id = uuid4()
        source = SourcePointer(
            source_uri="s3://bucket/key",
            page_number=1,
            region_id=uuid4(),
            text_version_id=uuid4(),
            text_start=2,
            text_end=4,
        )

        derived = RetrievalSourceSpan(
            document_id=document_id,
            sequence_number=0,
            document_start=0,
            document_end=2,
            role="body",
            source=source,
        )
        repeated = RetrievalSourceSpan(
            document_id=document_id,
            sequence_number=0,
            document_start=0,
            document_end=2,
            role="body",
            source=source,
        )
        changed_sequence = RetrievalSourceSpan(
            document_id=document_id,
            sequence_number=1,
            document_start=0,
            document_end=2,
            role="body",
            source=source,
        )
        changed_offset_source = SourcePointer.model_validate(
            {**source.model_dump(), "text_start": 3, "text_end": 5}
        )
        changed_offset = RetrievalSourceSpan(
            document_id=document_id,
            sequence_number=0,
            document_start=0,
            document_end=2,
            role="body",
            source=changed_offset_source,
        )

        self.assertTrue(derived.citation_id.startswith("citation:"))
        self.assertEqual(derived.citation_id, repeated.citation_id)
        self.assertNotEqual(derived.citation_id, changed_sequence.citation_id)
        self.assertNotEqual(derived.citation_id, changed_offset.citation_id)
        with self.assertRaisesRegex(ValidationError, "deterministic provenance"):
            RetrievalSourceSpan(
                citation_id="caller-selected",
                document_id=document_id,
                sequence_number=0,
                document_start=0,
                document_end=2,
                role="body",
                source=source,
            )

    def test_reviewed_coherent_unit_hit_round_trips_ordered_atomic_sources(self):
        revision_id = uuid4()
        coherent_unit_id = uuid4()
        first_region_id = uuid4()
        second_region_id = uuid4()
        spans = [
            RetrievalSourceSpan(
                document_id=revision_id,
                sequence_number=0,
                document_start=0,
                document_end=2,
                role="headline",
                source=SourcePointer(
                    source_uri="s3://bucket/volume.pdf",
                    page_id=uuid4(),
                    image_uri="artifacts/page-1.png",
                    page_number=1,
                    region_id=first_region_id,
                    text_version_id=uuid4(),
                    text_selection_id=uuid4(),
                    text_start=1,
                    text_end=3,
                ),
            ),
            RetrievalSourceSpan(
                document_id=revision_id,
                sequence_number=1,
                document_start=3,
                document_end=7,
                role="body",
                source=SourcePointer(
                    source_uri="s3://bucket/volume.pdf",
                    page_id=uuid4(),
                    image_uri="artifacts/page-2.png",
                    page_number=2,
                    region_id=second_region_id,
                    text_version_id=uuid4(),
                    text_selection_id=uuid4(),
                    text_start=4,
                    text_end=8,
                ),
            ),
        ]
        response = RetrievalResponse(
            schema_version="1.1",
            query="女學",
            mode=RetrievalMode.HYBRID,
            hits=[
                RetrievalHit(
                    rank=1,
                    score=0.9,
                    target_kind="reviewed_coherent_unit",
                    target_id=revision_id,
                    coherent_unit_id=coherent_unit_id,
                    source=None,
                    sources=spans,
                    text="女子\n入學消息",
                )
            ],
        )

        parsed = RetrievalResponse.model_validate_json(response.model_dump_json())

        hit = parsed.hits[0]
        self.assertEqual(parsed.schema_version, "1.1")
        self.assertIsNone(hit.source)
        self.assertEqual(hit.target_id, revision_id)
        self.assertEqual(hit.coherent_unit_id, coherent_unit_id)
        self.assertEqual([span.sequence_number for span in hit.sources], [0, 1])
        self.assertEqual(
            [span.source.region_id for span in hit.sources],
            [first_region_id, second_region_id],
        )

    def test_reviewed_coherent_unit_requires_nonempty_sources(self):
        with self.assertRaisesRegex(ValidationError, "nonempty ordered sources"):
            RetrievalHit(
                rank=1,
                score=1,
                target_kind="reviewed_coherent_unit",
                target_id=uuid4(),
                coherent_unit_id=uuid4(),
                source=None,
                sources=[],
                text="女子",
            )

    def test_reviewed_coherent_unit_requires_response_schema_1_1(self):
        revision_id = uuid4()
        source = SourcePointer(
            source_uri="s3://bucket/key", page_number=1, region_id=uuid4()
        )
        hit = RetrievalHit(
            rank=1,
            score=1,
            target_kind="reviewed_coherent_unit",
            target_id=revision_id,
            coherent_unit_id=uuid4(),
            source=None,
            sources=[
                RetrievalSourceSpan(
                    document_id=revision_id,
                    sequence_number=0,
                    document_start=0,
                    document_end=2,
                    role="body",
                    source=source,
                )
            ],
            text="女子",
        )

        with self.assertRaisesRegex(ValidationError, "require response schema 1.1"):
            RetrievalResponse(
                schema_version="1.0",
                query="女子",
                mode=RetrievalMode.HYBRID,
                hits=[hit],
            )

    def test_reviewed_coherent_unit_rejects_duplicate_citation_identity(self):
        revision_id = uuid4()
        source = SourcePointer(
            source_uri="s3://bucket/key", page_number=1, region_id=uuid4()
        )
        spans = [
            RetrievalSourceSpan(
                document_id=revision_id,
                sequence_number=0,
                document_start=sequence_number * 2,
                document_end=sequence_number * 2 + 1,
                role="body",
                source=source,
            )
            for sequence_number in range(2)
        ]

        with self.assertRaisesRegex(ValidationError, "citation_id values must be unique"):
            RetrievalHit(
                rank=1,
                score=1,
                target_kind="reviewed_coherent_unit",
                target_id=revision_id,
                coherent_unit_id=uuid4(),
                source=None,
                sources=spans,
                text="女子",
            )

    def test_reviewed_coherent_unit_rejects_singular_source(self):
        source = SourcePointer(
            source_uri="s3://bucket/key", page_number=1, region_id=uuid4()
        )
        with self.assertRaisesRegex(ValidationError, "must not have a singular source"):
            RetrievalHit(
                rank=1,
                score=1,
                target_kind="reviewed_coherent_unit",
                target_id=uuid4(),
                coherent_unit_id=uuid4(),
                source=source,
                sources=[
                    RetrievalSourceSpan(
                        document_id=uuid4(),
                        sequence_number=0,
                        document_start=0,
                        document_end=2,
                        role="body",
                        source=source,
                    )
                ],
                text="女子",
            )

    def test_reviewed_coherent_unit_rejects_out_of_order_or_overlapping_sources(self):
        revision_id = uuid4()
        source = SourcePointer(
            source_uri="s3://bucket/key", page_number=1, region_id=uuid4()
        )
        spans = [
            RetrievalSourceSpan(
                document_id=revision_id,
                sequence_number=1,
                document_start=2,
                document_end=4,
                role="body",
                source=source,
            ),
            RetrievalSourceSpan(
                document_id=revision_id,
                sequence_number=0,
                document_start=1,
                document_end=3,
                role="headline",
                source=source,
            ),
        ]
        with self.assertRaisesRegex(ValidationError, "ordered and nonoverlapping"):
            RetrievalHit(
                rank=1,
                score=1,
                target_kind="reviewed_coherent_unit",
                target_id=revision_id,
                coherent_unit_id=uuid4(),
                source=None,
                sources=spans,
                text="女子",
            )


if __name__ == "__main__":
    unittest.main()
