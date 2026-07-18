from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from wic_history.gold_packet import (
    ACTIVE_REGION_SQL,
    NERAnnotationPacket,
    PacketAnnotationSubmission,
    PacketUnitAdjudication,
    PacketUnitAnnotation,
    SamplingReason,
    annotation_template,
    blinded_reviewer_view,
    build_packet_from_rows,
    finalize_packet,
    validate_packet_identity,
    verify_packet_files,
)
from wic_history.ner_gold import GoldAdjudication, ReviewerAnnotation


PAGE_ID = UUID("00000000-0000-0000-0000-000000000010")
OCR_RUN_ID = UUID("00000000-0000-0000-0000-000000000020")
DERIVATIVE_ID = UUID("00000000-0000-0000-0000-000000000030")


def region_row(index: int, text: str, confidence: float) -> dict:
    return {
        "region_id": UUID(f"00000000-0000-0000-0000-{index:012d}"),
        "source_ocr_run_id": OCR_RUN_ID,
        "page_id": PAGE_ID,
        "region_kind": "text",
        "reading_order": index,
        "polygon": {
            "points": [
                {"x": index * 10, "y": 0},
                {"x": index * 10 + 5, "y": 0},
                {"x": index * 10 + 5, "y": 20},
                {"x": index * 10, "y": 20},
            ]
        },
        "raw_text": text,
        "normalized_text": text,
        "confidence": confidence,
        "direction": "vertical",
        "page_number": 308,
        "volume_number": 219,
        "publication_year": 1925,
        "source_uri": "s3://example/v219.pdf",
        "source_sha256": "a" * 64,
        "derivative_id": DERIVATIVE_ID,
        "image_uri": "artifacts/gold-packet/page.png",
        "image_sha256": "b" * 64,
        "width": 100,
        "height": 200,
        "dpi": 300,
        "media_type": "image/png",
        "evidence_tier": "non_gold_lossless_pilot",
        "render_manifest_uri": "artifacts/gold-packet/manifest.jsonl",
        "selection_basis": "technical_default",
    }


def packet(max_units: int = 6) -> NERAnnotationPacket:
    rows = [
        region_row(1, "士女", 0.95),
        region_row(2, "女學生", 0.82),
        region_row(3, "申報", 0.65),
        region_row(4, "模糊", 0.30),
        region_row(5, "商務", 0.75),
        region_row(6, "新聞", 0.90),
    ]
    mentions = [
        {
            "region_id": rows[0]["region_id"],
            "run_id": UUID("00000000-0000-0000-0000-000000000101"),
            "model_name": "model-a",
            "entity_type": "person",
            "text_start": 0,
            "text_end": 2,
        },
        {
            "region_id": rows[0]["region_id"],
            "run_id": UUID("00000000-0000-0000-0000-000000000102"),
            "model_name": "model-b",
            "entity_type": "organization",
            "text_start": 0,
            "text_end": 2,
        },
        {
            "region_id": rows[1]["region_id"],
            "run_id": UUID("00000000-0000-0000-0000-000000000101"),
            "model_name": "model-a",
            "entity_type": "school",
            "text_start": 0,
            "text_end": 3,
        },
    ]
    return build_packet_from_rows(
        rows,
        mentions,
        dataset_id="packet-unit",
        ontology_version="women-history-zh-v1",
        max_units=max_units,
        context_radius=1,
        volume_number=219,
        page_number=308,
        generated_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
    )


class GoldPacketTests(unittest.TestCase):
    def test_global_scope_casts_nullable_database_filters(self):
        self.assertIn("CAST(%s AS integer)", ACTIVE_REGION_SQL)
        self.assertIn("archive.page_issue_assignment", ACTIVE_REGION_SQL)
        self.assertIn("evidence.coherent_unit_revision", ACTIVE_REGION_SQL)

    def test_context_does_not_cross_reviewed_coherent_unit_boundaries(self):
        rows = [
            region_row(index, text, 0.9)
            for index, text in enumerate(
                ["女學", "教育", "新聞", "商務", "廣告", "社會"], start=1
            )
        ]
        first_revision = UUID("00000000-0000-0000-0000-000000000401")
        second_revision = UUID("00000000-0000-0000-0000-000000000402")
        region_group = {}
        for index, row in enumerate(rows):
            revision_id = first_revision if index < 3 else second_revision
            row["coherent_unit_revision_id"] = revision_id
            row["issue_id"] = UUID("00000000-0000-0000-0000-000000000403")
            region_group[row["region_id"]] = revision_id

        result = build_packet_from_rows(
            rows,
            [],
            dataset_id="bounded-context",
            ontology_version="women-history-zh-v1",
            max_units=6,
            context_radius=5,
            generated_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        )

        for unit in result.units:
            expected = unit.reviewed_coherent_unit_revision_id
            context = unit.context_before + unit.context_after
            self.assertTrue(context)
            self.assertTrue(
                all(region_group[item.source_ocr_region_id] == expected for item in context)
            )

    def test_balanced_packet_is_candidate_only_and_content_addressed(self):
        result = packet()
        self.assertEqual(result.status, "annotation_candidate")
        self.assertFalse(result.benchmark_eligible)
        self.assertEqual(result.coverage.units, 6)
        self.assertEqual(result.coverage.decades, ["1920s"])
        self.assertIn(SamplingReason.WOMEN_THEME, result.units[0].selection_reasons)
        self.assertTrue(
            any(
                SamplingReason.NER_DISAGREEMENT in unit.selection_reasons
                for unit in result.units
            )
        )
        validate_packet_identity(result)
        regenerated = packet().model_copy(
            update={"generated_at": result.generated_at + timedelta(days=1)}
        )
        self.assertEqual(result.packet_id, regenerated.packet_id)
        validate_packet_identity(regenerated)

    def test_packet_tampering_is_detected(self):
        result = packet()
        result.units[0].target.raw_text = "changed"
        with self.assertRaises(ValueError):
            validate_packet_identity(result)

    def test_template_is_blank_and_cannot_be_mistaken_for_gold(self):
        template = annotation_template(packet(1))
        self.assertIsNone(template["units"][0]["adjudication"]["gold_region_id"])
        self.assertEqual(template["units"][0]["reviews"][0]["reviewer"], "")
        with self.assertRaises(ValueError):
            PacketAnnotationSubmission.model_validate(template)

    def test_reviewer_view_hides_sampling_and_model_disagreement_signals(self):
        view = blinded_reviewer_view(packet())
        serialized = str(view)
        self.assertEqual(view["status"], "blinded_annotation_view")
        self.assertNotIn("selection_reasons", serialized)
        self.assertNotIn("ner_disagreement", serialized)
        self.assertNotIn("candidate_model", serialized)
        self.assertEqual(len(view["units"]), 6)

    def test_finalization_requires_a_new_gold_identity_and_exact_packet(self):
        result = packet(1)
        unit = result.units[0]
        now = datetime(2026, 7, 18, tzinfo=timezone.utc)
        reviews = [
            ReviewerAnnotation(
                reviewer="reviewer-a",
                corrected_text=unit.target.raw_text,
                entities=[],
                annotated_at=now,
            ),
            ReviewerAnnotation(
                reviewer="reviewer-b",
                corrected_text=unit.target.raw_text,
                entities=[],
                annotated_at=now,
            ),
        ]

        def submission(gold_region_id: UUID) -> PacketAnnotationSubmission:
            return PacketAnnotationSubmission(
                packet_id=result.packet_id,
                units=[
                    PacketUnitAnnotation(
                        unit_id=unit.unit_id,
                        reviews=reviews,
                        adjudication=PacketUnitAdjudication(
                            adjudicator="adjudicator-c",
                            gold_region_id=gold_region_id,
                            corrected_text=unit.target.raw_text,
                            entities=[],
                            adjudicated_at=now,
                            page_genre="mixed",
                            layout="vertical",
                            scan_quality="moderate",
                        ),
                    )
                ],
            )

        with self.assertRaises(ValueError):
            finalize_packet(result, submission(unit.target.source_ocr_region_id))
        gold = finalize_packet(
            result,
            submission(UUID("00000000-0000-0000-0000-000000000999")),
        )
        self.assertEqual(gold.schema_version, "1.1")
        self.assertEqual(
            gold.snippets[0].source_ocr_region_id,
            unit.target.source_ocr_region_id,
        )
        self.assertNotEqual(
            gold.snippets[0].gold_region_id,
            gold.snippets[0].source_ocr_region_id,
        )

    def test_local_derivative_hash_is_verified(self):
        result = packet(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "artifacts/gold-packet/page.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"lossless-image")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            result.pages[0].image_sha256 = digest
            result.pages[0].source.image_sha256 = digest
            result.units[0].source.image_sha256 = digest
            result.packet_id = "0" * 64
            from wic_history.gold_packet import packet_identity

            result.packet_id = packet_identity(result)
            verify_packet_files(result, root)
            image.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                verify_packet_files(result, root)

    def test_gold_schema_rejects_model_region_as_gold_identity(self):
        result = packet(1)
        unit = result.units[0]
        now = datetime(2026, 7, 18, tzinfo=timezone.utc)
        data = {
            "schema_version": "1.1",
            "dataset_id": "bad-gold",
            "created_at": now.isoformat(),
            "ontology_version": "women-history-zh-v1",
            "snippets": [
                {
                    "snippet_id": "bad",
                    "gold_region_id": str(unit.target.source_ocr_region_id),
                    "source_ocr_run_id": str(unit.target.source_ocr_run_id),
                    "source_ocr_region_id": str(unit.target.source_ocr_region_id),
                    "source": unit.source.model_dump(mode="json"),
                    "raw_ocr_text": unit.target.raw_text,
                    "page_genre": "mixed",
                    "layout": "vertical",
                    "scan_quality": "moderate",
                    "reviews": [
                        ReviewerAnnotation(
                            reviewer="a",
                            corrected_text=unit.target.raw_text,
                            annotated_at=now,
                        ).model_dump(mode="json"),
                        ReviewerAnnotation(
                            reviewer="b",
                            corrected_text=unit.target.raw_text,
                            annotated_at=now,
                        ).model_dump(mode="json"),
                    ],
                    "adjudication": GoldAdjudication(
                        adjudicator="c",
                        corrected_text=unit.target.raw_text,
                        adjudicated_at=now,
                    ).model_dump(mode="json"),
                }
            ],
        }
        from wic_history.ner_gold import NERGoldSet

        with self.assertRaises(ValueError):
            NERGoldSet.model_validate(data)


if __name__ == "__main__":
    unittest.main()
