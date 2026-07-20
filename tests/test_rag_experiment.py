import hashlib
import unittest
from uuid import UUID

from wic_history.rag_experiment import (
    EXPORT_SQL,
    GRAPHRAG_REVISION,
    LIGHTRAG_REVISION,
    REVIEWED_UNIT_EXPORT_SQL,
    build_coherent_unit_documents,
    build_documents,
)


RAGTestValue = (
    UUID
    | str
    | int
    | float
    | None
    | dict[str, list[dict[str, int]]]
    | list[dict[str, str | int]]
)


def _row(
    region_id: str, page_id: str, text: str, reading_order: int
) -> dict[str, RAGTestValue]:
    return {
        "page_id": UUID(page_id),
        "page_number": 308,
        "source_image_uri": "s3://example/page.jpg",
        "source_image_sha256": "b" * 64,
        "derivative_id": UUID("00000000-0000-0000-0000-000000000020"),
        "evidence_tier": "historian_selected_gold",
        "ocr_selection_basis": "historian_approved",
        "run_id": UUID("00000000-0000-0000-0000-000000000030"),
        "volume_number": 219,
        "publication_year": 1925,
        "source_uri": "s3://example/volume.pdf",
        "source_sha256": "a" * 64,
        "region_id": UUID(region_id),
        "reading_order": reading_order,
        "region_kind": "line",
        "polygon": {"points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}]},
        "raw_text": text,
        "normalized_text": text,
        "confidence": 0.8,
        "ocr_model": "PP-OCRv6",
        "ocr_model_revision": "server",
    }


class RAGExperimentTests(unittest.TestCase):
    def test_build_documents_preserves_exact_region_offsets(self) -> None:
        page = "00000000-0000-0000-0000-000000000010"
        rows = [
            _row("00000000-0000-0000-0000-000000000001", page, "富紳", 1),
            _row("00000000-0000-0000-0000-000000000002", page, "淑女", 2),
        ]

        output = build_documents(rows)

        self.assertEqual(len(output), 1)
        document, citations = output[0]
        self.assertEqual(document["text"], "富紳\n淑女")
        self.assertEqual(document["metadata"]["region_count"], 2)
        self.assertEqual(citations[0]["exported_text"], "富紳")
        self.assertEqual(
            citations[0]["derivative_id"],
            "00000000-0000-0000-0000-000000000020",
        )
        self.assertEqual(
            document["text"][citations[0]["start_char"] : citations[0]["end_char"]],
            "富紳",
        )
        self.assertEqual(
            document["text"][citations[1]["start_char"] : citations[1]["end_char"]],
            "淑女",
        )

    def test_rag_comparator_revisions_are_full_git_hashes(self) -> None:
        self.assertEqual(len(GRAPHRAG_REVISION), 40)
        self.assertEqual(len(LIGHTRAG_REVISION), 40)

    def test_export_includes_only_active_ocr_selections(self) -> None:
        self.assertIn("JOIN evidence.page_ocr_selection", EXPORT_SQL)
        self.assertIn("selection.superseded_at IS NULL", EXPORT_SQL)
        self.assertIn("JOIN evidence.ocr_run_input", EXPORT_SQL)
        self.assertIn("CAST(%(volume_number)s AS integer)", EXPORT_SQL)
        self.assertIn("CAST(%(page_number)s AS integer)", EXPORT_SQL)

    def test_reviewed_export_requires_active_approved_revisions(self) -> None:
        self.assertIn("evidence.coherent_unit_revision", REVIEWED_UNIT_EXPORT_SQL)
        self.assertIn("segmentation_selection.superseded_at IS NULL", REVIEWED_UNIT_EXPORT_SQL)
        self.assertIn("revision.superseded_at IS NULL", REVIEWED_UNIT_EXPORT_SQL)
        self.assertIn("revision.unit_kind = 'article'", REVIEWED_UNIT_EXPORT_SQL)
        self.assertIn("JOIN evidence.region_text_selection", REVIEWED_UNIT_EXPORT_SQL)
        self.assertIn("text_selection.superseded_at IS NULL", REVIEWED_UNIT_EXPORT_SQL)
        self.assertIn("selected_version.review_status = 'reviewed'", REVIEWED_UNIT_EXPORT_SQL)
        self.assertIn("AND NOT EXISTS", REVIEWED_UNIT_EXPORT_SQL)
        self.assertIn("AS expected_span_count", REVIEWED_UNIT_EXPORT_SQL)

    def test_reviewed_unit_offsets_map_back_to_raw_ocr(self) -> None:
        base = _row(
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000010",
            "甲女雷乙",
            1,
        )
        base.update(
            {
                "revision_id": UUID("00000000-0000-0000-0000-000000000040"),
                "unit_id": UUID("00000000-0000-0000-0000-000000000041"),
                "issue_id": None,
                "unit_kind": "article",
                "title": "女學",
                "content_sha256": "c" * 64,
                "approved_by": "historian-a",
                "segmentation_selection_id": UUID("00000000-0000-0000-0000-000000000042"),
                "segmentation_review_id": UUID("00000000-0000-0000-0000-000000000043"),
                "span_sequence_number": 0,
                "span_text_start": 1,
                "span_text_end": 3,
                "span_role": "body",
                "selected_text": "甲女霍乙",
                "selected_text_sha256": hashlib.sha256(
                    "甲女霍乙".encode("utf-8")
                ).hexdigest(),
                "selected_text_version_id": UUID(
                    "00000000-0000-0000-0000-000000000044"
                ),
                "text_selection_id": UUID(
                    "00000000-0000-0000-0000-000000000045"
                ),
                "alignment_operations": [
                    {
                        "operation": "equal",
                        "source_start": 0,
                        "source_end": 1,
                        "target_start": 0,
                        "target_end": 1,
                    },
                    {
                        "operation": "replace",
                        "source_start": 1,
                        "source_end": 3,
                        "target_start": 1,
                        "target_end": 3,
                    },
                    {
                        "operation": "equal",
                        "source_start": 3,
                        "source_end": 4,
                        "target_start": 3,
                        "target_end": 4,
                    },
                ],
                "expected_span_count": 1,
            }
        )

        document, citations = build_coherent_unit_documents([base])[0]

        self.assertEqual(document["text"], "女霍")
        self.assertEqual(citations[0]["region_text_start"], 1)
        self.assertEqual(citations[0]["region_text_end"], 3)
        self.assertEqual(citations[0]["raw_region_text_start"], 1)
        self.assertEqual(citations[0]["raw_region_text_end"], 3)
        self.assertEqual(
            citations[0]["selected_text_version_id"],
            "00000000-0000-0000-0000-000000000044",
        )
        self.assertEqual(
            citations[0]["text_selection_id"],
            "00000000-0000-0000-0000-000000000045",
        )
        self.assertEqual(
            document["metadata"]["content_sha256"],
            hashlib.sha256("女霍".encode("utf-8")).hexdigest(),
        )
        self.assertNotEqual(
            document["metadata"]["content_sha256"],
            document["metadata"]["segmentation_content_sha256"],
        )
        self.assertEqual(
            document["text"][citations[0]["start_char"] : citations[0]["end_char"]],
            "女霍",
        )

    def test_reviewed_unit_rejects_rows_lost_across_provenance_joins(self) -> None:
        # Given: one materialized row from a revision that declares two source spans.
        base = _row(
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000010",
            "女子",
            1,
        )
        base.update(
            {
                "revision_id": UUID("00000000-0000-0000-0000-000000000040"),
                "unit_id": UUID("00000000-0000-0000-0000-000000000041"),
                "issue_id": None,
                "unit_kind": "article",
                "title": "女學",
                "content_sha256": "c" * 64,
                "approved_by": "historian-a",
                "segmentation_selection_id": UUID(
                    "00000000-0000-0000-0000-000000000042"
                ),
                "segmentation_review_id": UUID(
                    "00000000-0000-0000-0000-000000000043"
                ),
                "span_sequence_number": 0,
                "span_text_start": 0,
                "span_text_end": 2,
                "span_role": "body",
                "selected_text": "女子",
                "selected_text_sha256": hashlib.sha256(
                    "女子".encode("utf-8")
                ).hexdigest(),
                "selected_text_version_id": UUID(
                    "00000000-0000-0000-0000-000000000044"
                ),
                "text_selection_id": UUID(
                    "00000000-0000-0000-0000-000000000045"
                ),
                "alignment_operations": None,
                "expected_span_count": 2,
            }
        )

        # When/Then: export fails instead of presenting a partial article as complete.
        with self.assertRaisesRegex(ValueError, "cardinality"):
            _ = build_coherent_unit_documents([base])
