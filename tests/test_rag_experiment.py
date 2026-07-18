import unittest
from uuid import UUID

from wic_history.rag_experiment import (
    EXPORT_SQL,
    GRAPHRAG_REVISION,
    LIGHTRAG_REVISION,
    build_documents,
)


def _row(region_id: str, page_id: str, text: str, reading_order: int) -> dict:
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
