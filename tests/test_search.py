from __future__ import annotations

import unittest
from uuid import uuid4

from wic_history.search import region_document, region_index_body


class SearchProjectionTests(unittest.TestCase):
    def test_mapping_has_cjk_text_and_versioned_dense_vector(self):
        properties = region_index_body()["mappings"]["properties"]
        self.assertEqual(properties["raw_text"]["analyzer"], "cjk")
        self.assertEqual(properties["embedding"]["dimension"], 1024)
        self.assertEqual(region_index_body()["mappings"]["dynamic"], "strict")

    def test_document_keeps_citation_and_provenance(self):
        region_id = uuid4()
        row = {
            "region_id": region_id,
            "page_id": uuid4(),
            "run_id": uuid4(),
            "source_uri": "s3://bucket/volume.pdf",
            "source_image_uri": "s3://bucket/page.png",
            "volume_number": 3,
            "publication_year": 1874,
            "page_number": 12,
            "reading_order": 4,
            "region_kind": "text",
            "raw_text": "女子學校",
            "normalized_text": "女子學校",
            "confidence": 0.9,
            "language": "zh-Hant",
            "direction": "vertical",
            "polygon": {"points": [{"x": 1, "y": 2}, {"x": 3, "y": 2}, {"x": 3, "y": 4}]},
            "page_warnings": [],
            "ocr_model": "fixture",
            "ocr_model_revision": "1",
            "embedding_model": None,
            "embedding_model_revision": None,
            "embedding_text": None,
            "entity_ids": [],
            "claim_ids": [],
        }
        document = region_document(row, "2026-01-01T00:00:00Z")
        self.assertEqual(document["region_id"], str(region_id))
        self.assertEqual(document["source_uri"], "s3://bucket/volume.pdf")
        self.assertEqual(document["polygon"], row["polygon"])


if __name__ == "__main__":
    unittest.main()
