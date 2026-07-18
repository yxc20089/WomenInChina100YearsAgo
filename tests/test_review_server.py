from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wic_history.review_server import AnnotationStore


class AnnotationStoreTests(unittest.TestCase):
    def test_round_trip_and_atomic_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            store = AnnotationStore(path)
            annotation = store.update(
                "v219-p0308",
                {
                    "page_genre": "advertisement_classified",
                    "layout": "mixed",
                    "scan_quality": "clean",
                    "women_relevance": "explicit",
                    "gold_status": "include",
                    "reviewer": "tester",
                    "notes": "film and consumer-culture page",
                },
            )
            self.assertEqual(annotation["gold_status"], "include")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("v219-p0308", data["annotations"])
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_rejects_invalid_controlled_value(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AnnotationStore(Path(directory) / "annotations.json")
            with self.assertRaisesRegex(ValueError, "Invalid layout"):
                store.update("sample", {"layout": "diagonal"})


if __name__ == "__main__":
    unittest.main()
