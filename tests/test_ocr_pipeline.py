from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from wic_history.ocr_pipeline import (
    DetectedLine,
    create_ocr_artifact,
    deduplicate_lines,
    tile_bounds,
)


class OCRPipelineTests(unittest.TestCase):
    def test_tiles_cover_edges(self):
        tiles = tile_bounds(2471, 3584, tile_size=1200, overlap=120)
        self.assertEqual((tiles[0].left, tiles[0].top), (0, 0))
        self.assertEqual((tiles[-1].right, tiles[-1].bottom), (2471, 3584))
        self.assertGreater(len(tiles), 1)

    def test_deduplicates_same_text_and_overlap_only(self):
        lines = [
            DetectedLine("女子", 0.8, ((0, 0), (10, 0), (10, 10), (0, 10)), {}),
            DetectedLine("女子", 0.9, ((1, 1), (11, 1), (11, 11), (1, 11)), {}),
            DetectedLine("學校", 0.7, ((1, 1), (11, 1), (11, 11), (1, 11)), {}),
        ]
        kept = deduplicate_lines(lines)
        self.assertEqual(len(kept), 2)
        self.assertIn(0.9, [line.confidence for line in kept])

    def test_artifact_preserves_absolute_coordinates_and_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "page.jpg"
            Image.new("RGB", (150, 100), "white").save(image_path)

            def fake_predictor(image: Image.Image):
                return [
                    DetectedLine(
                        "女學生",
                        0.95,
                        ((5, 6), (45, 6), (45, 18), (5, 18)),
                        {"fixture": True},
                    )
                ]

            artifact = create_ocr_artifact(
                image_path=image_path,
                source_uri="s3://bucket/volume.pdf",
                page_number=3,
                volume_number=219,
                publication_year=1925,
                predictor=fake_predictor,
                tile_size=200,
                overlap=20,
                language="ch",
                screening_derivative=True,
                isolated_tiles=False,
                page_detector=None,
            )
            self.assertEqual(artifact.regions[0].raw_text, "女學生")
            self.assertEqual(artifact.regions[0].polygon.points[0].x, 5)
            self.assertIn("technical smoke test", artifact.warnings[-1])


if __name__ == "__main__":
    unittest.main()
