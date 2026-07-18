from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from wic_history.ocr_pipeline import (
    DetectedLine,
    create_ocr_artifact,
    deduplicate_lines,
    resolve_render_provenance,
    tile_bounds,
)
from wic_history.render_samples import sha256_file


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

    def test_lossless_manifest_supplies_source_hash_and_evidence_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("L", (20, 30), "white").save(image_path)
            manifest_path = root / "manifest.jsonl"
            source_sha256 = "b" * 64
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "rendered",
                        "render_path": str(image_path),
                        "render_sha256": sha256_file(image_path),
                        "source_object_sha256": source_sha256,
                        "source_uri": "s3://bucket/volume.pdf",
                        "volume_number": 219,
                        "publication_year": 1925,
                        "page_number": 308,
                        "selection": {"gold_status": "non_gold_pilot"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            resolved_hash, tier = resolve_render_provenance(
                image_path,
                manifest_path,
                source_uri="s3://bucket/volume.pdf",
                page_number=308,
                volume_number=219,
                publication_year=1925,
            )
            self.assertEqual(resolved_hash, source_sha256)
            self.assertEqual(tier, "non_gold_lossless_pilot")


if __name__ == "__main__":
    unittest.main()
