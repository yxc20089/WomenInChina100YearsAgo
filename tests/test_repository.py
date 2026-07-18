from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wic_history.evidence import OCRPageArtifact
from wic_history.repository import _ocr_evidence_tier, read_jsonl


class RepositoryTests(unittest.TestCase):
    def test_legacy_smoke_artifact_retains_screening_tier(self):
        artifact = OCRPageArtifact.model_validate_json(
            Path("artifacts/ocr-smoke/v219-p0308.ppocrv6.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(_ocr_evidence_tier(artifact), "screening_derivative")

    def test_jsonl_reader_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text('{"volume_number": 1}\n\n{"volume_number": 2}\n', encoding="utf-8")
            self.assertEqual(
                [item["volume_number"] for item in read_jsonl(path)],
                [1, 2],
            )

    def test_jsonl_reader_reports_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text('{}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"manifest\.jsonl:2"):
                list(read_jsonl(path))


if __name__ == "__main__":
    unittest.main()
