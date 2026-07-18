from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wic_history.render_samples import djvulibre_version, render_pdf_pages, sha256_file, write_results


class RenderSampleTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_render_pdf_page(self):
        import fitz

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.pdf"
            document = fitz.open()
            page = document.new_page(width=200, height=300)
            page.insert_text((30, 50), "screening page")
            document.save(source)
            document.close()

            candidates = [
                {
                    "sample_id": "v001-p0001",
                    "source_uri": "s3://bucket/sample.pdf",
                    "volume_number": "1",
                    "publication_year": "1872",
                    "page_number": "1",
                }
            ]
            results = render_pdf_pages(source, candidates, root / "output", 1, 72, 85)
            self.assertEqual(results[0]["status"], "rendered")
            self.assertTrue(Path(results[0]["render_path"]).exists())
            self.assertEqual(results[0]["render_width"], 200)

    def test_write_results_merges_incremental_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = {"sample_id": "v001-p0001", "volume_number": 1, "page_number": 1, "status": "unsupported_renderer"}
            second = {"sample_id": "v002-p0001", "volume_number": 2, "page_number": 1, "status": "unsupported_renderer"}
            write_results(root, [first])
            summary = write_results(root, [second])
            self.assertEqual(summary["candidate_count"], 2)
            self.assertEqual(summary["updated_candidate_count"], 1)

    def test_detects_installed_djvulibre_version(self):
        import shutil

        ddjvu = shutil.which("ddjvu")
        if not ddjvu:
            self.skipTest("DjVuLibre is not installed")
        self.assertRegex(djvulibre_version(ddjvu), r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
