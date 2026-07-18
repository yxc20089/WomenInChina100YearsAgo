from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from wic_history.gold_render import (
    extract_pdf_page,
    pilot_candidates,
    selected_candidates,
    write_lossless_results,
)


def candidate() -> dict[str, str]:
    return {
        "sample_id": "v001-p0001",
        "source_uri": "s3://bucket/volume-1.pdf",
        "source_key": "prefix/volume-1.pdf",
        "volume_number": "1",
        "publication_year": "1872",
        "page_number": "1",
    }


def png_bytes(color: int = 255) -> bytes:
    image = Image.new("L", (20, 30), color=color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 3, 8, 12), fill=0)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def make_pdf(path: Path, *, composite: bool = False) -> None:
    import fitz

    document = fitz.open()
    page = document.new_page(width=200, height=300)
    if composite:
        page.insert_image(fitz.Rect(0, 0, 100, 300), stream=png_bytes())
        page.insert_image(fitz.Rect(100, 0, 200, 300), stream=png_bytes(200))
    else:
        page.insert_image(page.rect, stream=png_bytes())
    document.save(path)
    document.close()


class GoldRenderTests(unittest.TestCase):
    def test_selection_requires_complete_named_review(self):
        annotations = {
            "schema_version": "1.0",
            "annotations": {
                "v001-p0001": {
                    "gold_status": "include",
                    "page_genre": "news_editorial",
                    "layout": "vertical",
                    "scan_quality": "clean",
                    "women_relevance": "explicit",
                    "reviewer": "historian-a",
                    "reviewed_at": "2026-07-18T00:00:00Z",
                }
            },
        }
        selected = selected_candidates([candidate()], annotations)
        self.assertEqual(selected[0]["selection"]["gold_status"], "include")
        annotations["annotations"]["v001-p0001"]["reviewer"] = ""
        with self.assertRaises(ValueError):
            selected_candidates([candidate()], annotations)

    def test_pilot_is_explicitly_not_gold(self):
        selected = pilot_candidates([candidate()], ["v001-p0001"])
        self.assertEqual(selected[0]["selection"]["gold_status"], "non_gold_pilot")

    def test_extracts_full_page_embedded_pixels_without_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            output = root / "output" / "page.png"
            make_pdf(source)
            result = extract_pdf_page(source, 1, output, 1)
            self.assertEqual(result["render_method"], "direct_embedded_raster_decode")
            self.assertEqual(result["geometric_transform"], "none")
            self.assertEqual(
                result["pixel_encoding_transform"],
                "source_codec_decode_then_lossless_png_reencode",
            )
            self.assertEqual((result["render_width"], result["render_height"]), (20, 30))
            self.assertEqual(len(result["decoded_pixel_sha256"]), 64)
            with Image.open(output) as image:
                self.assertEqual(image.size, (20, 30))

    def test_refuses_composited_pdf_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            make_pdf(source, composite=True)
            with self.assertRaisesRegex(ValueError, "composited visible content"):
                extract_pdf_page(source, 1, root / "page.png", 1)

    def test_summary_never_counts_pilot_as_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "images" / "v001" / "p0001.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"png")
            summary = write_lossless_results(
                root,
                [
                    {
                        "sample_id": "v001-p0001",
                        "volume_number": 1,
                        "page_number": 1,
                        "status": "rendered",
                        "render_path": str(image_path),
                        "selection": {"gold_status": "non_gold_pilot"},
                    }
                ],
            )
            self.assertEqual(summary["gold_pages"], 0)
            self.assertEqual(summary["non_gold_pilot_pages"], 1)


if __name__ == "__main__":
    unittest.main()
