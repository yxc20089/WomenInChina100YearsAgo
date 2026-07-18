from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from wic_history.gold_render import (
    extract_pdf_page,
    ingestion_candidate,
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


def make_image_mask_pdf(path: Path) -> bytes:
    """Create one full-page PDF stencil whose extracted bits invert when painted."""
    import fitz

    mask_bytes = bytes(
        [
            0b11110000,
            0b11110000,
            0b00001111,
            0b00001111,
            0b10101010,
            0b01010101,
            0b11111111,
            0b00000000,
        ]
    )
    document = fitz.open()
    page = document.new_page(width=8, height=8)
    image_xref = document.get_new_xref()
    document.update_object(
        image_xref,
        "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 "
        "/ImageMask true /BitsPerComponent 1 >>",
    )
    document.update_stream(image_xref, mask_bytes, compress=True)
    resource_kind, resource_value = document.xref_get_key(page.xref, "Resources")
    if resource_kind != "xref":
        raise AssertionError("test PDF page lacks an indirect resource dictionary")
    resource_xref = int(resource_value.split()[0])
    document.update_object(
        resource_xref,
        f"<< /XObject << /Im1 {image_xref} 0 R >> >>",
    )
    contents_xref = document.get_new_xref()
    document.update_object(contents_xref, "<< >>")
    document.update_stream(
        contents_xref,
        b"0 g q 8 0 0 8 0 0 cm /Im1 Do Q",
        compress=True,
    )
    document.xref_set_key(page.xref, "Contents", f"{contents_xref} 0 R")
    document.save(path)
    document.close()
    return mask_bytes


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

    def test_ingestion_candidate_is_explicitly_unreviewed(self):
        selected = ingestion_candidate(
            source_uri="s3://bucket/volume-1.pdf",
            source_key="prefix/volume-1.pdf",
            volume_number=1,
            publication_year=1872,
            page_number=1,
            job_id="00000000-0000-0000-0000-000000000001",
        )
        self.assertEqual(
            selected["selection"]["gold_status"], "unreviewed_ingestion"
        )
        self.assertIsNone(selected["selection"]["reviewer"])

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

    def test_applies_pdf_imagemask_paint_semantics_at_native_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mask.pdf"
            output = root / "page.png"
            make_image_mask_pdf(source)

            result = extract_pdf_page(source, 1, output, 1)

            self.assertEqual(
                result["render_method"],
                "native_resolution_pdf_imagemask_composite",
            )
            self.assertEqual(
                result["pixel_encoding_transform"],
                "source_codec_decode_then_pdf_imagemask_paint_then_lossless_png_reencode",
            )
            self.assertTrue(result["source_image_is_mask"])
            self.assertEqual(len(result["source_mask_decoded_pixel_sha256"]), 64)
            self.assertEqual((result["render_width"], result["render_height"]), (8, 8))

            import fitz

            with fitz.open(source) as document:
                xref = document[0].get_images(full=True)[0][0]
                extracted = Image.open(io.BytesIO(document.extract_image(xref)["image"]))
                extracted_pixels = bytes(extracted.convert("L").get_flattened_data())
            with Image.open(output) as painted:
                painted_pixels = bytes(painted.convert("L").get_flattened_data())
            self.assertEqual(
                painted_pixels,
                bytes(255 - value for value in extracted_pixels),
            )

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
            self.assertEqual(summary["unreviewed_ingestion_pages"], 0)


if __name__ == "__main__":
    unittest.main()
