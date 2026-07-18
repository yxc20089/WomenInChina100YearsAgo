from __future__ import annotations

import io
import unittest
from datetime import datetime, timezone

from wic_history.corpus_manifest import (
    inspect_object,
    extract_djvu_page_count,
    parse_volume_number,
    parse_classic_xref,
    potential_duplicate_groups,
    publication_year_for_volume,
    summarize,
    validate_bytes,
)


class FakeS3:
    def __init__(self, content: bytes):
        self.content = content

    def get_object(self, *, Bucket: str, Key: str, Range: str):
        bounds = Range.removeprefix("bytes=").split("-")
        start, end = int(bounds[0]), int(bounds[1])
        return {"Body": io.BytesIO(self.content[start : end + 1])}


class CorpusManifestTests(unittest.TestCase):
    def test_year_mapping_boundaries(self):
        self.assertEqual(publication_year_for_volume(1), 1872)
        self.assertEqual(publication_year_for_volume(199), 1924)
        self.assertEqual(publication_year_for_volume(230), 1926)
        self.assertEqual(publication_year_for_volume(400), 1949)
        self.assertIsNone(publication_year_for_volume(401))

    def test_parse_volume_number(self):
        self.assertEqual(parse_volume_number("sb_raw/申报影印本220.pdf"), 220)
        self.assertEqual(parse_volume_number("sb_raw/申报影印本94.djvu"), 94)
        self.assertIsNone(parse_volume_number("sb_raw/申报分年一览表.jpg"))

    def test_pdf_validation(self):
        status, checks, issues = validate_bytes(
            ".pdf", b"%PDF-1.7\n", b"some objects\nstartxref\n42\n%%EOF\n"
        )
        self.assertEqual(status, "ok_fast_checks")
        self.assertEqual(checks, {"signature_valid": True, "trailer_valid": True})
        self.assertEqual(issues, [])

    def test_truncated_pdf_validation(self):
        status, checks, issues = validate_bytes(".pdf", b"%PDF-1.5\n", b"endstream")
        self.assertEqual(status, "suspect")
        self.assertTrue(checks["signature_valid"])
        self.assertIn("missing_pdf_trailer_or_eof", issues)

    def test_parse_classic_xref(self):
        data = (
            b"xref\n0 3\n"
            b"0000000000 65535 f \n"
            b"0000000017 00000 n \n"
            b"0000000042 00000 n \n"
            b"trailer\n<< /Size 3 /Root 2 0 R >>\nstartxref\n99\n%%EOF"
        )
        offsets, root_id = parse_classic_xref(data)
        self.assertEqual(offsets, {1: 17, 2: 42})
        self.assertEqual(root_id, 2)

    def test_bundled_djvu_page_count(self):
        header = b"AT&TFORM" + b"\x00\x00\x10\x00" + b"DJVM" + b"DIRM" + b"\x00\x00\x00\x20" + b"\x81\x03\x49"
        count, status, issues = extract_djvu_page_count(header)
        self.assertEqual(count, 841)
        self.assertEqual(status, "read_from_djvu_directory")
        self.assertEqual(issues, [])

    def test_inspect_object_and_summary(self):
        content = b"%PDF-1.5\n" + (b"x" * 100) + b"startxref\n10\n%%EOF"
        obj = {
            "Key": "sb_raw/申报影印本199.pdf",
            "Size": len(content),
            "ETag": '"0123456789abcdef0123456789abcdef"',
            "LastModified": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "StorageClass": "STANDARD",
        }
        record = inspect_object(FakeS3(content), "bucket", obj)
        self.assertEqual(record.volume_number, 199)
        self.assertEqual(record.publication_year, 1924)
        self.assertEqual(record.integrity_status, "ok_fast_checks")
        self.assertTrue(record.etag_is_simple_md5_candidate)

        summary = summarize([record], "bucket", "sb_raw/")
        self.assertEqual(summary["object_count"], 1)
        self.assertEqual(summary["integrity_status_counts"], {"ok_fast_checks": 1})
        self.assertIn(200, summary["missing_volume_numbers"])

    def test_potential_duplicate_is_only_candidate(self):
        content = b"%PDF-1.5\nstartxref\n1\n%%EOF"
        base = {
            "Size": len(content),
            "ETag": '"0123456789abcdef0123456789abcdef"',
            "LastModified": datetime(2025, 1, 1, tzinfo=timezone.utc),
        }
        one = inspect_object(FakeS3(content), "bucket", {**base, "Key": "sb_raw/申报影印本1.pdf"})
        two = inspect_object(FakeS3(content), "bucket", {**base, "Key": "sb_raw/申报影印本2.pdf"})
        groups = potential_duplicate_groups([one, two])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["classification"], "potential_duplicate_not_content_verified")


if __name__ == "__main__":
    unittest.main()
