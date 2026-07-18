from __future__ import annotations

import unittest

from wic_history.benchmark_sample import create_plan, evenly_spaced_pages, select_volume


def record(volume: int, year: int, pages: int, extension: str = ".pdf"):
    return {
        "volume_number": volume,
        "publication_year": year,
        "page_count": pages,
        "integrity_status": "ok_fast_checks",
        "extension": extension,
        "source_uri": f"s3://bucket/volume-{volume}{extension}",
        "key": f"volume-{volume}{extension}",
    }


class BenchmarkSampleTests(unittest.TestCase):
    def test_evenly_spaced_pages_include_edges(self):
        pages = evenly_spaced_pages(100, 5)
        self.assertEqual(pages, [1, 26, 51, 75, 100])

    def test_evenly_spaced_pages_are_unique(self):
        pages = evenly_spaced_pages(599, 50)
        self.assertEqual(len(pages), 50)
        self.assertEqual(len(set(pages)), 50)
        self.assertEqual((pages[0], pages[-1]), (1, 599))

    def test_selects_nearest_median_volume(self):
        selected = select_volume(
            [record(1, 1924, 500), record(2, 1924, 700), record(3, 1924, 900)],
            "pilot",
            1924,
            ".pdf",
        )
        self.assertEqual(selected.record["volume_number"], 2)

    def test_create_plan(self):
        rows, selections = create_plan(
            [record(1, 1924, 10), record(2, 1908, 12, ".djvu")],
            strata=(("pilot", 1924, ".pdf"), ("djvu", 1908, ".djvu")),
            pages_per_volume=3,
        )
        self.assertEqual(len(selections), 2)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["gold_status"], "not_reviewed")


if __name__ == "__main__":
    unittest.main()

