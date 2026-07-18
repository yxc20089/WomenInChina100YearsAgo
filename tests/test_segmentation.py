import unittest
from uuid import UUID

from wic_history.segmentation import (
    ACTIVE_REGIONS_SQL,
    group_reading_order_windows,
    validate_span_coverage,
)


def row(index: int, text: str) -> dict:
    return {
        "region_id": UUID(f"00000000-0000-0000-0000-{index:012d}"),
        "reading_order": index,
        "raw_text": text,
        "normalized_text": text,
    }


class SegmentationTests(unittest.TestCase):
    def test_windows_are_ordered_bounded_and_complete(self):
        rows = [row(3, "三三"), row(1, "一一"), row(2, "二二"), row(4, "四四")]

        windows = group_reading_order_windows(rows, max_regions=2, max_characters=20)

        self.assertEqual([[item["reading_order"] for item in unit] for unit in windows], [[1, 2], [3, 4]])
        self.assertEqual(sum(len(unit) for unit in windows), len(rows))

    def test_character_limit_starts_a_new_nonempty_window(self):
        windows = group_reading_order_windows(
            [row(1, "女子學校"), row(2, "教育"), row(3, "新聞")],
            max_regions=10,
            max_characters=6,
        )
        self.assertEqual([[1], [2, 3]], [[item["reading_order"] for item in unit] for unit in windows])

    def test_proposals_bind_the_active_ocr_selection(self):
        self.assertIn("selection.selection_id AS source_ocr_selection_id", ACTIVE_REGIONS_SQL)
        self.assertIn("selection.superseded_at IS NULL", ACTIVE_REGIONS_SQL)

    def test_invalid_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            group_reading_order_windows([], max_regions=0, max_characters=10)

    def test_character_split_can_cross_two_units_without_losing_evidence(self):
        region = row(1, "女子學校")
        validate_span_coverage(
            [region],
            [
                {"region_id": region["region_id"], "text_start": 0, "text_end": 2},
                {"region_id": region["region_id"], "text_start": 2, "text_end": 4},
            ],
        )

    def test_gap_or_overlap_is_rejected(self):
        region = row(1, "女子學校")
        with self.assertRaises(ValueError):
            validate_span_coverage(
                [region],
                [
                    {"region_id": region["region_id"], "text_start": 0, "text_end": 2},
                    {"region_id": region["region_id"], "text_start": 3, "text_end": 4},
                ],
            )

    def test_empty_region_requires_one_accounting_span(self):
        region = row(1, "")
        with self.assertRaises(ValueError):
            validate_span_coverage(
                [region],
                [
                    {"region_id": region["region_id"], "text_start": 0, "text_end": 0},
                    {"region_id": region["region_id"], "text_start": 0, "text_end": 0},
                ],
            )
