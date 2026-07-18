from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from wic_history.evidence import (
    OCRPageArtifact,
    OCRRegion,
    Point,
    Polygon,
    ProcessingRun,
    RegionKind,
    RunKind,
    SourcePointer,
)
from wic_history.ocr_gold import (
    GoldOCRRegion,
    OCRGoldAdjudication,
    OCRGoldPage,
    OCRGoldSet,
    OCRReviewerAnnotation,
    polygon_iou,
    score_ocr_artifacts,
)


def box(left: float, top: float, right: float, bottom: float) -> Polygon:
    return Polygon(
        points=[
            Point(x=left, y=top),
            Point(x=right, y=top),
            Point(x=right, y=bottom),
            Point(x=left, y=bottom),
        ]
    )


def gold_regions() -> list[GoldOCRRegion]:
    return [
        GoldOCRRegion(
            region_id=uuid4(),
            kind=RegionKind.TEXT,
            polygon=box(0, 0, 40, 20),
            reading_order=0,
            transcription="王氏",
            direction="vertical",
        ),
        GoldOCRRegion(
            region_id=uuid4(),
            kind=RegionKind.TEXT,
            polygon=box(50, 0, 90, 20),
            reading_order=1,
            transcription="女塾",
            direction="vertical",
        ),
    ]


def gold_set() -> OCRGoldSet:
    now = datetime(2026, 7, 18, tzinfo=timezone.utc)
    return OCRGoldSet(
        dataset_id="ocr-unit-gold",
        created_at=now,
        pages=[
            OCRGoldPage(
                page_id="v1-p1",
                source=SourcePointer(
                    source_uri="s3://example/v1.pdf",
                    volume_number=1,
                    publication_year=1925,
                    page_number=1,
                ),
                image_uri="artifacts/gold/v1-p1.png",
                image_sha256="a" * 64,
                width=100,
                height=100,
                dpi=300,
                page_genre="news_editorial",
                layout="vertical",
                scan_quality="poor",
                reviews=[
                    OCRReviewerAnnotation(
                        reviewer="reviewer-a",
                        regions=gold_regions(),
                        annotated_at=now,
                    ),
                    OCRReviewerAnnotation(
                        reviewer="reviewer-b",
                        regions=gold_regions(),
                        annotated_at=now,
                    ),
                ],
                adjudication=OCRGoldAdjudication(
                    adjudicator="adjudicator-c",
                    regions=gold_regions(),
                    adjudicated_at=now,
                ),
            )
        ],
    )


def prediction() -> OCRPageArtifact:
    now = datetime(2026, 7, 18, tzinfo=timezone.utc)
    run = ProcessingRun(
        kind=RunKind.OCR,
        engine="unit",
        model_name="unit-ocr",
        model_revision="unit-revision",
        started_at=now,
        completed_at=now,
    )
    return OCRPageArtifact(
        source=SourcePointer(
            source_uri="s3://example/v1.pdf",
            volume_number=1,
            publication_year=1925,
            page_number=1,
        ),
        image_uri="artifacts/gold/v1-p1.png",
        image_sha256="a" * 64,
        width=100,
        height=100,
        dpi=300,
        run=run,
        regions=[
            OCRRegion(
                kind=RegionKind.TEXT,
                polygon=box(0, 0, 40, 20),
                reading_order=1,
                raw_text="王民",
                direction="vertical",
            ),
            OCRRegion(
                kind=RegionKind.ADVERTISEMENT,
                polygon=box(50, 0, 90, 20),
                reading_order=0,
                raw_text="女塾",
                direction="vertical",
            ),
            OCRRegion(
                kind=RegionKind.TEXT,
                polygon=box(105, 0, 115, 10),
                reading_order=2,
                raw_text="錯",
                direction="horizontal",
            ),
        ],
    )


class OCRGoldTests(unittest.TestCase):
    def test_convex_polygon_iou(self):
        self.assertAlmostEqual(polygon_iou(box(0, 0, 10, 10), box(5, 0, 15, 10)), 1 / 3)

    def test_scores_detection_text_layout_and_reading_order(self):
        report = score_ocr_artifacts(gold_set(), [prediction()])
        overall = report["overall"]
        self.assertAlmostEqual(overall["region_detection"]["precision"], 2 / 3)
        self.assertEqual(overall["region_detection"]["recall"], 1.0)
        self.assertEqual(overall["invalid_geometry_predictions"], 1)
        self.assertEqual(overall["mean_matched_iou"], 1.0)
        self.assertEqual(overall["matched_region_cer"], 0.25)
        self.assertEqual(overall["region_kind_accuracy"], 0.5)
        self.assertEqual(overall["text_direction_accuracy"], 1.0)
        self.assertEqual(overall["reading_order_pair_accuracy"], 0.0)
        self.assertEqual(report["by_scan_quality"]["poor"]["pages"], 1)

    def test_rejects_prediction_from_a_different_image(self):
        changed = prediction().model_copy(update={"image_sha256": "b" * 64})
        with self.assertRaises(ValueError):
            score_ocr_artifacts(gold_set(), [changed])

    def test_rejects_mixed_model_revisions(self):
        second = prediction()
        second.run.model_revision = "different-revision"
        second.source.page_number = 2
        gold = gold_set()
        second_page = gold.pages[0].model_copy(deep=True)
        second_page.page_id = "v1-p2"
        second_page.source.page_number = 2
        gold.pages.append(second_page)
        with self.assertRaises(ValueError):
            score_ocr_artifacts(gold, [prediction(), second])

    def test_gold_requires_distinct_reviewers(self):
        data = gold_set().model_dump(mode="json")
        data["pages"][0]["reviews"][1]["reviewer"] = "reviewer-a"
        with self.assertRaises(ValueError):
            OCRGoldSet.model_validate(data)


if __name__ == "__main__":
    unittest.main()
