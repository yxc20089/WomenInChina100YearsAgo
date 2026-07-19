"""Validate adjudicated OCR/layout gold pages and score OCR page artifacts."""

from __future__ import annotations

import argparse
import json
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence
from uuid import UUID

from pydantic import Field, model_validator

from .evidence import (
    OCRPageArtifact,
    Polygon,
    RegionKind,
    RunKind,
    SourcePointer,
    StrictModel,
)
from .ner_gold import character_error_distance


def character_error_counts(reference: str, prediction: str) -> dict[str, int]:
    """Return one deterministic minimum-edit substitution/deletion/insertion split.

    Deletions are printed characters missing from OCR; insertions are OCR
    characters absent from the print. Equal-cost alignments prefer a
    substitution, then deletion, then insertion so reports are reproducible.
    """
    rows = len(reference) + 1
    columns = len(prediction) + 1
    distance = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        distance[row][0] = row
    for column in range(columns):
        distance[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            distance[row][column] = min(
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
                distance[row - 1][column - 1]
                + (reference[row - 1] != prediction[column - 1]),
            )
    substitutions = deletions = insertions = 0
    row, column = len(reference), len(prediction)
    while row or column:
        if (
            row
            and column
            and reference[row - 1] == prediction[column - 1]
            and distance[row][column] == distance[row - 1][column - 1]
        ):
            row -= 1
            column -= 1
        elif (
            row
            and column
            and distance[row][column] == distance[row - 1][column - 1] + 1
        ):
            substitutions += 1
            row -= 1
            column -= 1
        elif row and distance[row][column] == distance[row - 1][column] + 1:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1
    return {
        "substitutions": substitutions,
        "missing_characters": deletions,
        "hallucinated_characters": insertions,
        "total_errors": substitutions + deletions + insertions,
    }


class GoldOCRRegion(StrictModel):
    region_id: UUID
    kind: RegionKind
    polygon: Polygon
    reading_order: int = Field(ge=0)
    transcription: str
    direction: Literal["vertical", "horizontal", "mixed", "unknown"]
    note: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_transcription(self) -> "GoldOCRRegion":
        if unicodedata.normalize("NFC", self.transcription) != self.transcription:
            raise ValueError("gold transcriptions must be Unicode NFC")
        return self


class OCRReviewerAnnotation(StrictModel):
    reviewer: str = Field(min_length=1, max_length=200)
    regions: list[GoldOCRRegion] = Field(default_factory=list)
    annotated_at: datetime
    notes: str | None = Field(default=None, max_length=5000)


class OCRGoldAdjudication(StrictModel):
    adjudicator: str = Field(min_length=1, max_length=200)
    regions: list[GoldOCRRegion] = Field(default_factory=list)
    adjudicated_at: datetime
    notes: str | None = Field(default=None, max_length=5000)


def _signed_area(points: list[tuple[float, float]]) -> float:
    return sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    ) / 2


def polygon_area(polygon: Polygon) -> float:
    return abs(_signed_area([(point.x, point.y) for point in polygon.points]))


def _is_convex(polygon: Polygon) -> bool:
    points = [(point.x, point.y) for point in polygon.points]
    if polygon_area(polygon) <= 1e-9:
        return False
    signs = []
    for index in range(len(points)):
        one = points[index]
        two = points[(index + 1) % len(points)]
        three = points[(index + 2) % len(points)]
        cross = (two[0] - one[0]) * (three[1] - two[1]) - (
            two[1] - one[1]
        ) * (three[0] - two[0])
        if abs(cross) > 1e-9:
            signs.append(cross > 0)
    return bool(signs) and all(sign == signs[0] for sign in signs)


def _validate_regions(
    regions: list[GoldOCRRegion], width: int, height: int, label: str
) -> None:
    region_ids = [region.region_id for region in regions]
    reading_orders = [region.reading_order for region in regions]
    if len(set(region_ids)) != len(region_ids):
        raise ValueError(f"{label} region IDs must be unique")
    if len(set(reading_orders)) != len(reading_orders):
        raise ValueError(f"{label} reading-order values must be unique")
    for index, region in enumerate(regions):
        if not _is_convex(region.polygon):
            raise ValueError(f"{label} region {index} must have positive convex geometry")
        if any(
            point.x > width or point.y > height for point in region.polygon.points
        ):
            raise ValueError(f"{label} region {index} falls outside the source image")


class OCRGoldPage(StrictModel):
    page_id: str = Field(min_length=1, max_length=300)
    source: SourcePointer
    image_uri: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    dpi: int | None = Field(default=None, gt=0)
    page_genre: Literal[
        "news_editorial",
        "advertisement_classified",
        "mixed",
        "photograph_caption",
        "table_market_schedule",
        "front_matter_index",
        "blank_other",
    ]
    layout: Literal["vertical", "horizontal", "mixed", "unknown"]
    scan_quality: Literal["clean", "moderate", "poor", "unusable"]
    reviews: list[OCRReviewerAnnotation] = Field(min_length=2)
    adjudication: OCRGoldAdjudication

    @model_validator(mode="after")
    def validate_page(self) -> "OCRGoldPage":
        reviewers = [review.reviewer for review in self.reviews]
        if len(set(reviewers)) != len(reviewers):
            raise ValueError("gold pages require distinct independent reviewers")
        for index, review in enumerate(self.reviews):
            _validate_regions(review.regions, self.width, self.height, f"review {index}")
        _validate_regions(
            self.adjudication.regions, self.width, self.height, "adjudication"
        )
        return self


class OCRGoldSet(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=300)
    created_at: datetime
    pages: list[OCRGoldPage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_pages(self) -> "OCRGoldSet":
        page_ids = [page.page_id for page in self.pages]
        page_keys = [(page.source.source_uri, page.source.page_number) for page in self.pages]
        if len(set(page_ids)) != len(page_ids):
            raise ValueError("gold page IDs must be unique")
        if len(set(page_keys)) != len(page_keys):
            raise ValueError("gold source/page identities must be unique")
        return self


def _cross(
    one: tuple[float, float],
    two: tuple[float, float],
    point: tuple[float, float],
) -> float:
    return (two[0] - one[0]) * (point[1] - one[1]) - (
        two[1] - one[1]
    ) * (point[0] - one[0])


def _line_intersection(
    one: tuple[float, float],
    two: tuple[float, float],
    three: tuple[float, float],
    four: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = one
    x2, y2 = two
    x3, y3 = three
    x4, y4 = four
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) <= 1e-12:
        return two
    determinant_one = x1 * y2 - y1 * x2
    determinant_two = x3 * y4 - y3 * x4
    return (
        (
            determinant_one * (x3 - x4)
            - (x1 - x2) * determinant_two
        )
        / denominator,
        (
            determinant_one * (y3 - y4)
            - (y1 - y2) * determinant_two
        )
        / denominator,
    )


def _convex_intersection(
    subject: list[tuple[float, float]], clip: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    output = subject
    orientation = 1 if _signed_area(clip) > 0 else -1
    for clip_start, clip_end in zip(clip, clip[1:] + clip[:1]):
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = orientation * _cross(clip_start, clip_end, previous) >= -1e-9
        for current in input_points:
            current_inside = orientation * _cross(clip_start, clip_end, current) >= -1e-9
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, clip_start, clip_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, clip_start, clip_end)
                )
            previous = current
            previous_inside = current_inside
    return output


def polygon_intersection_area(one: Polygon, two: Polygon) -> float:
    subject = [(point.x, point.y) for point in one.points]
    clip = [(point.x, point.y) for point in two.points]
    intersection = _convex_intersection(subject, clip)
    return abs(_signed_area(intersection)) if len(intersection) >= 3 else 0.0


def polygon_iou(one: Polygon, two: Polygon) -> float:
    intersection = polygon_intersection_area(one, two)
    union = polygon_area(one) + polygon_area(two) - intersection
    return intersection / union if union > 0 else 0.0


def _detection_metrics(matched: int, predicted: int, expected: int) -> dict[str, Any]:
    precision = matched / predicted if predicted else (1.0 if expected == 0 else 0.0)
    recall = matched / expected if expected else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": matched,
        "false_positive": predicted - matched,
        "false_negative": expected - matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _prediction_geometry_valid(
    polygon: Polygon, width: int, height: int
) -> bool:
    return (
        _is_convex(polygon)
        and all(point.x <= width and point.y <= height for point in polygon.points)
    )


def _score_page(
    page: OCRGoldPage,
    prediction: OCRPageArtifact | None,
    iou_threshold: float,
) -> dict[str, Any]:
    gold_regions = page.adjudication.regions
    predicted_regions = prediction.regions if prediction else []
    valid_prediction_indexes = [
        index
        for index, region in enumerate(predicted_regions)
        if _prediction_geometry_valid(region.polygon, page.width, page.height)
    ]
    candidates = sorted(
        [
            (
                polygon_iou(
                    gold.polygon, predicted_regions[predicted_index].polygon
                ),
                gold_index,
                predicted_index,
            )
            for gold_index, gold in enumerate(gold_regions)
            for predicted_index in valid_prediction_indexes
        ],
        reverse=True,
    )
    matched_gold: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gold_index, predicted_index in candidates:
        if iou < iou_threshold:
            break
        if gold_index in matched_gold or predicted_index in matched_predictions:
            continue
        matched_gold.add(gold_index)
        matched_predictions.add(predicted_index)
        matches.append((gold_index, predicted_index, iou))

    matched_character_errors = 0
    matched_reference_characters = 0
    kind_correct = 0
    kind_total = 0
    direction_correct = 0
    direction_total = 0
    matched_intersection_area = 0.0
    matched_error_counts = {
        "substitutions": 0,
        "missing_characters": 0,
        "hallucinated_characters": 0,
        "total_errors": 0,
    }
    for gold_index, predicted_index, _ in matches:
        gold = gold_regions[gold_index]
        predicted = predicted_regions[predicted_index]
        matched_character_errors += character_error_distance(
            gold.transcription, predicted.raw_text
        )
        matched_reference_characters += len(gold.transcription)
        for field, value in character_error_counts(
            gold.transcription, predicted.raw_text
        ).items():
            matched_error_counts[field] += value
        if gold.kind != RegionKind.UNKNOWN:
            kind_total += 1
            kind_correct += gold.kind == predicted.kind
        if gold.direction != "unknown":
            direction_total += 1
            direction_correct += gold.direction == predicted.direction
        matched_intersection_area += polygon_intersection_area(
            gold.polygon, predicted.polygon
        )

    order_pairs_correct = 0
    order_pairs_total = 0
    for first_index, first in enumerate(matches):
        for second in matches[first_index + 1 :]:
            gold_order = gold_regions[first[0]].reading_order - gold_regions[second[0]].reading_order
            predicted_order = (
                predicted_regions[first[1]].reading_order
                - predicted_regions[second[1]].reading_order
            )
            order_pairs_total += 1
            order_pairs_correct += gold_order * predicted_order > 0

    gold_text = "".join(
        region.transcription for region in sorted(gold_regions, key=lambda item: item.reading_order)
    )
    predicted_text = "".join(
        region.raw_text
        for region in sorted(predicted_regions, key=lambda item: item.reading_order)
    )
    page_character_errors = character_error_distance(gold_text, predicted_text)
    page_error_counts = character_error_counts(gold_text, predicted_text)
    complete_order_correct = (
        len(matches) == len(gold_regions) == len(predicted_regions)
        and order_pairs_correct == order_pairs_total
        and all(
            gold_regions[gold_index].reading_order
            == predicted_regions[predicted_index].reading_order
            for gold_index, predicted_index, _ in matches
        )
    )
    gold_polygon_area = sum(polygon_area(region.polygon) for region in gold_regions)
    return {
        "page_id": page.page_id,
        "source_uri": page.source.source_uri,
        "page_number": page.source.page_number,
        "publication_year": page.source.publication_year,
        "decade": (
            f"{(page.source.publication_year // 10) * 10}s"
            if page.source.publication_year is not None
            else "unknown"
        ),
        "page_genre": page.page_genre,
        "layout": page.layout,
        "scan_quality": page.scan_quality,
        "gold_regions": len(gold_regions),
        "predicted_regions": len(predicted_regions),
        "matched_regions": len(matches),
        "invalid_geometry_predictions": len(predicted_regions)
        - len(valid_prediction_indexes),
        "sum_matched_iou": sum(match[2] for match in matches),
        "gold_polygon_area": gold_polygon_area,
        "matched_intersection_area": matched_intersection_area,
        "matched_character_errors": matched_character_errors,
        "matched_error_counts": matched_error_counts,
        "matched_reference_characters": matched_reference_characters,
        "page_character_errors": page_character_errors,
        "page_error_counts": page_error_counts,
        "page_reference_characters": len(gold_text),
        "predicted_characters": len(predicted_text),
        "kind_correct": kind_correct,
        "kind_total": kind_total,
        "direction_correct": direction_correct,
        "direction_total": direction_total,
        "order_pairs_correct": order_pairs_correct,
        "order_pairs_total": order_pairs_total,
        "complete_reading_order_correct": int(complete_order_correct),
    }


def _aggregate_page_scores(page_scores: list[dict[str, Any]]) -> dict[str, Any]:
    def total(field: str) -> Any:
        return sum(page[field] for page in page_scores)

    gold_regions = total("gold_regions")
    predicted_regions = total("predicted_regions")
    matched_regions = total("matched_regions")
    matched_reference_characters = total("matched_reference_characters")
    page_reference_characters = total("page_reference_characters")
    kind_total = total("kind_total")
    direction_total = total("direction_total")
    order_pairs_total = total("order_pairs_total")
    gold_polygon_area = total("gold_polygon_area")
    matched_error_counts = {
        field: sum(page["matched_error_counts"][field] for page in page_scores)
        for field in (
            "substitutions",
            "missing_characters",
            "hallucinated_characters",
            "total_errors",
        )
    }
    page_error_counts = {
        field: sum(page["page_error_counts"][field] for page in page_scores)
        for field in matched_error_counts
    }
    return {
        "pages": len(page_scores),
        "region_detection": _detection_metrics(
            matched_regions, predicted_regions, gold_regions
        ),
        "invalid_geometry_predictions": total("invalid_geometry_predictions"),
        "mean_matched_iou": (
            total("sum_matched_iou") / matched_regions if matched_regions else None
        ),
        "gold_area_covered": (
            total("matched_intersection_area") / gold_polygon_area
            if gold_polygon_area
            else None
        ),
        "matched_region_cer": (
            total("matched_character_errors") / matched_reference_characters
            if matched_reference_characters
            else None
        ),
        "matched_character_errors": matched_error_counts,
        "reading_order_cer": (
            total("page_character_errors") / page_reference_characters
            if page_reference_characters
            else None
        ),
        "reading_order_character_errors": page_error_counts,
        "reference_characters": page_reference_characters,
        "predicted_characters": total("predicted_characters"),
        "region_kind_accuracy": (
            total("kind_correct") / kind_total if kind_total else None
        ),
        "text_direction_accuracy": (
            total("direction_correct") / direction_total
            if direction_total
            else None
        ),
        "reading_order_pair_accuracy": (
            total("order_pairs_correct") / order_pairs_total
            if order_pairs_total
            else None
        ),
        "complete_reading_order_accuracy": (
            total("complete_reading_order_correct") / len(page_scores)
            if page_scores
            else None
        ),
    }


def score_ocr_artifacts(
    gold: OCRGoldSet,
    predictions: list[OCRPageArtifact],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    pages_by_key = {
        (page.source.source_uri, page.source.page_number): page for page in gold.pages
    }
    predictions_by_key = {}
    for prediction in predictions:
        if prediction.run.kind != RunKind.OCR:
            raise ValueError("OCR scoring accepts only artifacts with run.kind=ocr")
        key = (prediction.source.source_uri, prediction.source.page_number)
        if key not in pages_by_key:
            raise ValueError(f"prediction page is not present in gold: {key}")
        if key in predictions_by_key:
            raise ValueError(f"duplicate prediction artifact for gold page: {key}")
        page = pages_by_key[key]
        if prediction.image_sha256 != page.image_sha256:
            raise ValueError(f"prediction image hash differs from gold page {page.page_id}")
        if prediction.width != page.width or prediction.height != page.height:
            raise ValueError(f"prediction dimensions differ from gold page {page.page_id}")
        predictions_by_key[key] = prediction

    page_scores = [
        _score_page(
            page,
            predictions_by_key.get((page.source.source_uri, page.source.page_number)),
            iou_threshold,
        )
        for page in gold.pages
    ]

    def by_stratum(field: str) -> dict[str, dict[str, Any]]:
        return {
            str(value): _aggregate_page_scores(
                [page for page in page_scores if page[field] == value]
            )
            for value in sorted({page[field] for page in page_scores}, key=str)
        }

    model_identities = {
        (prediction.run.model_name, prediction.run.model_revision)
        for prediction in predictions
    }
    if len(model_identities) > 1:
        raise ValueError("one OCR score report cannot mix model names or revisions")
    model_name, model_revision = next(iter(model_identities), (None, None))
    durations = [
        (prediction.run.completed_at - prediction.run.started_at).total_seconds()
        for prediction in predictions
        if prediction.run.completed_at is not None
    ]
    duration_seconds = sum(durations) if len(durations) == len(predictions) else None
    peak_memory_values = [
        prediction.run.configuration.get("peak_memory_bytes")
        for prediction in predictions
        if isinstance(prediction.run.configuration.get("peak_memory_bytes"), (int, float))
    ]
    return {
        "schema_version": "1.0",
        "dataset_id": gold.dataset_id,
        "iou_threshold": iou_threshold,
        "model_name": model_name,
        "model_revision": model_revision,
        "model_duration_seconds": duration_seconds,
        "pages_per_second": (
            len(predictions) / duration_seconds
            if duration_seconds and duration_seconds > 0
            else None
        ),
        "peak_memory_bytes": max(peak_memory_values) if peak_memory_values else None,
        "overall": _aggregate_page_scores(page_scores),
        "by_page_genre": by_stratum("page_genre"),
        "by_layout": by_stratum("layout"),
        "by_scan_quality": by_stratum("scan_quality"),
        "by_decade": by_stratum("decade"),
        "pages": page_scores,
        "warnings": [
            "Scores are valid only for source-resolution pages with independent review and adjudication.",
            "Matched-region CER isolates recognition after layout matching; reading-order CER also penalizes missing, extra, and misordered regions.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gold = OCRGoldSet.model_validate_json(args.gold.read_text(encoding="utf-8"))
    predictions = [
        OCRPageArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        for path in args.predictions
    ]
    report = score_ocr_artifacts(
        gold, predictions, iou_threshold=args.iou_threshold
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "region_f1": report["overall"]["region_detection"]["f1"],
                "matched_region_cer": report["overall"]["matched_region_cer"],
                "reading_order_cer": report["overall"]["reading_order_cer"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
