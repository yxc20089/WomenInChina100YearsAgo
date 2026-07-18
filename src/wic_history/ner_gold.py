"""Validate adjudicated NER gold data and score exact-offset NER artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence
from uuid import UUID

from pydantic import Field, model_validator

from .evidence import EntityType, NERArtifact, SourcePointer, StrictModel


class GoldEntitySpan(StrictModel):
    entity_type: EntityType
    corrected_start: int = Field(ge=0)
    corrected_end: int = Field(ge=0)
    corrected_text: str = Field(min_length=1)
    raw_start: int | None = Field(default=None, ge=0)
    raw_end: int | None = Field(default=None, ge=0)
    raw_text: str | None = None
    note: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_offsets(self) -> "GoldEntitySpan":
        if self.corrected_end <= self.corrected_start:
            raise ValueError("corrected entity spans must be non-empty")
        raw_values = (self.raw_start, self.raw_end, self.raw_text)
        if any(value is None for value in raw_values) and not all(
            value is None for value in raw_values
        ):
            raise ValueError("raw_start, raw_end and raw_text must be supplied together")
        if self.raw_start is not None and self.raw_end <= self.raw_start:
            raise ValueError("raw entity spans must be non-empty")
        return self


class ReviewerAnnotation(StrictModel):
    reviewer: str = Field(min_length=1, max_length=200)
    corrected_text: str
    entities: list[GoldEntitySpan] = Field(default_factory=list)
    annotated_at: datetime
    notes: str | None = Field(default=None, max_length=5000)


class GoldAdjudication(StrictModel):
    adjudicator: str = Field(min_length=1, max_length=200)
    corrected_text: str
    entities: list[GoldEntitySpan] = Field(default_factory=list)
    adjudicated_at: datetime
    notes: str | None = Field(default=None, max_length=5000)


def _validate_annotation_text(
    raw_ocr_text: str,
    corrected_text: str,
    entities: list[GoldEntitySpan],
    label: str,
) -> None:
    keys = set()
    for index, entity in enumerate(entities):
        if entity.corrected_end > len(corrected_text):
            raise ValueError(f"{label} entity {index} corrected offsets are out of range")
        if corrected_text[entity.corrected_start : entity.corrected_end] != entity.corrected_text:
            raise ValueError(f"{label} entity {index} disagrees with corrected-text offsets")
        if entity.raw_start is not None:
            if entity.raw_end > len(raw_ocr_text):
                raise ValueError(f"{label} entity {index} raw offsets are out of range")
            if raw_ocr_text[entity.raw_start : entity.raw_end] != entity.raw_text:
                raise ValueError(f"{label} entity {index} disagrees with raw-OCR offsets")
        key = (
            entity.corrected_start,
            entity.corrected_end,
            entity.entity_type.value,
        )
        if key in keys:
            raise ValueError(f"{label} contains a duplicate entity span")
        keys.add(key)


class GoldSnippet(StrictModel):
    snippet_id: str = Field(min_length=1, max_length=300)
    gold_region_id: UUID | None = None
    source_ocr_run_id: UUID | None = None
    source_ocr_region_id: UUID | None = None
    source: SourcePointer
    raw_ocr_text: str
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
    reviews: list[ReviewerAnnotation] = Field(min_length=2)
    adjudication: GoldAdjudication

    @model_validator(mode="after")
    def validate_reviews(self) -> "GoldSnippet":
        if self.source.region_id is None:
            raise ValueError("gold snippets require an OCR region UUID")
        if (
            self.source_ocr_region_id is not None
            and self.source.region_id != self.source_ocr_region_id
        ):
            raise ValueError(
                "source pointer region_id must equal the explicit source OCR region mapping"
            )
        reviewers = [review.reviewer for review in self.reviews]
        if len(set(reviewers)) != len(reviewers):
            raise ValueError("gold snippets require distinct independent reviewers")
        for index, review in enumerate(self.reviews):
            _validate_annotation_text(
                self.raw_ocr_text,
                review.corrected_text,
                review.entities,
                f"review {index}",
            )
        _validate_annotation_text(
            self.raw_ocr_text,
            self.adjudication.corrected_text,
            self.adjudication.entities,
            "adjudication",
        )
        return self


class NERGoldSet(StrictModel):
    schema_version: Literal["1.0", "1.1"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=300)
    created_at: datetime
    ontology_version: str = Field(min_length=1, max_length=100)
    snippets: list[GoldSnippet] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_sources(self) -> "NERGoldSet":
        snippet_ids = [snippet.snippet_id for snippet in self.snippets]
        region_ids = [_source_ocr_region_id(snippet) for snippet in self.snippets]
        gold_region_ids = [_gold_region_id(snippet) for snippet in self.snippets]
        if len(set(snippet_ids)) != len(snippet_ids):
            raise ValueError("snippet IDs must be unique")
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("each gold snippet must target a distinct source OCR region")
        if len(set(gold_region_ids)) != len(gold_region_ids):
            raise ValueError("each gold snippet must have a distinct model-independent region ID")
        if self.schema_version == "1.1" and any(
            snippet.gold_region_id is None
            or snippet.source_ocr_run_id is None
            or snippet.source_ocr_region_id is None
            for snippet in self.snippets
        ):
            raise ValueError(
                "NER gold schema 1.1 requires explicit gold-to-source OCR region mappings"
            )
        if self.schema_version == "1.1" and any(
            snippet.gold_region_id == snippet.source_ocr_region_id
            for snippet in self.snippets
        ):
            raise ValueError(
                "NER gold schema 1.1 forbids reusing model OCR region UUIDs as gold identities"
            )
        return self


def _source_ocr_region_id(snippet: GoldSnippet) -> UUID:
    return snippet.source_ocr_region_id or snippet.source.region_id


def _gold_region_id(snippet: GoldSnippet) -> UUID:
    return snippet.gold_region_id or _source_ocr_region_id(snippet)


SpanKey = tuple[UUID, int, int, str]


def character_error_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance over Unicode characters using bounded memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, 1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1]


def _metrics(true_positive: int, predicted: int, expected: int) -> dict[str, Any]:
    precision = true_positive / predicted if predicted else (1.0 if expected == 0 else 0.0)
    recall = true_positive / expected if expected else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": true_positive,
        "false_positive": predicted - true_positive,
        "false_negative": expected - true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _maximum_overlap_matches(expected: set[SpanKey], predicted: set[SpanKey]) -> int:
    expected_by_group: dict[tuple[UUID, str], list[SpanKey]] = {}
    predicted_by_group: dict[tuple[UUID, str], list[SpanKey]] = {}
    for key in expected:
        expected_by_group.setdefault((key[0], key[3]), []).append(key)
    for key in predicted:
        predicted_by_group.setdefault((key[0], key[3]), []).append(key)

    matches = 0
    for group, predicted_spans in predicted_by_group.items():
        expected_spans = expected_by_group.get(group, [])
        matched_prediction: dict[int, int] = {}

        def assign(expected_index: int, visited: set[int]) -> bool:
            expected_span = expected_spans[expected_index]
            for predicted_index, predicted_span in enumerate(predicted_spans):
                overlaps = max(expected_span[1], predicted_span[1]) < min(
                    expected_span[2], predicted_span[2]
                )
                if not overlaps or predicted_index in visited:
                    continue
                visited.add(predicted_index)
                if predicted_index not in matched_prediction or assign(
                    matched_prediction[predicted_index], visited
                ):
                    matched_prediction[predicted_index] = expected_index
                    return True
            return False

        for expected_index in range(len(expected_spans)):
            assign(expected_index, set())
        matches += len(matched_prediction)
    return matches


def _stratified_exact_metrics(
    labels_by_region: dict[UUID, str],
    expected: set[SpanKey],
    predicted: set[SpanKey],
    prediction_count_by_region: Counter[UUID],
) -> dict[str, dict[str, Any]]:
    results = {}
    for label in sorted(set(labels_by_region.values())):
        region_ids = {
            region_id
            for region_id, region_label in labels_by_region.items()
            if region_label == label
        }
        label_expected = {key for key in expected if key[0] in region_ids}
        label_predicted = {key for key in predicted if key[0] in region_ids}
        results[label] = _metrics(
            len(label_expected & label_predicted),
            sum(prediction_count_by_region[region_id] for region_id in region_ids),
            len(label_expected),
        )
    return results


def score_ner_artifact(
    gold: NERGoldSet,
    predictions: NERArtifact,
    input_text: Literal["corrected", "raw_ocr"],
    *,
    confidence_threshold: float = 0.0,
) -> dict[str, Any]:
    expected_variant = "corrected_text" if input_text == "corrected" else "raw_ocr"
    if (
        predictions.input_variant is not None
        and predictions.input_variant != expected_variant
    ):
        raise ValueError(
            "prediction input_variant disagrees with the requested scoring text"
        )
    snippets_by_region = {
        _source_ocr_region_id(snippet): snippet for snippet in gold.snippets
    }
    expected: set[SpanKey] = set()
    total_adjudicated = 0
    for snippet in gold.snippets:
        region_id = _gold_region_id(snippet)
        total_adjudicated += len(snippet.adjudication.entities)
        for entity in snippet.adjudication.entities:
            if input_text == "corrected":
                expected.add(
                    (
                        region_id,
                        entity.corrected_start,
                        entity.corrected_end,
                        entity.entity_type.value,
                    )
                )
            elif entity.raw_start is not None:
                expected.add(
                    (
                        region_id,
                        entity.raw_start,
                        entity.raw_end,
                        entity.entity_type.value,
                    )
                )

    predicted_keys: set[SpanKey] = set()
    invalid_predictions: list[dict[str, Any]] = []
    duplicate_predictions = 0
    prediction_count_by_region: Counter[UUID] = Counter()
    prediction_count_by_type: Counter[str] = Counter()
    thresholded_mentions = [
        mention
        for mention in predictions.mentions
        if mention.confidence is None or mention.confidence >= confidence_threshold
    ]
    for mention in thresholded_mentions:
        pointer = mention.source
        prediction_count_by_type[mention.entity_type.value] += 1
        snippet = snippets_by_region.get(pointer.region_id)
        reason = None
        if snippet is None:
            reason = "region_not_in_gold_set"
        elif pointer.text_start is None or pointer.text_end is None:
            reason = "missing_offsets"
        else:
            target_text = (
                snippet.adjudication.corrected_text
                if input_text == "corrected"
                else snippet.raw_ocr_text
            )
            if pointer.text_end > len(target_text):
                reason = "offsets_out_of_range"
            elif target_text[pointer.text_start : pointer.text_end] != mention.text:
                reason = "surface_disagrees_with_target_offsets"
        if reason:
            invalid_predictions.append(
                {
                    "mention_id": str(mention.mention_id),
                    "entity_type": mention.entity_type.value,
                    "reason": reason,
                }
            )
            continue
        gold_region_id = _gold_region_id(snippet)
        prediction_count_by_region[gold_region_id] += 1
        key = (
            gold_region_id,
            pointer.text_start,
            pointer.text_end,
            mention.entity_type.value,
        )
        if key in predicted_keys:
            duplicate_predictions += 1
        predicted_keys.add(key)

    exact_true_positive = len(expected & predicted_keys)
    exact = _metrics(exact_true_positive, len(thresholded_mentions), len(expected))
    relaxed_true_positive = _maximum_overlap_matches(expected, predicted_keys)
    relaxed = _metrics(relaxed_true_positive, len(thresholded_mentions), len(expected))
    entity_types = sorted(
        {key[3] for key in expected | predicted_keys} | set(prediction_count_by_type)
    )
    by_entity_type = {}
    for entity_type in entity_types:
        type_expected = {key for key in expected if key[3] == entity_type}
        type_predicted = {key for key in predicted_keys if key[3] == entity_type}
        by_entity_type[entity_type] = _metrics(
            len(type_expected & type_predicted),
            prediction_count_by_type[entity_type],
            len(type_expected),
        )

    total_corrected_characters = sum(
        len(snippet.adjudication.corrected_text) for snippet in gold.snippets
    )
    total_character_errors = sum(
        character_error_distance(
            snippet.adjudication.corrected_text, snippet.raw_ocr_text
        )
        for snippet in gold.snippets
    )
    duration_seconds = (
        (predictions.run.completed_at - predictions.run.started_at).total_seconds()
        if predictions.run.completed_at
        else None
    )
    input_region_count = predictions.run.configuration.get("input_region_count")
    input_character_count = predictions.run.configuration.get("input_character_count")
    by_scan_quality = _stratified_exact_metrics(
        {
            _gold_region_id(snippet): snippet.scan_quality
            for snippet in gold.snippets
        },
        expected,
        predicted_keys,
        prediction_count_by_region,
    )
    by_layout = _stratified_exact_metrics(
        {_gold_region_id(snippet): snippet.layout for snippet in gold.snippets},
        expected,
        predicted_keys,
        prediction_count_by_region,
    )
    by_page_genre = _stratified_exact_metrics(
        {
            _gold_region_id(snippet): snippet.page_genre
            for snippet in gold.snippets
        },
        expected,
        predicted_keys,
        prediction_count_by_region,
    )
    by_decade = _stratified_exact_metrics(
        {
            _gold_region_id(snippet): (
                f"{(snippet.source.publication_year // 10) * 10}s"
                if snippet.source.publication_year is not None
                else "unknown"
            )
            for snippet in gold.snippets
        },
        expected,
        predicted_keys,
        prediction_count_by_region,
    )
    return {
        "schema_version": "1.0",
        "dataset_id": gold.dataset_id,
        "ontology_version": gold.ontology_version,
        "input_text": input_text,
        "model_name": predictions.run.model_name,
        "model_revision": predictions.run.model_revision,
        "prediction_artifact_schema_version": predictions.schema_version,
        "input_sha256": predictions.input_sha256,
        "dataset_split": predictions.split_id,
        "model_duration_seconds": duration_seconds,
        "mentions_per_second": (
            len(predictions.mentions) / duration_seconds
            if duration_seconds and duration_seconds > 0
            else None
        ),
        "regions_per_second": (
            input_region_count / duration_seconds
            if input_region_count is not None and duration_seconds and duration_seconds > 0
            else None
        ),
        "unicode_characters_per_second": (
            input_character_count / duration_seconds
            if input_character_count is not None and duration_seconds and duration_seconds > 0
            else None
        ),
        "latency_p50_seconds": predictions.run.configuration.get(
            "latency_p50_seconds"
        ),
        "latency_p95_seconds": predictions.run.configuration.get(
            "latency_p95_seconds"
        ),
        "cold_start_seconds": predictions.run.configuration.get("cold_start_seconds"),
        "peak_memory_bytes": predictions.run.configuration.get("peak_memory_bytes"),
        "confidence_threshold": confidence_threshold,
        "snippets": len(gold.snippets),
        "gold_entities": len(expected),
        "total_adjudicated_entities": total_adjudicated,
        "raw_recoverable_entities": (
            len(expected) if input_text == "raw_ocr" else None
        ),
        "raw_recoverability": (
            len(expected) / total_adjudicated if input_text == "raw_ocr" and total_adjudicated else None
        ),
        "ocr_character_errors": total_character_errors,
        "ocr_reference_characters": total_corrected_characters,
        "ocr_cer": (
            total_character_errors / total_corrected_characters
            if total_corrected_characters
            else 0.0
        ),
        "predictions_before_evidence_validation": len(thresholded_mentions),
        "valid_unique_predictions": len(predicted_keys),
        "duplicate_predictions": duplicate_predictions,
        "invalid_evidence_predictions": len(invalid_predictions),
        "invalid_evidence_rate": (
            len(invalid_predictions) / len(thresholded_mentions)
            if thresholded_mentions
            else 0.0
        ),
        "invalid_prediction_examples": invalid_predictions[:100],
        "exact": exact,
        "relaxed_overlap": relaxed,
        "end_to_end_exact_recall": (
            exact_true_positive / total_adjudicated
            if input_text == "raw_ocr" and total_adjudicated
            else None
        ),
        "by_entity_type": by_entity_type,
        "by_scan_quality": by_scan_quality,
        "by_layout": by_layout,
        "by_page_genre": by_page_genre,
        "by_decade": by_decade,
        "warnings": [
            "Scores are valid only for independently annotated and adjudicated gold data.",
            "Raw-OCR exact recall over recoverable spans isolates NER; end-to-end recall also counts OCR-lost entities.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--input-text", choices=("corrected", "raw_ocr"), required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.confidence_threshold <= 1:
        raise SystemExit("--confidence-threshold must be between 0 and 1")
    gold = NERGoldSet.model_validate_json(args.gold.read_text(encoding="utf-8"))
    predictions = NERArtifact.model_validate_json(
        args.predictions.read_text(encoding="utf-8")
    )
    report = score_ner_artifact(
        gold,
        predictions,
        args.input_text,
        confidence_threshold=args.confidence_threshold,
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
                "exact_f1": report["exact"]["f1"],
                "relaxed_f1": report["relaxed_overlap"]["f1"],
                "invalid_evidence_rate": report["invalid_evidence_rate"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
