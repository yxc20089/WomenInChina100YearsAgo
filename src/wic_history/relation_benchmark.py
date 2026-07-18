"""Freeze, run, and score evidence-grounded relation extraction benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Literal, Sequence
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from .evidence import EntityType, SourcePointer, StrictModel
from .ner_gold import NERGoldSet
from .relation_pipeline import (
    RELATION_RULES,
    RegionEvidence,
    ReviewedMention,
    extract_region_claims,
)


RelationSplit = Literal["train", "development", "test"]
RelationInputVariant = Literal["corrected_text", "raw_ocr"]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _full_sha256(value: str | None) -> bool:
    return bool(
        value
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _immutable_revision(value: str) -> bool:
    forbidden = {"main", "master", "latest", "nightly", "dev"}
    parts = {part.lower() for part in value.replace("\\", "/").split("/")}
    return not bool(parts & forbidden)


class RelationPredicateDefinition(StrictModel):
    predicate: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    label_zh: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    subject_entity_types: list[EntityType] = Field(min_length=1)
    object_entity_types: list[EntityType] = Field(min_length=1)
    directional: bool = True

    @model_validator(mode="after")
    def validate_types(self) -> "RelationPredicateDefinition":
        if len(set(self.subject_entity_types)) != len(self.subject_entity_types):
            raise ValueError("predicate subject entity types must be unique")
        if len(set(self.object_entity_types)) != len(self.object_entity_types):
            raise ValueError("predicate object entity types must be unique")
        return self


class RelationGoldMention(StrictModel):
    mention_id: UUID
    entity_type: EntityType
    corrected_start: int = Field(ge=0)
    corrected_end: int = Field(ge=0)
    corrected_text: str = Field(min_length=1)
    raw_start: int | None = Field(default=None, ge=0)
    raw_end: int | None = Field(default=None, ge=0)
    raw_text: str | None = None

    @model_validator(mode="after")
    def validate_offsets(self) -> "RelationGoldMention":
        if self.corrected_end <= self.corrected_start:
            raise ValueError("corrected mention spans must be non-empty")
        raw_values = (self.raw_start, self.raw_end, self.raw_text)
        if any(value is None for value in raw_values) and not all(
            value is None for value in raw_values
        ):
            raise ValueError("raw mention start, end and text must be supplied together")
        if self.raw_start is not None and self.raw_end <= self.raw_start:
            raise ValueError("raw mention spans must be non-empty")
        return self


class RelationGoldEdge(StrictModel):
    subject_mention_id: UUID
    predicate: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    object_mention_id: UUID
    corrected_evidence_start: int = Field(ge=0)
    corrected_evidence_end: int = Field(ge=0)
    corrected_evidence_text: str = Field(min_length=1)
    raw_evidence_start: int | None = Field(default=None, ge=0)
    raw_evidence_end: int | None = Field(default=None, ge=0)
    raw_evidence_text: str | None = None

    @model_validator(mode="after")
    def validate_edge(self) -> "RelationGoldEdge":
        if self.subject_mention_id == self.object_mention_id:
            raise ValueError("relation arguments must be distinct mentions")
        if self.corrected_evidence_end <= self.corrected_evidence_start:
            raise ValueError("corrected relation evidence must be non-empty")
        raw_values = (
            self.raw_evidence_start,
            self.raw_evidence_end,
            self.raw_evidence_text,
        )
        if any(value is None for value in raw_values) and not all(
            value is None for value in raw_values
        ):
            raise ValueError("raw evidence start, end and text must be supplied together")
        if (
            self.raw_evidence_start is not None
            and self.raw_evidence_end <= self.raw_evidence_start
        ):
            raise ValueError("raw relation evidence must be non-empty")
        return self


class RelationReviewerAnnotation(StrictModel):
    reviewer: str = Field(min_length=1, max_length=200)
    annotated_at: datetime
    relations: list[RelationGoldEdge] = Field(default_factory=list)
    notes: str = Field(min_length=1, max_length=5000)


class RelationAdjudication(StrictModel):
    adjudicator: str = Field(min_length=1, max_length=200)
    adjudicated_at: datetime
    relations: list[RelationGoldEdge] = Field(default_factory=list)
    notes: str = Field(min_length=1, max_length=5000)


class RelationNERTextMapping(StrictModel):
    source_ner_snippet_id: str = Field(min_length=1, max_length=300)
    source_region_id: UUID
    unit_corrected_start: int = Field(ge=0)
    unit_corrected_end: int = Field(ge=0)
    snippet_corrected_start: int = Field(ge=0)
    snippet_corrected_end: int = Field(ge=0)
    unit_raw_start: int | None = Field(default=None, ge=0)
    unit_raw_end: int | None = Field(default=None, ge=0)
    snippet_raw_start: int | None = Field(default=None, ge=0)
    snippet_raw_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_mapping(self) -> "RelationNERTextMapping":
        if self.unit_corrected_end <= self.unit_corrected_start:
            raise ValueError("relation-to-NER corrected mappings must be non-empty")
        if self.snippet_corrected_end <= self.snippet_corrected_start:
            raise ValueError("NER corrected mapping spans must be non-empty")
        raw_values = (
            self.unit_raw_start,
            self.unit_raw_end,
            self.snippet_raw_start,
            self.snippet_raw_end,
        )
        if any(value is None for value in raw_values) and not all(
            value is None for value in raw_values
        ):
            raise ValueError("relation-to-NER raw mappings must be supplied together")
        if self.unit_raw_start is not None and (
            self.unit_raw_end <= self.unit_raw_start
            or self.snippet_raw_end <= self.snippet_raw_start
        ):
            raise ValueError("relation-to-NER raw mappings must be non-empty")
        return self


def _validate_source_pointer(pointer: SourcePointer) -> bool:
    return bool(
        pointer.region_id is not None
        and pointer.source_sha256 is not None
        and _full_sha256(pointer.source_sha256)
        and pointer.derivative_id is not None
        and pointer.image_sha256 is not None
        and pointer.evidence_tier is not None
        and pointer.volume_number is not None
        and pointer.publication_year is not None
        and pointer.polygon is not None
    )


def _validate_relation_annotation(
    unit: "RelationBenchmarkUnit",
    relations: list[RelationGoldEdge],
    label: str,
) -> None:
    mentions = {mention.mention_id: mention for mention in unit.mentions}
    seen = set()
    for index, relation in enumerate(relations):
        subject = mentions.get(relation.subject_mention_id)
        object_mention = mentions.get(relation.object_mention_id)
        if subject is None or object_mention is None:
            raise ValueError(f"{label} relation {index} references an unknown mention")
        key = (
            relation.subject_mention_id,
            relation.predicate,
            relation.object_mention_id,
        )
        if key in seen:
            raise ValueError(f"{label} contains a duplicate relation")
        seen.add(key)
        if relation.corrected_evidence_end > len(unit.corrected_text):
            raise ValueError(f"{label} corrected evidence offsets are out of range")
        corrected_quote = unit.corrected_text[
            relation.corrected_evidence_start : relation.corrected_evidence_end
        ]
        if corrected_quote != relation.corrected_evidence_text:
            raise ValueError(f"{label} corrected evidence disagrees with exact offsets")
        if (
            relation.corrected_evidence_start
            > min(subject.corrected_start, object_mention.corrected_start)
            or relation.corrected_evidence_end
            < max(subject.corrected_end, object_mention.corrected_end)
        ):
            raise ValueError(f"{label} corrected evidence must contain both arguments")
        if relation.raw_evidence_start is not None:
            if subject.raw_start is None or object_mention.raw_start is None:
                raise ValueError(f"{label} raw relation requires raw-recoverable arguments")
            if relation.raw_evidence_end > len(unit.raw_ocr_text):
                raise ValueError(f"{label} raw evidence offsets are out of range")
            raw_quote = unit.raw_ocr_text[
                relation.raw_evidence_start : relation.raw_evidence_end
            ]
            if raw_quote != relation.raw_evidence_text:
                raise ValueError(f"{label} raw evidence disagrees with exact offsets")
            if (
                relation.raw_evidence_start
                > min(subject.raw_start, object_mention.raw_start)
                or relation.raw_evidence_end
                < max(subject.raw_end, object_mention.raw_end)
            ):
                raise ValueError(f"{label} raw evidence must contain both arguments")


class RelationBenchmarkUnit(StrictModel):
    unit_id: str = Field(min_length=1, max_length=300)
    gold_unit_id: UUID
    coherent_unit_revision_id: UUID | None = None
    source_ner_mappings: list[RelationNERTextMapping] = Field(min_length=1)
    issue_id: str = Field(min_length=1, max_length=300)
    split: RelationSplit
    selected_by: str = Field(min_length=1, max_length=200)
    source_regions: list[SourcePointer] = Field(min_length=1)
    corrected_text: str
    raw_ocr_text: str
    mentions: list[RelationGoldMention] = Field(min_length=2)
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
    reviews: list[RelationReviewerAnnotation] = Field(min_length=2, max_length=2)
    adjudication: RelationAdjudication

    @model_validator(mode="after")
    def validate_unit(self) -> "RelationBenchmarkUnit":
        region_ids = [pointer.region_id for pointer in self.source_regions]
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("relation unit source regions must be unique")
        snippet_ids = [
            mapping.source_ner_snippet_id for mapping in self.source_ner_mappings
        ]
        if len(set(snippet_ids)) != len(snippet_ids):
            raise ValueError("relation unit NER snippet mappings must be unique")
        corrected_intervals = sorted(
            (mapping.unit_corrected_start, mapping.unit_corrected_end)
            for mapping in self.source_ner_mappings
        )
        if any(
            left_end > right_start
            for (_, left_end), (right_start, _) in zip(
                corrected_intervals, corrected_intervals[1:], strict=False
            )
        ):
            raise ValueError("relation unit corrected NER mappings cannot overlap")
        raw_intervals = sorted(
            (mapping.unit_raw_start, mapping.unit_raw_end)
            for mapping in self.source_ner_mappings
            if mapping.unit_raw_start is not None
        )
        if any(
            left_end > right_start
            for (_, left_end), (right_start, _) in zip(
                raw_intervals, raw_intervals[1:], strict=False
            )
        ):
            raise ValueError("relation unit raw NER mappings cannot overlap")
        for mapping in self.source_ner_mappings:
            if mapping.unit_corrected_end > len(self.corrected_text):
                raise ValueError("corrected NER mapping is outside the relation unit")
            if (
                mapping.unit_raw_end is not None
                and mapping.unit_raw_end > len(self.raw_ocr_text)
            ):
                raise ValueError("raw NER mapping is outside the relation unit")
        mention_ids = [mention.mention_id for mention in self.mentions]
        if len(set(mention_ids)) != len(mention_ids):
            raise ValueError("relation unit mention IDs must be unique")
        corrected_spans = [
            (mention.corrected_start, mention.corrected_end, mention.entity_type)
            for mention in self.mentions
        ]
        if len(set(corrected_spans)) != len(corrected_spans):
            raise ValueError("relation unit corrected mention spans must be unique")
        for mention in self.mentions:
            if mention.corrected_end > len(self.corrected_text):
                raise ValueError("corrected mention offsets are out of range")
            if (
                self.corrected_text[mention.corrected_start : mention.corrected_end]
                != mention.corrected_text
            ):
                raise ValueError("corrected mention text disagrees with exact offsets")
            if mention.raw_start is not None:
                if mention.raw_end > len(self.raw_ocr_text):
                    raise ValueError("raw mention offsets are out of range")
                if (
                    self.raw_ocr_text[mention.raw_start : mention.raw_end]
                    != mention.raw_text
                ):
                    raise ValueError("raw mention text disagrees with exact offsets")
        reviewers = [review.reviewer for review in self.reviews]
        if len(set(reviewers)) != 2:
            raise ValueError("relation units require two distinct reviewers")
        if self.adjudication.adjudicator in reviewers:
            raise ValueError("relation adjudicator must be independent of reviewers")
        for index, review in enumerate(self.reviews):
            _validate_relation_annotation(self, review.relations, f"review {index}")
        _validate_relation_annotation(self, self.adjudication.relations, "adjudication")
        return self


def relation_benchmark_eligibility_failures(
    predicates: list[RelationPredicateDefinition],
    units: list[RelationBenchmarkUnit],
) -> list[str]:
    failures = []
    if len(units) < 300:
        failures.append(f"Dataset has {len(units)} units; at least 300 are required.")
    issues = {unit.issue_id for unit in units}
    if len(issues) < 30:
        failures.append(f"Dataset has {len(issues)} issues; at least 30 are required.")
    selectors = {unit.selected_by for unit in units}
    if len(selectors) < 2 or "technical-smoke" in selectors:
        failures.append(
            "Eligible relation benchmarks require two historian selectors and no technical-smoke selector."
        )
    split_by_issue: dict[str, set[str]] = {}
    for unit in units:
        split_by_issue.setdefault(unit.issue_id, set()).add(unit.split)
    leaked_issues = sorted(
        issue_id for issue_id, splits in split_by_issue.items() if len(splits) > 1
    )
    if leaked_issues:
        failures.append(
            f"Dataset leaks {len(leaked_issues)} issues across train/development/test."
        )
    for split in ("train", "development", "test"):
        count = sum(unit.split == split for unit in units)
        if count < 50:
            failures.append(f"Dataset has {count} {split} units; at least 50 are required.")
    test_units = [unit for unit in units if unit.split == "test"]
    test_relations = [
        relation for unit in test_units for relation in unit.adjudication.relations
    ]
    if len(test_relations) < 50:
        failures.append(
            f"Test split has {len(test_relations)} positive relations; at least 50 are required."
        )
    negative_test_units = sum(
        not unit.adjudication.relations for unit in test_units
    )
    if negative_test_units < 10:
        failures.append(
            f"Test split has {negative_test_units} negative units; at least 10 are required."
        )
    for definition in predicates:
        count = sum(
            relation.predicate == definition.predicate for relation in test_relations
        )
        if count < 10:
            failures.append(
                f"Test split has {count} {definition.predicate} relations; at least 10 are required."
            )
    decades = {
        f"{(pointer.publication_year // 10) * 10}s"
        for unit in units
        for pointer in unit.source_regions
        if pointer.publication_year is not None
    }
    if len(decades) < 2:
        failures.append("Dataset must cover at least two publication decades.")
    incomplete_sources = sum(
        not _validate_source_pointer(pointer)
        for unit in units
        for pointer in unit.source_regions
    )
    if incomplete_sources:
        failures.append(
            f"Dataset has {incomplete_sources} source pointers without complete scan provenance."
        )
    missing_coherent_units = sum(
        unit.coherent_unit_revision_id is None for unit in units
    )
    if missing_coherent_units:
        failures.append(
            f"Dataset has {missing_coherent_units} units without an approved coherent-unit revision."
        )
    input_hashes = [
        canonical_sha256(
            {
                "corrected_text": unit.corrected_text,
                "raw_ocr_text": unit.raw_ocr_text,
                "mentions": [
                    mention.model_dump(mode="json") for mention in unit.mentions
                ],
                "source_ner_mappings": [
                    mapping.model_dump(mode="json")
                    for mapping in unit.source_ner_mappings
                ],
            }
        )
        for unit in units
    ]
    duplicate_inputs = len(input_hashes) - len(set(input_hashes))
    if duplicate_inputs:
        failures.append(f"Dataset has {duplicate_inputs} duplicate extraction inputs.")
    return failures


class RelationBenchmarkDataset(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=300)
    created_at: datetime
    ontology_version: str = Field(min_length=1, max_length=100)
    source_ner_gold_dataset_id: str = Field(min_length=1, max_length=300)
    source_ner_gold_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicates: list[RelationPredicateDefinition] = Field(min_length=1)
    units: list[RelationBenchmarkUnit] = Field(min_length=1)
    benchmark_eligible: bool
    eligibility_failures: list[str]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dataset(self) -> "RelationBenchmarkDataset":
        predicate_names = [definition.predicate for definition in self.predicates]
        if len(set(predicate_names)) != len(predicate_names):
            raise ValueError("relation predicate definitions must be unique")
        allowed_predicates = set(predicate_names)
        unknown = {
            relation.predicate
            for unit in self.units
            for annotation in [*unit.reviews, unit.adjudication]
            for relation in annotation.relations
            if relation.predicate not in allowed_predicates
        }
        if unknown:
            raise ValueError(
                "relation annotations use undefined predicates: "
                + ", ".join(sorted(unknown))
            )
        definitions = {item.predicate: item for item in self.predicates}
        for unit in self.units:
            mentions = {mention.mention_id: mention for mention in unit.mentions}
            for annotation in [*unit.reviews, unit.adjudication]:
                for relation in annotation.relations:
                    definition = definitions[relation.predicate]
                    subject_type = mentions[
                        relation.subject_mention_id
                    ].entity_type
                    object_type = mentions[relation.object_mention_id].entity_type
                    if subject_type not in definition.subject_entity_types:
                        raise ValueError("relation subject type violates its ontology")
                    if object_type not in definition.object_entity_types:
                        raise ValueError("relation object type violates its ontology")
        unit_ids = [unit.unit_id for unit in self.units]
        gold_ids = [unit.gold_unit_id for unit in self.units]
        if len(set(unit_ids)) != len(unit_ids) or len(set(gold_ids)) != len(gold_ids):
            raise ValueError("relation benchmark unit identities must be unique")
        expected_failures = relation_benchmark_eligibility_failures(
            self.predicates, self.units
        )
        if self.eligibility_failures != expected_failures:
            raise ValueError("relation benchmark eligibility disagrees with its contents")
        if self.benchmark_eligible != (not expected_failures):
            raise ValueError("benchmark_eligible disagrees with eligibility failures")
        return self


def build_relation_benchmark_dataset(
    dataset_id: str,
    ontology_version: str,
    source_ner_gold_dataset_id: str,
    source_ner_gold_sha256: str,
    predicates: list[RelationPredicateDefinition],
    units: list[RelationBenchmarkUnit],
    *,
    created_at: datetime | None = None,
) -> RelationBenchmarkDataset:
    failures = relation_benchmark_eligibility_failures(predicates, units)
    return RelationBenchmarkDataset(
        dataset_id=dataset_id,
        created_at=created_at or datetime.now(timezone.utc),
        ontology_version=ontology_version,
        source_ner_gold_dataset_id=source_ner_gold_dataset_id,
        source_ner_gold_sha256=source_ner_gold_sha256,
        predicates=predicates,
        units=units,
        benchmark_eligible=not failures,
        eligibility_failures=failures,
        warnings=[
            "Relation gold assumes the referenced NER gold has already been independently adjudicated.",
            "Relation candidates never become reviewed claims without a separate historian decision.",
        ],
    )


def relation_benchmark_dataset_sha256(dataset: RelationBenchmarkDataset) -> str:
    return canonical_sha256(dataset.model_dump(mode="json", exclude={"created_at"}))


def verify_relation_dataset_ner_gold(
    dataset: RelationBenchmarkDataset, ner_gold_bytes: bytes
) -> NERGoldSet:
    digest = hashlib.sha256(ner_gold_bytes).hexdigest()
    if digest != dataset.source_ner_gold_sha256:
        raise ValueError("relation dataset NER gold file hash disagrees with its manifest")
    ner_gold = NERGoldSet.model_validate_json(ner_gold_bytes)
    if ner_gold.dataset_id != dataset.source_ner_gold_dataset_id:
        raise ValueError("relation dataset NER gold ID disagrees with its manifest")
    snippets = {snippet.snippet_id: snippet for snippet in ner_gold.snippets}
    for unit in dataset.units:
        source_region_ids = {
            pointer.region_id for pointer in unit.source_regions
        }
        mapped_region_ids = set()
        resolved_mappings = []
        for mapping in unit.source_ner_mappings:
            snippet = snippets.get(mapping.source_ner_snippet_id)
            if snippet is None:
                raise ValueError("relation unit references an unknown NER gold snippet")
            source_region_id = snippet.source_ocr_region_id or snippet.source.region_id
            if mapping.source_region_id != source_region_id:
                raise ValueError(
                    "relation NER mapping region disagrees with its source snippet"
                )
            mapped_region_ids.add(source_region_id)
            if source_region_id not in source_region_ids:
                raise ValueError(
                    "relation unit NER snippet is not represented in its source regions"
                )
            corrected_text = snippet.adjudication.corrected_text
            if mapping.snippet_corrected_end > len(corrected_text):
                raise ValueError("relation unit corrected NER mapping is out of range")
            if (
                unit.corrected_text[
                    mapping.unit_corrected_start : mapping.unit_corrected_end
                ]
                != corrected_text[
                    mapping.snippet_corrected_start : mapping.snippet_corrected_end
                ]
            ):
                raise ValueError(
                    "relation unit corrected text disagrees with its exact NER mapping"
                )
            if mapping.unit_raw_start is not None:
                if mapping.snippet_raw_end > len(snippet.raw_ocr_text):
                    raise ValueError("relation unit raw NER mapping is out of range")
                if (
                    unit.raw_ocr_text[mapping.unit_raw_start : mapping.unit_raw_end]
                    != snippet.raw_ocr_text[
                        mapping.snippet_raw_start : mapping.snippet_raw_end
                    ]
                ):
                    raise ValueError(
                        "relation unit raw text disagrees with its exact NER mapping"
                    )
            resolved_mappings.append((mapping, snippet))
        if mapped_region_ids != source_region_ids:
            raise ValueError(
                "relation unit source regions and NER snippet mappings must agree exactly"
            )
        for mention in unit.mentions:
            matched = False
            for mapping, snippet in resolved_mappings:
                if not (
                    mapping.unit_corrected_start <= mention.corrected_start
                    and mention.corrected_end <= mapping.unit_corrected_end
                ):
                    continue
                expected_corrected_start = mapping.snippet_corrected_start + (
                    mention.corrected_start - mapping.unit_corrected_start
                )
                expected_corrected_end = mapping.snippet_corrected_start + (
                    mention.corrected_end - mapping.unit_corrected_start
                )
                for entity in snippet.adjudication.entities:
                    if (
                        entity.entity_type != mention.entity_type
                        or entity.corrected_start != expected_corrected_start
                        or entity.corrected_end != expected_corrected_end
                        or entity.corrected_text != mention.corrected_text
                    ):
                        continue
                    if entity.raw_start is None:
                        raw_matches = mention.raw_start is None
                    elif mapping.unit_raw_start is None or not (
                        mapping.snippet_raw_start <= entity.raw_start
                        and entity.raw_end <= mapping.snippet_raw_end
                    ):
                        raw_matches = False
                    else:
                        expected_raw_start = mapping.unit_raw_start + (
                            entity.raw_start - mapping.snippet_raw_start
                        )
                        expected_raw_end = mapping.unit_raw_start + (
                            entity.raw_end - mapping.snippet_raw_start
                        )
                        raw_matches = (
                            mention.raw_start == expected_raw_start
                            and mention.raw_end == expected_raw_end
                            and mention.raw_text == entity.raw_text
                        )
                    if raw_matches:
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                raise ValueError(
                    "relation mention is not an exact copy of an adjudicated NER span"
                )
    return ner_gold


class PredictedRelationArgument(StrictModel):
    entity_type: EntityType
    text_start: int = Field(ge=0)
    text_end: int = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_span(self) -> "PredictedRelationArgument":
        if self.text_end <= self.text_start:
            raise ValueError("predicted relation argument spans must be non-empty")
        return self


class PredictedRelation(StrictModel):
    subject: PredictedRelationArgument
    predicate: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    object: PredictedRelationArgument
    evidence_start: int = Field(ge=0)
    evidence_end: int = Field(ge=0)
    evidence_text: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_evidence(self) -> "PredictedRelation":
        if self.evidence_end <= self.evidence_start:
            raise ValueError("predicted relation evidence must be non-empty")
        return self


class RelationPredictionResult(StrictModel):
    unit_id: str = Field(min_length=1, max_length=300)
    gold_unit_id: UUID
    issue_id: str = Field(min_length=1, max_length=300)
    input_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    latency_seconds: float = Field(ge=0)
    relations: list[PredictedRelation] = Field(default_factory=list)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_usage(self) -> "RelationPredictionResult":
        if (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
            and self.prompt_tokens + self.completion_tokens != self.total_tokens
        ):
            raise ValueError("relation prediction token usage does not reconcile")
        if self.estimated_cost_usd is not None and (
            self.prompt_tokens is None or self.completion_tokens is None
        ):
            raise ValueError("relation prediction cost requires token usage")
        return self


class RelationPredictionArtifact(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["relation_benchmark_predictions"] = (
        "relation_benchmark_predictions"
    )
    artifact_id: UUID = Field(default_factory=uuid4)
    dataset_id: str = Field(min_length=1, max_length=300)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ner_gold_dataset_id: str = Field(min_length=1, max_length=300)
    source_ner_gold_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ner_gold_verified: bool
    split: RelationSplit
    input_variant: RelationInputVariant
    adapter_id: str = Field(min_length=1, max_length=300)
    provider: str = Field(min_length=1, max_length=200)
    model_name: str = Field(min_length=1, max_length=1000)
    model_revision: str = Field(min_length=1, max_length=500)
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    token_usage_applicable: bool
    started_at: datetime
    completed_at: datetime
    results: list[RelationPredictionResult] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifact(self) -> "RelationPredictionArtifact":
        if self.completed_at < self.started_at:
            raise ValueError("relation benchmark completion cannot precede start")
        if not _immutable_revision(self.model_revision):
            raise ValueError("relation benchmark model revision must be immutable")
        result_ids = [result.unit_id for result in self.results]
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("relation benchmark result IDs must be unique")
        if not self.token_usage_applicable and any(
            value is not None
            for result in self.results
            for value in (
                result.prompt_tokens,
                result.completion_tokens,
                result.total_tokens,
                result.estimated_cost_usd,
            )
        ):
            raise ValueError("non-generative relation artifacts cannot report token usage")
        return self


def _variant_text(unit: RelationBenchmarkUnit, variant: RelationInputVariant) -> str:
    return unit.corrected_text if variant == "corrected_text" else unit.raw_ocr_text


def _variant_mentions(
    unit: RelationBenchmarkUnit,
    variant: RelationInputVariant,
) -> list[tuple[RelationGoldMention, int, int, str]]:
    if variant == "corrected_text":
        return [
            (
                mention,
                mention.corrected_start,
                mention.corrected_end,
                mention.corrected_text,
            )
            for mention in unit.mentions
        ]
    return [
        (mention, mention.raw_start, mention.raw_end, mention.raw_text)
        for mention in unit.mentions
        if mention.raw_start is not None
    ]


def _variant_mapping_interval(
    mapping: RelationNERTextMapping, variant: RelationInputVariant
) -> tuple[int, int] | None:
    if variant == "corrected_text":
        return mapping.unit_corrected_start, mapping.unit_corrected_end
    if mapping.unit_raw_start is None:
        return None
    return mapping.unit_raw_start, mapping.unit_raw_end


def _rules_configuration(variant: RelationInputVariant) -> dict[str, Any]:
    return {
        "input_variant": variant,
        "reviewed_or_gold_mentions_only": True,
        "cue_location": "strictly_between_exact_argument_spans",
        "clause_boundary_policy": "reject_cross_punctuation",
        "rules": [
            {
                "rule_id": rule.rule_id,
                "subject_type": rule.subject_type,
                "object_type": rule.object_type,
                "predicate": rule.predicate,
                "cue_pattern": rule.cue.pattern,
                "maximum_argument_gap": rule.maximum_argument_gap,
            }
            for rule in RELATION_RULES
        ],
    }


def execute_relation_rule_baseline(
    dataset: RelationBenchmarkDataset,
    *,
    split: RelationSplit,
    input_variant: RelationInputVariant,
    code_revision: str,
    source_ner_gold_bytes: bytes | None = None,
    allow_ineligible_technical_run: bool = False,
) -> RelationPredictionArtifact:
    if not dataset.benchmark_eligible and not allow_ineligible_technical_run:
        raise ValueError(
            "relation benchmark is ineligible: "
            + "; ".join(dataset.eligibility_failures)
        )
    source_ner_gold_verified = source_ner_gold_bytes is not None
    if source_ner_gold_bytes is not None:
        verify_relation_dataset_ner_gold(dataset, source_ner_gold_bytes)
    if dataset.benchmark_eligible and not source_ner_gold_verified:
        raise ValueError("eligible relation runs require verified source NER gold bytes")
    if len(code_revision) != 40 or not all(
        character in "0123456789abcdef" for character in code_revision
    ):
        raise ValueError("code_revision must be a full lowercase git commit")
    units = [unit for unit in dataset.units if unit.split == split]
    if not units:
        raise ValueError(f"relation benchmark has no {split} units")
    started_at = datetime.now(timezone.utc)
    results = []
    for unit in units:
        began = time.perf_counter()
        text = _variant_text(unit, input_variant)
        mention_rows = _variant_mentions(unit, input_variant)
        mentions_by_id = {
            mention.mention_id: (mention, start, end, surface)
            for mention, start, end, surface in mention_rows
        }
        predictions = []
        sources_by_region = {
            pointer.region_id: pointer for pointer in unit.source_regions
        }
        for mapping in unit.source_ner_mappings:
            interval = _variant_mapping_interval(mapping, input_variant)
            if interval is None:
                continue
            unit_start, unit_end = interval
            source = sources_by_region.get(mapping.source_region_id)
            if source is None or not _validate_source_pointer(source):
                raise ValueError("rule baseline requires mapped source provenance")
            local_mentions = [
                ReviewedMention(
                    entity_id=mention.mention_id,
                    entity_type=mention.entity_type.value,
                    text=surface,
                    text_start=start - unit_start,
                    text_end=end - unit_start,
                )
                for mention, start, end, surface in mention_rows
                if unit_start <= start and end <= unit_end
            ]
            claims = extract_region_claims(
                RegionEvidence(
                    region_id=source.region_id,
                    raw_text=text[unit_start:unit_end],
                    polygon=source.polygon.model_dump(mode="json"),
                    source_uri=source.source_uri,
                    source_sha256=source.source_sha256,
                    derivative_id=source.derivative_id,
                    image_sha256=source.image_sha256,
                    evidence_tier=source.evidence_tier,
                    volume_number=source.volume_number,
                    publication_year=source.publication_year,
                    page_number=source.page_number,
                ),
                local_mentions,
                uuid4(),
            )
            for claim in claims:
                subject, subject_start, subject_end, subject_text = mentions_by_id[
                    claim.subject_entity_id
                ]
                object_mention, object_start, object_end, object_text = mentions_by_id[
                    claim.object_entity_id
                ]
                pointer = claim.evidence[0]
                predictions.append(
                    PredictedRelation(
                        subject=PredictedRelationArgument(
                            entity_type=subject.entity_type,
                            text_start=subject_start,
                            text_end=subject_end,
                            text=subject_text,
                        ),
                        predicate=claim.predicate,
                        object=PredictedRelationArgument(
                            entity_type=object_mention.entity_type,
                            text_start=object_start,
                            text_end=object_end,
                            text=object_text,
                        ),
                        evidence_start=unit_start + pointer.text_start,
                        evidence_end=unit_start + pointer.text_end,
                        evidence_text=claim.supporting_quote,
                        confidence=claim.confidence,
                    )
                )
        results.append(
            RelationPredictionResult(
                unit_id=unit.unit_id,
                gold_unit_id=unit.gold_unit_id,
                issue_id=unit.issue_id,
                input_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                latency_seconds=time.perf_counter() - began,
                relations=predictions,
            )
        )
    configuration = _rules_configuration(input_variant)
    warnings = [
        "Rule outputs are benchmark candidates, not reviewed historical claims.",
        "The high-precision rule baseline intentionally misses implicit and cross-clause relations.",
    ]
    if not dataset.benchmark_eligible:
        warnings.append(
            "INELIGIBLE TECHNICAL RUN: " + "; ".join(dataset.eligibility_failures)
        )
    if not source_ner_gold_verified:
        warnings.append("Source NER gold bytes were not verified for this technical run.")
    return RelationPredictionArtifact(
        dataset_id=dataset.dataset_id,
        dataset_sha256=relation_benchmark_dataset_sha256(dataset),
        source_ner_gold_dataset_id=dataset.source_ner_gold_dataset_id,
        source_ner_gold_sha256=dataset.source_ner_gold_sha256,
        source_ner_gold_verified=source_ner_gold_verified,
        split=split,
        input_variant=input_variant,
        adapter_id="reviewed-co-mention-rules-v2",
        provider="deterministic_local",
        model_name="historical-women-relation-rules",
        model_revision="relation-rules-v2",
        configuration_sha256=canonical_sha256(configuration),
        code_revision=code_revision,
        token_usage_applicable=False,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        results=results,
        warnings=warnings,
    )


def _metrics(true_positive: int, predicted: int, expected: int) -> dict[str, Any]:
    precision = true_positive / predicted if predicted else (1.0 if not expected else 0.0)
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


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _gold_relations_for_variant(
    unit: RelationBenchmarkUnit,
    variant: RelationInputVariant,
) -> list[RelationGoldEdge]:
    if variant == "corrected_text":
        return unit.adjudication.relations
    return [
        relation
        for relation in unit.adjudication.relations
        if relation.raw_evidence_start is not None
    ]


def _gold_evidence_interval(
    relation: RelationGoldEdge, variant: RelationInputVariant
) -> tuple[int, int]:
    if variant == "corrected_text":
        return relation.corrected_evidence_start, relation.corrected_evidence_end
    return relation.raw_evidence_start, relation.raw_evidence_end


def score_relation_benchmark(
    dataset: RelationBenchmarkDataset,
    artifact: RelationPredictionArtifact,
) -> dict[str, Any]:
    if artifact.dataset_id != dataset.dataset_id:
        raise ValueError("relation artifact dataset ID disagrees with dataset")
    dataset_sha256 = relation_benchmark_dataset_sha256(dataset)
    if artifact.dataset_sha256 != dataset_sha256:
        raise ValueError("relation artifact dataset hash disagrees with dataset")
    if (
        artifact.source_ner_gold_dataset_id != dataset.source_ner_gold_dataset_id
        or artifact.source_ner_gold_sha256 != dataset.source_ner_gold_sha256
    ):
        raise ValueError("relation artifact NER gold provenance disagrees with dataset")
    if dataset.benchmark_eligible and not artifact.source_ner_gold_verified:
        raise ValueError("eligible relation scores require verified source NER gold")
    units = [unit for unit in dataset.units if unit.split == artifact.split]
    if len(units) != len(artifact.results):
        raise ValueError("relation artifact result count disagrees with dataset split")
    expected_keys = set()
    expected_evidence: dict[tuple[Any, ...], tuple[int, int]] = {}
    valid_prediction_keys = set()
    predicted_count_by_predicate: Counter[str] = Counter()
    expected_count_by_predicate: Counter[str] = Counter()
    true_count_by_predicate: Counter[str] = Counter()
    invalid_predictions = []
    duplicate_predictions = 0
    exact_evidence_keys = set()
    overlap_evidence_keys = set()
    negative_units = 0
    negative_units_with_predictions = 0
    reviewer_intersection = 0
    reviewer_total_a = 0
    reviewer_total_b = 0
    stratified_counts: dict[str, dict[str, list[int]]] = {
        "issue": {},
        "decade": {},
        "layout": {},
        "scan_quality": {},
    }
    total_adjudicated_relations = sum(
        len(unit.adjudication.relations) for unit in units
    )
    raw_recoverable_relations = sum(
        relation.raw_evidence_start is not None
        for unit in units
        for relation in unit.adjudication.relations
    )
    predicate_definitions = {
        definition.predicate: definition for definition in dataset.predicates
    }
    allowed_predicates = set(predicate_definitions)
    for unit, result in zip(units, artifact.results, strict=True):
        expected_identity = (unit.unit_id, unit.gold_unit_id, unit.issue_id)
        result_identity = (result.unit_id, result.gold_unit_id, result.issue_id)
        if result_identity != expected_identity:
            raise ValueError("relation result identity disagrees with dataset")
        target_text = _variant_text(unit, artifact.input_variant)
        if result.input_text_sha256 != hashlib.sha256(
            target_text.encode("utf-8")
        ).hexdigest():
            raise ValueError("relation result input hash disagrees with dataset")
        mention_lookup = {
            (entity_type, start, end, text): mention.mention_id
            for mention, start, end, text in _variant_mentions(
                unit, artifact.input_variant
            )
            for entity_type in [mention.entity_type]
        }
        unit_expected = _gold_relations_for_variant(unit, artifact.input_variant)
        unit_expected_keys = set()
        if not unit.adjudication.relations:
            negative_units += 1
            if result.relations:
                negative_units_with_predictions += 1
        for relation in unit_expected:
            key = (
                unit.gold_unit_id,
                relation.subject_mention_id,
                relation.predicate,
                relation.object_mention_id,
            )
            expected_keys.add(key)
            unit_expected_keys.add(key)
            expected_evidence[key] = _gold_evidence_interval(
                relation, artifact.input_variant
            )
            expected_count_by_predicate[relation.predicate] += 1
        review_sets = [
            {
                (
                    relation.subject_mention_id,
                    relation.predicate,
                    relation.object_mention_id,
                )
                for relation in review.relations
            }
            for review in unit.reviews
        ]
        reviewer_intersection += len(review_sets[0] & review_sets[1])
        reviewer_total_a += len(review_sets[0])
        reviewer_total_b += len(review_sets[1])
        unit_valid_prediction_keys = set()
        for index, prediction in enumerate(result.relations):
            predicted_count_by_predicate[prediction.predicate] += 1
            reason = None
            if prediction.predicate not in allowed_predicates:
                reason = "predicate_outside_frozen_ontology"
            elif prediction.evidence_end > len(target_text):
                reason = "evidence_offsets_out_of_range"
            elif (
                target_text[prediction.evidence_start : prediction.evidence_end]
                != prediction.evidence_text
            ):
                reason = "evidence_text_disagrees_with_offsets"
            elif (
                prediction.evidence_start
                > min(prediction.subject.text_start, prediction.object.text_start)
                or prediction.evidence_end
                < max(prediction.subject.text_end, prediction.object.text_end)
            ):
                reason = "evidence_does_not_contain_arguments"
            subject_id = mention_lookup.get(
                (
                    prediction.subject.entity_type,
                    prediction.subject.text_start,
                    prediction.subject.text_end,
                    prediction.subject.text,
                )
            )
            object_id = mention_lookup.get(
                (
                    prediction.object.entity_type,
                    prediction.object.text_start,
                    prediction.object.text_end,
                    prediction.object.text,
                )
            )
            if reason is None and subject_id is None:
                reason = "subject_not_in_adjudicated_ner_mentions"
            if reason is None and object_id is None:
                reason = "object_not_in_adjudicated_ner_mentions"
            if reason is None:
                definition = predicate_definitions[prediction.predicate]
                if (
                    prediction.subject.entity_type
                    not in definition.subject_entity_types
                    or prediction.object.entity_type
                    not in definition.object_entity_types
                ):
                    reason = "argument_types_violate_predicate_ontology"
            if reason:
                invalid_predictions.append(
                    {
                        "unit_id": unit.unit_id,
                        "prediction_index": index,
                        "predicate": prediction.predicate,
                        "reason": reason,
                    }
                )
                continue
            key = (unit.gold_unit_id, subject_id, prediction.predicate, object_id)
            if key in valid_prediction_keys:
                duplicate_predictions += 1
            valid_prediction_keys.add(key)
            unit_valid_prediction_keys.add(key)
            if key in expected_keys:
                gold_start, gold_end = expected_evidence[key]
                if (prediction.evidence_start, prediction.evidence_end) == (
                    gold_start,
                    gold_end,
                ):
                    exact_evidence_keys.add(key)
                if max(prediction.evidence_start, gold_start) < min(
                    prediction.evidence_end, gold_end
                ):
                    overlap_evidence_keys.add(key)
        unit_true = len(unit_expected_keys & unit_valid_prediction_keys)
        first_source = unit.source_regions[0]
        labels = {
            "issue": unit.issue_id,
            "decade": (
                f"{(first_source.publication_year // 10) * 10}s"
                if first_source.publication_year is not None
                else "unknown"
            ),
            "layout": unit.layout,
            "scan_quality": unit.scan_quality,
        }
        for dimension, label in labels.items():
            counts = stratified_counts[dimension].setdefault(label, [0, 0, 0])
            counts[0] += unit_true
            counts[1] += len(result.relations)
            counts[2] += len(unit_expected_keys)
    true_keys = expected_keys & valid_prediction_keys
    for key in true_keys:
        true_count_by_predicate[key[2]] += 1
    total_predictions = sum(len(result.relations) for result in artifact.results)
    exact = _metrics(len(true_keys), total_predictions, len(expected_keys))
    evidence_exact = _metrics(
        len(exact_evidence_keys), total_predictions, len(expected_keys)
    )
    evidence_overlap = _metrics(
        len(overlap_evidence_keys), total_predictions, len(expected_keys)
    )
    by_predicate = {
        predicate: _metrics(
            true_count_by_predicate[predicate],
            predicted_count_by_predicate[predicate],
            expected_count_by_predicate[predicate],
        )
        for predicate in sorted(
            allowed_predicates | set(predicted_count_by_predicate)
        )
    }
    duration = (artifact.completed_at - artifact.started_at).total_seconds()
    latencies = [result.latency_seconds for result in artifact.results]
    review_denominator = reviewer_total_a + reviewer_total_b
    reviewer_pair_f1 = (
        2 * reviewer_intersection / review_denominator
        if review_denominator
        else 1.0
    )
    usage_results = [
        result
        for result in artifact.results
        if result.prompt_tokens is not None
        and result.completion_tokens is not None
        and result.total_tokens is not None
    ]
    usage_complete = len(usage_results) == len(artifact.results)
    costs = [result.estimated_cost_usd for result in artifact.results]
    costs_complete = all(cost is not None for cost in costs)

    def stratified_metrics(dimension: str) -> dict[str, dict[str, Any]]:
        return {
            label: _metrics(*counts)
            for label, counts in sorted(stratified_counts[dimension].items())
        }

    return {
        "schema_version": "1.0",
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset_sha256,
        "artifact_id": str(artifact.artifact_id),
        "split": artifact.split,
        "input_variant": artifact.input_variant,
        "adapter_id": artifact.adapter_id,
        "provider": artifact.provider,
        "model_name": artifact.model_name,
        "model_revision": artifact.model_revision,
        "configuration_sha256": artifact.configuration_sha256,
        "code_revision": artifact.code_revision,
        "source_ner_gold_verified": artifact.source_ner_gold_verified,
        "units": len(units),
        "expected_relations": len(expected_keys),
        "total_adjudicated_relations": total_adjudicated_relations,
        "raw_recoverable_relations": (
            raw_recoverable_relations
            if artifact.input_variant == "raw_ocr"
            else None
        ),
        "raw_relation_recoverability": (
            raw_recoverable_relations / total_adjudicated_relations
            if artifact.input_variant == "raw_ocr" and total_adjudicated_relations
            else None
        ),
        "predictions_before_validation": total_predictions,
        "valid_unique_predictions": len(valid_prediction_keys),
        "duplicate_predictions": duplicate_predictions,
        "invalid_evidence_predictions": len(invalid_predictions),
        "invalid_evidence_rate": (
            len(invalid_predictions) / total_predictions if total_predictions else 0.0
        ),
        "invalid_prediction_examples": invalid_predictions[:100],
        "exact_relation": exact,
        "end_to_end_raw_relation": (
            _metrics(len(true_keys), total_predictions, total_adjudicated_relations)
            if artifact.input_variant == "raw_ocr"
            else None
        ),
        "exact_relation_and_evidence": evidence_exact,
        "relation_with_overlapping_evidence": evidence_overlap,
        "by_predicate": by_predicate,
        "by_issue": stratified_metrics("issue"),
        "by_decade": stratified_metrics("decade"),
        "by_layout": stratified_metrics("layout"),
        "by_scan_quality": stratified_metrics("scan_quality"),
        "negative_units": negative_units,
        "negative_unit_false_positive_rate": (
            negative_units_with_predictions / negative_units
            if negative_units
            else None
        ),
        "pre_adjudication_relation_pair_f1": reviewer_pair_f1,
        "duration_seconds": duration,
        "latency_p50_seconds": median(latencies),
        "latency_p95_seconds": _percentile(latencies, 0.95),
        "unicode_characters_per_second": (
            sum(len(_variant_text(unit, artifact.input_variant)) for unit in units)
            / duration
            if duration > 0
            else None
        ),
        "token_usage_applicable": artifact.token_usage_applicable,
        "token_usage_complete_rate": (
            len(usage_results) / len(artifact.results)
            if artifact.token_usage_applicable
            else None
        ),
        "total_prompt_tokens": (
            sum(result.prompt_tokens for result in usage_results)
            if artifact.token_usage_applicable and usage_complete
            else None
        ),
        "total_completion_tokens": (
            sum(result.completion_tokens for result in usage_results)
            if artifact.token_usage_applicable and usage_complete
            else None
        ),
        "total_tokens": (
            sum(result.total_tokens for result in usage_results)
            if artifact.token_usage_applicable and usage_complete
            else None
        ),
        "estimated_cost_usd": (
            sum(cost for cost in costs if cost is not None)
            if artifact.token_usage_applicable and costs_complete
            else None
        ),
        "warnings": [
            "Exact relation F1 is conditional on independently adjudicated NER mentions and does not measure entity linking.",
            *(
                ["Raw exact relation F1 is conditional on raw-recoverable relations; end_to_end_raw_relation also counts relations lost before extraction."]
                if artifact.input_variant == "raw_ocr"
                else []
            ),
            "Evidence-span agreement is structural; historians must still judge whether the passage entails the relation.",
            "No benchmark score authorizes automatic claim acceptance or graph promotion.",
            *(
                ["Generative token usage is incomplete; totals and cost were not imputed."]
                if artifact.token_usage_applicable and not usage_complete
                else []
            ),
            *(
                ["Generative cost is incomplete because one or more case costs are unavailable."]
                if artifact.token_usage_applicable and not costs_complete
                else []
            ),
        ],
    }


def compare_relation_reports(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    *,
    label_a: str,
    label_b: str,
    bootstrap_seed: int = 17,
    bootstrap_resamples: int = 5000,
) -> dict[str, Any]:
    if not label_a or not label_b or label_a == label_b:
        raise ValueError("relation comparison labels must be non-empty and distinct")
    if bootstrap_resamples < 100:
        raise ValueError("relation comparison requires at least 100 resamples")
    identity_fields = ("dataset_id", "dataset_sha256", "split", "input_variant")
    if any(report_a.get(field) != report_b.get(field) for field in identity_fields):
        raise ValueError("relation reports do not use the same frozen evaluation slice")
    if report_a.get("artifact_id") == report_b.get("artifact_id"):
        raise ValueError("relation comparison requires distinct prediction artifacts")
    issue_a = report_a.get("by_issue")
    issue_b = report_b.get("by_issue")
    if not isinstance(issue_a, dict) or not isinstance(issue_b, dict):
        raise ValueError("relation reports lack issue-level sufficient statistics")
    if set(issue_a) != set(issue_b) or not issue_a:
        raise ValueError("relation reports do not cover identical issue clusters")
    issue_ids = sorted(issue_a)

    def validated_counts(value: Any) -> tuple[int, int, int]:
        if not isinstance(value, dict):
            raise ValueError("relation issue metric is malformed")
        counts = tuple(
            value.get(name)
            for name in ("true_positive", "false_positive", "false_negative")
        )
        if any(not isinstance(count, int) or count < 0 for count in counts):
            raise ValueError("relation issue counts must be nonnegative integers")
        true_positive, false_positive, false_negative = counts
        return (
            true_positive,
            true_positive + false_positive,
            true_positive + false_negative,
        )

    counts_a = {issue: validated_counts(issue_a[issue]) for issue in issue_ids}
    counts_b = {issue: validated_counts(issue_b[issue]) for issue in issue_ids}

    def aggregate_f1(
        counts: dict[str, tuple[int, int, int]], sampled_issues: list[str]
    ) -> float:
        true_positive = sum(counts[issue][0] for issue in sampled_issues)
        predicted = sum(counts[issue][1] for issue in sampled_issues)
        expected = sum(counts[issue][2] for issue in sampled_issues)
        return _metrics(true_positive, predicted, expected)["f1"]

    observed_a = aggregate_f1(counts_a, issue_ids)
    observed_b = aggregate_f1(counts_b, issue_ids)
    rng = random.Random(bootstrap_seed)
    differences = []
    for _ in range(bootstrap_resamples):
        sample = [issue_ids[rng.randrange(len(issue_ids))] for _ in issue_ids]
        differences.append(
            aggregate_f1(counts_b, sample) - aggregate_f1(counts_a, sample)
        )
    issue_differences = [
        aggregate_f1(counts_b, [issue]) - aggregate_f1(counts_a, [issue])
        for issue in issue_ids
    ]
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": report_a["dataset_id"],
        "dataset_sha256": report_a["dataset_sha256"],
        "split": report_a["split"],
        "input_variant": report_a["input_variant"],
        "label_a": label_a,
        "label_b": label_b,
        "artifact_id_a": report_a["artifact_id"],
        "artifact_id_b": report_b["artifact_id"],
        "issue_clusters": len(issue_ids),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "exact_f1_a": observed_a,
        "exact_f1_b": observed_b,
        "exact_f1_difference_b_minus_a": observed_b - observed_a,
        "exact_f1_difference_ci95": [
            _percentile(differences, 0.025),
            _percentile(differences, 0.975),
        ],
        "b_issue_wins": sum(value > 0 for value in issue_differences),
        "issue_ties": sum(value == 0 for value in issue_differences),
        "a_issue_wins": sum(value < 0 for value in issue_differences),
        "warnings": [
            "The interval resamples whole issues to preserve within-issue dependence.",
            "A relative F1 win cannot override invalid-evidence, unsupported-claim, rights, cost, or historian-review gates.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-rules")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--ner-gold", type=Path, required=True)
    run.add_argument("--split", choices=("train", "development", "test"), required=True)
    run.add_argument("--input-variant", choices=("corrected_text", "raw_ocr"), required=True)
    run.add_argument("--code-revision", required=True)
    run.add_argument("--allow-ineligible-technical-run", action="store_true")
    run.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--ner-gold", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--report-a", type=Path, required=True)
    compare.add_argument("--label-a", required=True)
    compare.add_argument("--report-b", type=Path, required=True)
    compare.add_argument("--label-b", required=True)
    compare.add_argument("--bootstrap-seed", type=int, default=17)
    compare.add_argument("--bootstrap-resamples", type=int, default=5000)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        report_a = json.loads(args.report_a.read_text(encoding="utf-8"))
        report_b = json.loads(args.report_b.read_text(encoding="utf-8"))
        result = compare_relation_reports(
            report_a,
            report_b,
            label_a=args.label_a,
            label_b=args.label_b,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(args.output)}, sort_keys=True))
        return 0
    dataset = RelationBenchmarkDataset.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    ner_gold_bytes = args.ner_gold.read_bytes()
    verify_relation_dataset_ner_gold(dataset, ner_gold_bytes)
    if args.command == "run-rules":
        result: StrictModel | dict[str, Any] = execute_relation_rule_baseline(
            dataset,
            split=args.split,
            input_variant=args.input_variant,
            code_revision=args.code_revision,
            source_ner_gold_bytes=ner_gold_bytes,
            allow_ineligible_technical_run=args.allow_ineligible_technical_run,
        )
    else:
        predictions = RelationPredictionArtifact.model_validate_json(
            args.predictions.read_text(encoding="utf-8")
        )
        result = score_relation_benchmark(dataset, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        result.model_dump(mode="json")
        if isinstance(result, StrictModel)
        else result
    )
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
