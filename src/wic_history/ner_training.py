"""Export adjudicated, issue-split NER gold into provenance-safe W2NER files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence
from uuid import UUID

from pydantic import Field, model_validator

from .evidence import EntityType, SourcePointer, StrictModel
from .ner_adapters.base import (
    BenchmarkInput,
    BenchmarkSplit,
    IssueSplitManifest,
    NERBenchmarkDataset,
    benchmark_dataset_sha256,
)
from .ner_benchmark import prepare_benchmark_dataset
from .ner_gold import GoldEntitySpan, GoldSnippet, NERGoldSet


W2NER_IMPLEMENTATION_REVISION = "a34ff841891919001080edefb50e14fa9dc15e1c"
BOUNDARY_CHARACTERS = frozenset("\n\r。！？!?；;")


def _full_commit(value: str) -> bool:
    return len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class EmpiricalSubstitution(StrictModel):
    corrected_character: str = Field(min_length=1, max_length=1)
    raw_ocr_character: str = Field(min_length=1, max_length=1)
    count: int = Field(ge=1)
    source_snippet_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_substitution(self) -> "EmpiricalSubstitution":
        if self.corrected_character == self.raw_ocr_character:
            raise ValueError("empirical substitutions must change a character")
        if self.source_snippet_ids != sorted(set(self.source_snippet_ids)):
            raise ValueError(
                "substitution source snippet IDs must be sorted and unique"
            )
        return self


class AugmentationConfiguration(StrictModel):
    method: Literal["training_only_length_preserving_empirical_substitution"] = (
        "training_only_length_preserving_empirical_substitution"
    )
    probability: float = Field(default=0.15, ge=0, le=1)
    augmented_copies_per_clean_record: int = Field(default=1, ge=0, le=20)
    seed: int = Field(default=17, ge=0, le=2**63 - 1)
    confusion_source: Literal["training_gold_recoverable_entity_alignments"] = (
        "training_gold_recoverable_entity_alignments"
    )
    insertion_deletion_policy: Literal["forbidden_without_explicit_edit_map"] = (
        "forbidden_without_explicit_edit_map"
    )


class SubstitutionEvent(StrictModel):
    record_character_index: int = Field(ge=0)
    corrected_character: str = Field(min_length=1, max_length=1)
    raw_ocr_character: str = Field(min_length=1, max_length=1)
    evidence_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_event(self) -> "SubstitutionEvent":
        if self.corrected_character == self.raw_ocr_character:
            raise ValueError("augmentation events must change a character")
        return self


class W2NEREntity(StrictModel):
    index: list[int] = Field(min_length=1)
    entity_type: EntityType
    snippet_start: int = Field(ge=0)
    snippet_end: int = Field(ge=0)
    record_start: int = Field(ge=0)
    record_end: int = Field(ge=0)
    source_surface: str = Field(min_length=1)
    training_surface: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entity(self) -> "W2NEREntity":
        if (
            self.snippet_end <= self.snippet_start
            or self.record_end <= self.record_start
        ):
            raise ValueError("W2NER entity spans must be non-empty")
        if self.index != sorted(set(self.index)):
            raise ValueError("W2NER entity token indices must be sorted and unique")
        return self


class W2NERTrainingRecord(StrictModel):
    record_id: str = Field(min_length=1, max_length=1000)
    snippet_id: str = Field(min_length=1, max_length=300)
    issue_id: str = Field(min_length=1, max_length=300)
    split: BenchmarkSplit
    input_variant: Literal["raw_ocr", "corrected_text"]
    augmentation_kind: Literal["clean", "empirical_substitution"]
    augmentation_copy: int = Field(ge=0)
    gold_region_id: UUID
    source_ocr_run_id: UUID
    source_ocr_region_id: UUID
    source: SourcePointer
    snippet_start: int = Field(ge=0)
    snippet_end: int = Field(ge=0)
    source_text: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_text: str = Field(min_length=1)
    training_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sentence: list[str] = Field(min_length=1)
    token_character_offsets: list[int] = Field(min_length=1)
    entities: list[W2NEREntity] = Field(default_factory=list)
    substitutions: list[SubstitutionEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_record(self) -> "W2NERTrainingRecord":
        if self.source.region_id != self.source_ocr_region_id:
            raise ValueError("training record must cite its exact source OCR region")
        if self.gold_region_id == self.source_ocr_region_id:
            raise ValueError("training record gold identity must be model-independent")
        if self.snippet_end <= self.snippet_start:
            raise ValueError("training record snippet range must be non-empty")
        if self.snippet_end - self.snippet_start != len(self.source_text):
            raise ValueError("training record snippet range disagrees with source text")
        if len(self.training_text) != len(self.source_text):
            raise ValueError(
                "training augmentation must preserve Unicode character length"
            )
        if _sha256_text(self.source_text) != self.source_text_sha256:
            raise ValueError("training record source-text hash disagrees with text")
        if _sha256_text(self.training_text) != self.training_text_sha256:
            raise ValueError("training record augmented-text hash disagrees with text")
        expected_offsets = [
            index
            for index, character in enumerate(self.training_text)
            if not character.isspace()
        ]
        if self.token_character_offsets != expected_offsets:
            raise ValueError(
                "W2NER token offsets must cover every non-whitespace character exactly"
            )
        if self.sentence != [self.training_text[index] for index in expected_offsets]:
            raise ValueError(
                "W2NER sentence tokens disagree with mapped source characters"
            )
        if any(len(token) != 1 for token in self.sentence):
            raise ValueError("W2NER export requires one Unicode character per token")

        if self.augmentation_kind == "clean":
            if self.augmentation_copy != 0 or self.substitutions:
                raise ValueError("clean records cannot contain augmentation metadata")
            if self.training_text != self.source_text:
                raise ValueError("clean training text must equal source text")
        else:
            if self.split != "train" or self.input_variant != "corrected_text":
                raise ValueError(
                    "empirical augmentation is allowed only on corrected training data"
                )
            if self.augmentation_copy < 1 or not self.substitutions:
                raise ValueError("augmented records require a positive copy and events")
            positions = [event.record_character_index for event in self.substitutions]
            if positions != sorted(set(positions)):
                raise ValueError(
                    "augmentation event positions must be sorted and unique"
                )
            reconstructed = list(self.source_text)
            for event in self.substitutions:
                if event.record_character_index >= len(reconstructed):
                    raise ValueError("augmentation event is outside the source text")
                if (
                    reconstructed[event.record_character_index]
                    != event.corrected_character
                ):
                    raise ValueError(
                        "augmentation event disagrees with source character"
                    )
                reconstructed[event.record_character_index] = event.raw_ocr_character
            if "".join(reconstructed) != self.training_text:
                raise ValueError("augmentation events do not reconstruct training text")

        grid_labels: dict[tuple[int, int], EntityType] = {}
        for entity in self.entities:
            if not (
                self.snippet_start
                <= entity.snippet_start
                < entity.snippet_end
                <= self.snippet_end
            ):
                raise ValueError(
                    "entity snippet offsets are outside its training record"
                )
            if (
                entity.record_start != entity.snippet_start - self.snippet_start
                or entity.record_end != entity.snippet_end - self.snippet_start
            ):
                raise ValueError("entity record and snippet offsets disagree")
            if (
                self.source_text[entity.record_start : entity.record_end]
                != entity.source_surface
            ):
                raise ValueError("entity source surface disagrees with exact offsets")
            if (
                self.training_text[entity.record_start : entity.record_end]
                != entity.training_surface
            ):
                raise ValueError("entity training surface disagrees with exact offsets")
            expected_indices = [
                token_index
                for token_index, character_offset in enumerate(
                    self.token_character_offsets
                )
                if entity.record_start <= character_offset < entity.record_end
            ]
            if not expected_indices:
                raise ValueError("an entity cannot contain only whitespace tokens")
            if entity.index != expected_indices:
                raise ValueError("entity token indices disagree with character offsets")
            grid_key = (entity.index[-1], entity.index[0])
            prior_type = grid_labels.get(grid_key)
            if prior_type is not None and prior_type != entity.entity_type:
                raise ValueError(
                    "W2NER cannot encode different types on the same tail-head grid cell"
                )
            grid_labels[grid_key] = entity.entity_type
        return self


class SnippetTrainingCoverage(StrictModel):
    snippet_id: str = Field(min_length=1, max_length=300)
    split: BenchmarkSplit
    adjudicated_entities: int = Field(ge=0)
    raw_recoverable_entities: int = Field(ge=0)
    raw_unrecoverable_entities: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_coverage(self) -> "SnippetTrainingCoverage":
        if (
            self.raw_recoverable_entities + self.raw_unrecoverable_entities
            != self.adjudicated_entities
        ):
            raise ValueError(
                "raw recoverability counts must cover adjudicated entities"
            )
        return self


class OmittedTrainingInput(StrictModel):
    snippet_id: str = Field(min_length=1, max_length=300)
    issue_id: str = Field(min_length=1, max_length=300)
    split: BenchmarkSplit
    input_variant: Literal["raw_ocr", "corrected_text"]
    snippet_start: int = Field(ge=0)
    snippet_end: int = Field(ge=0)
    reason: Literal["empty_text", "no_non_whitespace_model_tokens"]


class W2NERView(StrictModel):
    view_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,199}$")
    filename: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,199}\.json$")
    split: BenchmarkSplit
    input_variant: Literal["raw_ocr", "corrected_text"]
    augmentation_policy: Literal["clean_only", "clean_plus_empirical_substitution"]
    record_ids: list[str] = Field(min_length=1)
    native_json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def native_w2ner_record(record: W2NERTrainingRecord) -> dict[str, Any]:
    """Return only the two keys consumed by the pinned official W2NER loader."""
    return {
        "sentence": record.sentence,
        "ner": [
            {"index": entity.index, "type": entity.entity_type.value}
            for entity in record.entities
        ],
    }


def native_w2ner_json_bytes(records: Sequence[W2NERTrainingRecord]) -> bytes:
    payload = [native_w2ner_record(record) for record in records]
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _view_identity(
    split: BenchmarkSplit,
    input_variant: Literal["raw_ocr", "corrected_text"],
    augmentation_policy: Literal["clean_only", "clean_plus_empirical_substitution"],
) -> tuple[str, str]:
    suffix = "clean" if augmentation_policy == "clean_only" else "empirical-noise"
    view_id = f"{split}.{input_variant}.{suffix}"
    return view_id, f"{view_id}.json"


class W2NERTrainingExport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["w2ner_training_export"] = "w2ner_training_export"
    export_id: str = Field(min_length=1, max_length=300)
    generated_at: datetime
    project_code_revision: str
    w2ner_implementation_revision: str
    source_benchmark_dataset_id: str = Field(min_length=1, max_length=300)
    source_benchmark_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_gold_dataset_id: str = Field(min_length=1, max_length=300)
    source_gold_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ontology_version: str = Field(min_length=1, max_length=100)
    benchmark_eligible: bool
    technical_export: bool
    maximum_record_characters: int = Field(ge=16, le=1024)
    tokenization_policy: Literal[
        "unicode_codepoints_omit_whitespace_with_exact_character_map"
    ] = "unicode_codepoints_omit_whitespace_with_exact_character_map"
    augmentation: AugmentationConfiguration
    empirical_substitutions: list[EmpiricalSubstitution]
    coverage: list[SnippetTrainingCoverage] = Field(min_length=1)
    records: list[W2NERTrainingRecord] = Field(min_length=1)
    omitted_inputs: list[OmittedTrainingInput] = Field(default_factory=list)
    views: list[W2NERView] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_export(self) -> "W2NERTrainingExport":
        if not _full_commit(self.project_code_revision):
            raise ValueError("project code revision must be a full lowercase commit")
        if not _full_commit(self.w2ner_implementation_revision):
            raise ValueError(
                "W2NER implementation revision must be a full lowercase commit"
            )
        if self.technical_export != (not self.benchmark_eligible):
            raise ValueError(
                "technical export state must reflect benchmark eligibility"
            )

        record_ids = [record.record_id for record in self.records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("W2NER training record IDs must be unique")
        record_by_id = {record.record_id: record for record in self.records}
        train_snippets = {
            record.snippet_id for record in self.records if record.split == "train"
        }
        confusion_pairs = set()
        confusion_by_pair = {}
        for confusion in self.empirical_substitutions:
            key = (confusion.corrected_character, confusion.raw_ocr_character)
            if key in confusion_pairs:
                raise ValueError("empirical substitution pairs must be unique")
            confusion_pairs.add(key)
            confusion_by_pair[key] = confusion
            if not set(confusion.source_snippet_ids) <= train_snippets:
                raise ValueError(
                    "empirical substitutions may cite training snippets only"
                )
        for record in self.records:
            for event in record.substitutions:
                confusion = confusion_by_pair.get(
                    (event.corrected_character, event.raw_ocr_character)
                )
                if confusion is None or event.evidence_count != confusion.count:
                    raise ValueError(
                        "augmentation event must cite an exact empirical substitution"
                    )

        coverage_ids = [item.snippet_id for item in self.coverage]
        if len(set(coverage_ids)) != len(coverage_ids):
            raise ValueError("snippet training coverage IDs must be unique")

        view_ids = [view.view_id for view in self.views]
        filenames = [view.filename for view in self.views]
        if len(set(view_ids)) != len(view_ids) or len(set(filenames)) != len(filenames):
            raise ValueError("W2NER view IDs and filenames must be unique")

        expected_views: dict[str, tuple[str, list[W2NERTrainingRecord]]] = {}
        for split in ("train", "development", "test"):
            for input_variant in ("raw_ocr", "corrected_text"):
                clean_records = [
                    record
                    for record in self.records
                    if record.split == split
                    and record.input_variant == input_variant
                    and record.augmentation_kind == "clean"
                ]
                if not clean_records:
                    continue
                view_id, filename = _view_identity(split, input_variant, "clean_only")
                expected_views[view_id] = (filename, clean_records)
                if split == "train" and input_variant == "corrected_text":
                    all_records = [
                        record
                        for record in self.records
                        if record.split == split
                        and record.input_variant == input_variant
                    ]
                    view_id, filename = _view_identity(
                        split,
                        input_variant,
                        "clean_plus_empirical_substitution",
                    )
                    expected_views[view_id] = (filename, all_records)
        if set(view_ids) != set(expected_views):
            raise ValueError(
                "W2NER views do not exactly cover materializable record scopes"
            )
        for view in self.views:
            filename, records = expected_views[view.view_id]
            expected_policy = (
                "clean_plus_empirical_substitution"
                if view.view_id.endswith("empirical-noise")
                else "clean_only"
            )
            if (
                view.filename != filename
                or view.augmentation_policy != expected_policy
                or view.record_ids != [record.record_id for record in records]
            ):
                raise ValueError("W2NER view scope disagrees with training records")
            selected = [record_by_id[record_id] for record_id in view.record_ids]
            expected_sha256 = hashlib.sha256(
                native_w2ner_json_bytes(selected)
            ).hexdigest()
            if view.native_json_sha256 != expected_sha256:
                raise ValueError("W2NER native view hash disagrees with its records")
        return self


def w2ner_training_export_sha256(export: W2NERTrainingExport) -> str:
    return hashlib.sha256(
        json.dumps(
            export.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def derive_training_substitutions(
    gold: NERGoldSet, split_by_snippet: dict[str, BenchmarkSplit]
) -> list[EmpiricalSubstitution]:
    """Derive only unambiguous, length-preserving mappings from training mentions."""
    counts: Counter[tuple[str, str]] = Counter()
    sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for snippet in gold.snippets:
        if split_by_snippet.get(snippet.snippet_id) != "train":
            continue
        seen_alignments = set()
        for entity in snippet.adjudication.entities:
            if entity.raw_text is None or len(entity.raw_text) != len(
                entity.corrected_text
            ):
                continue
            alignment_key = (
                entity.corrected_start,
                entity.corrected_end,
                entity.raw_start,
                entity.raw_end,
            )
            if alignment_key in seen_alignments:
                continue
            seen_alignments.add(alignment_key)
            for corrected_character, raw_character in zip(
                entity.corrected_text, entity.raw_text, strict=True
            ):
                if corrected_character == raw_character:
                    continue
                key = (corrected_character, raw_character)
                counts[key] += 1
                sources[key].add(snippet.snippet_id)
    return [
        EmpiricalSubstitution(
            corrected_character=corrected_character,
            raw_ocr_character=raw_character,
            count=counts[(corrected_character, raw_character)],
            source_snippet_ids=sorted(sources[(corrected_character, raw_character)]),
        )
        for corrected_character, raw_character in sorted(counts)
    ]


def _entity_offsets(
    entity: GoldEntitySpan,
    input_variant: Literal["raw_ocr", "corrected_text"],
) -> tuple[int, int, str] | None:
    if input_variant == "corrected_text":
        return entity.corrected_start, entity.corrected_end, entity.corrected_text
    if entity.raw_start is None:
        return None
    return entity.raw_start, entity.raw_end, entity.raw_text


def _chunk_ranges(
    text: str,
    entity_spans: list[tuple[int, int]],
    maximum_characters: int,
) -> list[tuple[int, int]]:
    if any(end - start > maximum_characters for start, end in entity_spans):
        raise ValueError("an adjudicated entity exceeds the W2NER record size limit")
    ranges = []
    start = 0
    while start < len(text):
        hard_end = min(start + maximum_characters, len(text))
        end = hard_end
        crossing = [
            entity_start
            for entity_start, entity_end in entity_spans
            if entity_start < end < entity_end
        ]
        if crossing:
            end = min(crossing)
        elif end < len(text):
            lower = max(start + 1, start + (maximum_characters // 2))
            for candidate in range(end, lower - 1, -1):
                if text[candidate - 1] not in BOUNDARY_CHARACTERS:
                    continue
                if any(
                    entity_start < candidate < entity_end
                    for entity_start, entity_end in entity_spans
                ):
                    continue
                end = candidate
                break
        if end <= start:
            raise ValueError("unable to chunk text without splitting an entity")
        ranges.append((start, end))
        start = end
    return ranges


def _record_id(
    *,
    input_id: str,
    snippet_start: int,
    snippet_end: int,
    augmentation_copy: int,
) -> str:
    digest = hashlib.sha256(
        f"{input_id}\0{snippet_start}\0{snippet_end}\0{augmentation_copy}".encode()
    ).hexdigest()[:24]
    suffix = "clean" if augmentation_copy == 0 else f"aug-{augmentation_copy}"
    return f"{suffix}-{digest}"


def _make_clean_records(
    snippet: GoldSnippet,
    benchmark_input: BenchmarkInput,
    maximum_characters: int,
) -> tuple[list[W2NERTrainingRecord], list[OmittedTrainingInput]]:
    input_variant: Literal["raw_ocr", "corrected_text"] = benchmark_input.input_variant
    entity_data = []
    for entity in snippet.adjudication.entities:
        offsets = _entity_offsets(entity, input_variant)
        if offsets is not None:
            entity_data.append((entity, *offsets))
    text = benchmark_input.text
    if not text:
        return [], [
            OmittedTrainingInput(
                snippet_id=snippet.snippet_id,
                issue_id=benchmark_input.issue_id,
                split=benchmark_input.split,
                input_variant=input_variant,
                snippet_start=0,
                snippet_end=0,
                reason="empty_text",
            )
        ]
    ranges = _chunk_ranges(
        text,
        [(start, end) for _, start, end, _ in entity_data],
        maximum_characters,
    )
    records = []
    omitted = []
    for snippet_start, snippet_end in ranges:
        source_text = text[snippet_start:snippet_end]
        token_offsets = [
            index
            for index, character in enumerate(source_text)
            if not character.isspace()
        ]
        if not token_offsets:
            omitted.append(
                OmittedTrainingInput(
                    snippet_id=snippet.snippet_id,
                    issue_id=benchmark_input.issue_id,
                    split=benchmark_input.split,
                    input_variant=input_variant,
                    snippet_start=snippet_start,
                    snippet_end=snippet_end,
                    reason="no_non_whitespace_model_tokens",
                )
            )
            continue
        entities = []
        for entity, entity_start, entity_end, entity_text in entity_data:
            if not (snippet_start <= entity_start < entity_end <= snippet_end):
                continue
            record_start = entity_start - snippet_start
            record_end = entity_end - snippet_start
            entity_indices = [
                token_index
                for token_index, character_offset in enumerate(token_offsets)
                if record_start <= character_offset < record_end
            ]
            entities.append(
                W2NEREntity(
                    index=entity_indices,
                    entity_type=entity.entity_type,
                    snippet_start=entity_start,
                    snippet_end=entity_end,
                    record_start=record_start,
                    record_end=record_end,
                    source_surface=entity_text,
                    training_surface=entity_text,
                )
            )
        records.append(
            W2NERTrainingRecord(
                record_id=_record_id(
                    input_id=benchmark_input.input_id,
                    snippet_start=snippet_start,
                    snippet_end=snippet_end,
                    augmentation_copy=0,
                ),
                snippet_id=snippet.snippet_id,
                issue_id=benchmark_input.issue_id,
                split=benchmark_input.split,
                input_variant=input_variant,
                augmentation_kind="clean",
                augmentation_copy=0,
                gold_region_id=benchmark_input.gold_region_id,
                source_ocr_run_id=benchmark_input.source_ocr_run_id,
                source_ocr_region_id=benchmark_input.source_ocr_region_id,
                source=snippet.source,
                snippet_start=snippet_start,
                snippet_end=snippet_end,
                source_text=source_text,
                source_text_sha256=_sha256_text(source_text),
                training_text=source_text,
                training_text_sha256=_sha256_text(source_text),
                sentence=[source_text[index] for index in token_offsets],
                token_character_offsets=token_offsets,
                entities=entities,
            )
        )
    return records, omitted


def _deterministic_digest(
    seed: int,
    record_id: str,
    copy_number: int,
    character_index: int,
    character: str,
) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{record_id}\0{copy_number}\0{character_index}\0{character}".encode(
            "utf-8"
        )
    ).digest()


def _augment_record(
    record: W2NERTrainingRecord,
    substitutions: list[EmpiricalSubstitution],
    configuration: AugmentationConfiguration,
    copy_number: int,
) -> W2NERTrainingRecord | None:
    options: dict[str, list[EmpiricalSubstitution]] = defaultdict(list)
    for substitution in substitutions:
        options[substitution.corrected_character].append(substitution)
    threshold = int(configuration.probability * (2**64))
    training_characters = list(record.source_text)
    events = []
    for character_index, character in enumerate(record.source_text):
        candidates = options.get(character)
        if not candidates:
            continue
        digest = _deterministic_digest(
            configuration.seed,
            record.record_id,
            copy_number,
            character_index,
            character,
        )
        if int.from_bytes(digest[:8], "big") >= threshold:
            continue
        total = sum(candidate.count for candidate in candidates)
        selection = int.from_bytes(digest[8:16], "big") % total
        selected = candidates[-1]
        cumulative = 0
        for candidate in candidates:
            cumulative += candidate.count
            if selection < cumulative:
                selected = candidate
                break
        training_characters[character_index] = selected.raw_ocr_character
        events.append(
            SubstitutionEvent(
                record_character_index=character_index,
                corrected_character=character,
                raw_ocr_character=selected.raw_ocr_character,
                evidence_count=selected.count,
            )
        )
    if not events:
        return None
    training_text = "".join(training_characters)
    token_offsets = [
        index
        for index, character in enumerate(training_text)
        if not character.isspace()
    ]
    entities = [
        entity.model_copy(
            update={
                "training_surface": training_text[
                    entity.record_start : entity.record_end
                ]
            }
        )
        for entity in record.entities
    ]
    return W2NERTrainingRecord(
        **record.model_dump(
            exclude={
                "record_id",
                "augmentation_kind",
                "augmentation_copy",
                "training_text",
                "training_text_sha256",
                "sentence",
                "token_character_offsets",
                "entities",
                "substitutions",
            }
        ),
        record_id=_record_id(
            input_id=record.record_id,
            snippet_start=record.snippet_start,
            snippet_end=record.snippet_end,
            augmentation_copy=copy_number,
        ),
        augmentation_kind="empirical_substitution",
        augmentation_copy=copy_number,
        training_text=training_text,
        training_text_sha256=_sha256_text(training_text),
        sentence=[training_text[index] for index in token_offsets],
        token_character_offsets=token_offsets,
        entities=entities,
        substitutions=events,
    )


def _build_views(records: list[W2NERTrainingRecord]) -> list[W2NERView]:
    views = []
    for split in ("train", "development", "test"):
        for input_variant in ("raw_ocr", "corrected_text"):
            clean_records = [
                record
                for record in records
                if record.split == split
                and record.input_variant == input_variant
                and record.augmentation_kind == "clean"
            ]
            if not clean_records:
                continue
            view_id, filename = _view_identity(split, input_variant, "clean_only")
            views.append(
                W2NERView(
                    view_id=view_id,
                    filename=filename,
                    split=split,
                    input_variant=input_variant,
                    augmentation_policy="clean_only",
                    record_ids=[record.record_id for record in clean_records],
                    native_json_sha256=hashlib.sha256(
                        native_w2ner_json_bytes(clean_records)
                    ).hexdigest(),
                )
            )
            if split == "train" and input_variant == "corrected_text":
                all_records = [
                    record
                    for record in records
                    if record.split == split and record.input_variant == input_variant
                ]
                view_id, filename = _view_identity(
                    split,
                    input_variant,
                    "clean_plus_empirical_substitution",
                )
                views.append(
                    W2NERView(
                        view_id=view_id,
                        filename=filename,
                        split=split,
                        input_variant=input_variant,
                        augmentation_policy="clean_plus_empirical_substitution",
                        record_ids=[record.record_id for record in all_records],
                        native_json_sha256=hashlib.sha256(
                            native_w2ner_json_bytes(all_records)
                        ).hexdigest(),
                    )
                )
    return views


def build_w2ner_training_export(
    gold: NERGoldSet,
    dataset: NERBenchmarkDataset,
    *,
    export_id: str,
    project_code_revision: str,
    maximum_record_characters: int = 256,
    augmentation: AugmentationConfiguration | None = None,
    allow_ineligible_technical_export: bool = False,
    generated_at: datetime | None = None,
) -> W2NERTrainingExport:
    if gold.schema_version != "1.1":
        raise ValueError("W2NER training exports require NER gold schema 1.1")
    if not dataset.benchmark_eligible and not allow_ineligible_technical_export:
        raise ValueError(
            "benchmark dataset is ineligible for training export: "
            + "; ".join(dataset.eligibility_failures)
        )
    if (
        dataset.source_gold_dataset_id != gold.dataset_id
        or dataset.ontology_version != gold.ontology_version
    ):
        raise ValueError("benchmark dataset identity disagrees with NER gold")
    if not 16 <= maximum_record_characters <= 1024:
        raise ValueError("maximum record characters must be between 16 and 1024")
    configuration = augmentation or AugmentationConfiguration()

    snippets_by_id = {snippet.snippet_id: snippet for snippet in gold.snippets}
    inputs_by_scope = {
        (item.snippet_id, item.input_variant): item for item in dataset.inputs
    }
    expected_scopes = {
        (snippet.snippet_id, variant)
        for snippet in gold.snippets
        for variant in ("raw_ocr", "corrected_text")
    }
    if set(inputs_by_scope) != expected_scopes:
        raise ValueError(
            "W2NER export requires exactly paired raw and corrected inputs per snippet"
        )
    split_by_snippet = {}
    for snippet in gold.snippets:
        raw_input = inputs_by_scope[(snippet.snippet_id, "raw_ocr")]
        corrected_input = inputs_by_scope[(snippet.snippet_id, "corrected_text")]
        if (
            raw_input.split != corrected_input.split
            or raw_input.issue_id != corrected_input.issue_id
        ):
            raise ValueError("paired benchmark inputs disagree on issue split")
        if raw_input.text != snippet.raw_ocr_text:
            raise ValueError("raw benchmark input disagrees with exact gold text")
        if corrected_input.text != snippet.adjudication.corrected_text:
            raise ValueError("corrected benchmark input disagrees with adjudication")
        split_by_snippet[snippet.snippet_id] = raw_input.split

    substitutions = derive_training_substitutions(gold, split_by_snippet)
    records = []
    omitted = []
    for benchmark_input in dataset.inputs:
        if benchmark_input.input_variant not in {"raw_ocr", "corrected_text"}:
            continue
        snippet = snippets_by_id[benchmark_input.snippet_id]
        clean_records, clean_omitted = _make_clean_records(
            snippet, benchmark_input, maximum_record_characters
        )
        records.extend(clean_records)
        omitted.extend(clean_omitted)
        if (
            benchmark_input.split == "train"
            and benchmark_input.input_variant == "corrected_text"
            and configuration.augmented_copies_per_clean_record
        ):
            for clean_record in clean_records:
                seen_training_text = {clean_record.training_text}
                for copy_number in range(
                    1, configuration.augmented_copies_per_clean_record + 1
                ):
                    augmented = _augment_record(
                        clean_record, substitutions, configuration, copy_number
                    )
                    if (
                        augmented is None
                        or augmented.training_text in seen_training_text
                    ):
                        continue
                    seen_training_text.add(augmented.training_text)
                    records.append(augmented)

    coverage = []
    for snippet in gold.snippets:
        recoverable = sum(
            entity.raw_start is not None for entity in snippet.adjudication.entities
        )
        coverage.append(
            SnippetTrainingCoverage(
                snippet_id=snippet.snippet_id,
                split=split_by_snippet[snippet.snippet_id],
                adjudicated_entities=len(snippet.adjudication.entities),
                raw_recoverable_entities=recoverable,
                raw_unrecoverable_entities=(
                    len(snippet.adjudication.entities) - recoverable
                ),
            )
        )
    warnings = [
        "Native W2NER files contain only sentence and ner; manifest records retain every source mapping and hash.",
        "Whitespace is omitted from W2NER tokens because the pinned loader can emit no subword pieces for whitespace; token_character_offsets preserve exact reversible mapping.",
        "Raw OCR views label only recoverable mentions. End-to-end scoring must retain raw-unrecoverable mentions in its denominator.",
        "Augmentation is training-only and length-preserving. No generated record is reviewed historical evidence.",
    ]
    augmented_count = sum(
        record.augmentation_kind == "empirical_substitution" for record in records
    )
    if configuration.augmented_copies_per_clean_record and not substitutions:
        warnings.append(
            "No unambiguous training-only empirical substitutions were recoverable; the empirical-noise view equals the clean view."
        )
    elif configuration.augmented_copies_per_clean_record and not augmented_count:
        warnings.append(
            "Empirical substitutions exist, but the deterministic probability produced no distinct augmented records."
        )
    if not dataset.benchmark_eligible:
        warnings.append(
            "INELIGIBLE TECHNICAL EXPORT: " + "; ".join(dataset.eligibility_failures)
        )
    return W2NERTrainingExport(
        export_id=export_id,
        generated_at=generated_at or datetime.now(timezone.utc),
        project_code_revision=project_code_revision,
        w2ner_implementation_revision=W2NER_IMPLEMENTATION_REVISION,
        source_benchmark_dataset_id=dataset.dataset_id,
        source_benchmark_dataset_sha256=benchmark_dataset_sha256(dataset),
        source_gold_dataset_id=dataset.source_gold_dataset_id,
        source_gold_sha256=dataset.source_gold_sha256,
        split_manifest_sha256=dataset.split_manifest_sha256,
        ontology_version=dataset.ontology_version,
        benchmark_eligible=dataset.benchmark_eligible,
        technical_export=not dataset.benchmark_eligible,
        maximum_record_characters=maximum_record_characters,
        augmentation=configuration,
        empirical_substitutions=substitutions,
        coverage=coverage,
        records=records,
        omitted_inputs=omitted,
        views=_build_views(records),
        warnings=warnings,
    )


def materialize_w2ner_training_export(
    export: W2NERTrainingExport, output_directory: Path
) -> list[Path]:
    record_by_id = {record.record_id: record for record in export.records}
    payloads = {
        output_directory / "manifest.json": (
            export.model_dump_json(indent=2) + "\n"
        ).encode("utf-8")
    }
    for view in export.views:
        records = [record_by_id[record_id] for record_id in view.record_ids]
        payloads[output_directory / view.filename] = native_w2ner_json_bytes(records)
    existing = sorted(str(path) for path in payloads if path.exists())
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing W2NER export files: "
            + ", ".join(existing[:5])
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    for path, payload in payloads.items():
        path.write_bytes(payload)
    return sorted(payloads)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--export-id", required=True)
    parser.add_argument("--project-code-revision", required=True)
    parser.add_argument("--maximum-record-characters", type=int, default=256)
    parser.add_argument("--augmentation-probability", type=float, default=0.15)
    parser.add_argument("--augmented-copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-ineligible-technical-export", action="store_true")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gold_bytes = args.gold.read_bytes()
    manifest_bytes = args.split_manifest.read_bytes()
    gold = NERGoldSet.model_validate_json(gold_bytes)
    split_manifest = IssueSplitManifest.model_validate_json(manifest_bytes)
    dataset = prepare_benchmark_dataset(
        gold,
        hashlib.sha256(gold_bytes).hexdigest(),
        split_manifest,
        hashlib.sha256(manifest_bytes).hexdigest(),
        dataset_id=args.dataset_id,
        input_variants=["raw_ocr", "corrected_text"],
    )
    export = build_w2ner_training_export(
        gold,
        dataset,
        export_id=args.export_id,
        project_code_revision=args.project_code_revision,
        maximum_record_characters=args.maximum_record_characters,
        augmentation=AugmentationConfiguration(
            probability=args.augmentation_probability,
            augmented_copies_per_clean_record=args.augmented_copies,
            seed=args.seed,
        ),
        allow_ineligible_technical_export=args.allow_ineligible_technical_export,
    )
    paths = materialize_w2ner_training_export(export, args.output_directory)
    print(
        json.dumps(
            {
                "output_directory": str(args.output_directory),
                "files": [str(path) for path in paths],
                "training_export_sha256": w2ner_training_export_sha256(export),
                "benchmark_eligible": export.benchmark_eligible,
                "records": len(export.records),
                "augmented_records": sum(
                    record.augmentation_kind == "empirical_substitution"
                    for record in export.records
                ),
                "empirical_substitutions": len(export.empirical_substitutions),
                "warnings": export.warnings,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
