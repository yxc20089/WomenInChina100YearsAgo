"""Strict contracts shared by every NER benchmark adapter."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from ..evidence import (
    EntityMentionCandidate,
    EntityType,
    ProcessingRun,
    RunKind,
    SourcePointer,
    StrictModel,
)


BenchmarkSplit = Literal["train", "development", "test"]
BenchmarkInputVariant = Literal["raw_ocr", "corrected_text", "multimodal_transcript"]
CORE_BENCHMARK_ENTITY_TYPES = {
    EntityType.PERSON,
    EntityType.PLACE,
    EntityType.ORGANIZATION,
    EntityType.SCHOOL,
    EntityType.OCCUPATION,
    EntityType.DATE,
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


class SnippetSplitAssignment(StrictModel):
    snippet_id: str = Field(min_length=1, max_length=300)
    issue_id: str = Field(min_length=1, max_length=300)
    split: BenchmarkSplit


class IssueSplitManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=300)
    created_at: datetime
    assigned_by: str = Field(min_length=1, max_length=200)
    assignments: list[SnippetSplitAssignment] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_no_leakage(self) -> "IssueSplitManifest":
        snippet_ids = [item.snippet_id for item in self.assignments]
        if len(set(snippet_ids)) != len(snippet_ids):
            raise ValueError("split manifest snippet IDs must be unique")
        splits_by_issue: dict[str, set[str]] = {}
        for item in self.assignments:
            splits_by_issue.setdefault(item.issue_id, set()).add(item.split)
        leaked = sorted(
            issue for issue, splits in splits_by_issue.items() if len(splits) > 1
        )
        if leaked:
            raise ValueError(
                "an issue cannot cross benchmark splits: " + ", ".join(leaked[:10])
            )
        return self


class BenchmarkInput(StrictModel):
    input_id: str = Field(min_length=1, max_length=500)
    snippet_id: str = Field(min_length=1, max_length=300)
    issue_id: str = Field(min_length=1, max_length=300)
    split: BenchmarkSplit
    input_variant: BenchmarkInputVariant
    gold_region_id: UUID
    source_ocr_run_id: UUID
    source_ocr_region_id: UUID
    source: SourcePointer
    text: str
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> "BenchmarkInput":
        if self.source.region_id != self.source_ocr_region_id:
            raise ValueError("benchmark source pointer must cite the source OCR region")
        if self.gold_region_id == self.source_ocr_region_id:
            raise ValueError("benchmark gold identity must be model-independent")
        if hashlib.sha256(self.text.encode()).hexdigest() != self.text_sha256:
            raise ValueError("benchmark input text hash mismatch")
        return self


def benchmark_eligibility_failures(
    inputs: list[BenchmarkInput],
    reported_entity_types: list[EntityType],
    locked_test_mentions_by_type: dict[EntityType, int],
) -> list[str]:
    unique_snippets = {item.snippet_id for item in inputs}
    unique_issues = {item.issue_id for item in inputs}
    years = {
        item.source.publication_year
        for item in inputs
        if item.source.publication_year is not None
    }
    present_splits = {item.split for item in inputs}
    input_variants = {item.input_variant for item in inputs}
    failures = []
    if len(unique_snippets) < 500:
        failures.append(
            f"Dataset has {len(unique_snippets)} snippets; at least 500 are required."
        )
    if len(unique_issues) < 30:
        failures.append(
            f"Dataset has {len(unique_issues)} issues; at least 30 are required."
        )
    if present_splits != {"train", "development", "test"}:
        failures.append(
            "Dataset must contain train, development and test issue splits."
        )
    if input_variants != {"raw_ocr", "corrected_text"}:
        failures.append(
            "Scientific selection requires paired raw_ocr and corrected_text inputs."
        )
    issues_by_split = {
        split: {item.issue_id for item in inputs if item.split == split}
        for split in ("train", "development", "test")
    }
    if unique_issues:
        target_ratios = {"train": 0.6, "development": 0.2, "test": 0.2}
        for split, target in target_ratios.items():
            observed = len(issues_by_split[split]) / len(unique_issues)
            if abs(observed - target) > 0.1:
                failures.append(
                    f"{split} contains {observed:.1%} of issues; target is {target:.0%} ±10%."
                )
    if len({(year // 10) * 10 for year in years}) < 3:
        failures.append("Dataset covers fewer than three publication decades.")
    for entity_type in sorted(CORE_BENCHMARK_ENTITY_TYPES, key=lambda item: item.value):
        count = locked_test_mentions_by_type.get(entity_type, 0)
        if count < 100:
            failures.append(
                f"Locked test split has {count} {entity_type.value} mentions; "
                "100 are required."
            )
    for entity_type in sorted(
        set(reported_entity_types) - CORE_BENCHMARK_ENTITY_TYPES,
        key=lambda item: item.value,
    ):
        count = locked_test_mentions_by_type.get(entity_type, 0)
        if count < 30:
            failures.append(
                f"Locked test split has {count} {entity_type.value} mentions; "
                "30 are required for a reported rare type."
            )
    return failures


class NERBenchmarkDataset(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=300)
    generated_at: datetime
    ontology_version: str = Field(min_length=1, max_length=100)
    source_gold_dataset_id: str = Field(min_length=1, max_length=300)
    source_gold_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inputs: list[BenchmarkInput] = Field(min_length=1)
    reported_entity_types: list[EntityType]
    locked_test_mentions_by_type: dict[EntityType, int]
    benchmark_eligible: bool
    eligibility_failures: list[str]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dataset(self) -> "NERBenchmarkDataset":
        input_ids = [item.input_id for item in self.inputs]
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("benchmark input IDs must be unique")
        by_snippet_variant = [
            (item.snippet_id, item.input_variant) for item in self.inputs
        ]
        if len(set(by_snippet_variant)) != len(by_snippet_variant):
            raise ValueError("each snippet may appear once per input variant")
        issue_splits: dict[str, set[str]] = {}
        for item in self.inputs:
            issue_splits.setdefault(item.issue_id, set()).add(item.split)
        if any(len(splits) > 1 for splits in issue_splits.values()):
            raise ValueError("benchmark dataset leaks an issue across splits")
        expected_reported_types = sorted(
            set(self.reported_entity_types), key=lambda item: item.value
        )
        if self.reported_entity_types != expected_reported_types:
            raise ValueError(
                "reported entity types must be unique and canonically ordered"
            )
        if set(self.locked_test_mentions_by_type) != set(self.reported_entity_types):
            raise ValueError(
                "locked test counts must cover exactly the reported entity types"
            )
        if any(count < 0 for count in self.locked_test_mentions_by_type.values()):
            raise ValueError("locked test mention counts must be nonnegative")
        expected_failures = benchmark_eligibility_failures(
            self.inputs,
            self.reported_entity_types,
            self.locked_test_mentions_by_type,
        )
        if self.eligibility_failures != expected_failures:
            raise ValueError(
                "benchmark eligibility failures disagree with dataset evidence"
            )
        if self.benchmark_eligible != (not expected_failures):
            raise ValueError(
                "benchmark_eligible must be true exactly when eligibility_failures is empty"
            )
        return self


def benchmark_dataset_sha256(dataset: NERBenchmarkDataset) -> str:
    payload = dataset.model_dump(mode="json", exclude={"generated_at"})
    return canonical_sha256(payload)


class AdapterIdentity(StrictModel):
    adapter_id: str = Field(min_length=1, max_length=200)
    family: Literal[
        "rules",
        "gliner",
        "w2ner",
        "global_pointer",
        "crf",
        "open_ner",
        "structured_generation",
    ]
    model_name: str = Field(min_length=1, max_length=500)
    model_revision: str = Field(min_length=7, max_length=200)
    base_model_revision: str | None = Field(default=None, min_length=7, max_length=200)
    head_implementation_revision: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{40}$"
    )
    trained_head_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training_dataset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    continued_pretraining_dataset_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    license: str | None = Field(default=None, max_length=200)
    modalities: list[Literal["text", "image"]] = Field(min_length=1)
    runtime: str = Field(min_length=1, max_length=500)
    container_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    device: str = Field(min_length=1, max_length=100)
    dtype: str = Field(min_length=1, max_length=100)
    ontology_version: str = Field(min_length=1, max_length=100)
    prompt_schema_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    configuration: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pins(self) -> "AdapterIdentity":
        forbidden = {"main", "master", "latest", "nightly", "dev"}
        revision_parts = {
            part.lower() for part in self.model_revision.replace("\\", "/").split("/")
        }
        if revision_parts & forbidden:
            raise ValueError("model_revision must be immutable, not a moving label")
        if self.family != "rules" and not all(
            character in "0123456789abcdef" for character in self.model_revision.lower()
        ):
            raise ValueError(
                "downloaded model_revision must be an immutable hexadecimal commit"
            )
        if self.family != "rules" and len(self.model_revision) != 40:
            raise ValueError(
                "downloaded model_revision must be a full 40-character commit"
            )
        if self.base_model_revision is not None and (
            len(self.base_model_revision) != 40
            or not all(
                character in "0123456789abcdef"
                for character in self.base_model_revision.lower()
            )
        ):
            raise ValueError("base_model_revision must be a full immutable commit")
        if self.family in {"w2ner", "global_pointer", "crf"} and any(
            value is None
            for value in (
                self.head_implementation_revision,
                self.trained_head_sha256,
                self.training_dataset_sha256,
            )
        ):
            raise ValueError(
                "supervised adapters require head-code, trained-head and training-data hashes"
            )
        if (
            self.family == "structured_generation"
            and self.prompt_schema_revision is None
        ):
            raise ValueError("structured generation requires a prompt/schema revision")
        return self


class BenchmarkResult(StrictModel):
    input_id: str = Field(min_length=1, max_length=500)
    snippet_id: str = Field(min_length=1, max_length=300)
    issue_id: str = Field(min_length=1, max_length=300)
    gold_region_id: UUID
    source_ocr_run_id: UUID
    source_ocr_region_id: UUID
    input_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mentions: list[EntityMentionCandidate] = Field(default_factory=list)
    abstention_reason: str | None = Field(default=None, max_length=2000)
    latency_seconds: float = Field(ge=0)
    raw_output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    finish_reason: str | None = Field(default=None, max_length=200)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    invalid_outputs: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_result_state(self) -> "BenchmarkResult":
        if self.abstention_reason is not None and self.mentions:
            raise ValueError("an abstained result cannot contain mentions")
        if any(
            mention.source.region_id != self.source_ocr_region_id
            for mention in self.mentions
        ):
            raise ValueError("result mentions must cite the result source OCR region")
        if any(
            mention.attributes.get("benchmark_input_id") != self.input_id
            for mention in self.mentions
        ):
            raise ValueError("result mentions must cite the result benchmark input")
        if any(
            mention.attributes.get("source_ocr_run_id") != str(self.source_ocr_run_id)
            for mention in self.mentions
        ):
            raise ValueError("result mentions must cite the result source OCR run")
        if (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
            and self.prompt_tokens + self.completion_tokens != self.total_tokens
        ):
            raise ValueError("benchmark result token usage does not reconcile")
        return self


class BenchmarkPredictionArtifact(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["ner_benchmark_predictions"] = "ner_benchmark_predictions"
    artifact_id: UUID
    benchmark_dataset_id: str = Field(min_length=1, max_length=300)
    benchmark_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_gold_dataset_id: str = Field(min_length=1, max_length=300)
    source_gold_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: BenchmarkSplit
    input_variant: BenchmarkInputVariant
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ontology_version: str = Field(min_length=1, max_length=100)
    adapter: AdapterIdentity
    run: ProcessingRun
    source_ocr_run_ids: list[UUID] = Field(min_length=1)
    input_ids: list[str] = Field(min_length=1)
    results: list[BenchmarkResult] = Field(min_length=1)
    mentions: list[EntityMentionCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_predictions(self) -> "BenchmarkPredictionArtifact":
        if self.run.kind != RunKind.NER:
            raise ValueError("benchmark prediction run must have kind=ner")
        if self.run.run_id != self.artifact_id:
            raise ValueError("benchmark artifact and run IDs must match")
        if (
            self.run.engine != self.adapter.adapter_id
            or self.run.model_name != self.adapter.model_name
            or self.run.model_revision != self.adapter.model_revision
        ):
            raise ValueError("benchmark run identity must exactly match its adapter")
        if self.adapter.ontology_version != self.ontology_version:
            raise ValueError("adapter and artifact ontology versions disagree")
        if len(set(self.source_ocr_run_ids)) != len(self.source_ocr_run_ids):
            raise ValueError("source OCR run IDs must be unique")
        if len(set(self.input_ids)) != len(self.input_ids):
            raise ValueError("benchmark input IDs must be unique")
        result_ids = [result.input_id for result in self.results]
        if result_ids != self.input_ids:
            raise ValueError("results must cover every input exactly once and in order")
        source_region_ids = [result.source_ocr_region_id for result in self.results]
        if len(set(source_region_ids)) != len(source_region_ids):
            raise ValueError(
                "a benchmark split/variant may contain each source region once"
            )
        result_source_runs = {result.source_ocr_run_id for result in self.results}
        if result_source_runs != set(self.source_ocr_run_ids):
            raise ValueError(
                "artifact source OCR runs must exactly cover result provenance"
            )
        expected_input_sha256 = canonical_sha256(
            [
                {
                    "input_id": result.input_id,
                    "text_sha256": result.input_text_sha256,
                }
                for result in self.results
            ]
        )
        if expected_input_sha256 != self.input_sha256:
            raise ValueError("benchmark artifact input hash disagrees with its results")
        expected_run_configuration = {
            "benchmark_dataset_sha256": self.benchmark_dataset_sha256,
            "split": self.split,
            "input_variant": self.input_variant,
            "input_sha256": self.input_sha256,
            "input_region_count": len(self.results),
        }
        if any(
            self.run.configuration.get(key) != value
            for key, value in expected_run_configuration.items()
        ):
            raise ValueError(
                "benchmark run configuration disagrees with artifact scope"
            )
        result_mentions = [
            mention for result in self.results for mention in result.mentions
        ]
        if result_mentions != self.mentions:
            raise ValueError(
                "top-level mentions must exactly flatten per-input results"
            )
        if any(mention.run_id != self.run.run_id for mention in self.mentions):
            raise ValueError("all benchmark mentions must cite the benchmark run")
        if self.adapter.family == "structured_generation" and any(
            result.raw_output_sha256 is None or result.prompt_sha256 is None
            for result in self.results
        ):
            raise ValueError(
                "structured generation results require prompt and raw-output hashes"
            )
        return self
