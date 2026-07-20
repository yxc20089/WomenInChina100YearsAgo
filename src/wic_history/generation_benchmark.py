"""Freeze, execute, blind, and objectively score grounded-generation benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Literal, Sequence
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from .evidence import ScenarioContextBundle, SourcePointer, StrictModel
from .evaluation import QuestionCategory
from .generation import (
    ChatTurn,
    GenerationResponse,
    GenerationStatus,
    GenerationTask,
    OpenAICompatibleGenerator,
    TextGenerator,
    generate,
    generation_context_sha256,
)


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


def is_immutable_model_revision(value: str) -> bool:
    forbidden = {"main", "master", "latest", "nightly", "dev"}
    parts = {part.lower() for part in value.replace("\\", "/").split("/")}
    return not bool(parts & forbidden)


class GenerationBenchmarkCase(StrictModel):
    case_id: str = Field(min_length=1, max_length=300)
    category: QuestionCategory
    task: GenerationTask
    context: ScenarioContextBundle
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)
    answerable: bool
    expected_support_region_ids: list[UUID] = Field(default_factory=list)
    author: str = Field(min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=5000)

    @model_validator(mode="after")
    def validate_case(self) -> "GenerationBenchmarkCase":
        if self.history and self.task != GenerationTask.CHAT_ANSWER:
            raise ValueError("conversation history is allowed only for chat cases")
        if self.answerable != (self.category != QuestionCategory.UNANSWERABLE):
            raise ValueError("answerability must agree with the unanswerable category")
        if self.answerable and not self.expected_support_region_ids:
            raise ValueError("answerable generation cases require expected support regions")
        if not self.answerable and self.expected_support_region_ids:
            raise ValueError("unanswerable cases cannot declare expected support regions")
        if len(set(self.expected_support_region_ids)) != len(
            self.expected_support_region_ids
        ):
            raise ValueError("expected support region IDs must be unique")
        reviewed_sources = {
            source.region_id
            for item in self.context.evidence_items
            if item.epistemic_label == "directly_evidenced"
            for source in item.sources
            if source.region_id is not None
        }
        retrieved_sources = {
            span.source.region_id
            for hit in self.context.retrieved_context
            for span in hit.sources
            if span.source.region_id is not None
        }
        allowed_sources = reviewed_sources
        if self.task != GenerationTask.RECONSTRUCTED_SCENE:
            allowed_sources = reviewed_sources | retrieved_sources
        if not set(self.expected_support_region_ids).issubset(allowed_sources):
            raise ValueError("expected support must be present in the task's allowed context")
        if (
            self.answerable
            and self.task == GenerationTask.RECONSTRUCTED_SCENE
            and not any(
                item.epistemic_label == "directly_evidenced"
                for item in self.context.evidence_items
            )
        ):
            raise ValueError("answerable scene cases require reviewed claims")
        return self


def generation_benchmark_eligibility_failures(
    cases: list[GenerationBenchmarkCase],
) -> list[str]:
    failures = []
    if len(cases) < 30:
        failures.append(f"Dataset has {len(cases)} cases; at least 30 are required.")
    authors = {case.author for case in cases}
    if len(authors) < 2 or any(author == "technical-smoke" for author in authors):
        failures.append(
            "Eligible generation benchmarks require at least two historian authors and no technical-smoke author."
        )
    categories = {case.category for case in cases}
    missing_categories = sorted(
        set(QuestionCategory) - categories, key=lambda item: item.value
    )
    if missing_categories:
        failures.append(
            "Dataset lacks required question categories: "
            + ", ".join(category.value for category in missing_categories)
            + "."
        )
    unanswerable_count = sum(not case.answerable for case in cases)
    if unanswerable_count < 5:
        failures.append(
            f"Dataset has {unanswerable_count} unanswerable cases; at least 5 are required."
        )
    input_hashes = [
        generation_context_sha256(case.context, case.task, case.history)
        for case in cases
    ]
    duplicate_inputs = len(input_hashes) - len(set(input_hashes))
    if duplicate_inputs:
        failures.append(
            f"Dataset has {duplicate_inputs} duplicate task/context/history inputs."
        )
    for task in GenerationTask:
        task_count = sum(case.task == task for case in cases)
        if task_count < 5:
            failures.append(
                f"Dataset has {task_count} {task.value} cases; at least 5 are required."
            )
    pointers = [
        span.source
        for case in cases
        for hit in case.context.retrieved_context
        for span in hit.sources
    ] + [
        source
        for case in cases
        for item in case.context.evidence_items
        for source in item.sources
    ]
    incomplete_pointers = sum(
        pointer.region_id is None
        or pointer.source_sha256 is None
        or not re_full_sha256(pointer.source_sha256)
        or pointer.derivative_id is None
        or pointer.image_sha256 is None
        or pointer.evidence_tier is None
        or pointer.polygon is None
        for pointer in pointers
    )
    if incomplete_pointers:
        failures.append(
            f"Dataset has {incomplete_pointers} context pointers without complete scan provenance."
        )
    directly_reviewed_region_ids = {
        source.region_id
        for case in cases
        for item in case.context.evidence_items
        if item.epistemic_label == "directly_evidenced"
        for source in item.sources
        if source.region_id is not None
    }
    historian_gold_region_ids = {
        span.source.region_id
        for case in cases
        for hit in case.context.retrieved_context
        for span in hit.sources
        if span.source.region_id is not None
        and span.source.evidence_tier == "historian_selected_gold"
    }
    eligible_support = directly_reviewed_region_ids | historian_gold_region_ids
    unsupported_expected = sum(
        region_id not in eligible_support
        for case in cases
        for region_id in case.expected_support_region_ids
    )
    if unsupported_expected:
        failures.append(
            f"Dataset has {unsupported_expected} expected support regions that are neither directly reviewed nor historian-selected gold."
        )
    return failures


class GenerationBenchmarkDataset(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=300)
    generated_at: datetime
    created_by: str = Field(min_length=1, max_length=200)
    cases: list[GenerationBenchmarkCase] = Field(min_length=1)
    benchmark_eligible: bool
    eligibility_failures: list[str]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dataset(self) -> "GenerationBenchmarkDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("generation benchmark case IDs must be unique")
        expected_failures = generation_benchmark_eligibility_failures(self.cases)
        if self.eligibility_failures != expected_failures:
            raise ValueError("generation benchmark eligibility disagrees with case evidence")
        if self.benchmark_eligible != (not expected_failures):
            raise ValueError("benchmark_eligible disagrees with eligibility failures")
        return self


def build_generation_benchmark_dataset(
    dataset_id: str,
    created_by: str,
    cases: list[GenerationBenchmarkCase],
    *,
    generated_at: datetime | None = None,
) -> GenerationBenchmarkDataset:
    failures = generation_benchmark_eligibility_failures(cases)
    return GenerationBenchmarkDataset(
        dataset_id=dataset_id,
        generated_at=generated_at or datetime.now(timezone.utc),
        created_by=created_by,
        cases=cases,
        benchmark_eligible=not failures,
        eligibility_failures=failures,
        warnings=[
            "Transport and citation validation do not establish answer quality.",
            "Keep model identity hidden during independent human grading.",
        ],
    )


def generation_benchmark_dataset_sha256(dataset: GenerationBenchmarkDataset) -> str:
    return canonical_sha256(dataset.model_dump(mode="json", exclude={"generated_at"}))


class GenerationBenchmarkResult(StrictModel):
    case_id: str = Field(min_length=1, max_length=300)
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    latency_seconds: float = Field(ge=0)
    response: GenerationResponse


class GenerationBenchmarkArtifact(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["generation_benchmark_predictions"] = (
        "generation_benchmark_predictions"
    )
    artifact_id: UUID
    dataset_id: str = Field(min_length=1, max_length=300)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    provider: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=1000)
    model_revision: str = Field(min_length=1, max_length=500)
    generation_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: datetime
    completed_at: datetime
    results: list[GenerationBenchmarkResult] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifact(self) -> "GenerationBenchmarkArtifact":
        if self.completed_at < self.started_at:
            raise ValueError("benchmark completion cannot precede start")
        if not is_immutable_model_revision(self.model_revision):
            raise ValueError("benchmark model revision must be immutable")
        case_ids = [result.case_id for result in self.results]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("generation benchmark results must have unique case IDs")
        for result in self.results:
            response = result.response
            if response.context_sha256 != result.context_sha256 and response.status in {
                GenerationStatus.COMPLETED,
                GenerationStatus.REJECTED,
            }:
                raise ValueError("generated response context hash disagrees with result")
            if response.model is not None and response.model != self.model:
                raise ValueError("response model identity disagrees with artifact")
            if (
                response.model_revision is not None
                and response.model_revision != self.model_revision
            ):
                raise ValueError("response model revision disagrees with artifact")
            if response.provider is not None and response.provider != self.provider:
                raise ValueError("response provider disagrees with artifact")
            if (
                response.generation_configuration_sha256 is not None
                and response.generation_configuration_sha256
                != self.generation_configuration_sha256
            ):
                raise ValueError("response generation configuration disagrees with artifact")
        return self


def _generator_identity(generator: TextGenerator) -> tuple[str, str, str, str]:
    values = (
        generator.model_identity,
        getattr(generator, "model_revision", None),
        getattr(generator, "provider_kind", None),
        getattr(generator, "generation_configuration_sha256", None),
    )
    if any(value is None or not isinstance(value, str) or not value for value in values):
        raise ValueError(
            "benchmark generators require model, immutable revision, provider and configuration hash"
        )
    model, revision, provider, configuration_sha256 = values
    if not is_immutable_model_revision(revision):
        raise ValueError("benchmark model revision must be immutable")
    if not re_full_sha256(configuration_sha256):
        raise ValueError("generator configuration hash must be a lowercase SHA-256")
    return model, revision, provider, configuration_sha256


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def execute_generation_benchmark(
    dataset: GenerationBenchmarkDataset,
    generator: TextGenerator,
    *,
    code_revision: str,
    allow_ineligible_technical_run: bool = False,
) -> GenerationBenchmarkArtifact:
    if not dataset.benchmark_eligible and not allow_ineligible_technical_run:
        raise ValueError(
            "generation benchmark is ineligible: "
            + "; ".join(dataset.eligibility_failures)
        )
    if len(code_revision) != 40 or not all(
        character in "0123456789abcdef" for character in code_revision
    ):
        raise ValueError("code_revision must be a full lowercase git commit")
    model, model_revision, provider, configuration_sha256 = _generator_identity(
        generator
    )
    started_at = datetime.now(timezone.utc)
    results = []
    for case in dataset.cases:
        began = time.perf_counter()
        response = generate(case.context, case.task, generator, case.history)
        latency = time.perf_counter() - began
        results.append(
            GenerationBenchmarkResult(
                case_id=case.case_id,
                context_sha256=generation_context_sha256(
                    case.context, case.task, case.history
                ),
                latency_seconds=latency,
                response=response,
            )
        )
    completed_at = datetime.now(timezone.utc)
    warnings = [
        "Generation artifacts are experiment outputs, not historical evidence or reviewed narratives."
    ]
    if not dataset.benchmark_eligible:
        warnings.append(
            "INELIGIBLE TECHNICAL RUN: " + "; ".join(dataset.eligibility_failures)
        )
    return GenerationBenchmarkArtifact(
        artifact_id=uuid4(),
        dataset_id=dataset.dataset_id,
        dataset_sha256=generation_benchmark_dataset_sha256(dataset),
        code_revision=code_revision,
        provider=provider,
        model=model,
        model_revision=model_revision,
        generation_configuration_sha256=configuration_sha256,
        started_at=started_at,
        completed_at=completed_at,
        results=results,
        warnings=warnings,
    )


class GenerationCaseObjectiveScore(StrictModel):
    case_id: str
    task: GenerationTask
    category: QuestionCategory
    answerable: bool
    status: GenerationStatus
    expected_support_count: int = Field(ge=0)
    cited_support_count: int = Field(ge=0)
    expected_support_recall: float | None = Field(default=None, ge=0, le=1)
    citation_precision_against_expected: float | None = Field(
        default=None, ge=0, le=1
    )
    invalid_citation_count: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class GenerationObjectiveReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    dataset_id: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_id: UUID
    model: str
    model_revision: str
    question_count: int = Field(ge=1)
    answerable_completion_rate: float = Field(ge=0, le=1)
    structural_rejection_rate: float = Field(ge=0, le=1)
    abstention_rate: float = Field(ge=0, le=1)
    unavailable_rate: float = Field(ge=0, le=1)
    micro_expected_support_recall: float | None = Field(default=None, ge=0, le=1)
    micro_citation_precision_against_expected: float | None = Field(
        default=None, ge=0, le=1
    )
    latency_p50_seconds: float = Field(ge=0)
    latency_p95_seconds: float = Field(ge=0)
    token_usage_complete_rate: float = Field(ge=0, le=1)
    total_prompt_tokens: int | None = Field(default=None, ge=0)
    total_completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    results: list[GenerationCaseObjectiveScore]
    warnings: list[str]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def score_generation_benchmark(
    dataset: GenerationBenchmarkDataset,
    artifact: GenerationBenchmarkArtifact,
) -> GenerationObjectiveReport:
    if artifact.dataset_id != dataset.dataset_id:
        raise ValueError("generation artifact dataset ID disagrees with dataset")
    dataset_sha256 = generation_benchmark_dataset_sha256(dataset)
    if artifact.dataset_sha256 != dataset_sha256:
        raise ValueError("generation artifact dataset hash disagrees with dataset")
    if len(artifact.results) != len(dataset.cases):
        raise ValueError("generation artifact result count disagrees with dataset")
    objective_results = []
    total_expected = 0
    total_expected_cited = 0
    total_citations = 0
    answerable_completed = 0
    answerable_count = sum(case.answerable for case in dataset.cases)
    for case, result in zip(dataset.cases, artifact.results, strict=True):
        if result.case_id != case.case_id:
            raise ValueError("generation artifact case order disagrees with dataset")
        expected_context_sha256 = generation_context_sha256(
            case.context, case.task, case.history
        )
        if result.context_sha256 != expected_context_sha256:
            raise ValueError("generation result context hash disagrees with dataset")
        if result.response.task != case.task:
            raise ValueError("generation response task disagrees with dataset")
        cited = {
            source.region_id
            for source in result.response.citations
            if source.region_id is not None
        }
        reviewed_allowed = {
            source.region_id
            for item in case.context.evidence_items
            if item.epistemic_label == "directly_evidenced"
            for source in item.sources
            if source.region_id is not None
        }
        allowed = reviewed_allowed
        if case.task != GenerationTask.RECONSTRUCTED_SCENE:
            allowed = allowed | {
                span.source.region_id
                for hit in case.context.retrieved_context
                for span in hit.sources
                if span.source.region_id is not None
            }
        if not cited.issubset(allowed):
            raise ValueError("generation result cites a region outside its frozen context")
        expected = set(case.expected_support_region_ids)
        expected_cited = cited & expected
        if case.answerable:
            total_expected += len(expected)
            total_expected_cited += len(expected_cited)
            total_citations += len(cited)
            answerable_completed += result.response.status == GenerationStatus.COMPLETED
        objective_results.append(
            GenerationCaseObjectiveScore(
                case_id=case.case_id,
                task=case.task,
                category=case.category,
                answerable=case.answerable,
                status=result.response.status,
                expected_support_count=len(expected),
                cited_support_count=len(expected_cited),
                expected_support_recall=(
                    len(expected_cited) / len(expected) if expected else None
                ),
                citation_precision_against_expected=(
                    len(expected_cited) / len(cited) if cited and case.answerable else None
                ),
                invalid_citation_count=len(result.response.invalid_citation_ids),
                latency_seconds=result.latency_seconds,
                prompt_tokens=result.response.prompt_tokens,
                completion_tokens=result.response.completion_tokens,
                total_tokens=result.response.total_tokens,
                estimated_cost_usd=result.response.estimated_cost_usd,
            )
        )
    statuses = [result.response.status for result in artifact.results]
    latencies = [result.latency_seconds for result in artifact.results]
    provider_call_responses = [
        result.response
        for result in artifact.results
        if result.response.status
        in {GenerationStatus.COMPLETED, GenerationStatus.REJECTED}
    ]
    complete_usage = [
        response
        for response in provider_call_responses
        if response.prompt_tokens is not None
        and response.completion_tokens is not None
        and response.total_tokens is not None
    ]
    usage_complete = len(complete_usage) == len(provider_call_responses)
    costs = [response.estimated_cost_usd for response in provider_call_responses]
    costs_complete = all(cost is not None for cost in costs)
    return GenerationObjectiveReport(
        generated_at=datetime.now(timezone.utc),
        dataset_id=dataset.dataset_id,
        dataset_sha256=dataset_sha256,
        artifact_id=artifact.artifact_id,
        model=artifact.model,
        model_revision=artifact.model_revision,
        question_count=len(dataset.cases),
        answerable_completion_rate=(
            answerable_completed / answerable_count if answerable_count else 0.0
        ),
        structural_rejection_rate=statuses.count(GenerationStatus.REJECTED)
        / len(statuses),
        abstention_rate=statuses.count(GenerationStatus.ABSTAINED) / len(statuses),
        unavailable_rate=statuses.count(GenerationStatus.UNAVAILABLE) / len(statuses),
        micro_expected_support_recall=(
            total_expected_cited / total_expected if total_expected else None
        ),
        micro_citation_precision_against_expected=(
            total_expected_cited / total_citations if total_citations else None
        ),
        latency_p50_seconds=median(latencies),
        latency_p95_seconds=_percentile(latencies, 0.95),
        token_usage_complete_rate=(
            len(complete_usage) / len(provider_call_responses)
            if provider_call_responses
            else 1.0
        ),
        total_prompt_tokens=(
            sum(response.prompt_tokens for response in complete_usage)
            if usage_complete
            else None
        ),
        total_completion_tokens=(
            sum(response.completion_tokens for response in complete_usage)
            if usage_complete
            else None
        ),
        total_tokens=(
            sum(response.total_tokens for response in complete_usage)
            if usage_complete
            else None
        ),
        estimated_cost_usd=(
            sum(cost for cost in costs if cost is not None)
            if costs_complete
            else None
        ),
        results=objective_results,
        warnings=[
            "Objective scores measure structural acceptance and expected-region citation, not entailment, completeness, prose quality, or historical judgment.",
            "Unanswerable cases require blinded human grading; completion or abstention alone is not a correctness label.",
        ],
    )


class BlindedGenerationCase(StrictModel):
    blind_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    pairing_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    task: GenerationTask
    context: ScenarioContextBundle
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)
    output_status: GenerationStatus
    output: str
    citations: list[SourcePointer]
    validation_errors: list[str]
    warnings: list[str]


class BlindedGenerationPacket(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: list[BlindedGenerationCase]
    instructions: list[str]

    @model_validator(mode="after")
    def validate_packet(self) -> "BlindedGenerationPacket":
        blind_ids = [case.blind_id for case in self.cases]
        if len(set(blind_ids)) != len(blind_ids):
            raise ValueError("blinded generation case IDs must be unique")
        pairing_ids = [case.pairing_id for case in self.cases]
        if len(set(pairing_ids)) != len(pairing_ids):
            raise ValueError("blinded generation inputs must be unique")
        return self


def generation_artifact_sha256(artifact: GenerationBenchmarkArtifact) -> str:
    return canonical_sha256(
        artifact.model_dump(mode="json", exclude={"started_at", "completed_at"})
    )


def build_blinded_generation_packet(
    dataset: GenerationBenchmarkDataset,
    artifact: GenerationBenchmarkArtifact,
) -> BlindedGenerationPacket:
    score_generation_benchmark(dataset, artifact)
    artifact_sha256 = generation_artifact_sha256(artifact)
    cases = []
    for case, result in zip(dataset.cases, artifact.results, strict=True):
        cases.append(
            BlindedGenerationCase(
                blind_id=canonical_sha256(
                    {
                        "dataset_sha256": artifact.dataset_sha256,
                        "artifact_sha256": artifact_sha256,
                        "case_id": case.case_id,
                    }
                ),
                pairing_id=result.context_sha256,
                task=case.task,
                context=case.context,
                history=case.history,
                output_status=result.response.status,
                output=result.response.output,
                citations=result.response.citations,
                validation_errors=result.response.validation_errors,
                warnings=result.response.warnings,
            )
        )
    return BlindedGenerationPacket(
        dataset_id=dataset.dataset_id,
        dataset_sha256=artifact.dataset_sha256,
        prediction_artifact_sha256=artifact_sha256,
        cases=sorted(cases, key=lambda item: item.blind_id),
        instructions=[
            "Grade evidence entailment, completeness, epistemic separation, historical safety and usefulness without seeking model identity.",
            "A structurally completed response is not automatically correct; inspect every cited scan/context item.",
            "Use two independent reviewers and adjudicate disagreements before model comparison.",
        ],
    )


def blinded_generation_packet_sha256(packet: BlindedGenerationPacket) -> str:
    return canonical_sha256(packet.model_dump(mode="json"))


class HumanGenerationReview(StrictModel):
    blind_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1, max_length=200)
    reviewed_at: datetime
    evidence_entailment: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    epistemic_separation: int = Field(ge=1, le=5)
    historical_safety: int = Field(ge=1, le=5)
    researcher_usefulness: int = Field(ge=1, le=5)
    unsupported_claim_count: int = Field(ge=0)
    decision: Literal["pass", "fail", "needs_discussion"]
    notes: str = Field(min_length=1, max_length=5000)


class HumanGenerationAdjudication(StrictModel):
    blind_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    adjudicator: str = Field(min_length=1, max_length=200)
    adjudicated_at: datetime
    decision: Literal["pass", "fail"]
    evidence_entailment: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    epistemic_separation: int = Field(ge=1, le=5)
    historical_safety: int = Field(ge=1, le=5)
    researcher_usefulness: int = Field(ge=1, le=5)
    unsupported_claim_count: int = Field(ge=0)
    notes: str = Field(min_length=1, max_length=5000)


class HumanGenerationGradeSet(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    blinded_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviews: list[HumanGenerationReview] = Field(min_length=2)
    adjudications: list[HumanGenerationAdjudication] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grades(self) -> "HumanGenerationGradeSet":
        reviews_by_blind_id: dict[str, list[HumanGenerationReview]] = {}
        for review in self.reviews:
            reviews_by_blind_id.setdefault(review.blind_id, []).append(review)
        adjudication_ids = [item.blind_id for item in self.adjudications]
        if len(set(adjudication_ids)) != len(adjudication_ids):
            raise ValueError("each blind case requires exactly one adjudication")
        if set(reviews_by_blind_id) != set(adjudication_ids):
            raise ValueError("reviews and adjudications must cover the same blind cases")
        for blind_id, reviews in reviews_by_blind_id.items():
            reviewers = {review.reviewer for review in reviews}
            if len(reviews) != 2 or len(reviewers) != 2:
                raise ValueError(
                    f"blind case {blind_id} requires exactly two distinct reviewers"
                )
            adjudicator = next(
                item.adjudicator for item in self.adjudications if item.blind_id == blind_id
            )
            if adjudicator in reviewers:
                raise ValueError("adjudicator must be independent of both reviewers")
        return self


class HumanGenerationReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    blinded_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adjudicated_cases: int = Field(ge=1)
    review_decision_agreement_rate: float = Field(ge=0, le=1)
    mean_rating_absolute_difference: float = Field(ge=0, le=4)
    pass_rate: float = Field(ge=0, le=1)
    mean_evidence_entailment: float = Field(ge=1, le=5)
    mean_completeness: float = Field(ge=1, le=5)
    mean_epistemic_separation: float = Field(ge=1, le=5)
    mean_historical_safety: float = Field(ge=1, le=5)
    mean_researcher_usefulness: float = Field(ge=1, le=5)
    unsupported_claims: int = Field(ge=0)
    case_results: list["HumanGenerationCaseScore"] = Field(min_length=1)
    warnings: list[str]

    @model_validator(mode="after")
    def validate_report(self) -> "HumanGenerationReport":
        if len(self.case_results) != self.adjudicated_cases:
            raise ValueError("human report case count disagrees with aggregate")
        pairing_ids = [item.pairing_id for item in self.case_results]
        if len(set(pairing_ids)) != len(pairing_ids):
            raise ValueError("human report pairing IDs must be unique")
        count = len(self.case_results)

        def case_mean(field: str) -> float:
            return sum(getattr(item, field) for item in self.case_results) / count

        expected_values = {
            "pass_rate": sum(item.decision == "pass" for item in self.case_results)
            / count,
            "mean_evidence_entailment": case_mean("evidence_entailment"),
            "mean_completeness": case_mean("completeness"),
            "mean_epistemic_separation": case_mean("epistemic_separation"),
            "mean_historical_safety": case_mean("historical_safety"),
            "mean_researcher_usefulness": case_mean("researcher_usefulness"),
        }
        for field, expected in expected_values.items():
            if not math.isclose(getattr(self, field), expected, abs_tol=1e-12):
                raise ValueError(f"human report {field} disagrees with case results")
        if self.unsupported_claims != sum(
            item.unsupported_claim_count for item in self.case_results
        ):
            raise ValueError("human report unsupported claims disagree with case results")
        return self


class HumanGenerationCaseScore(StrictModel):
    pairing_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["pass", "fail"]
    evidence_entailment: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    epistemic_separation: int = Field(ge=1, le=5)
    historical_safety: int = Field(ge=1, le=5)
    researcher_usefulness: int = Field(ge=1, le=5)
    unsupported_claim_count: int = Field(ge=0)

    @property
    def mean_quality(self) -> float:
        return (
            self.evidence_entailment
            + self.completeness
            + self.epistemic_separation
            + self.historical_safety
            + self.researcher_usefulness
        ) / 5


def score_human_generation_grades(
    packet: BlindedGenerationPacket,
    grades: HumanGenerationGradeSet,
) -> HumanGenerationReport:
    if grades.dataset_sha256 != packet.dataset_sha256:
        raise ValueError("human grades dataset hash disagrees with blinded packet")
    if grades.prediction_artifact_sha256 != packet.prediction_artifact_sha256:
        raise ValueError("human grades artifact hash disagrees with blinded packet")
    packet_sha256 = blinded_generation_packet_sha256(packet)
    if grades.blinded_packet_sha256 != packet_sha256:
        raise ValueError("human grades packet hash disagrees with blinded packet")
    blind_ids = {case.blind_id for case in packet.cases}
    if {item.blind_id for item in grades.adjudications} != blind_ids:
        raise ValueError("human grades must adjudicate every blinded case exactly once")
    adjudications = grades.adjudications
    count = len(adjudications)
    reviews_by_blind_id: dict[str, list[HumanGenerationReview]] = {}
    for review in grades.reviews:
        reviews_by_blind_id.setdefault(review.blind_id, []).append(review)
    decision_agreements = sum(
        reviews[0].decision == reviews[1].decision
        for reviews in reviews_by_blind_id.values()
    )
    rating_fields = (
        "evidence_entailment",
        "completeness",
        "epistemic_separation",
        "historical_safety",
        "researcher_usefulness",
    )
    rating_differences = [
        abs(getattr(reviews[0], field) - getattr(reviews[1], field))
        for reviews in reviews_by_blind_id.values()
        for field in rating_fields
    ]

    def mean(field: str) -> float:
        return sum(getattr(item, field) for item in adjudications) / count

    packet_by_blind_id = {case.blind_id: case for case in packet.cases}
    case_results = [
        HumanGenerationCaseScore(
            pairing_id=packet_by_blind_id[item.blind_id].pairing_id,
            decision=item.decision,
            evidence_entailment=item.evidence_entailment,
            completeness=item.completeness,
            epistemic_separation=item.epistemic_separation,
            historical_safety=item.historical_safety,
            researcher_usefulness=item.researcher_usefulness,
            unsupported_claim_count=item.unsupported_claim_count,
        )
        for item in adjudications
    ]

    return HumanGenerationReport(
        dataset_sha256=packet.dataset_sha256,
        prediction_artifact_sha256=packet.prediction_artifact_sha256,
        blinded_packet_sha256=packet_sha256,
        adjudicated_cases=count,
        review_decision_agreement_rate=decision_agreements / count,
        mean_rating_absolute_difference=sum(rating_differences)
        / len(rating_differences),
        pass_rate=sum(item.decision == "pass" for item in adjudications) / count,
        mean_evidence_entailment=mean("evidence_entailment"),
        mean_completeness=mean("completeness"),
        mean_epistemic_separation=mean("epistemic_separation"),
        mean_historical_safety=mean("historical_safety"),
        mean_researcher_usefulness=mean("researcher_usefulness"),
        unsupported_claims=sum(item.unsupported_claim_count for item in adjudications),
        case_results=sorted(case_results, key=lambda item: item.pairing_id),
        warnings=[
            "Human ratings are valid only when reviewers were blind to model identity and inspected cited evidence.",
            "Compare models with paired case-level uncertainty; do not select from means alone.",
        ],
    )


class PairedHumanGenerationComparison(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_a: str = Field(min_length=1, max_length=200)
    label_b: str = Field(min_length=1, max_length=200)
    prediction_artifact_sha256_a: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_artifact_sha256_b: str = Field(pattern=r"^[0-9a-f]{64}$")
    paired_cases: int = Field(ge=1)
    bootstrap_seed: int
    bootstrap_resamples: int = Field(ge=100)
    mean_quality_difference_b_minus_a: float = Field(ge=-4, le=4)
    mean_quality_difference_ci95: tuple[float, float]
    pass_rate_difference_b_minus_a: float = Field(ge=-1, le=1)
    unsupported_claim_difference_b_minus_a: float
    b_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    a_wins: int = Field(ge=0)
    warnings: list[str]

    @model_validator(mode="after")
    def validate_comparison(self) -> "PairedHumanGenerationComparison":
        if self.label_a == self.label_b:
            raise ValueError("paired comparison labels must be distinct")
        if sum((self.b_wins, self.ties, self.a_wins)) != self.paired_cases:
            raise ValueError("paired win/tie counts must cover every case")
        lower, upper = self.mean_quality_difference_ci95
        if lower > upper or lower < -4 or upper > 4:
            raise ValueError("paired quality interval is invalid")
        return self


def compare_human_generation_reports(
    report_a: HumanGenerationReport,
    report_b: HumanGenerationReport,
    *,
    label_a: str,
    label_b: str,
    bootstrap_seed: int = 17,
    bootstrap_resamples: int = 5000,
) -> PairedHumanGenerationComparison:
    if report_a.dataset_sha256 != report_b.dataset_sha256:
        raise ValueError("human reports use different frozen datasets")
    if report_a.prediction_artifact_sha256 == report_b.prediction_artifact_sha256:
        raise ValueError("paired comparison requires two distinct prediction artifacts")
    if bootstrap_resamples < 100:
        raise ValueError("paired comparison requires at least 100 bootstrap resamples")
    cases_a = {item.pairing_id: item for item in report_a.case_results}
    cases_b = {item.pairing_id: item for item in report_b.case_results}
    if len(cases_a) != len(report_a.case_results) or len(cases_b) != len(
        report_b.case_results
    ):
        raise ValueError("human reports contain duplicate pairing IDs")
    if set(cases_a) != set(cases_b):
        raise ValueError("human reports do not cover identical generation inputs")
    ordered_ids = sorted(cases_a)
    quality_differences = [
        cases_b[pairing_id].mean_quality - cases_a[pairing_id].mean_quality
        for pairing_id in ordered_ids
    ]
    unsupported_differences = [
        cases_b[pairing_id].unsupported_claim_count
        - cases_a[pairing_id].unsupported_claim_count
        for pairing_id in ordered_ids
    ]
    pass_differences = [
        int(cases_b[pairing_id].decision == "pass")
        - int(cases_a[pairing_id].decision == "pass")
        for pairing_id in ordered_ids
    ]
    rng = random.Random(bootstrap_seed)
    count = len(ordered_ids)
    bootstrap_means = []
    for _ in range(bootstrap_resamples):
        sampled = [quality_differences[rng.randrange(count)] for _ in range(count)]
        bootstrap_means.append(sum(sampled) / count)
    mean_quality_difference = sum(quality_differences) / count
    return PairedHumanGenerationComparison(
        generated_at=datetime.now(timezone.utc),
        dataset_sha256=report_a.dataset_sha256,
        label_a=label_a,
        label_b=label_b,
        prediction_artifact_sha256_a=report_a.prediction_artifact_sha256,
        prediction_artifact_sha256_b=report_b.prediction_artifact_sha256,
        paired_cases=count,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        mean_quality_difference_b_minus_a=mean_quality_difference,
        mean_quality_difference_ci95=(
            _percentile(bootstrap_means, 0.025),
            _percentile(bootstrap_means, 0.975),
        ),
        pass_rate_difference_b_minus_a=sum(pass_differences) / count,
        unsupported_claim_difference_b_minus_a=sum(unsupported_differences) / count,
        b_wins=sum(difference > 0 for difference in quality_differences),
        ties=sum(difference == 0 for difference in quality_differences),
        a_wins=sum(difference < 0 for difference in quality_differences),
        warnings=[
            "The interval is a deterministic case-level bootstrap, not evidence that either model is historically safe.",
            "Do not select a model that fails an absolute safety gate even when its paired mean is higher.",
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--code-revision", required=True)
    run.add_argument("--allow-ineligible-technical-run", action="store_true")

    score = commands.add_parser("score-objective")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    blind = commands.add_parser("export-blind")
    blind.add_argument("--dataset", type=Path, required=True)
    blind.add_argument("--predictions", type=Path, required=True)
    blind.add_argument("--output", type=Path, required=True)

    human = commands.add_parser("score-human")
    human.add_argument("--packet", type=Path, required=True)
    human.add_argument("--grades", type=Path, required=True)
    human.add_argument("--output", type=Path, required=True)

    compare = commands.add_parser("compare-human")
    compare.add_argument("--report-a", type=Path, required=True)
    compare.add_argument("--label-a", required=True)
    compare.add_argument("--report-b", type=Path, required=True)
    compare.add_argument("--label-b", required=True)
    compare.add_argument("--bootstrap-seed", type=int, default=17)
    compare.add_argument("--bootstrap-resamples", type=int, default=5000)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def _write_json(path: Path, model: StrictModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        dataset = GenerationBenchmarkDataset.model_validate_json(
            args.dataset.read_text(encoding="utf-8")
        )
        generator = OpenAICompatibleGenerator.from_environment()
        if generator is None:
            raise SystemExit("generation provider is not configured")
        result: StrictModel = execute_generation_benchmark(
            dataset,
            generator,
            code_revision=args.code_revision,
            allow_ineligible_technical_run=args.allow_ineligible_technical_run,
        )
    elif args.command in {"score-objective", "export-blind"}:
        dataset = GenerationBenchmarkDataset.model_validate_json(
            args.dataset.read_text(encoding="utf-8")
        )
        artifact = GenerationBenchmarkArtifact.model_validate_json(
            args.predictions.read_text(encoding="utf-8")
        )
        result = (
            score_generation_benchmark(dataset, artifact)
            if args.command == "score-objective"
            else build_blinded_generation_packet(dataset, artifact)
        )
    elif args.command == "score-human":
        packet = BlindedGenerationPacket.model_validate_json(
            args.packet.read_text(encoding="utf-8")
        )
        grades = HumanGenerationGradeSet.model_validate_json(
            args.grades.read_text(encoding="utf-8")
        )
        result = score_human_generation_grades(packet, grades)
    else:
        report_a = HumanGenerationReport.model_validate_json(
            args.report_a.read_text(encoding="utf-8")
        )
        report_b = HumanGenerationReport.model_validate_json(
            args.report_b.read_text(encoding="utf-8")
        )
        result = compare_human_generation_reports(
            report_a,
            report_b,
            label_a=args.label_a,
            label_b=args.label_b,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    _write_json(args.output, result)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
