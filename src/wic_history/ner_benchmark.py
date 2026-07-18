"""Prepare and execute issue-split, model-neutral NER benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Protocol, Sequence
from uuid import uuid4

from .evidence import EntityMentionCandidate, ProcessingRun, RunKind
from .ner_adapters.base import (
    AdapterIdentity,
    BenchmarkInput,
    BenchmarkInputVariant,
    BenchmarkPredictionArtifact,
    BenchmarkResult,
    BenchmarkSplit,
    IssueSplitManifest,
    NERBenchmarkDataset,
    benchmark_dataset_sha256,
    benchmark_eligibility_failures,
    canonical_sha256,
)
from .ner_adapters.output import AdapterBatchOutput, AdapterItemOutput
from .ner_gold import NERGoldSet
from .ner_pipeline import (
    GLiNERPredictor,
    ONTOLOGY_VERSION,
    BatchPredictor,
    RulePredictor,
    SpanCandidate,
    merge_candidates,
)
from .ner_structured import (
    STRUCTURED_NER_PROMPT_SCHEMA_SHA256,
    STRUCTURED_NER_RESPONSE_FORMAT_SHA256,
    StructuredGenerationBenchmarkAdapter,
    validate_local_artifact,
    verify_ollama_model_digest,
)
from .generation import OpenAICompatibleGenerator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_benchmark_dataset(
    gold: NERGoldSet,
    gold_sha256: str,
    split_manifest: IssueSplitManifest,
    split_manifest_sha256: str,
    *,
    dataset_id: str,
    input_variants: Sequence[BenchmarkInputVariant],
    generated_at: datetime | None = None,
) -> NERBenchmarkDataset:
    """Freeze exact inputs while refusing model-region IDs as gold identity."""
    if gold.schema_version != "1.1":
        raise ValueError("scientific NER benchmarks require gold schema 1.1")
    if split_manifest.dataset_id != gold.dataset_id:
        raise ValueError("split manifest dataset_id must equal the gold dataset_id")
    if len(set(input_variants)) != len(input_variants) or not input_variants:
        raise ValueError("input variants must be nonempty and unique")
    assignment_by_snippet = {
        assignment.snippet_id: assignment for assignment in split_manifest.assignments
    }
    gold_snippet_ids = {snippet.snippet_id for snippet in gold.snippets}
    if set(assignment_by_snippet) != gold_snippet_ids:
        missing = sorted(gold_snippet_ids - set(assignment_by_snippet))
        extra = sorted(set(assignment_by_snippet) - gold_snippet_ids)
        raise ValueError(
            "split manifest must cover exactly the gold snippets; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    inputs = []
    for snippet in gold.snippets:
        if (
            snippet.gold_region_id is None
            or snippet.source_ocr_run_id is None
            or snippet.source_ocr_region_id is None
        ):
            raise ValueError("gold snippet lacks explicit source/gold identity mapping")
        assignment = assignment_by_snippet[snippet.snippet_id]
        for variant in input_variants:
            if variant == "raw_ocr":
                text = snippet.raw_ocr_text
            elif variant == "corrected_text":
                text = snippet.adjudication.corrected_text
            else:
                raise ValueError(
                    "multimodal_transcript preparation requires a separate image-aware manifest"
                )
            input_id = (
                f"{dataset_id}:{assignment.split}:{variant}:"
                f"{snippet.snippet_id}:{snippet.gold_region_id}"
            )
            inputs.append(
                BenchmarkInput(
                    input_id=input_id,
                    snippet_id=snippet.snippet_id,
                    issue_id=assignment.issue_id,
                    split=assignment.split,
                    input_variant=variant,
                    gold_region_id=snippet.gold_region_id,
                    source_ocr_run_id=snippet.source_ocr_run_id,
                    source_ocr_region_id=snippet.source_ocr_region_id,
                    source=snippet.source,
                    text=text,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )

    test_type_counts = {}
    all_types = set()
    for snippet in gold.snippets:
        entity_types = {entity.entity_type for entity in snippet.adjudication.entities}
        all_types.update(entity_types)
        if assignment_by_snippet[snippet.snippet_id].split == "test":
            for entity in snippet.adjudication.entities:
                entity_type = entity.entity_type
                test_type_counts[entity_type] = test_type_counts.get(entity_type, 0) + 1
    reported_entity_types = sorted(all_types, key=lambda item: item.value)
    locked_test_mentions_by_type = {
        entity_type: test_type_counts.get(entity_type, 0)
        for entity_type in reported_entity_types
    }
    failures = benchmark_eligibility_failures(
        inputs, reported_entity_types, locked_test_mentions_by_type
    )
    return NERBenchmarkDataset(
        dataset_id=dataset_id,
        generated_at=generated_at or datetime.now(timezone.utc),
        ontology_version=gold.ontology_version,
        source_gold_dataset_id=gold.dataset_id,
        source_gold_sha256=gold_sha256,
        split_manifest_sha256=split_manifest_sha256,
        inputs=inputs,
        reported_entity_types=reported_entity_types,
        locked_test_mentions_by_type=locked_test_mentions_by_type,
        benchmark_eligible=not failures,
        eligibility_failures=failures,
        warnings=[
            "Corrected text is an oracle NER arm; raw OCR is the end-to-end arm.",
            "No issue may cross splits. Test data must remain unopened during tuning.",
        ],
    )


class BenchmarkAdapter(Protocol):
    identity: AdapterIdentity

    def predict(self, texts: list[str]) -> "AdapterBatchOutput": ...


class PredictorBenchmarkAdapter:
    """Wrap existing exact-span batch predictors behind the common adapter boundary."""

    def __init__(
        self,
        identity: AdapterIdentity,
        predictors: list[BatchPredictor],
        *,
        threshold: float,
        batch_size: int,
    ):
        if not predictors:
            raise ValueError("at least one predictor is required")
        self.identity = identity
        self.predictors = predictors
        self.threshold = threshold
        self.batch_size = batch_size

    def predict(self, texts: list[str]) -> AdapterBatchOutput:
        all_candidates: list[list[SpanCandidate]] = []
        latencies = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            began = time.perf_counter()
            outputs = [
                predictor.predict(batch, self.threshold)
                for predictor in self.predictors
            ]
            merged = merge_candidates(*outputs)
            elapsed = time.perf_counter() - began
            all_candidates.extend(merged)
            latencies.extend([elapsed / len(batch)] * len(batch))
        return AdapterBatchOutput(
            [
                AdapterItemOutput(spans=spans, latency_seconds=latency)
                for spans, latency in zip(all_candidates, latencies, strict=True)
            ]
        )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def execute_benchmark(
    dataset: NERBenchmarkDataset,
    adapter: BenchmarkAdapter,
    *,
    split: BenchmarkSplit,
    input_variant: BenchmarkInputVariant,
    allow_ineligible_technical_run: bool = False,
) -> BenchmarkPredictionArtifact:
    if not dataset.benchmark_eligible and not allow_ineligible_technical_run:
        raise ValueError(
            "benchmark dataset is ineligible: "
            + "; ".join(dataset.eligibility_failures)
        )
    if adapter.identity.ontology_version != dataset.ontology_version:
        raise ValueError("adapter ontology does not match the benchmark dataset")
    inputs = [
        item
        for item in dataset.inputs
        if item.split == split and item.input_variant == input_variant
    ]
    if not inputs:
        raise ValueError(f"no {split}/{input_variant} inputs exist")
    texts = [item.text for item in inputs]
    run_id = uuid4()
    started_at = datetime.now(timezone.utc)
    batch_output = adapter.predict(texts)
    if len(batch_output.items) != len(inputs):
        raise ValueError("adapter output count does not match benchmark inputs")
    latencies = [item.latency_seconds for item in batch_output.items]
    if any(latency < 0 or not math.isfinite(latency) for latency in latencies):
        raise ValueError("adapter latency must be finite and nonnegative")
    completed_at = datetime.now(timezone.utc)
    results = []
    all_mentions = []
    for item, adapter_output in zip(inputs, batch_output.items, strict=True):
        mentions = []
        invalid_outputs = adapter_output.invalid_outputs
        for span in adapter_output.spans:
            if not 0 <= span.start < span.end <= len(item.text):
                invalid_outputs += 1
                continue
            if item.text[span.start : span.end] != span.text:
                invalid_outputs += 1
                continue
            mention = EntityMentionCandidate(
                entity_type=span.entity_type,
                text=span.text,
                normalized_text=span.text,
                source=item.source.model_copy(
                    update={"text_start": span.start, "text_end": span.end}
                ),
                confidence=span.score if span.confidence_available else None,
                run_id=run_id,
                attributes={
                    "benchmark_input_id": item.input_id,
                    "source_ocr_run_id": str(item.source_ocr_run_id),
                    "input_variant": item.input_variant,
                    "extractor": span.extractor,
                    "extractor_support": [
                        {"extractor": extractor, "raw_score": score}
                        for extractor, score in span.supports
                    ],
                    "candidate_only": True,
                    "calibrated": False,
                    "confidence_semantics": (
                        "uncalibrated_candidate_score"
                        if span.confidence_available
                        else "not_provided_by_adapter"
                    ),
                },
            )
            mentions.append(mention)
        all_mentions.extend(mentions)
        results.append(
            BenchmarkResult(
                input_id=item.input_id,
                snippet_id=item.snippet_id,
                issue_id=item.issue_id,
                gold_region_id=item.gold_region_id,
                source_ocr_run_id=item.source_ocr_run_id,
                source_ocr_region_id=item.source_ocr_region_id,
                input_text_sha256=item.text_sha256,
                mentions=mentions,
                abstention_reason=adapter_output.abstention_reason,
                latency_seconds=adapter_output.latency_seconds,
                raw_output_sha256=adapter_output.raw_output_sha256
                or canonical_sha256(
                    [
                        {
                            "start": span.start,
                            "end": span.end,
                            "text": span.text,
                            "type": span.entity_type.value,
                            "score": span.score,
                            "extractor": span.extractor,
                        }
                        for span in adapter_output.spans
                    ]
                ),
                prompt_sha256=adapter_output.prompt_sha256,
                finish_reason=adapter_output.finish_reason,
                prompt_tokens=adapter_output.prompt_tokens,
                completion_tokens=adapter_output.completion_tokens,
                total_tokens=adapter_output.total_tokens,
                invalid_outputs=invalid_outputs,
            )
        )
    input_sha256 = canonical_sha256(
        [
            {"input_id": item.input_id, "text_sha256": item.text_sha256}
            for item in inputs
        ]
    )
    prompt_token_values = [result.prompt_tokens for result in results]
    completion_token_values = [result.completion_tokens for result in results]
    total_token_values = [result.total_tokens for result in results]
    run = ProcessingRun(
        run_id=run_id,
        kind=RunKind.NER,
        engine=adapter.identity.adapter_id,
        model_name=adapter.identity.model_name,
        model_revision=adapter.identity.model_revision,
        software_version=adapter.identity.runtime,
        configuration={
            "benchmark_dataset_sha256": benchmark_dataset_sha256(dataset),
            "split": split,
            "input_variant": input_variant,
            "input_sha256": input_sha256,
            "input_region_count": len(inputs),
            "input_character_count": sum(len(text) for text in texts),
            "latency_p50_seconds": median(latencies),
            "latency_p95_seconds": _percentile(latencies, 0.95),
            "device": adapter.identity.device,
            "dtype": adapter.identity.dtype,
            "invalid_outputs": sum(result.invalid_outputs for result in results),
            "prompt_tokens": (
                sum(prompt_token_values)
                if all(value is not None for value in prompt_token_values)
                else None
            ),
            "completion_tokens": (
                sum(completion_token_values)
                if all(value is not None for value in completion_token_values)
                else None
            ),
            "total_tokens": (
                sum(total_token_values)
                if all(value is not None for value in total_token_values)
                else None
            ),
            "token_usage_complete_results": sum(
                result.prompt_tokens is not None
                and result.completion_tokens is not None
                and result.total_tokens is not None
                for result in results
            ),
        },
        started_at=started_at,
        completed_at=completed_at,
    )
    warnings = [
        "Benchmark predictions are experiment artifacts and must never be ingested as reviewed mentions."
    ]
    if not dataset.benchmark_eligible:
        warnings.append(
            "INELIGIBLE TECHNICAL RUN: " + "; ".join(dataset.eligibility_failures)
        )
    return BenchmarkPredictionArtifact(
        artifact_id=run_id,
        benchmark_dataset_id=dataset.dataset_id,
        benchmark_dataset_sha256=benchmark_dataset_sha256(dataset),
        source_gold_dataset_id=dataset.source_gold_dataset_id,
        source_gold_sha256=dataset.source_gold_sha256,
        split_manifest_sha256=dataset.split_manifest_sha256,
        split=split,
        input_variant=input_variant,
        input_sha256=input_sha256,
        ontology_version=dataset.ontology_version,
        adapter=adapter.identity,
        run=run,
        source_ocr_run_ids=sorted({item.source_ocr_run_id for item in inputs}, key=str),
        input_ids=[item.input_id for item in inputs],
        results=results,
        mentions=all_mentions,
        warnings=warnings,
    )


def _adapter_from_args(args: argparse.Namespace) -> BenchmarkAdapter:
    if args.adapter == "rules":
        identity = AdapterIdentity(
            adapter_id="historical-women-zh-rules-v1",
            family="rules",
            model_name="historical-women-zh-rules",
            model_revision="rules-v1",
            license="project-code",
            modalities=["text"],
            runtime="python-re",
            code_revision=args.code_revision,
            device="cpu",
            dtype="deterministic",
            ontology_version=ONTOLOGY_VERSION,
            configuration={"threshold": args.threshold},
        )
        predictors: list[BatchPredictor] = [RulePredictor()]
    elif args.adapter == "gliner":
        if not args.model or not args.revision or not args.license:
            raise ValueError("GLiNER requires --model, --revision and --license")
        identity = AdapterIdentity(
            adapter_id=f"gliner:{args.model}",
            family="gliner",
            model_name=args.model,
            model_revision=args.revision,
            license=args.license,
            modalities=["text"],
            runtime="gliner-0.2.27",
            code_revision=args.code_revision,
            device="cpu",
            dtype="float32",
            ontology_version=ONTOLOGY_VERSION,
            configuration={
                "threshold": args.threshold,
                "batch_size": args.batch_size,
                "word_splitter_language": args.word_splitter_language,
                "flat_ner": False,
                "multi_label": True,
            },
        )
        predictors = [
            GLiNERPredictor(
                args.model,
                args.revision,
                args.batch_size,
                args.word_splitter_language,
                flat_ner=False,
                multi_label=True,
            )
        ]
        return PredictorBenchmarkAdapter(
            identity, predictors, threshold=args.threshold, batch_size=args.batch_size
        )
    else:
        required = {
            "--model": args.model,
            "--revision": args.revision,
            "--license": args.license,
            "--base-url": args.base_url,
            "--served-model": args.served_model,
            "--runtime-version": args.runtime_version,
            "--local-artifact-sha256": args.local_artifact_sha256,
            "--quantization": args.quantization,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "structured generation requires " + ", ".join(sorted(missing))
            )
        local_artifact_sha256 = args.local_artifact_sha256
        generator = OpenAICompatibleGenerator(
            args.base_url,
            args.served_model,
            api_key=os.environ.get("NER_LLM_API_KEY"),
            model_revision=local_artifact_sha256,
            timeout_seconds=args.timeout_seconds,
            max_output_tokens=args.max_output_tokens,
            seed=args.seed,
            allow_remote=args.allow_remote_model_endpoint,
        )
        runtime_verification: dict[str, object]
        if args.runtime_name == "ollama":
            if not args.ollama_manifest_digest:
                raise ValueError(
                    "Ollama structured generation requires --ollama-manifest-digest"
                )
            verification = verify_ollama_model_digest(
                generator,
                args.ollama_manifest_digest,
                args.runtime_version,
            )
            if verification.observed_digest[7:] != local_artifact_sha256:
                raise ValueError(
                    "Ollama manifest digest must equal local artifact SHA-256"
                )
            runtime_verification = verification.model_dump(mode="json")
        else:
            if args.local_model_artifact is None:
                raise ValueError(
                    "LM Studio structured generation requires --local-model-artifact"
                )
            validate_local_artifact(args.local_model_artifact, local_artifact_sha256)
            runtime_verification = {
                "artifact_path": str(args.local_model_artifact.resolve()),
                "artifact_sha256": local_artifact_sha256,
            }
        identity = AdapterIdentity(
            adapter_id=f"structured-ner:{args.runtime_name}:{args.served_model}",
            family="structured_generation",
            model_name=args.model,
            model_revision=args.revision,
            license=args.license,
            modalities=["text"],
            runtime=f"{args.runtime_name}-{args.runtime_version}",
            code_revision=args.code_revision,
            device=args.device,
            dtype=args.quantization,
            ontology_version=ONTOLOGY_VERSION,
            prompt_schema_revision=STRUCTURED_NER_PROMPT_SCHEMA_SHA256,
            configuration={
                "base_url": generator.base_url,
                "served_model": args.served_model,
                "runtime_name": args.runtime_name,
                "runtime_version": args.runtime_version,
                "runtime_verification": runtime_verification,
                "local_artifact_sha256": local_artifact_sha256,
                "quantization": args.quantization,
                "temperature": 0,
                "top_p": 1,
                "reasoning_effort": "none",
                "seed": args.seed,
                "max_output_tokens": args.max_output_tokens,
                "timeout_seconds": args.timeout_seconds,
                "response_format_sha256": STRUCTURED_NER_RESPONSE_FORMAT_SHA256,
                "remote_data_egress_allowed": args.allow_remote_model_endpoint,
            },
        )
        structured_adapter = StructuredGenerationBenchmarkAdapter(identity, generator)
        canary = structured_adapter.run_schema_canary(args.schema_canary_repetitions)
        if not canary.deterministic:
            raise RuntimeError(
                "structured NER schema canary was nondeterministic; aborting benchmark"
            )
        return structured_adapter.with_canary(canary)

    return PredictorBenchmarkAdapter(
        identity, predictors, threshold=args.threshold, batch_size=args.batch_size
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--gold", type=Path, required=True)
    prepare.add_argument("--split-manifest", type=Path, required=True)
    prepare.add_argument("--dataset-id", required=True)
    prepare.add_argument(
        "--input-variant",
        action="append",
        choices=("raw_ocr", "corrected_text"),
        required=True,
    )
    prepare.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--split", choices=("train", "development", "test"), required=True)
    run.add_argument(
        "--input-variant", choices=("raw_ocr", "corrected_text"), required=True
    )
    run.add_argument(
        "--adapter",
        choices=("rules", "gliner", "structured-generation"),
        required=True,
    )
    run.add_argument("--model")
    run.add_argument("--revision")
    run.add_argument("--license")
    run.add_argument("--threshold", type=float, default=0.45)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--word-splitter-language")
    run.add_argument("--base-url")
    run.add_argument("--served-model")
    run.add_argument(
        "--runtime-name", choices=("ollama", "lm_studio"), default="ollama"
    )
    run.add_argument("--runtime-version")
    run.add_argument("--local-artifact-sha256")
    run.add_argument("--local-model-artifact", type=Path)
    run.add_argument("--ollama-manifest-digest")
    run.add_argument("--quantization")
    run.add_argument("--device", default="local-runtime")
    run.add_argument("--timeout-seconds", type=float, default=120)
    run.add_argument("--max-output-tokens", type=int, default=2048)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--schema-canary-repetitions", type=int, default=3)
    run.add_argument("--allow-remote-model-endpoint", action="store_true")
    run.add_argument("--code-revision", required=True)
    run.add_argument("--allow-ineligible-technical-run", action="store_true")
    run.add_argument("--confirm-locked-test-evaluation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        gold = NERGoldSet.model_validate_json(args.gold.read_text(encoding="utf-8"))
        manifest = IssueSplitManifest.model_validate_json(
            args.split_manifest.read_text(encoding="utf-8")
        )
        dataset = prepare_benchmark_dataset(
            gold,
            sha256_file(args.gold),
            manifest,
            sha256_file(args.split_manifest),
            dataset_id=args.dataset_id,
            input_variants=args.input_variant,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            dataset.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        payload = {
            "output": str(args.output),
            "dataset_sha256": benchmark_dataset_sha256(dataset),
            "inputs": len(dataset.inputs),
            "benchmark_eligible": dataset.benchmark_eligible,
            "eligibility_failures": dataset.eligibility_failures,
        }
    else:
        if not 0 <= args.threshold <= 1 or args.batch_size < 1:
            raise SystemExit("threshold must be 0–1 and batch-size must be positive")
        if args.split == "test" and not args.confirm_locked_test_evaluation:
            raise SystemExit(
                "test split is locked; add --confirm-locked-test-evaluation only after "
                "freezing the model, threshold and configuration"
            )
        dataset = NERBenchmarkDataset.model_validate_json(
            args.dataset.read_text(encoding="utf-8")
        )
        adapter = _adapter_from_args(args)
        artifact = execute_benchmark(
            dataset,
            adapter,
            split=args.split,
            input_variant=args.input_variant,
            allow_ineligible_technical_run=args.allow_ineligible_technical_run,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        duration = (artifact.run.completed_at - artifact.run.started_at).total_seconds()
        payload = {
            "output": str(args.output),
            "mentions": len(artifact.mentions),
            "inputs": len(artifact.input_ids),
            "duration_seconds": duration,
            "invalid_outputs": sum(
                result.invalid_outputs for result in artifact.results
            ),
            "warnings": artifact.warnings,
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
