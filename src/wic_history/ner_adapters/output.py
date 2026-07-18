"""Rich per-input outputs shared by deterministic and generative NER adapters."""

from __future__ import annotations

from dataclasses import dataclass

from ..ner_pipeline import SpanCandidate


@dataclass(frozen=True, slots=True)
class AdapterItemOutput:
    spans: list[SpanCandidate]
    latency_seconds: float
    raw_output_sha256: str | None = None
    prompt_sha256: str | None = None
    invalid_outputs: int = 0
    abstention_reason: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AdapterBatchOutput:
    items: list[AdapterItemOutput]
