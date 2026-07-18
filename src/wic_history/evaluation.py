"""Run reproducible, citation-aware retrieval evaluations."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import UUID

from pydantic import Field, model_validator

from .embedding_pipeline import BGEEmbedder, DEFAULT_MODEL, DEFAULT_REVISION
from .evidence import RetrievalMode, RetrievalResponse, StrictModel
from .search import DEFAULT_ALIAS, dense_search, hybrid_search, lexical_search


class QuestionCategory(StrEnum):
    EXACT_LOOKUP = "exact_lookup"
    OCR_VARIANT = "ocr_variant"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    CORPUS_THEME = "corpus_theme"
    UNANSWERABLE = "unanswerable"
    CITATION_TRACE = "citation_trace"


class EvaluationQuestion(StrictModel):
    question_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: QuestionCategory
    answerable: bool = True
    expected_region_ids: list[UUID] = Field(default_factory=list)
    year_start: int | None = Field(default=None, ge=1872, le=1949)
    year_end: int | None = Field(default=None, ge=1872, le=1949)
    author: str
    notes: str | None = None

    @model_validator(mode="after")
    def validate_judgment(self) -> "EvaluationQuestion":
        if self.answerable and not self.expected_region_ids:
            raise ValueError("answerable retrieval questions require expected_region_ids")
        if not self.answerable and self.expected_region_ids:
            raise ValueError("unanswerable questions cannot declare expected regions")
        if self.year_start and self.year_end and self.year_end < self.year_start:
            raise ValueError("year_end cannot precede year_start")
        return self


class QuestionResult(StrictModel):
    question_id: str
    category: QuestionCategory
    answerable: bool
    expected_region_ids: list[UUID]
    retrieved_region_ids: list[UUID]
    relevant_retrieved: int
    recall_at_k: float | None
    reciprocal_rank: float | None
    citation_pointer_rate: float
    response: RetrievalResponse


class EvaluationReport(StrictModel):
    schema_version: str = "1.0"
    generated_at: datetime
    mode: RetrievalMode
    index: str
    limit: int
    question_set: str
    question_count: int
    scored_answerable_questions: int
    macro_recall_at_k: float | None
    mean_reciprocal_rank: float | None
    citation_pointer_rate: float
    results: list[QuestionResult]
    warnings: list[str] = Field(default_factory=list)


def load_questions(path: Path) -> list[EvaluationQuestion]:
    questions = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                questions.append(EvaluationQuestion.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid evaluation question at {path}:{line_number}: {exc}") from exc
    if not questions:
        raise ValueError("Evaluation question set is empty")
    identifiers = [question.question_id for question in questions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Evaluation question IDs must be unique")
    return questions


def score_question(
    question: EvaluationQuestion, response: RetrievalResponse
) -> QuestionResult:
    retrieved = [hit.source.region_id for hit in response.hits if hit.source.region_id is not None]
    expected = set(question.expected_region_ids)
    relevant = [region_id for region_id in retrieved if region_id in expected]
    recall = len(set(relevant)) / len(expected) if expected else None
    reciprocal_rank = None
    if expected:
        reciprocal_rank = next(
            (1.0 / rank for rank, region_id in enumerate(retrieved, 1) if region_id in expected),
            0.0,
        )
    citation_count = sum(
        hit.source.region_id is not None and hit.source.polygon is not None
        for hit in response.hits
    )
    citation_rate = citation_count / len(response.hits) if response.hits else 1.0
    return QuestionResult(
        question_id=question.question_id,
        category=question.category,
        answerable=question.answerable,
        expected_region_ids=question.expected_region_ids,
        retrieved_region_ids=retrieved,
        relevant_retrieved=len(set(relevant)),
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        citation_pointer_rate=citation_rate,
        response=response,
    )


def evaluate(
    questions: list[EvaluationQuestion],
    retrieve: Callable[[EvaluationQuestion], RetrievalResponse],
    *,
    mode: RetrievalMode,
    index: str,
    limit: int,
    question_set: str,
) -> EvaluationReport:
    results = [score_question(question, retrieve(question)) for question in questions]
    scored = [result for result in results if result.answerable]
    recalls = [result.recall_at_k for result in scored if result.recall_at_k is not None]
    reciprocal_ranks = [
        result.reciprocal_rank for result in scored if result.reciprocal_rank is not None
    ]
    hit_count = sum(len(result.response.hits) for result in results)
    citation_rate = (
        sum(result.citation_pointer_rate * len(result.response.hits) for result in results)
        / hit_count
        if hit_count
        else 1.0
    )
    warnings = []
    if all(question.author == "technical-smoke" for question in questions):
        warnings.append(
            "This is a technical smoke set, not a historian-authored quality evaluation."
        )
    return EvaluationReport(
        generated_at=datetime.now(timezone.utc),
        mode=mode,
        index=index,
        limit=limit,
        question_set=question_set,
        question_count=len(questions),
        scored_answerable_questions=len(scored),
        macro_recall_at_k=sum(recalls) / len(recalls) if recalls else None,
        mean_reciprocal_rank=(
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None
        ),
        citation_pointer_rate=citation_rate,
        results=results,
        warnings=warnings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opensearch-url", default=os.environ.get("OPENSEARCH_URL", "http://127.0.0.1:9200"))
    parser.add_argument("--index", default=DEFAULT_ALIAS)
    parser.add_argument("--mode", choices=("lexical", "dense", "hybrid"), default="lexical")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    questions = load_questions(args.questions)
    mode = RetrievalMode(args.mode)
    embedder = BGEEmbedder(args.model, args.revision) if mode != RetrievalMode.LEXICAL else None

    def retrieve(question: EvaluationQuestion) -> RetrievalResponse:
        positional: tuple[Any, ...] = (
            args.opensearch_url,
            question.query,
            args.index,
            args.limit,
            question.year_start,
            question.year_end,
        )
        if mode == RetrievalMode.LEXICAL:
            return lexical_search(*positional)
        if mode == RetrievalMode.DENSE:
            return dense_search(
                args.opensearch_url,
                question.query,
                embedder,
                args.index,
                args.limit,
                question.year_start,
                question.year_end,
            )
        return hybrid_search(
            args.opensearch_url,
            question.query,
            embedder,
            args.index,
            args.limit,
            year_start=question.year_start,
            year_end=question.year_end,
        )

    report = evaluate(
        questions,
        retrieve,
        mode=mode,
        index=args.index,
        limit=args.limit,
        question_set=str(args.questions),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    summary = {
        key: value
        for key, value in report.model_dump(mode="json").items()
        if key
        in {
            "mode",
            "question_count",
            "macro_recall_at_k",
            "mean_reciprocal_rank",
            "citation_pointer_rate",
            "warnings",
        }
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
