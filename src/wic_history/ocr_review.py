"""Validate append-preserving human review ledgers for OCR hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from datetime import date
from pathlib import Path
from typing import Literal, Sequence
from uuid import UUID

from pydantic import Field, model_validator

from .evidence import StrictModel


class ReviewCropBox(StrictModel):
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(gt=0)
    bottom: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_extent(self) -> "ReviewCropBox":
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("crop box must have positive width and height")
        return self


class OCRReviewTarget(StrictModel):
    case_id: str = Field(min_length=1, max_length=100)
    volume_number: int = Field(ge=1)
    page_number: int = Field(ge=1)
    source_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    crop_uri: str = Field(min_length=1)
    crop_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    crop_box: ReviewCropBox


class OCRModelHypothesis(StrictModel):
    hypothesis_id: UUID
    engine: str = Field(min_length=1, max_length=200)
    model_name: str = Field(min_length=1, max_length=300)
    model_revision: str = Field(min_length=1, max_length=300)
    raw_text: str
    raw_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_raw_text_hash(self) -> "OCRModelHypothesis":
        digest = hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()
        if digest != self.raw_text_sha256:
            raise ValueError("raw_text_sha256 does not match the exact model text")
        return self


class OCRContextEvidence(StrictModel):
    description: str = Field(min_length=1, max_length=2000)
    crop_uri: str | None = None
    crop_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_optional_crop(self) -> "OCRContextEvidence":
        if (self.crop_uri is None) != (self.crop_sha256 is None):
            raise ValueError("context crop URI and SHA-256 must be provided together")
        return self


class SingleHumanOCRReview(StrictModel):
    review_id: UUID
    reviewer_role: str = Field(min_length=1, max_length=200)
    reviewed_on: date
    decision: Literal["confirmed", "corrected", "partial", "context_resolved"]
    provided_text: str = Field(min_length=1)
    source_script_transcription: str = Field(min_length=1)
    evidence: list[OCRContextEvidence] = Field(default_factory=list)
    superseded_proposals: list[str] = Field(default_factory=list)
    unresolved_note: str | None = Field(default=None, min_length=1, max_length=2000)
    gold_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_review(self) -> "SingleHumanOCRReview":
        if self.decision == "partial" and self.unresolved_note is None:
            raise ValueError("partial OCR review requires an unresolved_note")
        if self.decision != "partial" and self.unresolved_note is not None:
            raise ValueError("only a partial OCR review may carry an unresolved_note")
        for label, text in (
            ("provided_text", self.provided_text),
            ("source_script_transcription", self.source_script_transcription),
        ):
            if unicodedata.normalize("NFC", text) != text:
                raise ValueError(f"{label} must be Unicode NFC")
        return self


class OCRReviewCase(StrictModel):
    target: OCRReviewTarget
    hypotheses: list[OCRModelHypothesis] = Field(min_length=1)
    reviews: list[SingleHumanOCRReview] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "OCRReviewCase":
        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        review_ids = [item.review_id for item in self.reviews]
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError("OCR hypothesis IDs must be unique within a case")
        if len(set(review_ids)) != len(review_ids):
            raise ValueError("OCR review IDs must be unique within a case")
        return self


class OCRReviewLedger(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    created_on: date
    gold_status: Literal["single_review_not_gold"] = "single_review_not_gold"
    cases: list[OCRReviewCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_cases(self) -> "OCRReviewLedger":
        case_ids = [item.target.case_id for item in self.cases]
        review_ids = [
            review.review_id for case in self.cases for review in case.reviews
        ]
        hypothesis_ids = [
            hypothesis.hypothesis_id
            for case in self.cases
            for hypothesis in case.hypotheses
        ]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("OCR review case IDs must be unique")
        if len(set(review_ids)) != len(review_ids):
            raise ValueError("OCR review IDs must be globally unique")
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError("OCR hypothesis IDs must be globally unique")
        return self


def load_review_ledger(path: Path) -> OCRReviewLedger:
    return OCRReviewLedger.model_validate_json(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ledger = load_review_ledger(args.input)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "cases": len(ledger.cases),
                "hypotheses": sum(len(item.hypotheses) for item in ledger.cases),
                "reviews": sum(len(item.reviews) for item in ledger.cases),
                "gold_status": ledger.gold_status,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
