"""Benchmark-gated NER candidate extraction over coordinate-preserving OCR."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

from .evidence import (
    EntityMentionCandidate,
    EntityType,
    NERArtifact,
    OCRPageArtifact,
    ProcessingRun,
    RunKind,
    SourcePointer,
)


DEFAULT_MODEL = "urchade/gliner_multi-v2.1"
DEFAULT_MODEL_REVISION = "443d26d654e0324125a96bebd8e796c14ff2efe6"
CHALLENGER_MODEL = "gliner-community/gliner_large-v2.5"
CHALLENGER_REVISION = "3d6d1760be1c591069f85f207fced9214df8b15f"

MODEL_LABELS: dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "place": EntityType.PLACE,
    "address": EntityType.ADDRESS,
    "organization": EntityType.ORGANIZATION,
    "school": EntityType.SCHOOL,
    "occupation": EntityType.OCCUPATION,
    "role title": EntityType.ROLE_TITLE,
    "publication": EntityType.PUBLICATION,
    "event": EntityType.EVENT,
    "date": EntityType.DATE,
    "product": EntityType.PRODUCT,
}


@dataclass(frozen=True, slots=True)
class SpanCandidate:
    start: int
    end: int
    text: str
    entity_type: EntityType
    score: float
    extractor: str


class BatchPredictor(Protocol):
    def predict(self, texts: list[str], threshold: float) -> list[list[SpanCandidate]]: ...


RULES: tuple[tuple[str, EntityType, re.Pattern[str], float], ...] = (
    (
        "republic_date",
        EntityType.DATE,
        re.compile(r"民國[〇零一二三四五六七八九十百\d]{1,5}年(?:[〇零一二三四五六七八九十\d]{1,3}月)?(?:[〇零一二三四五六七八九十\d]{1,3}日)?"),
        0.95,
    ),
    (
        "organization_suffix",
        EntityType.ORGANIZATION,
        re.compile(r"[\u3400-\u9fff]{2,14}(?:公司|銀行|報館|公會|協會|學會|醫院|工廠)"),
        0.72,
    ),
    (
        "school_suffix",
        EntityType.SCHOOL,
        re.compile(r"[\u3400-\u9fff]{2,14}(?:女子學校|女校|大學|中學|小學|學堂)"),
        0.76,
    ),
    (
        "known_publication",
        EntityType.PUBLICATION,
        re.compile(r"申報|新聞報|時報|婦女雜誌"),
        0.98,
    ),
    (
        "women_role_term",
        EntityType.OCCUPATION,
        re.compile(r"女學生|女工|女醫師|女教師|女記者|女演員|護士|產婆"),
        0.90,
    ),
)


class RulePredictor:
    def predict(self, texts: list[str], threshold: float = 0.0) -> list[list[SpanCandidate]]:
        outputs: list[list[SpanCandidate]] = []
        for text in texts:
            spans = []
            for rule_name, entity_type, pattern, score in RULES:
                if score < threshold:
                    continue
                for match in pattern.finditer(text):
                    spans.append(
                        SpanCandidate(
                            match.start(),
                            match.end(),
                            match.group(),
                            entity_type,
                            score,
                            f"rule:{rule_name}",
                        )
                    )
            outputs.append(spans)
        return outputs


class FixedStanzaLanguageSplitter:
    """Use a declared corpus language instead of detecting each noisy OCR region."""

    def __init__(self, language: str):
        try:
            import stanza
        except ImportError as exc:  # pragma: no cover - optional challenger
            raise RuntimeError("Install the NER extra: uv sync --extra ner") from exc
        try:
            self.pipeline = stanza.Pipeline(
                language,
                processors="tokenize",
                verbose=False,
                download_method=None,
                tokenize_no_ssplit=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Stanza tokenizer '{language}' is unavailable. Download it explicitly with "
                f"python -c \"import stanza; stanza.download('{language}', processors='tokenize')\""
            ) from exc

    def __call__(self, text: str):
        for sentence in self.pipeline(text).sentences:
            for word in sentence.words:
                yield word.text, word.start_char, word.end_char


class GLiNERPredictor:
    def __init__(
        self,
        model_name: str,
        revision: str,
        batch_size: int = 8,
        word_splitter_language: str | None = None,
    ):
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - minimal installations
            raise RuntimeError("Install the NER extra: uv sync --extra ner") from exc
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.model = GLiNER.from_pretrained(
            model_name,
            revision=revision,
            map_location="cpu",
        )
        if word_splitter_language:
            words_splitter = getattr(self.model.data_processor, "words_splitter", None)
            if words_splitter is None or not hasattr(words_splitter, "splitter"):
                raise RuntimeError("This GLiNER model does not expose a replaceable word splitter")
            words_splitter.splitter = FixedStanzaLanguageSplitter(word_splitter_language)
        self.model.eval()

    def predict(self, texts: list[str], threshold: float) -> list[list[SpanCandidate]]:
        outputs = self.model.inference(
            texts,
            list(MODEL_LABELS),
            threshold=threshold,
            flat_ner=True,
            multi_label=False,
            batch_size=self.batch_size,
        )
        results = []
        for text, entities in zip(texts, outputs, strict=True):
            spans = []
            for entity in entities:
                start = int(entity["start"])
                end = int(entity["end"])
                label = str(entity["label"]).lower().replace("_", " ")
                if label not in MODEL_LABELS or not 0 <= start < end <= len(text):
                    continue
                if text[start:end] != entity["text"]:
                    continue
                spans.append(
                    SpanCandidate(
                        start,
                        end,
                        entity["text"],
                        MODEL_LABELS[label],
                        float(entity["score"]),
                        f"gliner:{self.model_name}@{self.revision}",
                    )
                )
            results.append(spans)
        return results


def merge_candidates(*candidate_sets: list[list[SpanCandidate]]) -> list[list[SpanCandidate]]:
    if not candidate_sets:
        return []
    merged: list[list[SpanCandidate]] = []
    for per_text in zip(*candidate_sets, strict=True):
        by_key: dict[tuple[int, int, EntityType], SpanCandidate] = {}
        for candidate in (item for group in per_text for item in group):
            key = (candidate.start, candidate.end, candidate.entity_type)
            current = by_key.get(key)
            if current is None or candidate.score > current.score:
                by_key[key] = candidate
        merged.append(sorted(by_key.values(), key=lambda item: (item.start, item.end, item.entity_type)))
    return merged


def create_ner_artifact(
    ocr: OCRPageArtifact,
    predictors: list[BatchPredictor],
    model_name: str,
    model_revision: str,
    threshold: float,
    batch_size: int,
    word_splitter_language: str | None = None,
    max_regions: int | None = None,
) -> NERArtifact:
    started_at = datetime.now(timezone.utc)
    eligible = [region for region in ocr.regions if len(region.raw_text.strip()) >= 2]
    if max_regions is not None:
        eligible = eligible[:max_regions]
    texts = [region.raw_text for region in eligible]
    predicted_sets = [predictor.predict(texts, threshold) for predictor in predictors]
    candidates = merge_candidates(*predicted_sets)
    run = ProcessingRun(
        kind=RunKind.NER,
        engine="rules+GLiNER" if len(predictors) > 1 else type(predictors[0]).__name__,
        model_name=model_name,
        model_revision=model_revision,
        software_version="gliner-0.2.27",
        configuration={
            "threshold": threshold,
            "batch_size": batch_size,
            "labels": list(MODEL_LABELS),
            "rule_set": "historical-women-zh-v1",
            "word_splitter_language": word_splitter_language,
            "max_regions": max_regions,
        },
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
    )
    mentions = []
    for region, spans in zip(eligible, candidates, strict=True):
        for span in spans:
            mentions.append(
                EntityMentionCandidate(
                    entity_type=span.entity_type,
                    text=span.text,
                    normalized_text=span.text,
                    source=SourcePointer(
                        source_uri=ocr.source.source_uri,
                        source_sha256=ocr.source.source_sha256,
                        volume_number=ocr.source.volume_number,
                        publication_year=ocr.source.publication_year,
                        page_number=ocr.source.page_number,
                        region_id=region.region_id,
                        polygon=region.polygon,
                        text_start=span.start,
                        text_end=span.end,
                    ),
                    confidence=span.score,
                    run_id=run.run_id,
                    attributes={"extractor": span.extractor, "candidate_only": True},
                )
            )
    warnings = [
        "All NER outputs are machine candidates and require benchmark validation or review before entity linking."
    ]
    if max_regions is not None:
        warnings.append(
            f"Technical compatibility subset: only the first {max_regions} eligible OCR regions were processed."
        )
    warnings.extend(ocr.warnings)
    return NERArtifact(
        source_ocr_run_id=ocr.run.run_id,
        run=run,
        mentions=mentions,
        warnings=warnings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--word-splitter-language",
        help="Force a pre-downloaded Stanza tokenizer language, e.g. zh-hant",
    )
    parser.add_argument("--max-regions", type=int, help="Bound a technical compatibility run")
    parser.add_argument("--rules-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.max_regions is not None and args.max_regions < 1:
        raise SystemExit("--max-regions must be positive")
    ocr = OCRPageArtifact.model_validate_json(args.ocr_artifact.read_text(encoding="utf-8"))
    predictors: list[BatchPredictor] = [RulePredictor()]
    if args.rules_only:
        model_name = "historical-women-zh-rules"
        model_revision = "1"
    else:
        predictors.append(
            GLiNERPredictor(
                args.model,
                args.revision,
                args.batch_size,
                args.word_splitter_language,
            )
        )
        model_name = f"{args.model}+historical-women-zh-rules"
        model_revision = f"{args.revision}+rules-v1"
    artifact = create_ner_artifact(
        ocr,
        predictors,
        model_name,
        model_revision,
        args.threshold,
        args.batch_size,
        args.word_splitter_language,
        args.max_regions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "mentions": len(artifact.mentions), "warnings": artifact.warnings},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
