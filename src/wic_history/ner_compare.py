"""Compare NER candidate artifacts without pretending disagreement is accuracy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .evidence import EntityMentionCandidate, NERArtifact


def mention_key(mention: EntityMentionCandidate) -> tuple[str, int, int, str]:
    source = mention.source
    if source.region_id is None or source.text_start is None or source.text_end is None:
        raise ValueError("NER comparisons require region IDs and exact offsets")
    return (
        str(source.region_id),
        source.text_start,
        source.text_end,
        mention.entity_type.value,
    )


def _compact(mention: EntityMentionCandidate) -> dict[str, Any]:
    return {
        "region_id": str(mention.source.region_id),
        "text_start": mention.source.text_start,
        "text_end": mention.source.text_end,
        "text": mention.text,
        "entity_type": mention.entity_type.value,
        "confidence": mention.confidence,
        "extractor": mention.attributes.get("extractor"),
    }


def _summary(artifact: NERArtifact) -> dict[str, Any]:
    duration = None
    if artifact.run.completed_at:
        duration = (artifact.run.completed_at - artifact.run.started_at).total_seconds()
    extractors = Counter(
        "rule" if str(mention.attributes.get("extractor", "")).startswith("rule:") else "model"
        for mention in artifact.mentions
    )
    return {
        "artifact_schema_version": artifact.schema_version,
        "input_variant": artifact.input_variant,
        "input_sha256": artifact.input_sha256,
        "dataset_id": artifact.dataset_id,
        "split_id": artifact.split_id,
        "ontology_version": artifact.ontology_version,
        "adapter_id": artifact.adapter_id,
        "model_name": artifact.run.model_name,
        "model_revision": artifact.run.model_revision,
        "configuration": artifact.run.configuration,
        "duration_seconds": duration,
        "mentions": len(artifact.mentions),
        "model_mentions": extractors["model"],
        "rule_mentions": extractors["rule"],
        "by_entity_type": dict(sorted(Counter(item.entity_type.value for item in artifact.mentions).items())),
        "mean_span_characters": (
            sum(len(item.text) for item in artifact.mentions) / len(artifact.mentions)
            if artifact.mentions
            else 0.0
        ),
        "mean_confidence": (
            sum(item.confidence for item in artifact.mentions) / len(artifact.mentions)
            if artifact.mentions
            else None
        ),
    }


def compare_artifacts(left: NERArtifact, right: NERArtifact) -> dict[str, Any]:
    if left.source_ocr_run_id != right.source_ocr_run_id:
        raise ValueError("NER comparisons require the same source OCR run")
    identity_fields = (
        "input_variant",
        "input_sha256",
        "dataset_id",
        "split_id",
        "ontology_version",
    )
    if any(getattr(left, field) != getattr(right, field) for field in identity_fields):
        raise ValueError("NER comparisons require byte-identical input and benchmark identity")
    left_by_key = {mention_key(mention): mention for mention in left.mentions}
    right_by_key = {mention_key(mention): mention for mention in right.mentions}
    left_keys = set(left_by_key)
    right_keys = set(right_by_key)
    intersection = left_keys & right_keys
    union = left_keys | right_keys
    left_spans = {(key[0], key[1], key[2]) for key in left_keys}
    right_spans = {(key[0], key[1], key[2]) for key in right_keys}

    def ranked(items: list[EntityMentionCandidate]) -> list[dict[str, Any]]:
        return [_compact(item) for item in sorted(items, key=lambda value: value.confidence, reverse=True)[:25]]

    return {
        "schema_version": "1.0",
        "source_ocr_run_id": str(left.source_ocr_run_id),
        "left": _summary(left),
        "right": _summary(right),
        "agreement": {
            "exact_span_and_type": len(intersection),
            "same_span_any_type": len(left_spans & right_spans),
            "candidate_jaccard": len(intersection) / len(union) if union else 1.0,
        },
        "common": ranked([left_by_key[key] for key in intersection]),
        "left_only_high_confidence": ranked(
            [left_by_key[key] for key in left_keys - right_keys]
        ),
        "right_only_high_confidence": ranked(
            [right_by_key[key] for key in right_keys - left_keys]
        ),
        "warnings": [
            "This report measures candidate volume and agreement, not precision, recall, or historical correctness.",
            "Only double-reviewed gold spans can select a production NER model or threshold.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    left = NERArtifact.model_validate_json(args.left.read_text(encoding="utf-8"))
    right = NERArtifact.model_validate_json(args.right.read_text(encoding="utf-8"))
    report = compare_artifacts(left, right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "left_mentions": report["left"]["mentions"],
                "right_mentions": report["right"]["mentions"],
                **report["agreement"],
                "warnings": report["warnings"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
