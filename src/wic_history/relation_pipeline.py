"""Propose grounded relation candidates from reviewed, linked co-mentions."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from .evidence import (
    ClaimArtifact,
    ClaimCandidate,
    Point,
    Polygon,
    ProcessingRun,
    RunKind,
    SourcePointer,
)


@dataclass(frozen=True, slots=True)
class ReviewedMention:
    entity_id: UUID
    entity_type: str
    text: str
    text_start: int
    text_end: int


@dataclass(frozen=True, slots=True)
class RegionEvidence:
    region_id: UUID
    raw_text: str
    polygon: dict[str, Any]
    source_uri: str
    source_sha256: str
    derivative_id: UUID
    image_sha256: str
    evidence_tier: str
    volume_number: int
    publication_year: int
    page_number: int


@dataclass(frozen=True, slots=True)
class RelationRule:
    rule_id: str
    subject_type: str
    object_type: str
    predicate: str
    cue: re.Pattern[str]
    maximum_argument_gap: int = 40


RELATION_RULES: tuple[RelationRule, ...] = (
    RelationRule(
        "person-school-explicit-cue-v2",
        "person",
        "school",
        "attended_school",
        re.compile(r"入學|就讀|畢業|肄業"),
    ),
    RelationRule(
        "person-organization-explicit-cue-v2",
        "person",
        "organization",
        "affiliated_with",
        re.compile(r"任職|就任|擔任|服務於|加入"),
    ),
    RelationRule(
        "person-place-explicit-cue-v2",
        "person",
        "place",
        "resided_in",
        re.compile(r"居於|住於|寓|遷居|居住"),
    ),
)


def _validate_reviewed_mentions(
    region: RegionEvidence, mentions: list[ReviewedMention]
) -> None:
    for mention in mentions:
        if not 0 <= mention.text_start < mention.text_end <= len(region.raw_text):
            raise ValueError("reviewed mention offsets are outside the cited OCR region")
        if region.raw_text[mention.text_start : mention.text_end] != mention.text:
            raise ValueError("reviewed mention text disagrees with its exact OCR offsets")


def _cue_between_arguments(
    region: RegionEvidence,
    subject: ReviewedMention,
    object_mention: ReviewedMention,
    rule: RelationRule,
) -> re.Match[str] | None:
    left, right = sorted((subject, object_mention), key=lambda item: item.text_start)
    if left.text_end > right.text_start:
        return None
    if right.text_start - left.text_end > rule.maximum_argument_gap:
        return None
    intervening_text = region.raw_text[left.text_end : right.text_start]
    if re.search(r"[，。；！？\n\r]", intervening_text):
        return None
    return rule.cue.search(region.raw_text, left.text_end, right.text_start)


def extract_region_claims(
    region: RegionEvidence,
    mentions: list[ReviewedMention],
    run_id: UUID,
) -> list[ClaimCandidate]:
    _validate_reviewed_mentions(region, mentions)
    claims = []
    seen: set[tuple[Any, str, Any]] = set()
    for rule in RELATION_RULES:
        subjects = [
            mention for mention in mentions if mention.entity_type == rule.subject_type
        ]
        objects = [
            mention for mention in mentions if mention.entity_type == rule.object_type
        ]
        for subject in subjects:
            for object_mention in objects:
                cue = _cue_between_arguments(region, subject, object_mention, rule)
                if cue is None:
                    continue
                key = (
                    subject.entity_id,
                    rule.predicate,
                    object_mention.entity_id,
                )
                if key in seen or subject.entity_id == object_mention.entity_id:
                    continue
                seen.add(key)
                evidence_start = min(subject.text_start, object_mention.text_start)
                evidence_end = max(subject.text_end, object_mention.text_end)
                claims.append(
                    ClaimCandidate(
                        subject_entity_id=subject.entity_id,
                        predicate=rule.predicate,
                        object_entity_id=object_mention.entity_id,
                        confidence=None,
                        evidence=[
                            SourcePointer(
                                source_uri=region.source_uri,
                                source_sha256=region.source_sha256,
                                derivative_id=region.derivative_id,
                                image_sha256=region.image_sha256,
                                evidence_tier=region.evidence_tier,
                                volume_number=region.volume_number,
                                publication_year=region.publication_year,
                                page_number=region.page_number,
                                region_id=region.region_id,
                                polygon=Polygon(
                                    points=[Point.model_validate(point) for point in region.polygon["points"]]
                                ),
                                text_start=evidence_start,
                                text_end=evidence_end,
                            )
                        ],
                        supporting_quote=region.raw_text[
                            evidence_start:evidence_end
                        ],
                        run_id=run_id,
                    )
                )
    return claims


def create_claim_artifact(database_url: str) -> ClaimArtifact:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    started_at = datetime.now(timezone.utc)
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            """
            SELECT m.region_id, m.entity_id, m.entity_type, m.mention_text,
                   m.text_start, m.text_end, r.raw_text, r.polygon,
                   s.source_uri, s.sha256 AS source_sha256,
                   input.derivative_id, derivative.image_sha256,
                   derivative.evidence_tier,
                   v.volume_number, v.publication_year, p.page_number
            FROM evidence.entity_mention m
            JOIN evidence.entity e USING (entity_id)
            JOIN evidence.ocr_region r USING (region_id)
            JOIN archive.page p USING (page_id)
            JOIN archive.volume v USING (volume_id)
            JOIN archive.source_object s USING (source_object_id)
            JOIN evidence.ocr_run_input input
              ON input.run_id = r.run_id AND input.page_id = r.page_id
            JOIN archive.page_derivative derivative
              ON derivative.derivative_id = input.derivative_id
             AND derivative.page_id = input.page_id
            WHERE m.mention_status = 'reviewed' AND e.entity_status = 'reviewed'
            ORDER BY m.region_id, m.text_start
            """
        ).fetchall()
    by_region: dict[Any, list[ReviewedMention]] = defaultdict(list)
    regions: dict[Any, RegionEvidence] = {}
    for row in rows:
        by_region[row["region_id"]].append(
            ReviewedMention(
                row["entity_id"],
                row["entity_type"],
                row["mention_text"],
                row["text_start"],
                row["text_end"],
            )
        )
        regions[row["region_id"]] = RegionEvidence(
            row["region_id"],
            row["raw_text"],
            row["polygon"],
            row["source_uri"],
            row["source_sha256"],
            row["derivative_id"],
            row["image_sha256"],
            row["evidence_tier"],
            row["volume_number"],
            row["publication_year"],
            row["page_number"],
        )
    run = ProcessingRun(
        kind=RunKind.RELATION,
        engine="reviewed-co-mention-rules",
        model_name="historical-women-relations",
        model_revision="2",
        configuration={
            "reviewed_mentions_only": True,
            "evidence_requirement": "minimal_exact_argument_and_between-cue_span",
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "predicate": rule.predicate,
                    "subject_type": rule.subject_type,
                    "object_type": rule.object_type,
                    "cue_pattern": rule.cue.pattern,
                    "maximum_argument_gap": rule.maximum_argument_gap,
                }
                for rule in RELATION_RULES
            ],
        },
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
    )
    claims = [
        claim
        for region_id, mentions in by_region.items()
        for claim in extract_region_claims(regions[region_id], mentions, run.run_id)
    ]
    warnings = [
        "Relation outputs are candidates; cues and co-occurrence do not prove the relationship."
    ]
    if not rows:
        warnings.append("No reviewed linked mentions exist, so relation extraction safely abstained.")
    return ClaimArtifact(run=run, claims=claims, warnings=warnings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    artifact = create_claim_artifact(args.database_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "claims": len(artifact.claims), "warnings": artifact.warnings},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
