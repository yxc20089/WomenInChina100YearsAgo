"""Compare paired NER scores with issue-cluster bootstrap uncertainty."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence


MATCHED_FIELDS = (
    "dataset_id",
    "ontology_version",
    "input_text",
    "dataset_split",
    "benchmark_dataset_sha256",
    "source_gold_sha256",
    "input_sha256",
    "confidence_threshold",
)


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = (2 * true_positive) + false_positive + false_negative
    return (2 * true_positive) / denominator if denominator else 1.0


def _validate_issue_metrics(report: dict[str, Any], label: str) -> dict[str, Any]:
    if report.get("source_gold_sha256_verified") is not True:
        raise ValueError(f"{label} score must verify the exact gold file SHA-256")
    if report.get("benchmark_dataset_sha256_verified") is not True:
        raise ValueError(f"{label} score must verify the frozen benchmark dataset")
    issues = report.get("by_issue")
    if not isinstance(issues, dict) or not issues:
        raise ValueError(f"{label} score lacks issue-level benchmark metrics")
    for issue_id, metrics in issues.items():
        if not isinstance(issue_id, str) or not issue_id:
            raise ValueError(f"{label} score contains an invalid issue ID")
        if not isinstance(metrics, dict):
            raise ValueError(f"{label} issue metrics must be objects")
        for field in ("true_positive", "false_positive", "false_negative"):
            value = metrics.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} issue metric {field} must be nonnegative integer")
    exact = report.get("exact")
    if not isinstance(exact, dict):
        raise ValueError(f"{label} score lacks aggregate exact metrics")
    for field in ("true_positive", "false_positive", "false_negative"):
        if sum(metrics[field] for metrics in issues.values()) != exact.get(field):
            raise ValueError(
                f"{label} issue metrics do not reconcile with aggregate {field}"
            )
    return issues


def compare_score_reports(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 1729,
) -> dict[str, Any]:
    """Estimate challenger-minus-baseline exact-F1 uncertainty by issue."""
    if bootstrap_samples < 1_000:
        raise ValueError("at least 1,000 bootstrap samples are required")
    for field in MATCHED_FIELDS:
        if baseline.get(field) != challenger.get(field):
            raise ValueError(f"paired score reports disagree on {field}")
    baseline_issues = _validate_issue_metrics(baseline, "baseline")
    challenger_issues = _validate_issue_metrics(challenger, "challenger")
    if set(baseline_issues) != set(challenger_issues):
        raise ValueError("paired score reports must cover exactly the same issues")

    issue_ids = sorted(baseline_issues)
    generator = random.Random(seed)

    def aggregate(report_issues: dict[str, Any], sampled: list[str]) -> float:
        counts = {
            field: sum(report_issues[issue_id][field] for issue_id in sampled)
            for field in ("true_positive", "false_positive", "false_negative")
        }
        return _f1(**counts)

    observed_baseline = aggregate(baseline_issues, issue_ids)
    observed_challenger = aggregate(challenger_issues, issue_ids)
    deltas = []
    for _ in range(bootstrap_samples):
        sampled = [generator.choice(issue_ids) for _ in issue_ids]
        deltas.append(
            aggregate(challenger_issues, sampled)
            - aggregate(baseline_issues, sampled)
        )
    deltas.sort()
    lower_index = max(0, math.floor(0.025 * bootstrap_samples))
    upper_index = min(bootstrap_samples - 1, math.ceil(0.975 * bootstrap_samples) - 1)
    return {
        "schema_version": "1.0",
        "comparison": "challenger_minus_baseline_exact_span_and_type_micro_f1",
        "dataset_id": baseline["dataset_id"],
        "benchmark_dataset_sha256": baseline["benchmark_dataset_sha256"],
        "source_gold_sha256": baseline["source_gold_sha256"],
        "input_sha256": baseline["input_sha256"],
        "input_text": baseline["input_text"],
        "dataset_split": baseline["dataset_split"],
        "confidence_threshold": baseline["confidence_threshold"],
        "issue_clusters": len(issue_ids),
        "baseline": {
            "adapter_id": baseline.get("adapter_id"),
            "model_name": baseline.get("model_name"),
            "model_revision": baseline.get("model_revision"),
            "exact_f1": observed_baseline,
        },
        "challenger": {
            "adapter_id": challenger.get("adapter_id"),
            "model_name": challenger.get("model_name"),
            "model_revision": challenger.get("model_revision"),
            "exact_f1": observed_challenger,
        },
        "exact_f1_delta": observed_challenger - observed_baseline,
        "paired_issue_cluster_bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "lower_95": deltas[lower_index],
            "upper_95": deltas[upper_index],
        },
        "warnings": [
            "A positive interval does not waive absolute quality, evidence-integrity, cost, or historian-review gates.",
            *(
                ["Fewer than six test issue clusters makes this interval highly unstable."]
                if len(issue_ids) < 6
                else []
            ),
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--challenger-score", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline = json.loads(args.baseline_score.read_text(encoding="utf-8"))
    challenger = json.loads(args.challenger_score.read_text(encoding="utf-8"))
    report = compare_score_reports(
        baseline,
        challenger,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "exact_f1_delta": report["exact_f1_delta"],
                "lower_95": report["paired_issue_cluster_bootstrap"]["lower_95"],
                "upper_95": report["paired_issue_cluster_bootstrap"]["upper_95"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
