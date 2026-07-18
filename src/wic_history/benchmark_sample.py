"""Create a deterministic benchmark-page screening plan from a corpus manifest."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_STRATA: tuple[tuple[str, int, str | None], ...] = (
    ("early_1872", 1872, ".pdf"),
    ("turn_of_century_1900", 1900, ".pdf"),
    ("djvu_1908", 1908, ".djvu"),
    ("pre_pilot_1920", 1920, ".pdf"),
    ("pilot_1924", 1924, ".pdf"),
    ("pilot_1925", 1925, ".pdf"),
    ("pilot_1926", 1926, ".pdf"),
    ("mid_1930s", 1935, ".pdf"),
    ("wartime_1940", 1940, ".pdf"),
    ("late_1949", 1949, ".pdf"),
)


@dataclass(frozen=True, slots=True)
class SelectedVolume:
    stratum: str
    target_year: int
    record: dict[str, Any]


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_volume(
    records: Sequence[dict[str, Any]],
    stratum: str,
    year: int,
    extension: str | None,
) -> SelectedVolume:
    eligible = [
        record
        for record in records
        if record.get("publication_year") == year
        and record.get("page_count")
        and record.get("integrity_status") == "ok_fast_checks"
        and (extension is None or record.get("extension") == extension)
    ]
    if not eligible:
        raise ValueError(f"No eligible volume for stratum {stratum!r}")
    median_pages = statistics.median(record["page_count"] for record in eligible)
    chosen = min(
        eligible,
        key=lambda record: (
            abs(record["page_count"] - median_pages),
            record.get("volume_number") or 0,
        ),
    )
    return SelectedVolume(stratum=stratum, target_year=year, record=chosen)


def evenly_spaced_pages(page_count: int, count: int) -> list[int]:
    """Return stable 1-based pages spanning the complete volume."""
    if page_count <= 0 or count <= 0:
        return []
    if count >= page_count:
        return list(range(1, page_count + 1))
    if count == 1:
        return [(page_count + 1) // 2]
    pages = {
        round(index * (page_count - 1) / (count - 1)) + 1
        for index in range(count)
    }
    # Rounding should normally preserve the requested count. Fill any rare gaps
    # deterministically from the start rather than introducing randomness.
    if len(pages) < count:
        pages.update(page for page in range(1, page_count + 1) if page not in pages)
    return sorted(pages)[:count]


def create_plan(
    records: Sequence[dict[str, Any]],
    strata: Iterable[tuple[str, int, str | None]] = DEFAULT_STRATA,
    pages_per_volume: int = 50,
) -> tuple[list[dict[str, Any]], list[SelectedVolume]]:
    selections = [select_volume(records, *stratum) for stratum in strata]
    rows: list[dict[str, Any]] = []
    for selection in selections:
        record = selection.record
        for page_number in evenly_spaced_pages(record["page_count"], pages_per_volume):
            rows.append(
                {
                    "sample_id": f"v{record['volume_number']:03d}-p{page_number:04d}",
                    "stratum": selection.stratum,
                    "target_year": selection.target_year,
                    "publication_year": record["publication_year"],
                    "volume_number": record["volume_number"],
                    "page_number": page_number,
                    "volume_page_count": record["page_count"],
                    "extension": record["extension"],
                    "source_uri": record["source_uri"],
                    "source_key": record["key"],
                    "selection_stage": "visual_screening_candidate",
                    "gold_status": "not_reviewed",
                    "page_genre": "unassigned_until_visual_review",
                    "quality_stratum": "unassigned_until_visual_review",
                }
            )
    return rows, selections


def write_plan(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    selections: Sequence[SelectedVolume],
    manifest_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "candidate_pages.csv"
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "plan_version": "1.0",
        "source_manifest": str(manifest_path),
        "candidate_page_count": len(rows),
        "selected_volume_count": len(selections),
        "purpose": "Candidate pool for visual screening before 150-250 page gold-set selection",
        "selected_volumes": [
            {
                "stratum": selection.stratum,
                "target_year": selection.target_year,
                "volume_number": selection.record["volume_number"],
                "publication_year": selection.record["publication_year"],
                "page_count": selection.record["page_count"],
                "extension": selection.record["extension"],
                "source_uri": selection.record["source_uri"],
            }
            for selection in selections
        ],
        "required_review": [
            "Render candidates and assign page genre and scan-quality strata.",
            "Select 150-250 gold pages with deliberate coverage of layout and degradation.",
            "Do not treat this mechanically sampled pool as the final gold set.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/corpus-audit/manifest.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/benchmark-sample")
    )
    parser.add_argument("--pages-per-volume", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = read_manifest(args.manifest)
    rows, selections = create_plan(records, pages_per_volume=args.pages_per_volume)
    write_plan(args.output_dir, rows, selections, args.manifest)
    print(f"Wrote {len(rows)} candidate pages from {len(selections)} volumes to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

