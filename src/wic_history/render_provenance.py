"""Validate one immutable page render against its lossless manifest record."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .render_samples import sha256_file


def resolve_render_provenance(
    image_path: Path,
    manifest_path: Path,
    *,
    source_uri: str,
    page_number: int,
    volume_number: int | None,
    publication_year: int | None,
    supplied_source_sha256: str | None = None,
    artifact_root: Path | None = None,
) -> tuple[str, str]:
    """Return the verified source hash and evidence tier for one page image."""
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    def resolve_record_path(row: dict[str, Any]) -> Path:
        value = Path(row.get("render_path", ""))
        if not value.is_absolute() and artifact_root is not None:
            value = artifact_root / value
        return value.resolve()

    matches = [row for row in rows if resolve_record_path(row) == image_path.resolve()]
    if len(matches) != 1:
        raise ValueError("render manifest must contain exactly one record for the OCR image")
    record = matches[0]
    if record.get("status") != "rendered":
        raise ValueError("render manifest record is not successfully rendered")
    expected = {
        "source_uri": source_uri,
        "page_number": page_number,
        "volume_number": volume_number,
        "publication_year": publication_year,
    }
    for key, value in expected.items():
        if value is not None and record.get(key) != value:
            raise ValueError(f"render manifest {key} disagrees with the OCR request")
    if sha256_file(image_path) != record.get("render_sha256"):
        raise ValueError("OCR image bytes disagree with the render manifest hash")
    source_sha256 = record.get("source_object_sha256")
    if not isinstance(source_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_sha256
    ):
        raise ValueError("render manifest lacks a valid full source-object SHA-256")
    if supplied_source_sha256 and supplied_source_sha256 != source_sha256:
        raise ValueError("supplied source SHA-256 disagrees with the render manifest")
    selection_status = (record.get("selection") or {}).get("gold_status")
    evidence_tiers = {
        "include": "historian_selected_gold",
        "non_gold_pilot": "non_gold_lossless_pilot",
        "unreviewed_ingestion": "unreviewed_input",
    }
    if selection_status not in evidence_tiers:
        raise ValueError(
            "render manifest does not contain an eligible gold, pilot, or ingestion selection"
        )
    return source_sha256, evidence_tiers[selection_status]
