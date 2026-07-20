"""Export one citation-preserving corpus for reproducible RAG comparisons.

The export intentionally contains no generated entities, relations, or summaries.
Every compared system receives the same selected page or approved coherent-unit
text. A sidecar maps exact character spans back to immutable OCR regions so
downstream answers can be grounded again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .reviewed_text_materializer import (
    ReviewedSpanInput,
    materialize_reviewed_article,
)


GRAPHRAG_VERSION = "3.1.1"
GRAPHRAG_REVISION = "14a00ad88fc33cf2b52f4f113f25807556f8e25e"
LIGHTRAG_VERSION = "1.5.4"
LIGHTRAG_REVISION = "9a45b64c2ee25b1d806e90db926a8af37480bb16"


@dataclass(frozen=True, slots=True)
class RAGExportResult:
    output_dir: str
    input_unit: str
    documents: int
    source_regions: int
    exported_regions: int
    omitted_empty_regions: int
    text_characters: int
    manifest_sha256: str


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _page_document(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Cannot export an empty page")
    first = rows[0]
    page_id = str(first["page_id"])
    pieces: list[str] = []
    citations: list[dict[str, Any]] = []
    position = 0
    for row in rows:
        text = (row.get("normalized_text") or row.get("raw_text") or "").strip()
        if not text:
            continue
        if pieces:
            pieces.append("\n")
            position += 1
        start = position
        pieces.append(text)
        position += len(text)
        citations.append(
            {
                "document_id": page_id,
                "region_id": str(row["region_id"]),
                "start_char": start,
                "end_char": position,
                "reading_order": row["reading_order"],
                "region_kind": row["region_kind"],
                "polygon": row["polygon"],
                "ocr_confidence": row["confidence"],
                "raw_text": row["raw_text"],
                "exported_text": text,
                "source_uri": row["source_uri"],
                "source_sha256": row["source_sha256"],
                "ocr_run_id": str(row["run_id"]),
                "derivative_id": str(row["derivative_id"]),
                "source_image_uri": row["source_image_uri"],
                "source_image_sha256": row["source_image_sha256"],
                "evidence_tier": row["evidence_tier"],
                "ocr_selection_basis": row["ocr_selection_basis"],
                "volume_number": row["volume_number"],
                "publication_year": row["publication_year"],
                "page_number": row["page_number"],
            }
        )
    text = "".join(pieces)
    document = {
        "id": page_id,
        "title": (
            f"Shen Bao volume {first['volume_number']}, "
            f"page {first['page_number']} ({first['publication_year']})"
        ),
        "text": text,
        "metadata": {
            "page_id": page_id,
            "volume_number": first["volume_number"],
            "publication_year": first["publication_year"],
            "page_number": first["page_number"],
            "source_uri": first["source_uri"],
            "source_sha256": first["source_sha256"],
            "ocr_run_id": str(first["run_id"]),
            "derivative_id": str(first["derivative_id"]),
            "source_image_uri": first["source_image_uri"],
            "source_image_sha256": first["source_image_sha256"],
            "evidence_tier": first["evidence_tier"],
            "ocr_selection_basis": first["ocr_selection_basis"],
            "ocr_model": first["ocr_model"],
            "ocr_model_revision": first["ocr_model_revision"],
            "region_count": len(citations),
        },
    }
    return document, citations


def _coherent_unit_document(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Cannot export an empty coherent unit")
    first = rows[0]
    expected_span_counts = {row["expected_span_count"] for row in rows}
    if len(expected_span_counts) != 1 or expected_span_counts != {len(rows)}:
        raise ValueError(
            "reviewed article span cardinality changed across provenance joins"
        )
    document_id = str(first["revision_id"])
    sources = tuple(
        ReviewedSpanInput(
            sequence_number=row["span_sequence_number"],
            region_id=row["region_id"],
            page_id=row["page_id"],
            raw_text=row["raw_text"],
            raw_start=row["span_text_start"],
            raw_end=row["span_text_end"],
            selected_text_version_id=row["selected_text_version_id"],
            selected_text_sha256=row["selected_text_sha256"],
            selection_id=row["text_selection_id"],
            selected_text=row["selected_text"],
            role=row["span_role"],
            alignment_operations=(
                tuple(row["alignment_operations"])
                if row["alignment_operations"] is not None
                else None
            ),
        )
        for row in rows
    )
    canonical = materialize_reviewed_article(
        first["revision_id"], first["unit_kind"], sources
    )
    rows_by_sequence = {row["span_sequence_number"]: row for row in rows}
    citations: list[dict[str, Any]] = []
    for span in canonical.spans:
        row = rows_by_sequence[span.sequence_number]
        citations.append(
            {
                "document_id": document_id,
                "coherent_unit_id": str(row["unit_id"]),
                "coherent_unit_revision_id": document_id,
                "region_id": str(span.region_id),
                "start_char": span.composite_start,
                "end_char": span.composite_end,
                "region_text_start": span.selected_start,
                "region_text_end": span.selected_end,
                "raw_region_text_start": span.raw_start,
                "raw_region_text_end": span.raw_end,
                "sequence_number": span.sequence_number,
                "role": span.role,
                "polygon": row["polygon"],
                "ocr_confidence": row["confidence"],
                "raw_text": row["raw_text"],
                "exported_text": span.text,
                "selected_text_version_id": str(span.selected_text_version_id),
                "selected_text_sha256": span.selected_text_sha256,
                "text_selection_id": str(span.selection_id),
                "source_uri": row["source_uri"],
                "source_sha256": row["source_sha256"],
                "ocr_run_id": str(row["run_id"]),
                "derivative_id": str(row["derivative_id"]),
                "source_image_uri": row["source_image_uri"],
                "source_image_sha256": row["source_image_sha256"],
                "evidence_tier": row["evidence_tier"],
                "ocr_selection_basis": row["ocr_selection_basis"],
                "segmentation_selection_id": str(row["segmentation_selection_id"]),
                "segmentation_review_id": str(row["segmentation_review_id"]),
                "approved_by": row["approved_by"],
                "issue_id": str(row["issue_id"]) if row["issue_id"] else None,
                "volume_number": row["volume_number"],
                "publication_year": row["publication_year"],
                "page_number": row["page_number"],
            }
        )
    document = {
        "id": document_id,
        "title": first["title"] or f"Reviewed {first['unit_kind']} {first['unit_id']}",
        "text": canonical.content,
        "metadata": {
            "coherent_unit_id": str(first["unit_id"]),
            "coherent_unit_revision_id": document_id,
            "unit_kind": first["unit_kind"],
            "issue_id": str(first["issue_id"]) if first["issue_id"] else None,
            "approved_by": first["approved_by"],
            "approval_selection_id": str(first["segmentation_selection_id"]),
            "segmentation_review_id": str(first["segmentation_review_id"]),
            "content_sha256": canonical.content_sha256,
            "input_sha256": canonical.input_sha256,
            "segmentation_content_sha256": first["content_sha256"],
            "region_span_count": len(citations),
        },
    }
    return document, citations


def build_documents(rows: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Group ordered region rows into page documents with exact offset maps."""
    output: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    current_page: Any = None
    page_rows: list[dict[str, Any]] = []
    for row in rows:
        if current_page is not None and row["page_id"] != current_page:
            output.append(_page_document(page_rows))
            page_rows = []
        current_page = row["page_id"]
        page_rows.append(row)
    if page_rows:
        output.append(_page_document(page_rows))
    return output


def build_coherent_unit_documents(
    rows: Iterable[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Group approved spans by immutable coherent-unit revision."""
    documents, skipped = build_coherent_unit_documents_isolated(rows)
    if skipped:
        revision_id, reason = skipped[0]
        raise ValueError(f"coherent unit {revision_id} is unmaterializable: {reason}")
    return documents


def build_coherent_unit_documents_isolated(
    rows: Iterable[dict[str, Any]],
) -> tuple[
    list[tuple[dict[str, Any], list[dict[str, Any]]]], list[tuple[str, str]]
]:
    """Group approved spans by revision, isolating per-article failures.

    An article whose reviewed text cannot be materialized (ambiguous
    alignment, hash mismatch, span-cardinality drift) is returned in the
    skipped list with its reason instead of failing the whole export: a
    single damaged article must abstain into review, never block operations
    over the rest of the corpus.
    """
    output: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    skipped: list[tuple[str, str]] = []

    def emit(revision_rows: list[dict[str, Any]]) -> None:
        try:
            output.append(_coherent_unit_document(revision_rows))
        except ValueError as error:  # ReviewedTextMaterializationError included
            skipped.append((str(revision_rows[0]["revision_id"]), str(error)))

    current_revision: Any = None
    revision_rows: list[dict[str, Any]] = []
    for row in rows:
        if current_revision is not None and row["revision_id"] != current_revision:
            emit(revision_rows)
            revision_rows = []
        current_revision = row["revision_id"]
        revision_rows.append(row)
    if revision_rows:
        emit(revision_rows)
    return output, skipped


EXPORT_SQL = """
    SELECT p.page_id, p.page_number, r.run_id,
           derivative.derivative_id,
           derivative.image_uri AS source_image_uri,
           derivative.image_sha256 AS source_image_sha256,
           derivative.evidence_tier,
           selection.selection_basis AS ocr_selection_basis,
           v.volume_number, v.publication_year, s.source_uri,
           s.sha256 AS source_sha256,
           r.region_id, r.reading_order, r.region_kind, r.polygon,
           r.raw_text, r.normalized_text, r.confidence,
           pr.model_name AS ocr_model, pr.model_revision AS ocr_model_revision
    FROM evidence.ocr_region r
    JOIN archive.page p USING (page_id)
    JOIN archive.volume v USING (volume_id)
    JOIN archive.source_object s USING (source_object_id)
    JOIN evidence.processing_run pr USING (run_id)
    JOIN evidence.page_ocr_selection selection
      ON selection.page_id = r.page_id
     AND selection.run_id = r.run_id
     AND selection.superseded_at IS NULL
    JOIN evidence.ocr_run_input input
      ON input.run_id = selection.run_id
     AND input.page_id = selection.page_id
     AND input.derivative_id = selection.derivative_id
    JOIN archive.page_derivative derivative
      ON derivative.derivative_id = input.derivative_id
     AND derivative.page_id = input.page_id
    WHERE (CAST(%(volume_number)s AS integer) IS NULL
           OR v.volume_number = CAST(%(volume_number)s AS integer))
      AND (CAST(%(page_number)s AS integer) IS NULL
           OR p.page_number = CAST(%(page_number)s AS integer))
    ORDER BY v.volume_number, p.page_number, r.reading_order, r.region_id
"""


REVIEWED_UNIT_EXPORT_SQL = """
    WITH eligible_revision AS (
        SELECT DISTINCT revision.revision_id
        FROM evidence.coherent_unit_revision revision
        JOIN evidence.coherent_unit_span span USING (revision_id)
        JOIN evidence.ocr_region region USING (region_id)
        JOIN archive.page page USING (page_id)
        JOIN archive.volume volume USING (volume_id)
        JOIN evidence.page_article_segmentation_selection segmentation_selection
          ON segmentation_selection.selection_id = revision.approval_selection_id
         AND segmentation_selection.superseded_at IS NULL
        WHERE revision.superseded_at IS NULL
          AND revision.unit_kind = 'article'
          AND NOT EXISTS (
              SELECT 1
              FROM evidence.coherent_unit_span required_span
              LEFT JOIN evidence.region_text_selection required_selection
                ON required_selection.region_id = required_span.region_id
               AND required_selection.superseded_at IS NULL
              LEFT JOIN evidence.text_version required_version
                ON required_version.text_version_id = required_selection.text_version_id
               AND required_version.review_status = 'reviewed'
              WHERE required_span.revision_id = revision.revision_id
                AND required_version.text_version_id IS NULL
          )
          AND (CAST(%(volume_number)s AS integer) IS NULL
               OR volume.volume_number = CAST(%(volume_number)s AS integer))
          AND (CAST(%(page_number)s AS integer) IS NULL
               OR page.page_number = CAST(%(page_number)s AS integer))
    )
    SELECT revision.revision_id, revision.unit_id, revision.issue_id,
           revision.unit_kind, revision.title, revision.content_sha256,
           revision.approved_by,
           (
               SELECT count(*)
               FROM evidence.coherent_unit_span expected_span
               WHERE expected_span.revision_id = revision.revision_id
           ) AS expected_span_count,
           segmentation_selection.selection_id AS segmentation_selection_id,
           segmentation_selection.review_id AS segmentation_review_id,
           span.sequence_number AS span_sequence_number,
           span.text_start AS span_text_start, span.text_end AS span_text_end,
           span.role AS span_role,
           page.page_id, page.page_number, region.run_id,
           derivative.derivative_id,
           derivative.image_uri AS source_image_uri,
           derivative.image_sha256 AS source_image_sha256,
           derivative.evidence_tier,
           ocr_selection.selection_basis AS ocr_selection_basis,
           volume.volume_number, volume.publication_year, source.source_uri,
           source.sha256 AS source_sha256,
           region.region_id, region.reading_order, region.region_kind,
           COALESCE(span.polygon, region.polygon) AS polygon,
           region.raw_text, region.normalized_text, region.confidence,
           text_selection.selection_id AS text_selection_id,
           selected_version.text_version_id AS selected_text_version_id,
           selected_version.text_content AS selected_text,
           selected_version.text_sha256 AS selected_text_sha256,
           alignment.operations AS alignment_operations,
           run.model_name AS ocr_model, run.model_revision AS ocr_model_revision
    FROM eligible_revision eligible
    JOIN evidence.coherent_unit_revision revision USING (revision_id)
    JOIN evidence.coherent_unit_span span USING (revision_id)
    JOIN evidence.ocr_region region USING (region_id)
    JOIN archive.page page USING (page_id)
    JOIN archive.volume volume USING (volume_id)
    JOIN archive.source_object source USING (source_object_id)
    JOIN evidence.processing_run run ON run.run_id = region.run_id
    JOIN evidence.region_text_selection text_selection
      ON text_selection.region_id = region.region_id
     AND text_selection.superseded_at IS NULL
    JOIN evidence.text_version selected_version
      ON selected_version.text_version_id = text_selection.text_version_id
     AND selected_version.review_status = 'reviewed'
    JOIN evidence.text_version raw_version
      ON raw_version.region_id = region.region_id
     AND raw_version.variant = 'raw_ocr'
     AND raw_version.text_content = region.raw_text
    LEFT JOIN evidence.text_version_alignment alignment
      ON alignment.source_text_version_id = raw_version.text_version_id
     AND alignment.target_text_version_id = selected_version.text_version_id
    JOIN evidence.page_ocr_selection ocr_selection
      ON ocr_selection.page_id = region.page_id
     AND ocr_selection.run_id = region.run_id
     AND ocr_selection.superseded_at IS NULL
    JOIN evidence.ocr_run_input input
      ON input.run_id = ocr_selection.run_id
     AND input.page_id = ocr_selection.page_id
     AND input.derivative_id = ocr_selection.derivative_id
    JOIN archive.page_derivative derivative
      ON derivative.derivative_id = input.derivative_id
     AND derivative.page_id = input.page_id
    JOIN evidence.page_article_segmentation_selection segmentation_selection
      ON segmentation_selection.selection_id = revision.approval_selection_id
     AND segmentation_selection.superseded_at IS NULL
    WHERE revision.superseded_at IS NULL
    ORDER BY revision.revision_id, span.sequence_number
"""


def export_rag_corpus(
    database_url: str,
    output_dir: Path,
    *,
    volume_number: int | None = None,
    page_number: int | None = None,
    input_unit: str = "ocr_page",
    overwrite: bool = False,
) -> RAGExportResult:
    """Export authoritative source text with citations, never generated claims."""
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc

    managed_names = {
        "documents.jsonl",
        "citations.jsonl",
        "experiment-manifest.json",
    }
    documents_dir = output_dir / "documents"
    if output_dir.exists() and not overwrite:
        occupied = any((output_dir / name).exists() for name in managed_names) or (
            documents_dir.exists() and any(documents_dir.iterdir())
        )
        if occupied:
            raise FileExistsError(f"Export already exists at {output_dir}; pass overwrite=True")
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old_document in documents_dir.glob("*.txt"):
            old_document.unlink()

    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        if input_unit not in {"ocr_page", "reviewed_coherent_unit"}:
            raise ValueError("input_unit must be ocr_page or reviewed_coherent_unit")
        rows = connection.execute(
            REVIEWED_UNIT_EXPORT_SQL if input_unit == "reviewed_coherent_unit" else EXPORT_SQL,
            {"volume_number": volume_number, "page_number": page_number},
        ).fetchall()
    documents = (
        build_coherent_unit_documents(rows)
        if input_unit == "reviewed_coherent_unit"
        else build_documents(rows)
    )
    if not documents:
        raise ValueError(f"No {input_unit} records match the requested export scope")

    documents_path = output_dir / "documents.jsonl"
    citations_path = output_dir / "citations.jsonl"
    region_count = 0
    text_characters = 0
    with documents_path.open("w", encoding="utf-8", newline="\n") as document_file, (
        citations_path.open("w", encoding="utf-8", newline="\n")
    ) as citation_file:
        for document, citations in documents:
            document_file.write(_json_line(document))
            (documents_dir / f"{document['id']}.txt").write_text(
                document["text"], encoding="utf-8", newline="\n"
            )
            text_characters += len(document["text"])
            for citation in citations:
                citation_file.write(_json_line(citation))
            region_count += len(citations)

    omitted_empty_regions = len(rows) - region_count
    manifest = {
        "schema_version": "1.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"volume_number": volume_number, "page_number": page_number},
        "input_unit": input_unit,
        "ocr_run_policy": "active_page_selection_only",
        "segmentation_policy": (
            "active_historian_approved_coherent_unit_revisions_only"
            if input_unit == "reviewed_coherent_unit"
            else "none_page_smoke_test"
        ),
        "citation_contract": "citations.jsonl maps document character offsets to OCR regions",
        "counts": {
            "documents": len(documents),
            "source_regions": len(rows),
            "exported_regions": region_count,
            "omitted_empty_regions": omitted_empty_regions,
            "text_characters": text_characters,
        },
        "files": {
            "documents_jsonl": {
                "path": "documents.jsonl",
                "sha256": _sha256(documents_path),
            },
            "citations_jsonl": {
                "path": "citations.jsonl",
                "sha256": _sha256(citations_path),
            },
            "plain_text_directory": "documents/",
        },
        "systems": {
            "hybrid_baseline": {"implementation": "wic_history.search", "status": "implemented"},
            "lightrag": {
                "package": f"lightrag-hku=={LIGHTRAG_VERSION}",
                "git_revision": LIGHTRAG_REVISION,
                "status": "bounded_experiment",
            },
            "microsoft_graphrag": {
                "package": f"graphrag=={GRAPHRAG_VERSION}",
                "git_revision": GRAPHRAG_REVISION,
                "query_modes": ["global", "drift"],
                "status": "bounded_experiment",
            },
            "lazygraphrag": {
                "package": None,
                "status": "tracked_not_installable",
                "reason": "No reproducible LazyGraphRAG mode is exposed by the OSS GraphRAG CLI",
            },
        },
        "warnings": (
            [
                "Page units are temporary until reviewed article segmentation exists.",
                "OCR text is machine-generated and must not be treated as a reviewed historical claim.",
            ]
            if input_unit == "ocr_page"
            else [
                "Article text comes only from active historian-reviewed text selections.",
            ]
        ) + [
            "RAG-generated entities, relations, communities, and summaries are disposable projections.",
            "OCR regions with empty normalized and raw text are omitted and counted in the manifest.",
        ],
    }
    manifest_path = output_dir / "experiment-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return RAGExportResult(
        output_dir=str(output_dir),
        input_unit=input_unit,
        documents=len(documents),
        source_regions=len(rows),
        exported_regions=region_count,
        omitted_empty_regions=omitted_empty_regions,
        text_characters=text_characters,
        manifest_sha256=_sha256(manifest_path),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--volume", type=int)
    parser.add_argument("--page", type=int)
    parser.add_argument(
        "--unit", choices=("ocr_page", "reviewed_coherent_unit"), default="ocr_page"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("--database-url or DATABASE_URL is required")
    result = export_rag_corpus(
        args.database_url,
        args.output,
        volume_number=args.volume,
        page_number=args.page,
        input_unit=args.unit,
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
