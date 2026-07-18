"""Export one citation-preserving corpus for reproducible RAG comparisons.

The export intentionally contains no generated entities, relations, or summaries.
Every system receives the same page text. A sidecar maps exact character spans
back to immutable OCR regions so downstream answers can be grounded again.
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


GRAPHRAG_VERSION = "3.1.1"
GRAPHRAG_REVISION = "14a00ad88fc33cf2b52f4f113f25807556f8e25e"
LIGHTRAG_VERSION = "1.5.4"
LIGHTRAG_REVISION = "9a45b64c2ee25b1d806e90db926a8af37480bb16"


@dataclass(frozen=True, slots=True)
class RAGExportResult:
    output_dir: str
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


def export_rag_corpus(
    database_url: str,
    output_dir: Path,
    *,
    volume_number: int | None = None,
    page_number: int | None = None,
    overwrite: bool = False,
) -> RAGExportResult:
    """Export authoritative OCR text and region citations, never generated claims."""
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
        rows = connection.execute(
            EXPORT_SQL,
            {"volume_number": volume_number, "page_number": page_number},
        ).fetchall()
    pages = build_documents(rows)
    if not pages:
        raise ValueError("No OCR regions match the requested export scope")

    documents_path = output_dir / "documents.jsonl"
    citations_path = output_dir / "citations.jsonl"
    region_count = 0
    text_characters = 0
    with documents_path.open("w", encoding="utf-8", newline="\n") as document_file, (
        citations_path.open("w", encoding="utf-8", newline="\n")
    ) as citation_file:
        for document, citations in pages:
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
        "input_unit": "ocr_page",
        "ocr_run_policy": "active_page_selection_only",
        "citation_contract": "citations.jsonl maps page-text character offsets to OCR regions",
        "counts": {
            "documents": len(pages),
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
        "warnings": [
            "Page units are temporary until reviewed article segmentation exists.",
            "OCR text is machine-generated and must not be treated as a reviewed historical claim.",
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
        documents=len(pages),
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
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
