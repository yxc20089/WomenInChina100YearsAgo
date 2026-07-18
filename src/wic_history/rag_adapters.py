"""Validate shared RAG exports and bridge them to pinned experiment engines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .rag_experiment import (
    GRAPHRAG_REVISION,
    GRAPHRAG_VERSION,
)


@dataclass(frozen=True, slots=True)
class ExportValidationResult:
    export_dir: str
    documents: int
    citations: int
    source_regions: int
    omitted_empty_regions: int


@dataclass(frozen=True, slots=True)
class GraphRAGWorkspaceResult:
    workspace: str
    documents: int
    source_manifest_sha256: str
    next_commands: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def validate_export(export_dir: Path) -> ExportValidationResult:
    manifest_path = export_dir / "experiment-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for file_key in ("documents_jsonl", "citations_jsonl"):
        file_record = manifest["files"][file_key]
        path = export_dir / file_record["path"]
        if _sha256(path) != file_record["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}")
    document_rows = _read_jsonl(export_dir / manifest["files"]["documents_jsonl"]["path"])
    documents = {row["id"]: row for row in document_rows}
    if len(documents) != len(document_rows):
        raise ValueError("RAG document IDs must be unique")
    for document_id, document in documents.items():
        plain_path = export_dir / "documents" / f"{document_id}.txt"
        if plain_path.read_text(encoding="utf-8") != document["text"]:
            raise ValueError(f"Plain text differs from JSONL document {document_id}")
    citations = _read_jsonl(export_dir / manifest["files"]["citations_jsonl"]["path"])
    for citation in citations:
        document = documents.get(citation["document_id"])
        if document is None:
            raise ValueError(f"Citation references unknown document {citation['document_id']}")
        observed = document["text"][citation["start_char"] : citation["end_char"]]
        if observed != citation["exported_text"]:
            raise ValueError(f"Citation offset mismatch for region {citation['region_id']}")
    counts = manifest["counts"]
    expected = counts["exported_regions"]
    if len(citations) != expected or len(documents) != counts["documents"]:
        raise ValueError("Manifest counts do not match exported records")
    if counts["source_regions"] != expected + counts["omitted_empty_regions"]:
        raise ValueError("Manifest region accounting is incomplete")
    return ExportValidationResult(
        str(export_dir),
        len(documents),
        len(citations),
        counts["source_regions"],
        counts["omitted_empty_regions"],
    )


def prepare_graphrag_workspace(
    export_dir: Path, workspace: Path, *, overwrite: bool = False
) -> GraphRAGWorkspaceResult:
    validation = validate_export(export_dir)
    manifest_path = export_dir / "experiment-manifest.json"
    input_dir = workspace / "input"
    metadata_path = workspace / "wic-experiment.json"
    managed_exists = metadata_path.exists() or (
        input_dir.exists() and any(input_dir.glob("*.txt"))
    )
    if managed_exists and not overwrite:
        raise FileExistsError(f"GraphRAG workspace already exists at {workspace}")
    input_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old_file in input_dir.glob("*.txt"):
            old_file.unlink()
    for source in sorted((export_dir / "documents").glob("*.txt")):
        shutil.copy2(source, input_dir / source.name)
    source_manifest_sha256 = _sha256(manifest_path)
    metadata = {
        "schema_version": "1.0",
        "source_export": str(export_dir),
        "source_manifest_sha256": source_manifest_sha256,
        "graphrag": {
            "package": f"graphrag=={GRAPHRAG_VERSION}",
            "git_revision": GRAPHRAG_REVISION,
        },
        "warnings": [
            "Generated entities, claims, relationships and summaries are experimental projections.",
            "Run prompt tuning for Traditional Chinese historical newspapers before scoring quality.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    root = str(workspace)
    shell_root = shlex.quote(root)
    package = f"graphrag=={GRAPHRAG_VERSION}"
    commands = [
        f"uvx --from {package} graphrag init --root {shell_root}",
        (
            f"uvx --from {package} graphrag prompt-tune --root {shell_root} "
            "--language 'Traditional Chinese' --domain 'historical Chinese newspaper'"
        ),
        f"uvx --from {package} graphrag index --root {shell_root} --method standard",
        f"uvx --from {package} graphrag query --root {shell_root} --method global '<question>'",
        f"uvx --from {package} graphrag query --root {shell_root} --method drift '<question>'",
    ]
    return GraphRAGWorkspaceResult(
        root, validation.documents, source_manifest_sha256, commands
    )


class LightRAGClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                body = response.read(16 * 1024 * 1024)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"LightRAG request failed for {path}: {exc}") from exc
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LightRAG returned invalid JSON for {path}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"LightRAG returned a non-object response for {path}")
        return parsed

    def insert_documents(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        return self._post(
            "/documents/texts",
            {
                "texts": [document["text"] for document in documents],
                "file_sources": [f"wic/{document['id']}.txt" for document in documents],
            },
        )

    def query_data(self, query: str, mode: str = "mix") -> dict[str, Any]:
        if mode not in {"local", "global", "hybrid", "naive", "mix"}:
            raise ValueError(f"Unsupported LightRAG query mode: {mode}")
        return self._post(
            "/query/data",
            {"query": query, "mode": mode, "include_references": True},
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--export", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-graphrag")
    prepare.add_argument("--export", type=Path, required=True)
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")
    load = subparsers.add_parser("load-lightrag")
    load.add_argument("--export", type=Path, required=True)
    load.add_argument(
        "--base-url", default=os.environ.get("LIGHTRAG_URL", "http://127.0.0.1:9621")
    )
    load.add_argument("--api-key", default=os.environ.get("LIGHTRAG_API_KEY"))
    query = subparsers.add_parser("query-lightrag")
    query.add_argument("query")
    query.add_argument(
        "--mode", choices=("local", "global", "hybrid", "naive", "mix"), default="mix"
    )
    query.add_argument(
        "--base-url", default=os.environ.get("LIGHTRAG_URL", "http://127.0.0.1:9621")
    )
    query.add_argument("--api-key", default=os.environ.get("LIGHTRAG_API_KEY"))
    query.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        print(json.dumps(asdict(validate_export(args.export)), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "prepare-graphrag":
        result = prepare_graphrag_workspace(
            args.export, args.workspace, overwrite=args.overwrite
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    client = LightRAGClient(args.base_url, api_key=args.api_key)
    if args.command == "load-lightrag":
        validate_export(args.export)
        documents = _read_jsonl(args.export / "documents.jsonl")
        response = client.insert_documents(documents)
    else:
        response = client.query_data(args.query, args.mode)
    rendered = json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
