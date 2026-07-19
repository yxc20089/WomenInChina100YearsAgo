"""Run the reviewed-unit semantic E2E path with the single central model config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from .model_config import load_pipeline_model_configuration
from .semantic_repository import (
    load_resolution_mentions,
    load_reviewed_coherent_text,
    persist_local_resolution,
    persist_semantic_extraction,
    semantic_multimodal_context,
)
from .semantic_tasks import SemanticTaskResult, build_verified_semantic_client


def _write_task(path: Path, result: SemanticTaskResult[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "task": result.task,
        "prompt_sha256": result.prompt_sha256,
        "prompt_schema_sha256": result.prompt_schema_sha256,
        "response_format_sha256": result.response_format_sha256,
        "raw_output_sha256": result.raw_output_sha256,
        "raw_output": result.raw_output,
        "finish_reason": result.finish_reason,
        "token_usage": {
            "prompt": result.prompt_tokens,
            "completion": result.completion_tokens,
            "total": result.total_tokens,
        },
        "response": result.response.model_dump(mode="json"),
    }
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_semantic_e2e(
    database_url: str,
    coherent_unit_revision_id: UUID,
    output_dir: Path,
    *,
    model_config_path: str | None = None,
) -> dict[str, Any]:
    """Run exactly one extraction call, then one bounded resolution call."""
    model_configuration = load_pipeline_model_configuration(model_config_path)
    bundle = load_reviewed_coherent_text(database_url, coherent_unit_revision_id)
    client = build_verified_semantic_client(model_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_segments, page_images = semantic_multimodal_context(bundle)
    extraction = client.extract_evidence(
        coherent_text=bundle.content,
        segments=text_segments,
        page_images=page_images,
    )
    extraction_path = output_dir / "01-semantic-extraction.json"
    _write_task(extraction_path, extraction)
    extraction_persisted = persist_semantic_extraction(
        database_url,
        bundle,
        extraction,
        model_config_path=model_config_path,
        artifact_uri=extraction_path.as_posix(),
    )

    resolution_run = None
    resolution = None
    if extraction_persisted.mention_ids:
        mentions = load_resolution_mentions(
            database_url, bundle, extraction_persisted.mention_ids
        )
        resolution = client.resolve_local_identities(
            coherent_text=bundle.content,
            segments=text_segments,
            page_images=page_images,
            mentions=mentions,
        )
        resolution_path = output_dir / "02-local-resolution.json"
        _write_task(resolution_path, resolution)
        resolution_run = persist_local_resolution(
            database_url,
            bundle,
            resolution,
            mention_ids=extraction_persisted.mention_ids,
            model_config_path=model_config_path,
            artifact_uri=resolution_path.as_posix(),
        )

    receipt = {
        "schema_version": "1.0",
        "coherent_unit_revision_id": str(coherent_unit_revision_id),
        "coherent_input_sha256": bundle.input_sha256,
        "pipeline_model_configuration": model_configuration.source_path.as_posix(),
        "pipeline_model_configuration_sha256": model_configuration.sha256,
        "semantic_model": {
            "model_name": model_configuration.semantic.model_name,
            "model_revision": model_configuration.semantic.model_revision,
            "served_model": model_configuration.semantic.served_model,
            "ollama_manifest_digest": model_configuration.semantic.ollama_manifest_digest,
            "quantization": model_configuration.semantic.quantization,
            "runtime_version": model_configuration.semantic.runtime_version,
            "acceleration": model_configuration.semantic.acceleration,
        },
        "runs": {
            "semantic_extraction": extraction_persisted.run.run_id,
            "local_resolution": resolution_run.run_id if resolution_run else None,
        },
        "counts": {
            "retained_mention_occurrences": extraction_persisted.run.records,
            "local_resolution_memberships": (
                resolution_run.records if resolution_run else 0
            ),
            "unresolved_mention_occurrences": (
                len(resolution.response.unresolved_mention_ids) if resolution else 0
            ),
            "event_candidates": len(extraction_persisted.events),
            "semantic_model_calls": 2 if resolution is not None else 1,
            "page_images_supplied_per_call": len(page_images),
        },
        "publication_state": "candidate_only",
        "next_required_gate": (
            "Historian review of text, coherent-unit membership, mention spans, "
            "identity resolutions, and events before reviewed graph projection."
        ),
    }
    receipt_path = output_dir / "receipt.json"
    temporary = receipt_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--coherent-unit-revision-id", type=UUID, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        help=(
            "Path to the complete model config; omitted means "
            "config/pipeline-models.toml or WIC_PIPELINE_MODEL_CONFIG."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("--database-url or DATABASE_URL is required")
    receipt = run_semantic_e2e(
        args.database_url,
        args.coherent_unit_revision_id,
        args.output_dir,
        model_config_path=args.model_config,
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
