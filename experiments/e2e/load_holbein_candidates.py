#!/usr/bin/env python3
"""Translate an accepted Holbein model-step artifact into candidate DB rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from wic_history.evidence import (
    EntityMentionCandidate,
    EntityType,
    NERArtifact,
    Point,
    Polygon,
    ProcessingRun,
    RunKind,
    SourcePointer,
)
from wic_history.model_config import load_pipeline_model_configuration
from wic_history.repository import ingest_ner_artifact

from holbein_v219_p0308 import (
    MODEL_SCHEMA,
    SOURCE,
    SYSTEM_PROMPT,
    exact_candidates,
    validate_response,
)


INGESTED_LABELS = {
    "ROLE_TITLE": (EntityType.ROLE_TITLE, "title_reference"),
    "PERSON_REFERENCE": (EntityType.PERSON, "shortened_surname"),
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_ner_artifact(
    model_artifact: dict[str, Any],
    *,
    model_config_path: str | Path | None = None,
) -> NERArtifact:
    pipeline_configuration = load_pipeline_model_configuration(model_config_path)
    semantic = pipeline_configuration.semantic
    if model_artifact.get("source") != SOURCE:
        raise ValueError("model artifact source identity differs from the pinned example")
    candidates = model_artifact.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("model artifact has no deterministic candidate list")
    if candidates != exact_candidates(SOURCE["region_text"]):
        raise ValueError("model artifact candidates differ from deterministic candidates")
    model_result = model_artifact.get("model_result")
    if not isinstance(model_result, dict):
        raise ValueError("model artifact has no model result")
    response = model_result.get("response")
    validation_errors = validate_response(response, candidates)
    if validation_errors or not model_artifact.get("accepted_for_ingestion"):
        raise ValueError(
            "model artifact is not eligible for ingestion: " + "; ".join(validation_errors)
        )
    model = model_artifact.get("model")
    if model != semantic.served_model:
        raise ValueError(f"model {model!r} differs from the selected semantic model")
    if model_artifact.get("pipeline_model_configuration_sha256") != pipeline_configuration.sha256:
        raise ValueError("model artifact was not produced by the selected model configuration")
    model_identity = model_artifact.get("model_identity")
    expected_identity = {
        "model_name": semantic.model_name,
        "model_revision": semantic.model_revision,
        "ollama_manifest_digest": semantic.ollama_manifest_digest,
        "model_blob_sha256": semantic.model_blob_sha256,
        "quantization": semantic.quantization,
        "runtime_name": semantic.runtime_name,
        "runtime_version": semantic.runtime_version,
        "acceleration": semantic.acceleration,
    }
    if model_identity != expected_identity:
        raise ValueError("model artifact identity differs from the selected configuration")
    revision = semantic.model_blob_sha256

    page_namespace = UUID(SOURCE["page_id"])
    invocation_identity = canonical_sha256(
        {
            "created_at": model_artifact.get("created_at"),
            "model_raw_content_sha256": model_result.get("raw_content_sha256"),
        }
    )
    run_id = uuid5(
        page_namespace,
        f"e2e-ner:{model}:{revision}:{invocation_identity}",
    )
    artifact_id = uuid5(
        page_namespace,
        f"e2e-ner-artifact:{model}:{revision}:{invocation_identity}",
    )
    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    decision_by_id = {
        item["candidate_id"]: item["label"] for item in response["decisions"]
    }
    input_identity = {
        "source_ocr_run_id": SOURCE["ocr_run_id"],
        "region_id": SOURCE["region_id"],
        "region_text": SOURCE["region_text"],
        "candidates": candidates,
    }
    prompt_schema_revision = canonical_sha256(
        {
            "protocol_version": "holbein-e2e-model-step-v1",
            "system_prompt": SYSTEM_PROMPT,
            "response_schema": MODEL_SCHEMA,
        }
    )
    completed_at = datetime.fromisoformat(model_artifact["created_at"])
    started_at = completed_at - timedelta(
        seconds=float(model_result.get("elapsed_seconds") or 0)
    )
    polygon = Polygon(
        points=[Point.model_validate(point) for point in SOURCE["region_polygon"]["points"]]
    )

    mentions = []
    for candidate_id in ("C1", "C2"):
        label = decision_by_id[candidate_id]
        entity_type, form = INGESTED_LABELS[label]
        candidate = candidate_by_id[candidate_id]
        mention_id = uuid5(run_id, candidate_id)
        attributes: dict[str, Any] = {
            "candidate_id": candidate_id,
            "classification_label": label,
            "mention_form": form,
            "left_context": candidate["left_context"],
            "right_context": candidate["right_context"],
            "offset_authority": "deterministic_python_codepoint_index",
            "model_raw_content_sha256": model_result["raw_content_sha256"],
            "example_scope": "non_gold_lossless_pilot",
        }
        if candidate_id == "C2":
            attributes.update(
                {
                    "resolution_scope_required": "reviewed_coherent_unit_or_article",
                    "do_not_register_as_global_alias": True,
                }
            )
        mentions.append(
            EntityMentionCandidate(
                mention_id=mention_id,
                entity_type=entity_type,
                text=candidate["surface"],
                source=SourcePointer(
                    source_uri=SOURCE["source_uri"],
                    source_sha256=SOURCE["source_object_sha256"],
                    derivative_id=UUID("d5ce3a23-0cce-4308-bdce-98f9db44fa12"),
                    image_sha256=SOURCE["image_sha256"],
                    evidence_tier=SOURCE["evidence_tier"],
                    volume_number=SOURCE["volume_number"],
                    publication_year=SOURCE["publication_year"],
                    page_number=SOURCE["page_number"],
                    region_id=UUID(SOURCE["region_id"]),
                    polygon=polygon,
                    text_start=candidate["start"],
                    text_end=candidate["end"],
                ),
                confidence=None,
                run_id=run_id,
                attributes=attributes,
            )
        )

    return NERArtifact(
        schema_version="1.1",
        artifact_id=artifact_id,
        source_ocr_run_id=UUID(SOURCE["ocr_run_id"]),
        input_variant="raw_ocr",
        input_sha256=canonical_sha256(input_identity),
        dataset_id="holbein-v219-p0308-e2e",
        split_id="design-example",
        ontology_version="entity-schema-v0.1",
        adapter_id="bounded-candidate-qwen3.5-4b",
        prompt_schema_revision=prompt_schema_revision,
        run=ProcessingRun(
            run_id=run_id,
            kind=RunKind.NER,
            engine="Ollama",
            model_name=model,
            model_revision=revision,
            software_version=f"{semantic.runtime_name}-{semantic.runtime_version}",
            configuration={
                "temperature": semantic.temperature,
                "seed": semantic.seed,
                "thinking": semantic.thinking,
                "structured_output": "native_ollama_json_schema",
                "acceleration": semantic.acceleration,
                "pipeline_model_configuration_sha256": pipeline_configuration.sha256,
                "offsets_supplied_by_model": False,
                "model_raw_content_sha256": model_result["raw_content_sha256"],
                "validation_policy": "whole_response_rejection",
                "scope": "executable_design_example_not_benchmark_or_gold",
            },
            started_at=started_at,
            completed_at=completed_at,
        ),
        mentions=mentions,
        warnings=[
            "This is a non-gold lossless pilot and all stored mentions remain candidates.",
            "霍 requires reviewed article/coherent-unit context before entity resolution.",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_artifact", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/e2e/holbein-v219-p0308.ner.json"),
    )
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--model-config",
        help="Complete model configuration; individual model overrides are not accepted",
    )
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("--database-url or DATABASE_URL is required")

    model_artifact = json.loads(args.model_artifact.read_text(encoding="utf-8"))
    artifact = build_ner_artifact(model_artifact, model_config_path=args.model_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        artifact.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    result = ingest_ner_artifact(args.database_url, args.output)
    print(
        json.dumps(
            {
                "artifact_id": result.artifact_id,
                "run_id": result.run_id,
                "mentions_verified": result.mentions_verified,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
