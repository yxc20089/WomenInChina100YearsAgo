from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from PIL import Image

from wic_history.evidence import EntityLinkArtifact, NERArtifact, OCRPageArtifact
from wic_history.ingestion_jobs import (
    DEFAULT_CONFIGURATION,
    GLINER_NER_CONFIGURATION,
    JobLease,
    canonical_sha256,
)
from wic_history.ingestion_worker import (
    PageJobContext,
    _link_artifact_matches,
    _ner_artifact_matches,
    _ocr_artifact_matches,
    _parent_ner,
    _render_manifest_execution,
    _validate_link_coverage,
    WorkerRunResult,
    build_parser,
    execute_entity_link,
    execute_ner,
    resolve_workspace_path,
    run_loop,
    run_one,
    stage_output_dir,
)
from wic_history.link_pipeline import create_link_artifact
from wic_history.ner_pipeline import ner_input_sha256
from wic_history.render_samples import sha256_file


def context(stage: str, configuration: dict | None = None) -> PageJobContext:
    return PageJobContext(
        job_id="00000000-0000-0000-0000-000000000001",
        stage=stage,
        configuration=configuration or DEFAULT_CONFIGURATION[stage],
        source_uri="s3://bucket/prefix/volume.pdf",
        bucket="bucket",
        object_key="prefix/volume.pdf",
        media_type="application/pdf",
        size_bytes=100,
        etag="etag",
        source_sha256="a" * 64,
        integrity_status="ok_fast_checks",
        volume_number=219,
        publication_year=1925,
        page_number=308,
        page_count=599,
        parent_stage=None,
        parent_artifact_uri=None,
        parent_output_sha256=None,
        parent_result=None,
    )


class IngestionWorkerTests(unittest.TestCase):
    def test_bounded_loop_summarizes_work_and_idle_backoff(self):
        results = iter(
            [
                WorkerRunResult("1", "ocr", "completed", True, "old.json"),
                WorkerRunResult("2", "ner", "pending"),
                WorkerRunResult(None, None, "idle"),
                WorkerRunResult(None, None, "idle"),
            ]
        )
        sleeps: list[float] = []

        def runner(*_args, **_kwargs):
            return next(results)

        summary = run_loop(
            "postgresql://example",
            worker_id="worker",
            max_jobs=10,
            idle_polls=2,
            poll_seconds=0.25,
            runner=runner,
            sleep=sleeps.append,
        )
        self.assertEqual(summary.attempts, 2)
        self.assertEqual(summary.by_status, {"completed": 1, "idle": 2, "pending": 1})
        self.assertEqual(summary.adopted_jobs, 1)
        self.assertEqual(summary.stop_reason, "idle_polls")
        self.assertEqual(sleeps, [0.25])

    def test_loop_limits_are_positive_and_cli_is_opt_in(self):
        with self.assertRaisesRegex(ValueError, "max_jobs"):
            run_loop(
                "postgresql://example",
                worker_id="worker",
                max_jobs=0,
                idle_polls=1,
                poll_seconds=0,
            )
        args = build_parser().parse_args(
            ["--worker", "worker", "--loop", "--max-jobs", "25"]
        )
        self.assertTrue(args.loop)
        self.assertEqual(args.max_jobs, 25)

    def test_worker_observes_operator_cancellation_during_execution(self):
        lease = JobLease(
            job_id="00000000-0000-0000-0000-000000000001",
            batch_id="00000000-0000-0000-0000-000000000002",
            stage="render_lossless",
            scope_kind="page",
            volume_number=219,
            page_number=308,
            input_fingerprint="a" * 64,
            configuration=DEFAULT_CONFIGURATION["render_lossless"],
            attempt_count=1,
            max_attempts=3,
            lease_owner="worker",
            lease_expires_at="2026-07-18T00:00:00+00:00",
        )

        def cancelled_executor(*_args, **_kwargs):
            raise RuntimeError("lease revoked")

        with (
            patch("wic_history.ingestion_worker.claim_job", return_value=lease),
            patch("wic_history.ingestion_worker.start_job"),
            patch(
                "wic_history.ingestion_worker.load_job_context",
                return_value=context("render_lossless"),
            ),
            patch(
                "wic_history.ingestion_worker.fail_job",
                side_effect=ValueError("lease absent"),
            ),
            patch(
                "wic_history.ingestion_worker.load_job_status",
                return_value="cancelled",
            ),
        ):
            result = run_one(
                "postgresql://example",
                worker_id="worker",
                workspace_root=Path.cwd(),
                cache_dir=Path("/tmp/wic-source-cache"),
                lease_seconds=30,
                executor=cancelled_executor,
            )
        self.assertEqual(result.status, "cancelled")

    def test_artifact_paths_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                resolve_workspace_path(root, "artifacts/page.json"),
                (root / "artifacts/page.json").resolve(),
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_workspace_path(root, "../outside.json")

    def test_stage_output_is_namespaced_by_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = stage_output_dir(context("ocr"), Path(directory))
            self.assertEqual(output.name, "00000000-0000-0000-0000-000000000001")
            self.assertEqual(output.parent.name, "jobs")

    def test_unreviewed_render_manifest_is_adoptable_but_not_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "artifacts" / "page.png"
            image_path.parent.mkdir(parents=True)
            Image.new("L", (20, 30), "white").save(image_path)
            manifest_path = root / "artifacts" / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "rendered",
                        "render_path": image_path.relative_to(root).as_posix(),
                        "render_sha256": sha256_file(image_path),
                        "source_object_sha256": "a" * 64,
                        "source_uri": "s3://bucket/prefix/volume.pdf",
                        "volume_number": 219,
                        "publication_year": 1925,
                        "page_number": 308,
                        "selection": {"gold_status": "unreviewed_ingestion"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            execution = _render_manifest_execution(
                manifest_path, context("render_lossless"), root
            )
            self.assertTrue(execution.adopted)
            self.assertEqual(
                execution.result["render_sha256"], sha256_file(image_path)
            )

    def test_existing_ocr_requires_exact_model_and_image(self):
        legacy = OCRPageArtifact.model_validate_json(
            Path("artifacts/ocr-pilot/v219-p0308.lossless.ppocrv6.json").read_text(
                encoding="utf-8"
            )
        )
        candidate = context(
            "ocr",
            {
                **DEFAULT_CONFIGURATION["ocr"],
                "output_root": "artifacts/test",
            },
        )
        candidate = replace(
            candidate,
            source_uri=legacy.source.source_uri,
            parent_stage="layout",
            parent_output_sha256="c" * 64,
        )
        self.assertFalse(
            _ocr_artifact_matches(legacy, candidate, legacy.image_sha256)
        )
        artifact = legacy.model_copy(
            update={
                "run": legacy.run.model_copy(
                    update={
                        "engine": candidate.configuration["engine"],
                        "model_name": candidate.configuration["model"],
                        "model_revision": candidate.configuration["revision"],
                        "configuration": {
                            "pipeline": candidate.configuration["pipeline"],
                            "toolkit_name": candidate.configuration["toolkit"],
                            "toolkit_revision": candidate.configuration[
                                "toolkit_revision"
                            ],
                            "language": candidate.configuration["language"],
                            "role": "materialize_accepted_hunyuan_spotting_json",
                            "fallback_allowed": False,
                            "reinference_performed": False,
                            "tiling_used": False,
                            "layout_artifact_sha256": "c" * 64,
                            "pipeline_model_configuration_sha256": candidate.configuration[
                                "pipeline_model_configuration_sha256"
                            ],
                        }
                    }
                )
            }
        )
        self.assertTrue(
            _ocr_artifact_matches(artifact, candidate, artifact.image_sha256)
        )
        self.assertFalse(_ocr_artifact_matches(artifact, candidate, "b" * 64))

    def test_existing_ner_requires_exact_bounded_configuration(self):
        artifact = NERArtifact.model_validate_json(
            Path("artifacts/ner-pilot/v219-p0308.lossless.gliner-x.first50.json").read_text(
                encoding="utf-8"
            )
        )
        configuration = {
            **GLINER_NER_CONFIGURATION,
            "max_regions": 50,
            "batch_size": 2,
        }
        candidate = context("ner", configuration)
        self.assertTrue(
            _ner_artifact_matches(
                artifact, candidate, str(artifact.source_ocr_run_id)
            )
        )
        wrong = context("ner", {**configuration, "max_regions": 25})
        self.assertFalse(
            _ner_artifact_matches(artifact, wrong, str(artifact.source_ocr_run_id))
        )

    def test_structured_ner_worker_verifies_publishes_and_recovers_exact_plan(self):
        ocr = OCRPageArtifact.model_validate_json(
            Path("artifacts/ocr-pilot/v219-p0308.lossless.ppocrv6.json").read_text(
                encoding="utf-8"
            )
        )
        source = NERArtifact.model_validate_json(
            Path("artifacts/ner-pilot/v219-p0308.lossless.gliner-x.first50.json").read_text(
                encoding="utf-8"
            )
        )
        configuration = {
            "adapter": "structured_generation",
            "model": "Qwen/Qwen3.5-0.8B",
            "revision": "2" * 40,
            "license": "Apache-2.0",
            "ontology_version": "women-history-zh-v1",
            "input_variant": "raw_ocr",
            "max_regions": 2,
            "dataset_id": None,
            "split_id": None,
            "base_url": "http://127.0.0.1:11434/v1",
            "served_model": "qwen3.5:0.8b",
            "runtime_name": "ollama",
            "runtime_version": "0.24.0",
            "runtime_executable": "/usr/local/bin/ollama",
            "runtime_executable_sha256": "3" * 64,
            "ollama_manifest_digest": "sha256:" + ("4" * 64),
            "quantization": "Q8_0",
            "device": "local-runtime",
            "seed": 42,
            "max_output_tokens": 512,
            "timeout_seconds": 120,
            "schema_canary_repetitions": 3,
            "expected_canary_raw_output_sha256": "5" * 64,
            "code_revision": "6" * 40,
            "prompt_schema_revision": "7" * 64,
            "response_format_sha256": "8" * 64,
            "region_chunk_size": 8,
            "output_root": "artifacts/ingestion-ner",
            "status": "candidate_only",
        }
        identity = {
            "adapter_id": "structured-ner:ollama:qwen3.5:0.8b",
            "family": "structured_generation",
            "model_name": configuration["model"],
            "model_revision": configuration["revision"],
            "license": configuration["license"],
            "code_revision": configuration["code_revision"],
            "prompt_schema_revision": configuration["prompt_schema_revision"],
            "dtype": configuration["quantization"],
            "configuration": {
                "base_url": configuration["base_url"],
                "served_model": configuration["served_model"],
                "runtime_version": configuration["runtime_version"],
                "runtime_executable_sha256": configuration[
                    "runtime_executable_sha256"
                ],
                "ollama_manifest_digest": configuration[
                    "ollama_manifest_digest"
                ],
                "quantization": configuration["quantization"],
                "seed": configuration["seed"],
                "max_output_tokens": configuration["max_output_tokens"],
                "response_format_sha256": configuration[
                    "response_format_sha256"
                ],
                "schema_canary": {
                    "raw_output_sha256s": [
                        configuration["expected_canary_raw_output_sha256"]
                    ]
                    * configuration["schema_canary_repetitions"],
                    "deterministic": True,
                    "required_span_verified": True,
                },
            },
        }
        eligible = [
            region for region in ocr.regions if len(region.raw_text.strip()) >= 2
        ][:2]
        run = source.run.model_copy(
            update={
                "engine": identity["adapter_id"],
                "model_name": configuration["model"],
                "model_revision": configuration["revision"],
                "configuration": {
                    "adapter_identity": identity,
                    "max_regions": 2,
                    "region_chunk_size": 8,
                    "input_region_count": 2,
                    "regions_attempted": 2,
                    "regions_succeeded": 1,
                    "regions_abstained": 1,
                    "invalid_outputs": 1,
                    "job_configuration_sha256": canonical_sha256(configuration),
                },
            }
        )
        structured = source.model_copy(
            update={
                "source_ocr_run_id": ocr.run.run_id,
                "input_variant": "raw_ocr",
                "input_sha256": ner_input_sha256(ocr.run.run_id, eligible),
                "dataset_id": f"ocr-run:{ocr.run.run_id}",
                "split_id": "technical_pilot",
                "ontology_version": configuration["ontology_version"],
                "adapter_id": identity["adapter_id"],
                "prompt_schema_revision": configuration["prompt_schema_revision"],
                "run": run,
                "mentions": [],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_path = root / "artifacts" / "ocr.json"
            parent_path.parent.mkdir(parents=True)
            parent_path.write_text(ocr.model_dump_json(indent=2), encoding="utf-8")
            candidate = replace(
                context("ner", configuration),
                parent_stage="ocr",
                parent_artifact_uri="artifacts/ocr.json",
                parent_output_sha256=sha256_file(parent_path),
                parent_result={"ocr_run_id": str(ocr.run.run_id)},
            )

            with (
                patch("wic_history.ingestion_worker._existing_ner", return_value=None),
                patch(
                    "wic_history.ingestion_worker.build_verified_structured_ner_adapter",
                    return_value=(SimpleNamespace(), SimpleNamespace()),
                ) as build,
                patch(
                    "wic_history.ingestion_worker.create_structured_ner_artifact",
                    return_value=structured,
                ),
                patch(
                    "wic_history.ingestion_worker.reverify_structured_ner_runtime"
                ) as reverify,
                patch(
                    "wic_history.ingestion_worker.ingest_ner_artifact",
                    return_value=SimpleNamespace(
                        run_id=str(run.run_id), mentions_verified=0
                    ),
                ),
            ):
                fresh = execute_ner("postgresql://example", candidate, root)
                reused = execute_ner("postgresql://example", candidate, root)

            self.assertFalse(fresh.adopted)
            self.assertTrue(reused.adopted)
            self.assertEqual(fresh.result["regions_attempted"], 2)
            self.assertEqual(fresh.result["regions_abstained"], 1)
            self.assertEqual(build.call_count, 1)
            reverify.assert_called_once()

    def test_entity_link_parent_requires_exact_ner_artifact_and_result(self):
        artifact = NERArtifact.model_validate_json(
            Path("artifacts/ner-pilot/v219-p0308.lossless.gliner-x.first50.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifacts" / "ner.json"
            path.parent.mkdir(parents=True)
            path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
            candidate = replace(
                context("entity_link"),
                parent_stage="ner",
                parent_artifact_uri="artifacts/ner.json",
                parent_output_sha256=sha256_file(path),
                parent_result={
                    "ner_run_id": str(artifact.run.run_id),
                    "mentions": len(artifact.mentions),
                },
            )

            loaded_path, loaded = _parent_ner(candidate, root)
            self.assertEqual(loaded_path, path.resolve())
            self.assertEqual(loaded.run.run_id, artifact.run.run_id)
            wrong = replace(
                candidate,
                parent_result={
                    **candidate.parent_result,
                    "mentions": len(artifact.mentions) + 1,
                },
            )
            with self.assertRaisesRegex(ValueError, "mention count"):
                _parent_ner(wrong, root)

    def test_link_coverage_requires_one_nil_for_every_parent_mention(self):
        source = NERArtifact.model_validate_json(
            Path("artifacts/ner-pilot/v219-p0308.lossless.gliner-x.first50.json").read_text(
                encoding="utf-8"
            )
        )
        parent = source.model_copy(update={"mentions": source.mentions[:2]})
        links = create_link_artifact(parent, [], top_k=5)
        _validate_link_coverage(links, parent, top_k=5)

        missing = links.model_copy(update={"links": links.links[:-1]})
        with self.assertRaisesRegex(ValueError, "cover every"):
            _validate_link_coverage(missing, parent, top_k=5)
        duplicate_nil = links.model_copy(
            update={
                "links": [
                    *links.links,
                    links.links[0].model_copy(update={"link_id": uuid4()}),
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly one NIL"):
            _validate_link_coverage(duplicate_nil, parent, top_k=5)

    def test_entity_link_worker_generates_ingests_and_recovers_job_local_artifact(self):
        parent = NERArtifact.model_validate_json(
            Path("artifacts/ner-pilot/v219-p0308.lossless.gliner-x.first50.json").read_text(
                encoding="utf-8"
            )
        ).model_copy(update={"mentions": []})
        # The live remote semantic provider disables the entity_link stage
        # profile (no verifiable local runtime), so the worker recovery
        # mechanics are exercised with a frozen resolver-free configuration.
        self.assertIsNone(DEFAULT_CONFIGURATION["entity_link"])
        baseline_configuration = {
            "engine": "exact-alias+character-similarity",
            "candidate_generator_revision": "1",
            "top_k": 5,
            "fuzzy_threshold": 0.72,
            "reviewed_entities_only": True,
            "nil_required": True,
            "resolver": "none",
            "identity_mutation": False,
            "output_root": "artifacts/ingestion-links",
            "status": "candidate_only",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_path = root / "artifacts" / "ner.json"
            parent_path.parent.mkdir(parents=True)
            parent_path.write_text(parent.model_dump_json(indent=2), encoding="utf-8")
            candidate = replace(
                context("entity_link", baseline_configuration),
                parent_stage="ner",
                parent_artifact_uri="artifacts/ner.json",
                parent_output_sha256=sha256_file(parent_path),
                parent_result={"ner_run_id": str(parent.run.run_id), "mentions": 0},
            )

            def fake_link_main(arguments):
                output = Path(arguments[arguments.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    create_link_artifact(parent, []).model_dump_json(indent=2),
                    encoding="utf-8",
                )
                return 0

            with (
                patch("wic_history.ingestion_worker.link_main", side_effect=fake_link_main),
                patch(
                    "wic_history.ingestion_worker.ingest_link_artifact",
                    return_value=SimpleNamespace(links_verified=0),
                ),
            ):
                fresh = execute_entity_link("postgresql://example", candidate, root)
                reused = execute_entity_link("postgresql://example", candidate, root)

            self.assertFalse(fresh.adopted)
            self.assertTrue(reused.adopted)
            self.assertEqual(fresh.result["identity_mutations"], 0)
            self.assertEqual(fresh.result["nil_links"], 0)
            artifact = EntityLinkArtifact.model_validate_json(
                (root / fresh.artifact_uri).read_text(encoding="utf-8")
            )
            self.assertTrue(
                _link_artifact_matches(
                    artifact, candidate, str(parent.run.run_id)
                )
            )


if __name__ == "__main__":
    unittest.main()
