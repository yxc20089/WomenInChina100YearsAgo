from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import UUID

from wic_history.evidence import EntityType, OCRPageArtifact, SourcePointer
from wic_history.generation import OpenAICompatibleGenerator
from wic_history.ner_adapters.base import (
    AdapterIdentity,
    BenchmarkPredictionArtifact,
    IssueSplitManifest,
    SnippetSplitAssignment,
)
from wic_history.ner_benchmark import (
    execute_benchmark,
    main as benchmark_main,
    prepare_benchmark_dataset,
)
from wic_history.ner_gold import (
    GoldAdjudication,
    GoldEntitySpan,
    GoldSnippet,
    NERGoldSet,
    ReviewerAnnotation,
)
from wic_history.ner_structured import (
    LocalStructuredNERConfiguration,
    STRUCTURED_NER_PROMPT_SCHEMA_SHA256,
    STRUCTURED_NER_RESPONSE_FORMAT_SHA256,
    StructuredGenerationBenchmarkAdapter,
    build_verified_structured_ner_adapter,
    create_structured_ner_artifact,
    parse_structured_ner_content,
    prepare_structured_ner_messages,
    validate_local_artifact,
    verify_ollama_model_digest,
)


NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)
MODEL_REVISION = "b" * 40
CODE_REVISION = "c" * 40
LOCAL_ARTIFACT_SHA256 = "a" * 64
OLLAMA_DIGEST = "sha256:" + LOCAL_ARTIFACT_SHA256
SERVED_MODEL = "project/qwen35-08b-q8-ner-v1"


def _identity(base_url: str, **configuration_updates: object) -> AdapterIdentity:
    configuration = {
        "base_url": base_url,
        "served_model": SERVED_MODEL,
        "runtime_name": "ollama",
        "runtime_version": "0.32.0",
        "runtime_verification": {"observed_digest": OLLAMA_DIGEST},
        "local_artifact_sha256": LOCAL_ARTIFACT_SHA256,
        "quantization": "Q8_0",
        "temperature": 0,
        "top_p": 1,
        "reasoning_effort": "none",
        "seed": 42,
        "max_output_tokens": 256,
        "timeout_seconds": 10,
        "response_format_sha256": STRUCTURED_NER_RESPONSE_FORMAT_SHA256,
        "remote_data_egress_allowed": False,
    }
    configuration.update(configuration_updates)
    return AdapterIdentity(
        adapter_id="structured-ner:ollama:project-qwen35",
        family="structured_generation",
        model_name="Qwen/Qwen3.5-0.8B",
        model_revision=MODEL_REVISION,
        license="Apache-2.0",
        modalities=["text"],
        runtime="ollama-0.32.0",
        code_revision=CODE_REVISION,
        device="cpu",
        dtype="Q8_0",
        ontology_version="women-history-zh-v1",
        prompt_schema_revision=STRUCTURED_NER_PROMPT_SCHEMA_SHA256,
        configuration=configuration,
    )


def _generator(base_url: str) -> OpenAICompatibleGenerator:
    return OpenAICompatibleGenerator(
        base_url,
        SERVED_MODEL,
        model_revision=LOCAL_ARTIFACT_SHA256,
        timeout_seconds=10,
        max_output_tokens=256,
        seed=42,
    )


def _gold() -> NERGoldSet:
    raw_text = "臺北王女士任教"
    entity = GoldEntitySpan(
        entity_type=EntityType.PERSON,
        corrected_start=2,
        corrected_end=5,
        corrected_text="王女士",
        raw_start=2,
        raw_end=5,
        raw_text="王女士",
    )
    annotation = {
        "corrected_text": raw_text,
        "entities": [entity],
        "annotated_at": NOW,
    }
    return NERGoldSet(
        schema_version="1.1",
        dataset_id="structured-gold",
        created_at=NOW,
        ontology_version="women-history-zh-v1",
        snippets=[
            GoldSnippet(
                snippet_id="snippet-1",
                gold_region_id=UUID(int=101),
                source_ocr_run_id=UUID(int=201),
                source_ocr_region_id=UUID(int=301),
                source=SourcePointer(
                    source_uri="s3://example/v1.pdf",
                    publication_year=1926,
                    page_number=1,
                    region_id=UUID(int=301),
                ),
                raw_ocr_text=raw_text,
                page_genre="news_editorial",
                layout="vertical",
                scan_quality="moderate",
                reviews=[
                    ReviewerAnnotation(reviewer="reviewer-a", **annotation),
                    ReviewerAnnotation(reviewer="reviewer-b", **annotation),
                ],
                adjudication=GoldAdjudication(
                    adjudicator="adjudicator-c",
                    corrected_text=raw_text,
                    entities=[entity],
                    adjudicated_at=NOW,
                ),
            )
        ],
    )


def _dataset():
    manifest = IssueSplitManifest(
        dataset_id="structured-gold",
        created_at=NOW,
        assigned_by="historian",
        assignments=[
            SnippetSplitAssignment(
                snippet_id="snippet-1", issue_id="issue-1", split="test"
            )
        ],
    )
    return prepare_benchmark_dataset(
        _gold(),
        "1" * 64,
        manifest,
        "2" * 64,
        dataset_id="structured-benchmark",
        input_variants=["raw_ocr", "corrected_text"],
        generated_at=NOW,
    )


class LocalModelServer:
    def __init__(
        self,
        *,
        nondeterministic: bool = False,
        redirect_tags: bool = False,
        raw_digest: bool = False,
    ):
        self.requests: list[dict[str, object]] = []
        self.nondeterministic = nondeterministic
        self.redirect_tags = redirect_tags
        self.raw_digest = raw_digest
        self.redirect_target_requests = 0
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/version":
                    body = json.dumps({"version": "0.32.0"}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/api/tags" and parent.redirect_tags:
                    self.send_response(302)
                    self.send_header("Location", "/captured")
                    self.end_headers()
                    return
                if self.path == "/captured":
                    parent.redirect_target_requests += 1
                    self.send_response(200)
                    self.end_headers()
                    return
                if self.path != "/api/tags":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(
                    {
                        "models": [
                            {
                                "name": SERVED_MODEL + ":latest",
                                "model": SERVED_MODEL + ":latest",
                                "digest": (
                                    LOCAL_ARTIFACT_SHA256
                                    if parent.raw_digest
                                    else OLLAMA_DIGEST
                                ),
                                "details": {
                                    "family": "qwen35",
                                    "parameter_size": "873.44M",
                                    "quantization_level": "Q8_0",
                                },
                            }
                        ]
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(length))
                parent.requests.append({"path": self.path, "payload": payload})
                user_payload = json.loads(payload["messages"][-1]["content"])
                source_text = user_payload["source_text"]
                entities = []
                if "王女士" in source_text:
                    entities.append(
                        {
                            "type": "person",
                            "surface": "王女士",
                        }
                    )
                content = json.dumps(
                    {"entities": entities},
                    ensure_ascii=False,
                    sort_keys=not (
                        parent.nondeterministic and len(parent.requests) % 2 == 0
                    ),
                    separators=(",", ":")
                    if not (parent.nondeterministic and len(parent.requests) % 2 == 0)
                    else (", ", ": "),
                )
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class StructuredNERTests(unittest.TestCase):
    def test_prompt_preserves_exact_traditional_text_and_pins_schema(self):
        messages, prompt_sha256 = prepare_structured_ner_messages("臺灣女醫師")
        user_payload = json.loads(messages[-1]["content"])

        self.assertEqual(user_payload["source_text"], "臺灣女醫師")
        self.assertNotIn("台湾", messages[-1]["content"])
        self.assertEqual(len(prompt_sha256), 64)
        self.assertEqual(len(STRUCTURED_NER_PROMPT_SCHEMA_SHA256), 64)

    def test_parser_rejects_normalization_unknown_labels_absent_and_duplicates(
        self,
    ):
        content = json.dumps(
            {
                "entities": [
                    {"type": "place", "surface": "臺北"},
                    {"type": "place", "surface": "台北"},
                    {"type": "unknown", "surface": "王女士"},
                    {"type": "person", "surface": "王女士"},
                    {"type": "person", "surface": "王女士"},
                    {"type": "person", "surface": "不存在"},
                ]
            },
            ensure_ascii=False,
        )
        parsed = parse_structured_ner_content(content, "臺北王女士任教")

        self.assertEqual(len(parsed.spans), 2)
        self.assertEqual(parsed.invalid_outputs, 4)
        self.assertEqual(parsed.spans[0]["text"], "臺北")
        self.assertEqual(parsed.spans[1]["entity_type"], EntityType.PERSON)

        ambiguous = parse_structured_ner_content(
            '{"entities":[{"type":"person","surface":"王女士"}]}',
            "王女士與王女士",
        )
        self.assertEqual(ambiguous.spans, [])
        self.assertEqual(ambiguous.invalid_outputs, 1)

        rejected = parse_structured_ner_content(
            '{"entities":[],"explanation":"ignore schema"}', "原文"
        )
        self.assertEqual(rejected.invalid_outputs, 1)
        self.assertIsNotNone(rejected.rejection_reason)

    def test_adapter_preserves_raw_prompt_usage_and_missing_confidence(self):
        with LocalModelServer() as server:
            adapter = StructuredGenerationBenchmarkAdapter(
                _identity(server.base_url), _generator(server.base_url)
            )
            artifact = execute_benchmark(
                _dataset(),
                adapter,
                split="test",
                input_variant="raw_ocr",
                allow_ineligible_technical_run=True,
            )

        result = artifact.results[0]
        self.assertEqual(len(result.mentions), 1)
        self.assertEqual(result.mentions[0].text, "王女士")
        self.assertIsNone(result.mentions[0].confidence)
        self.assertEqual(
            result.mentions[0].attributes["confidence_semantics"],
            "not_provided_by_adapter",
        )
        self.assertEqual(result.invalid_outputs, 0)
        self.assertEqual(result.total_tokens, 15)
        self.assertEqual(artifact.run.configuration["total_tokens"], 15)
        self.assertIsNotNone(result.raw_output_sha256)
        self.assertIsNotNone(result.prompt_sha256)
        self.assertEqual(
            server.requests[0]["payload"]["response_format"]["type"], "json_schema"
        )
        self.assertEqual(server.requests[0]["payload"]["seed"], 42)
        self.assertEqual(server.requests[0]["payload"]["top_p"], 1)
        self.assertEqual(server.requests[0]["payload"]["reasoning_effort"], "none")

    def test_canary_reports_nondeterminism_without_averaging(self):
        with LocalModelServer(nondeterministic=True) as server:
            adapter = StructuredGenerationBenchmarkAdapter(
                _identity(server.base_url), _generator(server.base_url)
            )
            result = adapter.run_schema_canary(2)

        self.assertFalse(result.deterministic)
        self.assertEqual(len(set(result.raw_output_sha256s)), 2)

    def test_ollama_digest_check_is_exact_and_does_not_follow_redirects(self):
        with LocalModelServer() as server:
            verification = verify_ollama_model_digest(
                _generator(server.base_url), OLLAMA_DIGEST, "0.32.0"
            )
            self.assertEqual(verification.observed_digest, OLLAMA_DIGEST)
            self.assertEqual(verification.observed_runtime_version, "0.32.0")
            with self.assertRaisesRegex(RuntimeError, "runtime version mismatch"):
                verify_ollama_model_digest(
                    _generator(server.base_url), OLLAMA_DIGEST, "0.31.0"
                )

        with LocalModelServer(raw_digest=True) as server:
            verification = verify_ollama_model_digest(
                _generator(server.base_url), OLLAMA_DIGEST, "0.32.0"
            )
            self.assertEqual(verification.observed_digest, OLLAMA_DIGEST)
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                verify_ollama_model_digest(
                    _generator(server.base_url),
                    "sha256:" + ("f" * 64),
                    "0.32.0",
                )

        with LocalModelServer(redirect_tags=True) as server:
            with self.assertRaisesRegex(RuntimeError, "tags request failed"):
                verify_ollama_model_digest(
                    _generator(server.base_url), OLLAMA_DIGEST, "0.32.0"
                )
            self.assertEqual(server.redirect_target_requests, 0)

    def test_local_artifact_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            path.write_bytes(b"pinned-model")
            digest = hashlib.sha256(b"pinned-model").hexdigest()
            self.assertEqual(validate_local_artifact(path, digest), digest)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_local_artifact(path, "f" * 64)

    def test_verified_factory_hashes_runtime_and_pins_canary(self):
        with tempfile.TemporaryDirectory() as temporary, LocalModelServer() as server:
            executable = Path(temporary) / "ollama"
            executable.write_bytes(b"pinned-runtime")
            executable_hash = hashlib.sha256(b"pinned-runtime").hexdigest()
            configuration = LocalStructuredNERConfiguration(
                model_name="Qwen/Qwen3.5-0.8B",
                model_revision=MODEL_REVISION,
                license="Apache-2.0",
                base_url=server.base_url,
                served_model=SERVED_MODEL,
                runtime_version="0.32.0",
                runtime_executable=executable,
                runtime_executable_sha256=executable_hash,
                ollama_manifest_digest=OLLAMA_DIGEST,
                quantization="Q8_0",
                code_revision=CODE_REVISION,
                max_output_tokens=256,
                schema_canary_repetitions=2,
            )

            adapter, verified = build_verified_structured_ner_adapter(configuration)

        self.assertTrue(verified.canary.deterministic)
        self.assertEqual(verified.executable_sha256, executable_hash)
        self.assertEqual(
            adapter.identity.configuration["ollama_manifest_digest"], OLLAMA_DIGEST
        )
        self.assertEqual(len(server.requests), 2)

    def test_production_artifact_retains_zero_mention_region_and_request_provenance(self):
        ocr = OCRPageArtifact.model_validate_json(
            Path("artifacts/ocr-pilot/v219-p0308.lossless.ppocrv6.json").read_text(
                encoding="utf-8"
            )
        )
        first = ocr.regions[0].model_copy(
            update={"raw_text": "王女士任教於上海女子學校。", "normalized_text": None}
        )
        second = ocr.regions[1].model_copy(
            update={"raw_text": "無名文字", "normalized_text": None}
        )
        ocr = ocr.model_copy(update={"regions": [first, second]})
        with LocalModelServer() as server:
            adapter = StructuredGenerationBenchmarkAdapter(
                _identity(server.base_url), _generator(server.base_url)
            )
            artifact = create_structured_ner_artifact(
                ocr, adapter, region_chunk_size=1
            )

        results = artifact.run.configuration["region_results"]
        self.assertEqual(len(results), 2)
        self.assertEqual([item["mention_count"] for item in results], [1, 0])
        self.assertEqual([item["status"] for item in results], ["ok", "ok"])
        self.assertEqual(artifact.mentions[0].text, "王女士")
        self.assertIsNone(artifact.mentions[0].confidence)
        self.assertEqual(
            artifact.mentions[0].attributes["confidence_semantics"],
            "not_provided_by_adapter",
        )
        self.assertIsNotNone(results[0]["prompt_sha256"])
        self.assertIsNotNone(results[0]["raw_output_sha256"])

    def test_structured_generation_cli_verifies_canary_and_writes_artifact(self):
        with tempfile.TemporaryDirectory() as temporary, LocalModelServer() as server:
            root = Path(temporary)
            dataset_path = root / "dataset.json"
            output_path = root / "predictions.json"
            dataset_path.write_text(
                _dataset().model_dump_json(indent=2), encoding="utf-8"
            )

            result = benchmark_main(
                [
                    "run",
                    "--dataset",
                    str(dataset_path),
                    "--output",
                    str(output_path),
                    "--split",
                    "test",
                    "--input-variant",
                    "raw_ocr",
                    "--adapter",
                    "structured-generation",
                    "--model",
                    "Qwen/Qwen3.5-0.8B",
                    "--revision",
                    MODEL_REVISION,
                    "--license",
                    "Apache-2.0",
                    "--base-url",
                    server.base_url,
                    "--served-model",
                    SERVED_MODEL,
                    "--runtime-name",
                    "ollama",
                    "--runtime-version",
                    "0.32.0",
                    "--local-artifact-sha256",
                    LOCAL_ARTIFACT_SHA256,
                    "--ollama-manifest-digest",
                    OLLAMA_DIGEST,
                    "--quantization",
                    "Q8_0",
                    "--max-output-tokens",
                    "256",
                    "--schema-canary-repetitions",
                    "2",
                    "--code-revision",
                    CODE_REVISION,
                    "--allow-ineligible-technical-run",
                    "--confirm-locked-test-evaluation",
                ]
            )

            self.assertEqual(result, 0)
            artifact = BenchmarkPredictionArtifact.model_validate_json(
                output_path.read_bytes()
            )
            self.assertEqual(artifact.adapter.family, "structured_generation")
            self.assertTrue(
                artifact.adapter.configuration["schema_canary"]["deterministic"]
            )
            self.assertEqual(len(artifact.mentions), 1)
            self.assertEqual(len(server.requests), 3)

    def test_lm_studio_cli_hashes_the_exact_local_gguf_as_a_distinct_system(self):
        with tempfile.TemporaryDirectory() as temporary, LocalModelServer() as server:
            root = Path(temporary)
            dataset_path = root / "dataset.json"
            output_path = root / "predictions.json"
            model_path = root / "Qwen3.5-0.8B-Q8_0.gguf"
            model_bytes = b"test-only-pinned-gguf"
            model_path.write_bytes(model_bytes)
            model_sha256 = hashlib.sha256(model_bytes).hexdigest()
            dataset_path.write_text(
                _dataset().model_dump_json(indent=2), encoding="utf-8"
            )

            result = benchmark_main(
                [
                    "run",
                    "--dataset",
                    str(dataset_path),
                    "--output",
                    str(output_path),
                    "--split",
                    "test",
                    "--input-variant",
                    "raw_ocr",
                    "--adapter",
                    "structured-generation",
                    "--model",
                    "Qwen/Qwen3.5-0.8B",
                    "--revision",
                    MODEL_REVISION,
                    "--license",
                    "Apache-2.0",
                    "--base-url",
                    server.base_url,
                    "--served-model",
                    "qwen35-lmstudio-q8",
                    "--runtime-name",
                    "lm_studio",
                    "--runtime-version",
                    "0.4.19-build-2",
                    "--local-artifact-sha256",
                    model_sha256,
                    "--local-model-artifact",
                    str(model_path),
                    "--quantization",
                    "Q8_0",
                    "--max-output-tokens",
                    "256",
                    "--schema-canary-repetitions",
                    "2",
                    "--code-revision",
                    CODE_REVISION,
                    "--allow-ineligible-technical-run",
                    "--confirm-locked-test-evaluation",
                ]
            )

            self.assertEqual(result, 0)
            artifact = BenchmarkPredictionArtifact.model_validate_json(
                output_path.read_bytes()
            )
            self.assertEqual(
                artifact.adapter.configuration["runtime_name"], "lm_studio"
            )
            self.assertEqual(
                artifact.adapter.configuration["runtime_verification"][
                    "artifact_sha256"
                ],
                model_sha256,
            )
            self.assertEqual(len(artifact.mentions), 1)
            self.assertEqual(len(server.requests), 3)


if __name__ == "__main__":
    unittest.main()
