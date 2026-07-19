from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from wic_history.e2e_pipeline import run_semantic_e2e
from wic_history.semantic_repository import (
    SemanticExtractionPersistResult,
    SemanticPersistResult,
)
from wic_history.semantic_tasks import (
    LocalResolutionResponse,
    SemanticExtractionResponse,
    SemanticTaskResult,
)


def _result(response, task: str) -> SemanticTaskResult:
    return SemanticTaskResult(
        response=response,
        task=task,
        prompt_sha256="a" * 64,
        prompt_schema_sha256="b" * 64,
        response_format_sha256="c" * 64,
        raw_output_sha256="d" * 64,
        raw_output='{"raw":"qwen-output"}',
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
    )


def test_runner_uses_exactly_extraction_then_resolution_for_mentions(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    mention_id = UUID(int=1)
    extraction = _result(
        SemanticExtractionResponse(mentions=[], event_evidence=[], events=[]),
        "semantic_extraction",
    )
    resolution = _result(
        LocalResolutionResponse(clusters=[], unresolved_mention_ids=[mention_id]),
        "local_resolution",
    )

    class FakeClient:
        def extract_evidence(self, **kwargs):
            calls.append("extraction")
            assert kwargs["page_images"] == ["immutable-image"]
            return extraction

        def resolve_local_identities(self, **kwargs):
            calls.append("resolution")
            assert [item.mention_id for item in kwargs["mentions"]] == [mention_id]
            assert kwargs["page_images"] == ["immutable-image"]
            return resolution

    semantic_identity = {
        "provider": "ollama",
        "endpoint": "http://127.0.0.1:11434/v1",
        "served_model": "qwen3.5:4b",
        "model_name": "Qwen3.5-4B",
        "model_revision": "revision",
        "ollama_manifest_digest": "sha256:" + "e" * 64,
        "model_blob_sha256": "sha256:" + "e" * 64,
        "quantization": "Q4_K_M",
        "runtime_name": "ollama",
        "runtime_version": "0.15.4",
        "acceleration": "none",
    }
    semantic_config = SimpleNamespace(
        provenance_identity=lambda: dict(semantic_identity),
    )
    monkeypatch.setattr(
        "wic_history.e2e_pipeline.load_pipeline_model_configuration",
        lambda _path: SimpleNamespace(
            source_path=Path("config/pipeline-models.toml"),
            sha256="f" * 64,
            semantic=semantic_config,
        ),
    )
    monkeypatch.setattr(
        "wic_history.e2e_pipeline.load_reviewed_coherent_text",
        lambda *_args: SimpleNamespace(content="reviewed", input_sha256="1" * 64),
    )
    monkeypatch.setattr(
        "wic_history.e2e_pipeline.build_verified_semantic_client",
        lambda _path: FakeClient(),
    )
    monkeypatch.setattr(
        "wic_history.e2e_pipeline.semantic_multimodal_context",
        lambda _bundle: (["selected-segment"], ["immutable-image"]),
    )
    monkeypatch.setattr(
        "wic_history.e2e_pipeline.persist_semantic_extraction",
        lambda *_args, **_kwargs: SemanticExtractionPersistResult(
            run=SemanticPersistResult("extraction-run", 1, False),
            mention_ids=(mention_id,),
            events=(),
        ),
    )
    monkeypatch.setattr(
        "wic_history.e2e_pipeline.load_resolution_mentions",
        lambda *_args: [SimpleNamespace(mention_id=mention_id)],
    )
    monkeypatch.setattr(
        "wic_history.e2e_pipeline.persist_local_resolution",
        lambda *_args, **_kwargs: SemanticPersistResult("resolution-run", 0, False),
    )

    receipt = run_semantic_e2e("postgresql://unused", UUID(int=2), tmp_path / "out")

    assert calls == ["extraction", "resolution"]
    assert receipt["counts"]["semantic_model_calls"] == 2
    assert receipt["semantic_model"] == semantic_identity
    assert receipt["runs"] == {
        "semantic_extraction": "extraction-run",
        "local_resolution": "resolution-run",
    }
    assert '"raw_output": "{\\"raw\\":\\"qwen-output\\"}"' in (
        tmp_path / "out" / "01-semantic-extraction.json"
    ).read_text(encoding="utf-8")
