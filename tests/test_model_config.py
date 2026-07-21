from pathlib import Path

import pytest
from pydantic import ValidationError

from wic_history.model_config import (
    LAYOUT_PARSE_PROMPT,
    SPOTTING_JSON_PROMPT,
    load_pipeline_model_configuration,
)


OPENROUTER_SEMANTIC_SECTION = """\
[semantic]
provider = "openrouter"
base_url = "https://openrouter.ai/api/v1"
served_model = "z-ai/glm-4.6v"
model_name = "zai-org/GLM-4.6V"
api_key_environment_variable = "OPENROUTER_API_KEY"
model_revision_status = "not_available"
weight_hash_status = "not_available"
quantization_status = "not_disclosed_by_provider"
thinking = false
temperature = 0.0
seed = 42
context_length = 131072
max_output_tokens = 4096
timeout_seconds = 120.0
structured_output = "openai_response_format_json_schema"

"""


def _write_openrouter_configuration(
    tmp_path: Path, mutate=lambda section: section
) -> Path:
    source = Path("config/pipeline-models.toml").read_text(encoding="utf-8")
    start = source.index("[semantic]")
    end = source.index("[retrieval.passage_embedding]")
    path = tmp_path / "openrouter-models.toml"
    path.write_text(
        source[:start] + mutate(OPENROUTER_SEMANTIC_SECTION) + source[end:],
        encoding="utf-8",
    )
    return path


def test_default_configuration_has_one_pinned_ocr_and_layout_model() -> None:
    source = Path("config/pipeline-models.toml").read_text(encoding="utf-8")
    configuration = load_pipeline_model_configuration()

    assert "\n[layout]\n" not in source
    assert "\n[ocr.primary]\n" not in source
    assert "\n[ocr.difficult]\n" not in source
    assert source.count("\n[ocr]\n") == 1
    assert configuration.ocr.model_name == "tencent/HunyuanOCR"
    assert configuration.ocr.model_revision == "de8f10ad2f00a0cefd790b526de8a65dcfdb3205"
    assert configuration.ocr.toolkit_revision == "a1ce1099db98edceb153710536af23edf4391cf0"
    assert configuration.ocr.pipeline == "spotting_json+layout_parse"
    assert configuration.ocr.spotting_prompt == SPOTTING_JSON_PROMPT
    assert configuration.ocr.layout_prompt == LAYOUT_PARSE_PROMPT
    assert configuration.ocr.runtime == "transformers-cuda"
    assert configuration.ocr.confidence_status == "not_emitted_by_model"
    assert configuration.ocr.confidence_calibration == "not_available"
    # Read-only compatibility views do not select alternate models.
    assert configuration.layout is configuration.ocr
    assert not hasattr(configuration.ocr, "primary")
    assert not hasattr(configuration.ocr, "difficult")
    assert len(configuration.sha256) == 64


def test_non_ocr_model_selections_remain_pinned() -> None:
    configuration = load_pipeline_model_configuration()

    assert configuration.semantic.provider == "openrouter"
    assert configuration.semantic.served_model == "anthropic/claude-opus-4.8"
    assert configuration.frontier_ocr is not None
    assert configuration.frontier_ocr.served_model == "anthropic/claude-opus-4.8"
    assert configuration.retrieval.passage_embedding.dimension == 1024
    assert configuration.identity.embedding.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert configuration.identity.reranker.model_name == "Qwen/Qwen3-Reranker-0.6B"
    assert configuration.identity.enabled is False


@pytest.mark.parametrize(
    "old,new",
    [
        (
            'model_revision = "de8f10ad2f00a0cefd790b526de8a65dcfdb3205"',
            'model_revision = "main"',
        ),
        (
            'spotting_prompt = "检测并识别图中所有的文字行',
            'spotting_prompt = "请识别图中所有的文字行',
        ),
    ],
)
def test_configuration_rejects_unpinned_revision_or_prompt(
    tmp_path: Path, old: str, new: str
) -> None:
    source = Path("config/pipeline-models.toml").read_text(encoding="utf-8")
    path = tmp_path / "models.toml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_pipeline_model_configuration(path)


def test_openrouter_semantic_provider_records_unavailable_provenance(
    tmp_path: Path,
) -> None:
    configuration = load_pipeline_model_configuration(
        _write_openrouter_configuration(tmp_path)
    )
    semantic = configuration.semantic

    assert semantic.provider == "openrouter"
    assert semantic.base_url == "https://openrouter.ai/api/v1"
    assert semantic.served_model == "z-ai/glm-4.6v"
    assert semantic.api_key_environment_variable == "OPENROUTER_API_KEY"
    # Hosted routing exposes no immutable revision or weight hashes; the
    # configuration must say so explicitly instead of fabricating them.
    identity = semantic.provenance_identity()
    assert identity["model_revision"] == "not_available"
    assert identity["weight_hashes"] == "not_available"
    assert identity["quantization"] == "not_disclosed_by_provider"
    assert not hasattr(semantic, "ollama_manifest_digest")
    assert not hasattr(semantic, "runtime_executable")


def test_ollama_semantic_identity_keeps_exact_local_provenance() -> None:
    # the ollama provider remains supported; its exact local provenance is
    # exercised through the frozen qwen pilot configuration
    pilot = Path("experiments/e2e/pilot-models.toml")
    identity = load_pipeline_model_configuration(pilot).semantic.provenance_identity()

    assert identity["provider"] == "ollama"
    assert identity["served_model"] == "qwen3.5:4b"
    assert identity["ollama_manifest_digest"].startswith("sha256:")
    assert identity["model_blob_sha256"].startswith("sha256:")


@pytest.mark.parametrize(
    "old,new",
    [
        # Any endpoint other than the pinned HTTPS OpenRouter origin is refused.
        (
            'base_url = "https://openrouter.ai/api/v1"',
            'base_url = "http://openrouter.ai/api/v1"',
        ),
        (
            'base_url = "https://openrouter.ai/api/v1"',
            'base_url = "https://models.example/v1"',
        ),
        # Claiming a pinned revision the provider cannot expose is fabrication.
        (
            'model_revision_status = "not_available"',
            'model_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"\n'
            'model_revision_status = "not_available"',
        ),
        (
            'weight_hash_status = "not_available"',
            'weight_hash_status = "verified"',
        ),
        # The API key belongs in the environment, never in configuration.
        (
            'api_key_environment_variable = "OPENROUTER_API_KEY"',
            'api_key = "sk-or-verysecret"\n'
            'api_key_environment_variable = "OPENROUTER_API_KEY"',
        ),
    ],
)
def test_openrouter_semantic_rejects_unpinned_or_fabricated_fields(
    tmp_path: Path, old: str, new: str
) -> None:
    path = _write_openrouter_configuration(
        tmp_path, mutate=lambda section: section.replace(old, new, 1)
    )
    with pytest.raises(ValidationError):
        load_pipeline_model_configuration(path)


def test_semantic_provider_field_admits_only_known_providers(
    tmp_path: Path,
) -> None:
    path = _write_openrouter_configuration(
        tmp_path,
        mutate=lambda section: section.replace(
            'provider = "openrouter"', 'provider = "openai"', 1
        ),
    )
    with pytest.raises(ValidationError):
        load_pipeline_model_configuration(path)


def test_configuration_hash_changes_with_complete_file(tmp_path: Path) -> None:
    source = Path("config/pipeline-models.toml").read_text(encoding="utf-8")
    original = tmp_path / "original.toml"
    changed = tmp_path / "changed.toml"
    original.write_text(source, encoding="utf-8")
    changed.write_text(
        source.replace('timeout_seconds = 300.0', 'timeout_seconds = 299.0'),
        encoding="utf-8",
    )

    assert (
        load_pipeline_model_configuration(original).sha256
        != load_pipeline_model_configuration(changed).sha256
    )
