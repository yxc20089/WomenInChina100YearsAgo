from pathlib import Path

import pytest
from pydantic import ValidationError

from wic_history.model_config import (
    LAYOUT_PARSE_PROMPT,
    SPOTTING_JSON_PROMPT,
    load_pipeline_model_configuration,
)


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

    assert configuration.semantic.model_name == "Qwen/Qwen3.5-4B"
    assert configuration.semantic.served_model == "qwen3.5:4b"
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


def test_configuration_hash_changes_with_complete_file(tmp_path: Path) -> None:
    source = Path("config/pipeline-models.toml").read_text(encoding="utf-8")
    original = tmp_path / "original.toml"
    changed = tmp_path / "changed.toml"
    original.write_text(source, encoding="utf-8")
    changed.write_text(
        source.replace('keep_alive = "30m"', 'keep_alive = "20m"'),
        encoding="utf-8",
    )

    assert (
        load_pipeline_model_configuration(original).sha256
        != load_pipeline_model_configuration(changed).sha256
    )
