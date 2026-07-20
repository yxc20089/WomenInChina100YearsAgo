from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest

from wic_history.article_embedding_vectors import BGEArticleEncoder


def test_bge_encoder_disables_background_safetensors_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: Transformers would otherwise start a non-daemon conversion download.
    observed: list[str | None] = []

    class Tokenizer:
        model_max_length: int = 10

        def num_special_tokens_to_add(self, *, pair: bool) -> int:
            assert pair is False
            return 2

    class Model:
        tokenizer: Tokenizer = Tokenizer()
        max_seq_length: int = 10

        def __init__(self, _model_name: str, *, revision: str, device: str) -> None:
            assert (revision, device) == ("revision", "cpu")
            observed.append(os.environ.get("DISABLE_SAFETENSORS_CONVERSION"))

    class SentenceTransformerModule(ModuleType):
        SentenceTransformer: type[Model] = Model

    module = SentenceTransformerModule("sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)

    # When: the pinned article encoder loads a legacy PyTorch checkpoint.
    _ = BGEArticleEncoder("model", "revision")

    # Then: no conversion thread can escape, and caller environment is restored.
    assert observed == ["1"]
    assert "DISABLE_SAFETENSORS_CONVERSION" not in os.environ
    monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", "0")
    _ = BGEArticleEncoder("model", "revision")
    assert observed == ["1", "1"]
    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "0"


class ConstructorFailure(RuntimeError):
    pass


@pytest.mark.parametrize("initial_value", [None, "caller-value"])
def test_bge_encoder_restores_environment_when_constructor_fails(
    monkeypatch: pytest.MonkeyPatch,
    initial_value: str | None,
) -> None:
    observed: list[str | None] = []

    class FailingModel:
        def __init__(self, _model_name: str, *, revision: str, device: str) -> None:
            assert (revision, device) == ("revision", "cpu")
            observed.append(os.environ.get("DISABLE_SAFETENSORS_CONVERSION"))
            raise ConstructorFailure("constructor failed")

    class SentenceTransformerModule(ModuleType):
        SentenceTransformer: type[FailingModel] = FailingModel

    module = SentenceTransformerModule("sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    if initial_value is None:
        monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    else:
        monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", initial_value)

    with pytest.raises(ConstructorFailure, match="constructor failed"):
        _ = BGEArticleEncoder("model", "revision")

    assert observed == ["1"]
    if initial_value is None:
        assert "DISABLE_SAFETENSORS_CONVERSION" not in os.environ
    else:
        assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == initial_value
