from __future__ import annotations

import sys
import types
from uuid import UUID

from wic_history.identity_batch import build_parser
from wic_history.identity_models import (
    IDENTITY_RERANK_INSTRUCTION,
    QwenIdentityEmbedder,
    QwenIdentityReranker,
    identity_profile_text,
)
from wic_history.model_config import load_pipeline_model_configuration


class _Values(list):
    def tolist(self):
        return list(self)


def test_identity_profile_serialization_is_stable_and_evidence_bounded() -> None:
    profile = {
        "identity_profile_id": UUID(int=1),
        "entity_type": "person",
        "name_surfaces": ["孫文", "逸仙"],
        "evidence_span_ids": [UUID(int=2)],
        "attributes": {"context": "孫文，字逸仙", "mention_form": "full_name"},
        "created_at": "ignored",
    }
    rendered = identity_profile_text(profile)
    assert "孫文" in rendered
    assert "逸仙" in rendered
    assert str(UUID(int=2)) in rendered
    assert "identity_profile_id" not in rendered
    assert "created_at" not in rendered


def test_identity_adapters_load_only_centrally_pinned_models(monkeypatch) -> None:
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append(("embedding", model_name, kwargs))

        def encode(self, texts, **kwargs):
            return _Values([[0.0] * 1024 for _ in texts])

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            calls.append(("reranker", model_name, kwargs))

        def predict(self, pairs, **kwargs):
            return _Values([0.75 for _ in pairs])

    fake_sentence_transformers = types.SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer,
        CrossEncoder=FakeCrossEncoder,
    )
    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Sigmoid=lambda: "sigmoid")
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    configuration = load_pipeline_model_configuration()
    embedder = QwenIdentityEmbedder(configuration)
    reranker = QwenIdentityReranker(configuration)
    vectors = embedder.encode_documents([{"entity_type": "person", "name_surfaces": ["孫文"]}])
    scores = reranker.score_pairs(
        [
            (
                {"entity_type": "person", "name_surfaces": ["孫文"]},
                {"entity_type": "person", "name_surfaces": ["孫中山"]},
            )
        ]
    )

    assert len(vectors[0]) == configuration.identity.embedding.dimension
    assert scores == [0.75]
    assert calls[0] == (
        "embedding",
        configuration.identity.embedding.model_name,
        {"revision": configuration.identity.embedding.model_revision},
    )
    assert calls[1][0:2] == (
        "reranker",
        configuration.identity.reranker.model_name,
    )
    assert calls[1][2]["revision"] == configuration.identity.reranker.model_revision
    assert calls[1][2]["prompts"]["identity"] == IDENTITY_RERANK_INSTRUCTION


def test_identity_cli_has_one_complete_config_override_and_no_model_flags() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--model-config" in option_strings
    assert "--model" not in option_strings
    assert "--embedding-model" not in option_strings
    assert "--reranker-model" not in option_strings
