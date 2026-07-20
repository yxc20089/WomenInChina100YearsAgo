from __future__ import annotations

from typing import NamedTuple, final

from .article_embedding_vectors import (
    pinned_window_configuration,
    window_configuration_sha256,
)
from .embedding_pipeline import BGEEmbedder
from .model_config import load_pipeline_model_configuration


class CoherentQueryIdentity(NamedTuple):
    model_name: str
    model_revision: str
    configuration_sha256: str


def pinned_coherent_query_identity() -> CoherentQueryIdentity:
    """Query-embedder identity resolved from config/pipeline-models.toml.

    The pinned config file is the sole source of model identities; callers
    (API, CLI) must not accept model names or revisions from the
    environment or command line.
    """
    model = load_pipeline_model_configuration().retrieval.passage_embedding
    return CoherentQueryIdentity(
        model.model_name,
        model.model_revision,
        window_configuration_sha256(pinned_window_configuration()),
    )


@final
class PinnedQueryEmbedder:
    __slots__ = ("configuration_sha256", "model_name", "model_revision")

    def __init__(
        self, model_name: str, model_revision: str, configuration_sha256: str
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.configuration_sha256 = configuration_sha256

    def encode_query(self, query: str) -> list[float]:
        return BGEEmbedder(self.model_name, self.model_revision).encode_query(query)
