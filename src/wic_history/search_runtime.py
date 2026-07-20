from __future__ import annotations

from typing import final

from .embedding_pipeline import BGEEmbedder


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
