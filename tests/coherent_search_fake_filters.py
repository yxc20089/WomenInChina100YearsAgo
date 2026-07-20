from __future__ import annotations


def matches_dense_identity(body: str, document: str) -> bool:
    for field in (
        "embedding_model",
        "embedding_model_revision",
        "embedding_configuration_sha256",
    ):
        prefix = f'"{field}":"'
        start = body.find(prefix)
        if start < 0:
            return False
        start += len(prefix)
        end = body.find('"', start)
        if end < 0 or f'"{field}": "{body[start:end]}"' not in document:
            return False
    return True
