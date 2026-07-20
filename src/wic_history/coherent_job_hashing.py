from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from .coherent_search_contracts import JsonValue


def coherent_sha256(value: Mapping[str, JsonValue]) -> str:
    return hashlib.sha256(
        json.dumps(value, default=str, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
