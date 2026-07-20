from __future__ import annotations

import pytest

from tests.coherent_search_support import (
    OpenSearchFake,
    TestJson,
    manifest,
)
from wic_history.coherent_search import (
    COHERENT_ALIAS,
    CoherentProjectionError,
    project_coherent_units,
)


def test_bulk_failure_preserves_old_alias(search_server: str) -> None:
    OpenSearchFake.fail_bulk = True
    with pytest.raises(CoherentProjectionError, match="bulk"):
        _ = project_coherent_units(search_server, manifest())
    assert OpenSearchFake.aliases[COHERENT_ALIAS] == "wic-coherent-units-build-old"


def test_misleading_bulk_success_preserves_old_alias(search_server: str) -> None:
    OpenSearchFake.misleading_bulk = True
    with pytest.raises(CoherentProjectionError, match="bulk"):
        _ = project_coherent_units(search_server, manifest())
    assert OpenSearchFake.aliases[COHERENT_ALIAS] == "wic-coherent-units-build-old"


def test_projection_requires_acknowledged_alias_update(search_server: str) -> None:
    OpenSearchFake.alias_update_envelope = {"acknowledged": False}
    with pytest.raises(CoherentProjectionError, match="alias update"):
        _ = project_coherent_units(search_server, manifest())


def test_projection_rejects_failed_refresh_before_alias_move(
    search_server: str,
) -> None:
    OpenSearchFake.refresh_envelope = {
        "_shards": {"total": 2, "successful": 1, "failed": 1}
    }
    with pytest.raises(CoherentProjectionError, match="refresh"):
        _ = project_coherent_units(search_server, manifest())
    assert OpenSearchFake.aliases[COHERENT_ALIAS] == "wic-coherent-units-build-old"


@pytest.mark.parametrize(
    ("envelope_name", "envelope"),
    [
        ("bulk_envelope", []),
        ("count_envelope", {"count": "one"}),
        ("alias_lookup_envelope", []),
        ("alias_update_envelope", []),
    ],
)
def test_projection_parses_malformed_envelopes_as_typed_errors(
    search_server: str, envelope_name: str, envelope: TestJson
) -> None:
    setattr(OpenSearchFake, envelope_name, envelope)
    with pytest.raises(CoherentProjectionError, match="OpenSearch"):
        _ = project_coherent_units(search_server, manifest())
