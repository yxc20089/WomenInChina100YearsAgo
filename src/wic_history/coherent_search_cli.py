"""Register and execute reviewed coherent-unit CLI operations."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, Literal

from .coherent_search import (
    SearchSpec,
    coherent_dense_search,
    coherent_hybrid_search,
    coherent_lexical_search,
    project_coherent_units,
)
from .search_manifest import CoherentProjectionPins, load_coherent_projection_manifest
from .search_runtime import PinnedQueryEmbedder

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from .evidence import RetrievalResponse


REVIEWED_COHERENT_UNIT: Final = "reviewed_coherent_unit"
COHERENT_PROJECTION_PINS_ERROR: Final = (
    "coherent projection requires --snapshot-sha256"
)
COHERENT_QUERY_PINS_ERROR: Final = (
    "coherent dense/hybrid query is missing its pinned embedding identity"
)


@dataclass(frozen=True, slots=True)
class CoherentProjectionArguments:
    """Typed coherent projection inputs parsed by the shared CLI."""

    database_url: str
    opensearch_url: str
    model_name: str | None
    model_revision: str | None
    configuration_sha256: str | None
    snapshot_sha256: str | None


@dataclass(frozen=True, slots=True)
class CoherentQueryArguments:
    """Typed coherent query inputs parsed by the shared CLI."""

    opensearch_url: str
    query: str
    limit: int
    year_start: int | None
    year_end: int | None
    mode: Literal["lexical", "dense", "hybrid"]
    model_name: str | None
    model_revision: str | None
    configuration_sha256: str | None


def register_coherent_project_arguments(parser: argparse.ArgumentParser) -> None:
    """Register coherent projection options on the project subcommand."""
    _ = parser.add_argument(
        "--unit",
        choices=("region", REVIEWED_COHERENT_UNIT),
        default="region",
    )
    _ = parser.add_argument("--snapshot-sha256")


def register_coherent_query_arguments(parser: argparse.ArgumentParser) -> None:
    """Register coherent retrieval options on the query subcommand."""
    _ = parser.add_argument(
        "--unit",
        choices=("region", REVIEWED_COHERENT_UNIT),
        default="region",
    )


def run_coherent_projection(args: CoherentProjectionArguments) -> int:
    """Project reviewed coherent units and emit the result as JSON."""
    model_name = args.model_name
    model_revision = args.model_revision
    configuration_sha256 = args.configuration_sha256
    snapshot_sha256 = args.snapshot_sha256
    if (
        not model_name
        or not model_revision
        or not configuration_sha256
        or not snapshot_sha256
    ):
        raise SystemExit(COHERENT_PROJECTION_PINS_ERROR)
    manifest = load_coherent_projection_manifest(
        args.database_url,
        CoherentProjectionPins(
            model_name,
            model_revision,
            configuration_sha256,
            snapshot_sha256,
        ),
    )
    result = project_coherent_units(args.opensearch_url, manifest)
    _ = sys.stdout.write(f"{json.dumps(asdict(result), ensure_ascii=False)}\n")
    return 0


def _pinned_query_embedder(args: CoherentQueryArguments) -> PinnedQueryEmbedder:
    model_name = args.model_name
    model_revision = args.model_revision
    configuration_sha256 = args.configuration_sha256
    if not model_name or not model_revision or not configuration_sha256:
        raise SystemExit(COHERENT_QUERY_PINS_ERROR)
    return PinnedQueryEmbedder(
        model_name,
        model_revision,
        configuration_sha256,
    )


def run_coherent_query(args: CoherentQueryArguments) -> int:
    """Run coherent retrieval and emit the response as formatted JSON."""
    spec = SearchSpec(
        args.query,
        limit=args.limit,
        year_min=args.year_start,
        year_max=args.year_end,
        model_name=args.model_name,
        model_revision=args.model_revision,
        configuration_sha256=args.configuration_sha256,
    )
    retrievals: dict[
        Literal["lexical", "dense", "hybrid"],
        Callable[[], RetrievalResponse],
    ] = {
        "lexical": lambda: coherent_lexical_search(args.opensearch_url, spec),
        "dense": lambda: coherent_dense_search(
            args.opensearch_url,
            spec,
            _pinned_query_embedder(args),
        ),
        "hybrid": lambda: coherent_hybrid_search(
            args.opensearch_url,
            spec,
            _pinned_query_embedder(args),
        ),
    }
    response = retrievals[args.mode]()
    _ = sys.stdout.write(f"{response.model_dump_json(indent=2)}\n")
    return 0
