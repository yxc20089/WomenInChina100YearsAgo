from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import Literal, Protocol, runtime_checkable

from .coherent_search_contracts import JsonValue

SNAPSHOT_LOCK = "wic-coherent-active-snapshot-v1"


class Result(Protocol):
    def fetchone(self) -> Mapping[str, JsonValue] | None: ...
    def fetchall(self) -> list[dict[str, JsonValue]]: ...


class Cursor(Protocol):
    def __enter__(self) -> Cursor: ...
    def __exit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> Literal[False]: ...
    def executemany(self, query: str, params: object) -> object: ...


class DatabaseConnection(Protocol):
    def __enter__(self) -> DatabaseConnection: ...
    def __exit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> Literal[False]: ...
    def execute(self, query: str, params: object = None) -> Result: ...
    def cursor(self) -> Cursor: ...


def lock_coherent_mutation(connection: DatabaseConnection) -> None:
    _ = connection.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (SNAPSHOT_LOCK,)
    )


@runtime_checkable
class PsycopgModule(Protocol):
    Error: type[Exception]

    def connect(
        self, database_url: str, *, row_factory: object = None
    ) -> DatabaseConnection: ...


def database_clients() -> tuple[PsycopgModule, object]:
    module: ModuleType = importlib.import_module("psycopg")
    rows: ModuleType = importlib.import_module("psycopg.rows")
    if not isinstance(module, PsycopgModule):
        raise RuntimeError("psycopg client is invalid")
    return module, getattr(rows, "dict_row")
