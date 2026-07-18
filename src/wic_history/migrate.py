"""Apply ordered PostgreSQL migrations with checksums and advisory locking."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Sequence


MIGRATION_LOCK_ID = 864219274


def migration_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.sql") if path.is_file())


def apply_migrations(database_url: str, directory: Path) -> list[str]:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc

    applied: list[str] = []
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_ID,))
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS public.schema_migration (
                    migration_name text PRIMARY KEY,
                    sha256 text NOT NULL,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            for path in migration_files(directory):
                sql = path.read_text(encoding="utf-8")
                checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                row = connection.execute(
                    "SELECT sha256 FROM public.schema_migration WHERE migration_name = %s",
                    (path.name,),
                ).fetchone()
                if row:
                    if row[0] != checksum:
                        raise RuntimeError(f"Applied migration changed: {path.name}")
                    continue
                # Migration files contain their own BEGIN/COMMIT so the same
                # files remain valid for PostgreSQL container initialization.
                connection.execute(sql)
                connection.execute(
                    "INSERT INTO public.schema_migration(migration_name, sha256) VALUES (%s, %s)",
                    (path.name, checksum),
                )
                applied.append(path.name)
        finally:
            connection.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_ID,))
    return applied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL"), help="PostgreSQL connection URL"
    )
    parser.add_argument("--directory", type=Path, default=Path("db/migrations"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    applied = apply_migrations(args.database_url, args.directory)
    print("Applied migrations: " + (", ".join(applied) if applied else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
