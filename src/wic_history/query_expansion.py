"""Reviewed historical-synonym expansion for archive queries.

Modern query vocabulary and 1920s newspaper vocabulary diverge (女士 vs
淑女/士女), and the CJK bigram analyzer gives them no lexical overlap. This
module loads a versioned, historian-reviewed lexicon and reports which
reviewed variants apply to a query. It never calls a model: only reviewed
entries expand, expansion is directional (headword to variants), and a
malformed lexicon raises instead of silently broadening retrieval.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import final

DEFAULT_LEXICON_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "historical-synonyms.toml"
)
_MIN_HEADWORD_LENGTH = 2


@final
class ExpansionLexiconError(ValueError):
    """The lexicon file is malformed; expansion must fail closed."""


@dataclass(frozen=True, slots=True)
class ExpansionVariant:
    term: str
    weight: float
    note: str


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    headword: str
    variants: tuple[ExpansionVariant, ...]


@dataclass(frozen=True, slots=True)
class ExpansionLexicon:
    version: str
    sha256: str
    entries: tuple[LexiconEntry, ...]


@dataclass(frozen=True, slots=True)
class QueryExpansion:
    """Reviewed variants that apply to one query, with lexicon identity."""

    lexicon_version: str
    lexicon_sha256: str
    matches: tuple[LexiconEntry, ...]


def _validate_variant(headword: str, payload: object) -> ExpansionVariant:
    if not isinstance(payload, dict):
        raise ExpansionLexiconError(f"variant of {headword!r} must be a table")
    term = payload.get("term")
    weight = payload.get("weight")
    note = payload.get("note", "")
    unknown = set(payload) - {"term", "weight", "note"}
    if unknown:
        raise ExpansionLexiconError(
            f"variant of {headword!r} has unknown keys {sorted(unknown)}"
        )
    if not isinstance(term, str) or not term.strip():
        raise ExpansionLexiconError(f"variant of {headword!r} needs a non-empty term")
    if term == headword:
        raise ExpansionLexiconError(f"{headword!r} must not list itself as a variant")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        raise ExpansionLexiconError(f"variant {term!r} needs a numeric weight")
    if not 0 < float(weight) < 1:
        raise ExpansionLexiconError(
            f"variant {term!r} weight must be strictly between 0 and 1 so exact "
            "matches always outrank synonym-only matches"
        )
    if not isinstance(note, str):
        raise ExpansionLexiconError(f"variant {term!r} note must be a string")
    return ExpansionVariant(term, float(weight), note)


def load_expansion_lexicon(
    path: Path | str = DEFAULT_LEXICON_PATH,
) -> ExpansionLexicon:
    """Parse and validate the reviewed lexicon, failing closed on any defect."""
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ExpansionLexiconError(f"lexicon is not valid TOML: {error}") from error
    version = payload.pop("version", None)
    if not isinstance(version, str) or not version.strip():
        raise ExpansionLexiconError("lexicon requires a non-empty string version")
    entries: list[LexiconEntry] = []
    for headword, body in payload.items():
        if len(headword) < _MIN_HEADWORD_LENGTH:
            raise ExpansionLexiconError(
                f"headword {headword!r} is shorter than {_MIN_HEADWORD_LENGTH} "
                "characters; single characters would expand almost every query"
            )
        if not isinstance(body, dict) or set(body) != {"variants"}:
            raise ExpansionLexiconError(
                f"entry {headword!r} must contain exactly a variants list"
            )
        raw_variants = body["variants"]
        if not isinstance(raw_variants, list) or not raw_variants:
            raise ExpansionLexiconError(f"entry {headword!r} needs at least one variant")
        variants = tuple(
            _validate_variant(headword, variant) for variant in raw_variants
        )
        terms = [variant.term for variant in variants]
        if len(set(terms)) != len(terms):
            raise ExpansionLexiconError(f"entry {headword!r} lists a duplicate variant")
        entries.append(LexiconEntry(headword, variants))
    return ExpansionLexicon(version.strip(), sha256, tuple(entries))


def expand_query(query: str, lexicon: ExpansionLexicon) -> QueryExpansion | None:
    """Reviewed expansion for a query, or None when no headword applies.

    Directional by construction: a headword expands to its variants; a query
    containing only a variant never expands back to the headword.
    """
    matches = tuple(
        entry for entry in lexicon.entries if entry.headword in query
    )
    if not matches:
        return None
    return QueryExpansion(lexicon.version, lexicon.sha256, matches)
