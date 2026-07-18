"""Qualify a pinned tokenizer for evidence-grade Unicode character offsets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from pydantic import Field, model_validator

from .evidence import StrictModel
from .ner_adapters.base import canonical_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _full_commit(value: str) -> bool:
    return len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


class TokenizerProbeSpan(StrictModel):
    label: str = Field(min_length=1, max_length=200)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_probe(self) -> "TokenizerProbeSpan":
        if self.end <= self.start:
            raise ValueError("tokenizer probe spans must be non-empty")
        return self


class TokenizerFixtureCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,99}$")
    description: str = Field(min_length=1, max_length=1000)
    text: str = Field(min_length=1)
    probes: list[TokenizerProbeSpan] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case(self) -> "TokenizerFixtureCase":
        keys = set()
        for probe in self.probes:
            if probe.end > len(self.text):
                raise ValueError("tokenizer probe offsets are outside the fixture text")
            if self.text[probe.start : probe.end] != probe.text:
                raise ValueError("tokenizer probe text disagrees with exact offsets")
            key = (probe.start, probe.end, probe.label)
            if key in keys:
                raise ValueError("tokenizer fixture contains a duplicate probe")
            keys.add(key)
        return self


class TokenizerQualificationFixture(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    fixture_id: str = Field(min_length=1, max_length=200)
    created_at: datetime
    cases: list[TokenizerFixtureCase] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_fixture(self) -> "TokenizerQualificationFixture":
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("tokenizer fixture case IDs must be unique")
        return self


class TokenizerFileRecord(StrictModel):
    path: str = Field(min_length=1, max_length=1000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class TokenObservation(StrictModel):
    token_index: int = Field(ge=0)
    token_id: int = Field(ge=0)
    token: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    surface: str
    is_unknown: bool
    is_virtual_prefix: bool = False


class ProbeObservation(StrictModel):
    label: str
    expected_start: int = Field(ge=0)
    expected_end: int = Field(ge=0)
    expected_text: str
    token_indices: list[int]
    recovered_start: int | None = Field(default=None, ge=0)
    recovered_end: int | None = Field(default=None, ge=0)
    recovered_text: str | None = None
    exact: bool


class TokenizerCaseResult(StrictModel):
    case_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization_changed: bool
    input_characters: int = Field(ge=1)
    tokens: list[TokenObservation]
    probes: list[ProbeObservation]
    uncovered_non_whitespace_indices: list[int]
    multiply_covered_indices: list[int]
    unknown_token_indices: list[int]
    virtual_prefix_token_indices: list[int]
    failures: list[str]
    passed: bool

    @model_validator(mode="after")
    def validate_result(self) -> "TokenizerCaseResult":
        if self.passed != (not self.failures):
            raise ValueError("tokenizer case pass state disagrees with failures")
        if self.unknown_token_indices != [
            token.token_index for token in self.tokens if token.is_unknown
        ]:
            raise ValueError(
                "tokenizer unknown-token indices disagree with observations"
            )
        virtual_prefix_indices = [
            token.token_index for token in self.tokens if token.is_virtual_prefix
        ]
        if self.virtual_prefix_token_indices != virtual_prefix_indices:
            raise ValueError(
                "tokenizer virtual-prefix indices disagree with observations"
            )
        if len(virtual_prefix_indices) > 1:
            raise ValueError("a tokenizer case may contain at most one virtual prefix")
        if virtual_prefix_indices:
            virtual = self.tokens[virtual_prefix_indices[0]]
            if (
                virtual.token_index != 0
                or virtual.token != "▁"
                or virtual.start != 0
                or virtual.end != 1
                or len(self.tokens) < 2
                or self.tokens[1].token_index != 1
                or self.tokens[1].start != virtual.start
                or self.tokens[1].end != virtual.end
                or self.tokens[1].surface != virtual.surface
                or self.tokens[1].is_virtual_prefix
            ):
                raise ValueError("virtual prefix does not satisfy the recorded policy")
        return self


class TokenizerQualificationArtifact(StrictModel):
    schema_version: Literal["1.1"] = "1.1"
    artifact_kind: Literal["ner_tokenizer_offset_qualification"] = (
        "ner_tokenizer_offset_qualification"
    )
    generated_at: datetime
    fixture_id: str
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_name: str = Field(min_length=1, max_length=1000)
    model_revision: str
    code_revision: str
    tokenizer_class: str = Field(min_length=1, max_length=500)
    tokenizer_is_fast: bool
    tokenizer_files: list[TokenizerFileRecord] = Field(min_length=1)
    tokenizer_file_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformers_version: str = Field(min_length=1, max_length=100)
    tokenizers_version: str = Field(min_length=1, max_length=100)
    virtual_prefix_policy: Literal["standalone_sentencepiece_prefix_duplicate_v1"] = (
        "standalone_sentencepiece_prefix_duplicate_v1"
    )
    results: list[TokenizerCaseResult] = Field(min_length=1)
    global_failures: list[str]
    warnings: list[str]
    passed: bool

    @model_validator(mode="after")
    def validate_artifact(self) -> "TokenizerQualificationArtifact":
        if not _full_commit(self.model_revision):
            raise ValueError("tokenizer model revision must be a full lowercase commit")
        if not _full_commit(self.code_revision):
            raise ValueError("tokenizer code revision must be a full lowercase commit")
        paths = [record.path for record in self.tokenizer_files]
        if len(set(paths)) != len(paths):
            raise ValueError("tokenizer file manifest paths must be unique")
        expected_manifest_sha256 = canonical_sha256(
            [record.model_dump(mode="json") for record in self.tokenizer_files]
        )
        if self.tokenizer_file_manifest_sha256 != expected_manifest_sha256:
            raise ValueError("tokenizer file manifest hash disagrees with its records")
        expected_passed = (
            self.tokenizer_is_fast
            and not self.global_failures
            and all(result.passed for result in self.results)
        )
        if self.passed != expected_passed:
            raise ValueError(
                "tokenizer qualification pass state disagrees with results"
            )
        return self


class OffsetTokenizer(Protocol):
    is_fast: bool
    unk_token: str | None
    backend_tokenizer: Any

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]: ...

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]: ...


def _normalized_text(tokenizer: OffsetTokenizer, text: str) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    normalizer = getattr(backend, "normalizer", None)
    if normalizer is None:
        return text
    return normalizer.normalize_str(text)


def _is_virtual_prefix_token(
    index: int,
    token: str,
    offsets: list[Any],
    token_strings: list[str],
) -> bool:
    """Recognize one audited SentencePiece prefix marker with no unique source span."""
    if index != 0 or token != "▁" or len(offsets) < 2 or len(token_strings) < 2:
        return False
    current, following = offsets[0], offsets[1]
    if not (
        isinstance(current, (list, tuple))
        and isinstance(following, (list, tuple))
        and len(current) == 2
        and len(following) == 2
        and all(isinstance(value, int) for value in (*current, *following))
    ):
        return False
    return (
        tuple(current) == (0, 1)
        and tuple(following) == tuple(current)
        and token_strings[1] != "▁"
    )


def _case_result(
    case: TokenizerFixtureCase, tokenizer: OffsetTokenizer
) -> TokenizerCaseResult:
    failures = []
    normalized = _normalized_text(tokenizer, case.text)
    if normalized != case.text:
        failures.append("tokenizer normalization changes source Unicode")
    try:
        encoded = tokenizer(
            case.text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        token_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        token_strings = list(tokenizer.convert_ids_to_tokens(token_ids))
    except Exception as exc:
        failures.append(f"tokenizer did not return offset mappings: {exc}")
        token_ids, offsets, token_strings = [], [], []
    if not (len(token_ids) == len(offsets) == len(token_strings)):
        failures.append("token IDs, strings and offsets have different lengths")
    observations = []
    coverage = [0] * len(case.text)
    previous_end = 0
    for index, (token_id, token, offset) in enumerate(
        zip(token_ids, token_strings, offsets, strict=False)
    ):
        if (
            not isinstance(offset, (list, tuple))
            or len(offset) != 2
            or not all(isinstance(value, int) for value in offset)
        ):
            failures.append(f"token {index} has a malformed offset")
            continue
        start, end = offset
        if not 0 <= start < end <= len(case.text):
            failures.append(f"token {index} has an empty or out-of-range offset")
            continue
        is_virtual_prefix = _is_virtual_prefix_token(
            index, str(token), offsets, token_strings
        )
        if not is_virtual_prefix and start < previous_end:
            failures.append(f"token {index} overlaps or precedes the prior token")
        if not is_virtual_prefix:
            previous_end = max(previous_end, end)
            for character_index in range(start, end):
                coverage[character_index] += 1
        observations.append(
            TokenObservation(
                token_index=index,
                token_id=int(token_id),
                token=str(token),
                start=start,
                end=end,
                surface=case.text[start:end],
                is_unknown=(
                    tokenizer.unk_token is not None and token == tokenizer.unk_token
                ),
                is_virtual_prefix=is_virtual_prefix,
            )
        )
    uncovered = [
        index
        for index, count in enumerate(coverage)
        if count == 0 and not case.text[index].isspace()
    ]
    multiply_covered = [index for index, count in enumerate(coverage) if count > 1]
    if uncovered:
        failures.append("token offsets do not cover every non-whitespace character")
    if multiply_covered:
        failures.append("token offsets cover one or more characters multiple times")

    probes = []
    for probe in case.probes:
        matching = [
            token
            for token in observations
            if not token.is_virtual_prefix
            and max(token.start, probe.start) < min(token.end, probe.end)
        ]
        token_indices = [token.token_index for token in matching]
        if matching:
            recovered_start = min(token.start for token in matching)
            recovered_end = max(token.end for token in matching)
            recovered_text = case.text[recovered_start:recovered_end]
            contiguous = token_indices == list(
                range(token_indices[0], token_indices[-1] + 1)
            )
            exact = (
                contiguous
                and recovered_start == probe.start
                and recovered_end == probe.end
                and recovered_text == probe.text
            )
        else:
            recovered_start = recovered_end = None
            recovered_text = None
            exact = False
        if not exact:
            failures.append(f"probe '{probe.label}' does not round-trip exactly")
        probes.append(
            ProbeObservation(
                label=probe.label,
                expected_start=probe.start,
                expected_end=probe.end,
                expected_text=probe.text,
                token_indices=token_indices,
                recovered_start=recovered_start,
                recovered_end=recovered_end,
                recovered_text=recovered_text,
                exact=exact,
            )
        )
    failures = list(dict.fromkeys(failures))
    return TokenizerCaseResult(
        case_id=case.case_id,
        input_sha256=hashlib.sha256(case.text.encode("utf-8")).hexdigest(),
        normalized_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        normalization_changed=normalized != case.text,
        input_characters=len(case.text),
        tokens=observations,
        probes=probes,
        uncovered_non_whitespace_indices=uncovered,
        multiply_covered_indices=multiply_covered,
        unknown_token_indices=[
            token.token_index for token in observations if token.is_unknown
        ],
        virtual_prefix_token_indices=[
            token.token_index for token in observations if token.is_virtual_prefix
        ],
        failures=failures,
        passed=not failures,
    )


def qualify_tokenizer(
    fixture: TokenizerQualificationFixture,
    tokenizer: OffsetTokenizer,
    *,
    fixture_sha256: str,
    model_name: str,
    model_revision: str,
    code_revision: str,
    tokenizer_files: list[TokenizerFileRecord],
    transformers_version: str,
    tokenizers_version: str,
    generated_at: datetime | None = None,
) -> TokenizerQualificationArtifact:
    global_failures = []
    if not tokenizer.is_fast:
        global_failures.append("tokenizer is not a fast tokenizer with source offsets")
    results = [_case_result(case, tokenizer) for case in fixture.cases]
    unknown_count = sum(len(result.unknown_token_indices) for result in results)
    virtual_prefix_count = sum(
        len(result.virtual_prefix_token_indices) for result in results
    )
    warnings = [
        "This artifact qualifies tokenizer offset integrity only; it is not NER accuracy evidence.",
        "Passing does not authorize model selection, entity acceptance, or graph promotion.",
    ]
    if unknown_count:
        warnings.append(
            f"Tokenizer emitted {unknown_count} unknown tokens; offsets may still pass, but representation quality requires benchmark evaluation."
        )
    if virtual_prefix_count:
        warnings.append(
            f"Tokenizer emitted {virtual_prefix_count} standalone SentencePiece prefix markers with duplicated first-character offsets. They were retained in the audit record but excluded from character coverage and span alignment under standalone_sentencepiece_prefix_duplicate_v1; downstream adapters must apply the same policy."
        )
    manifest_sha256 = canonical_sha256(
        [record.model_dump(mode="json") for record in tokenizer_files]
    )
    return TokenizerQualificationArtifact(
        generated_at=generated_at or datetime.now(timezone.utc),
        fixture_id=fixture.fixture_id,
        fixture_sha256=fixture_sha256,
        model_name=model_name,
        model_revision=model_revision,
        code_revision=code_revision,
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_is_fast=bool(tokenizer.is_fast),
        tokenizer_files=tokenizer_files,
        tokenizer_file_manifest_sha256=manifest_sha256,
        transformers_version=transformers_version,
        tokenizers_version=tokenizers_version,
        results=results,
        global_failures=global_failures,
        warnings=warnings,
        passed=not global_failures and all(result.passed for result in results),
    )


def load_pinned_tokenizer(
    model_name: str, model_revision: str, *, local_files_only: bool = False
) -> tuple[OffsetTokenizer, list[TokenizerFileRecord]]:
    if not _full_commit(model_revision):
        raise ValueError("tokenizer model revision must be a full lowercase commit")
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Tokenizer qualification requires the project ner optional dependencies"
        ) from exc
    snapshot = Path(
        snapshot_download(
            repo_id=model_name,
            revision=model_revision,
            allow_patterns=[
                "config.json",
                "tokenizer*",
                "*.model",
                "vocab*",
                "merges.txt",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
            ignore_patterns=["*.bin", "*.safetensors"],
            local_files_only=local_files_only,
        )
    )
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    if not files:
        raise ValueError("pinned tokenizer snapshot contains no files")
    records = [
        TokenizerFileRecord(
            path=path.relative_to(snapshot).as_posix(),
            sha256=_sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in files
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    return tokenizer, records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixture_bytes = args.fixture.read_bytes()
    fixture = TokenizerQualificationFixture.model_validate_json(fixture_bytes)
    tokenizer, file_records = load_pinned_tokenizer(
        args.model,
        args.revision,
        local_files_only=args.local_files_only,
    )
    artifact = qualify_tokenizer(
        fixture,
        tokenizer,
        fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
        model_name=args.model,
        model_revision=args.revision,
        code_revision=args.code_revision,
        tokenizer_files=file_records,
        transformers_version=importlib.metadata.version("transformers"),
        tokenizers_version=importlib.metadata.version("tokenizers"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": artifact.passed,
                "cases": len(artifact.results),
                "failed_cases": sum(not result.passed for result in artifact.results),
                "tokenizer_file_manifest_sha256": artifact.tokenizer_file_manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0 if artifact.passed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
