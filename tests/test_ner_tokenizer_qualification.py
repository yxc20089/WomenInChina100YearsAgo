from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from wic_history.ner_tokenizer_qualification import (
    TokenizerFileRecord,
    TokenizerFixtureCase,
    TokenizerProbeSpan,
    TokenizerQualificationArtifact,
    TokenizerQualificationFixture,
    load_pinned_tokenizer,
    qualify_tokenizer,
)


NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)


class IdentityNormalizer:
    @staticmethod
    def normalize_str(text: str) -> str:
        return text


class ChangingNormalizer:
    @staticmethod
    def normalize_str(text: str) -> str:
        return text.replace("士", "土")


class Backend:
    def __init__(self, normalizer):
        self.normalizer = normalizer


class CharacterTokenizer:
    is_fast = True
    unk_token = "[UNK]"

    def __init__(self, normalizer=IdentityNormalizer()):
        self.backend_tokenizer = Backend(normalizer)
        self._tokens: dict[int, str] = {}

    def __call__(self, text: str, **kwargs):
        input_ids = []
        offsets = []
        for index, character in enumerate(text):
            if character.isspace():
                continue
            token_id = index + 1
            self._tokens[token_id] = character
            input_ids.append(token_id)
            offsets.append((index, index + 1))
        return {"input_ids": input_ids, "offset_mapping": offsets}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self._tokens[token_id] for token_id in token_ids]


def fixture() -> TokenizerQualificationFixture:
    text = "宋女士入學\n上海女子學校。"
    return TokenizerQualificationFixture(
        fixture_id="traditional-chinese-offset-fixture-v1",
        created_at=NOW,
        cases=[
            TokenizerFixtureCase(
                case_id="traditional-news",
                description="Traditional Chinese without word spaces and with a line break.",
                text=text,
                probes=[
                    TokenizerProbeSpan(label="person", start=0, end=3, text="宋女士"),
                    TokenizerProbeSpan(label="school", start=6, end=12, text="上海女子學校"),
                ],
            )
        ],
    )


def file_records() -> list[TokenizerFileRecord]:
    return [
        TokenizerFileRecord(path="tokenizer.json", sha256="1" * 64, size_bytes=10)
    ]


class TokenizerQualificationTests(unittest.TestCase):
    def test_character_tokenizer_preserves_exact_unicode_offsets(self):
        artifact = qualify_tokenizer(
            fixture(),
            CharacterTokenizer(),
            fixture_sha256="2" * 64,
            model_name="example/tokenizer",
            model_revision="a" * 40,
            code_revision="b" * 40,
            tokenizer_files=file_records(),
            transformers_version="5.5.3",
            tokenizers_version="0.22.1",
            generated_at=NOW,
        )
        self.assertTrue(artifact.passed)
        self.assertEqual(artifact.results[0].uncovered_non_whitespace_indices, [])
        self.assertTrue(all(probe.exact for probe in artifact.results[0].probes))
        self.assertEqual(artifact.results[0].tokens[0].surface, "宋")

    def test_normalization_drift_is_a_hard_failure(self):
        artifact = qualify_tokenizer(
            fixture(),
            CharacterTokenizer(ChangingNormalizer()),
            fixture_sha256="2" * 64,
            model_name="example/tokenizer",
            model_revision="a" * 40,
            code_revision="b" * 40,
            tokenizer_files=file_records(),
            transformers_version="5.5.3",
            tokenizers_version="0.22.1",
            generated_at=NOW,
        )
        self.assertFalse(artifact.passed)
        self.assertTrue(artifact.results[0].normalization_changed)
        self.assertIn("normalization", artifact.results[0].failures[0])

    def test_fixture_and_artifact_states_cannot_be_forged(self):
        data = fixture().model_dump(mode="json")
        data["cases"][0]["probes"][0]["text"] = "錯字"
        with self.assertRaisesRegex(ValidationError, "exact offsets"):
            TokenizerQualificationFixture.model_validate(data)

        artifact = qualify_tokenizer(
            fixture(),
            CharacterTokenizer(ChangingNormalizer()),
            fixture_sha256="2" * 64,
            model_name="example/tokenizer",
            model_revision="a" * 40,
            code_revision="b" * 40,
            tokenizer_files=file_records(),
            transformers_version="5.5.3",
            tokenizers_version="0.22.1",
            generated_at=NOW,
        )
        changed = artifact.model_dump(mode="json")
        changed["passed"] = True
        with self.assertRaisesRegex(ValidationError, "pass state"):
            TokenizerQualificationArtifact.model_validate(changed)

    def test_moving_model_revision_is_rejected_before_download(self):
        with self.assertRaisesRegex(ValueError, "full lowercase commit"):
            load_pinned_tokenizer("example/tokenizer", "main")


if __name__ == "__main__":
    unittest.main()
