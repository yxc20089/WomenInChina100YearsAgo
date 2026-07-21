from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wic_history.query_expansion import (
    DEFAULT_LEXICON_PATH,
    ExpansionLexiconError,
    expand_query,
    load_expansion_lexicon,
)
from wic_history.search import _expansion_explanation, _lexical_query_body


def _write(directory: str, content: str) -> Path:
    path = Path(directory) / "historical-synonyms.toml"
    _ = path.write_text(content, encoding="utf-8")
    return path


VALID = """
version = "test-1"

["女士"]
variants = [
  { term = "淑女", weight = 0.5, note = "gentry lady" },
  { term = "士女", weight = 0.4 },
]
"""


class LexiconValidationTests(unittest.TestCase):
    def test_reviewed_lexicon_in_config_loads_and_hashes(self):
        lexicon = load_expansion_lexicon(DEFAULT_LEXICON_PATH)
        self.assertTrue(lexicon.version)
        self.assertRegex(lexicon.sha256, "^[0-9a-f]{64}$")
        headwords = [entry.headword for entry in lexicon.entries]
        self.assertIn("女士", headwords)
        for entry in lexicon.entries:
            for variant in entry.variants:
                self.assertLess(variant.weight, 1.0)
                self.assertGreater(variant.weight, 0.0)

    def test_valid_lexicon_parses(self):
        with TemporaryDirectory() as directory:
            lexicon = load_expansion_lexicon(_write(directory, VALID))
        self.assertEqual(lexicon.version, "test-1")
        self.assertEqual(lexicon.entries[0].headword, "女士")
        self.assertEqual(lexicon.entries[0].variants[0].term, "淑女")

    def test_missing_version_fails_closed(self):
        content = '["女士"]\nvariants = [ { term = "淑女", weight = 0.5 } ]\n'
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ExpansionLexiconError, "version"):
                _ = load_expansion_lexicon(_write(directory, content))

    def test_weight_at_or_above_one_fails_closed(self):
        content = (
            'version = "1"\n["女士"]\n'
            'variants = [ { term = "淑女", weight = 1.0 } ]\n'
        )
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ExpansionLexiconError, "between 0 and 1"):
                _ = load_expansion_lexicon(_write(directory, content))

    def test_self_reference_fails_closed(self):
        content = (
            'version = "1"\n["女士"]\n'
            'variants = [ { term = "女士", weight = 0.5 } ]\n'
        )
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ExpansionLexiconError, "itself"):
                _ = load_expansion_lexicon(_write(directory, content))

    def test_duplicate_variant_fails_closed(self):
        content = (
            'version = "1"\n["女士"]\nvariants = [\n'
            '  { term = "淑女", weight = 0.5 },\n'
            '  { term = "淑女", weight = 0.3 },\n]\n'
        )
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ExpansionLexiconError, "duplicate"):
                _ = load_expansion_lexicon(_write(directory, content))

    def test_single_character_headword_fails_closed(self):
        content = 'version = "1"\n["女"]\nvariants = [ { term = "淑女", weight = 0.5 } ]\n'
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ExpansionLexiconError, "shorter"):
                _ = load_expansion_lexicon(_write(directory, content))

    def test_unknown_variant_key_fails_closed(self):
        content = (
            'version = "1"\n["女士"]\n'
            'variants = [ { term = "淑女", weight = 0.5, source = "llm" } ]\n'
        )
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ExpansionLexiconError, "unknown keys"):
                _ = load_expansion_lexicon(_write(directory, content))


class ExpansionTests(unittest.TestCase):
    def _lexicon(self):
        with TemporaryDirectory() as directory:
            return load_expansion_lexicon(_write(directory, VALID))

    def test_headword_in_query_expands(self):
        expansion = expand_query("女士", self._lexicon())
        assert expansion is not None
        self.assertEqual(expansion.lexicon_version, "test-1")
        self.assertEqual(
            [variant.term for variant in expansion.matches[0].variants],
            ["淑女", "士女"],
        )

    def test_expansion_is_directional(self):
        # a variant never expands back to its headword
        self.assertIsNone(expand_query("淑女", self._lexicon()))
        self.assertIsNone(expand_query("富紳淑女", self._lexicon()))

    def test_unrelated_query_does_not_expand(self):
        self.assertIsNone(expand_query("女學生", self._lexicon()))


class LexicalQueryBodyTests(unittest.TestCase):
    def test_no_expansion_keeps_single_clause_and_no_named_queries(self):
        body = _lexical_query_body("女學生", 10, [], None)
        clauses = body["query"]["bool"]["should"]
        self.assertEqual(len(clauses), 1)
        self.assertNotIn("_name", clauses[0]["multi_match"])

    def test_expansion_adds_weighted_named_clauses_below_original(self):
        with TemporaryDirectory() as directory:
            lexicon = load_expansion_lexicon(_write(directory, VALID))
        expansion = expand_query("女士", lexicon)
        body = _lexical_query_body("女士", 10, [], expansion)
        clauses = body["query"]["bool"]["should"]
        self.assertEqual(clauses[0]["multi_match"]["query"], "女士")
        self.assertEqual(clauses[0]["multi_match"]["_name"], "query")
        self.assertNotIn("boost", clauses[0]["multi_match"])  # full weight
        expanded = clauses[1:]
        self.assertEqual(
            [clause["multi_match"]["query"] for clause in expanded],
            ["淑女", "士女"],
        )
        for clause in expanded:
            self.assertLess(clause["multi_match"]["boost"], 1.0)
            self.assertTrue(clause["multi_match"]["_name"].startswith("expansion:"))
        self.assertEqual(body["query"]["bool"]["minimum_should_match"], 1)


class ExplanationTests(unittest.TestCase):
    def _expansion(self):
        with TemporaryDirectory() as directory:
            lexicon = load_expansion_lexicon(_write(directory, VALID))
        return expand_query("女士", lexicon)

    def test_synonym_assisted_hit_reports_matched_term_and_lexicon_identity(self):
        item = {"matched_queries": ["expansion:女士->淑女"]}
        explanation = _expansion_explanation(item, self._expansion())
        self.assertEqual(
            explanation["query_expansion"]["matched_terms"], ["女士->淑女"]
        )
        self.assertEqual(
            explanation["query_expansion"]["lexicon_version"], "test-1"
        )
        self.assertRegex(
            explanation["query_expansion"]["lexicon_sha256"], "^[0-9a-f]{64}$"
        )

    def test_exact_match_hit_carries_no_expansion_explanation(self):
        # the hit matched the original query clause only
        item = {"matched_queries": ["query"]}
        self.assertEqual(_expansion_explanation(item, self._expansion()), {})

    def test_no_expansion_keeps_explanation_untouched(self):
        self.assertEqual(_expansion_explanation({"matched_queries": []}, None), {})


if __name__ == "__main__":
    unittest.main()
