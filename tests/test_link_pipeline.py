from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import uuid4

from wic_history.evidence import EntityType
from wic_history.link_pipeline import AuthorityEntity, candidate_links, normalize_name


class LinkPipelineTests(unittest.TestCase):
    def test_normalization_preserves_historical_characters(self):
        self.assertEqual(normalize_name(" 宋・慶齡 "), "宋慶齡")

    def test_exact_candidate_and_nil_are_both_retained(self):
        mention = SimpleNamespace(
            mention_id=uuid4(),
            text="宋慶齡",
            normalized_text="宋慶齡",
            entity_type=EntityType.PERSON,
        )
        entity_id = uuid4()
        catalog = [
            AuthorityEntity(
                entity_id,
                EntityType.PERSON,
                "宋慶齡",
                "宋慶齡",
                "https://example.test/person/1",
                ("宋庆龄",),
            )
        ]
        links = candidate_links(mention, catalog, uuid4())
        self.assertEqual(links[0].entity_id, entity_id)
        self.assertEqual(links[0].score, 1.0)
        self.assertTrue(links[-1].nil_candidate)


if __name__ == "__main__":
    unittest.main()
