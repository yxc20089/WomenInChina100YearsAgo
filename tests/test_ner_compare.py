from __future__ import annotations

import unittest
from uuid import UUID, uuid4

from wic_history.evidence import EntityMentionCandidate, EntityType, SourcePointer
from wic_history.ner_compare import mention_key


class NERCompareTests(unittest.TestCase):
    def test_mention_key_is_grounded_in_region_offsets_and_type(self):
        region_id = UUID("00000000-0000-0000-0000-000000000001")
        mention = EntityMentionCandidate(
            entity_type=EntityType.PERSON,
            text="王氏",
            source=SourcePointer(
                source_uri="s3://example/volume.pdf",
                page_number=1,
                region_id=region_id,
                text_start=2,
                text_end=4,
            ),
            confidence=0.5,
            run_id=uuid4(),
        )
        self.assertEqual(mention_key(mention), (str(region_id), 2, 4, "person"))
