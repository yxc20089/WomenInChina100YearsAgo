from __future__ import annotations

import unittest
from uuid import uuid4

from wic_history.relation_pipeline import RegionEvidence, ReviewedMention, extract_region_claims


class RelationPipelineTests(unittest.TestCase):
    def test_grounded_rule_requires_cue_and_reviewed_types(self):
        person_id = uuid4()
        school_id = uuid4()
        region = RegionEvidence(
            uuid4(),
            "宋女士入學上海女子學校",
            {"points": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}]},
            "s3://bucket/volume.pdf",
            1,
            1872,
            3,
        )
        mentions = [
            ReviewedMention(person_id, "person", "宋女士", 0, 3),
            ReviewedMention(school_id, "school", "上海女子學校", 5, 11),
        ]
        claims = extract_region_claims(region, mentions, uuid4())
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].predicate, "attended_school")
        self.assertEqual(claims[0].supporting_quote, region.raw_text)
        self.assertEqual(claims[0].evidence[0].text_end, len(region.raw_text))

    def test_no_cue_means_no_claim(self):
        region = RegionEvidence(
            uuid4(),
            "某女士與某學校",
            {"points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]},
            "s3://bucket/volume.pdf",
            1,
            1872,
            3,
        )
        mentions = [
            ReviewedMention(uuid4(), "person", "某女士", 0, 3),
            ReviewedMention(uuid4(), "school", "某學校", 4, 7),
        ]
        self.assertEqual(extract_region_claims(region, mentions, uuid4()), [])


if __name__ == "__main__":
    unittest.main()
