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
            "1" * 64,
            uuid4(),
            "2" * 64,
            "historian_selected_gold",
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
        self.assertEqual(claims[0].supporting_quote, "宋女士入學上海女子學校")
        self.assertEqual(claims[0].evidence[0].text_end, len(region.raw_text))
        self.assertEqual(claims[0].evidence[0].source_sha256, "1" * 64)
        self.assertIsNone(claims[0].confidence)

    def test_no_cue_means_no_claim(self):
        region = RegionEvidence(
            uuid4(),
            "某女士與某學校",
            {"points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]},
            "s3://bucket/volume.pdf",
            "1" * 64,
            uuid4(),
            "2" * 64,
            "historian_selected_gold",
            1,
            1872,
            3,
        )
        mentions = [
            ReviewedMention(uuid4(), "person", "某女士", 0, 3),
            ReviewedMention(uuid4(), "school", "某學校", 4, 7),
        ]
        self.assertEqual(extract_region_claims(region, mentions, uuid4()), [])

    def test_cue_must_be_between_the_exact_argument_pair(self):
        region = RegionEvidence(
            uuid4(),
            "甲女士入學乙學校，丙女士丁學校",
            {"points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]},
            "s3://bucket/volume.pdf",
            "1" * 64,
            uuid4(),
            "2" * 64,
            "historian_selected_gold",
            1,
            1925,
            3,
        )
        person_a, school_a, person_b, school_b = (uuid4() for _ in range(4))
        mentions = [
            ReviewedMention(person_a, "person", "甲女士", 0, 3),
            ReviewedMention(school_a, "school", "乙學校", 5, 8),
            ReviewedMention(person_b, "person", "丙女士", 9, 12),
            ReviewedMention(school_b, "school", "丁學校", 12, 15),
        ]
        claims = extract_region_claims(region, mentions, uuid4())
        self.assertEqual(
            [(claim.subject_entity_id, claim.object_entity_id) for claim in claims],
            [(person_a, school_a)],
        )

    def test_reviewed_mentions_must_match_the_cited_ocr(self):
        region = RegionEvidence(
            uuid4(),
            "宋女士入學上海女子學校",
            {"points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]},
            "s3://bucket/volume.pdf",
            "1" * 64,
            uuid4(),
            "2" * 64,
            "historian_selected_gold",
            1,
            1925,
            3,
        )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            extract_region_claims(
                region,
                [ReviewedMention(uuid4(), "person", "錯字", 0, 3)],
                uuid4(),
            )


if __name__ == "__main__":
    unittest.main()
