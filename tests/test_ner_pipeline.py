from __future__ import annotations

import unittest

from wic_history.evidence import EntityType
from wic_history.ner_pipeline import RulePredictor, SpanCandidate, merge_candidates


class NERPipelineTests(unittest.TestCase):
    def test_rules_return_exact_offsets(self):
        text = "上海女子學校女學生讀申報"
        spans = RulePredictor().predict([text])[0]
        for span in spans:
            self.assertEqual(text[span.start : span.end], span.text)
        self.assertTrue(any(span.entity_type == EntityType.SCHOOL for span in spans))
        self.assertTrue(any(span.text == "女學生" for span in spans))
        self.assertTrue(any(span.text == "申報" for span in spans))

    def test_merge_retains_highest_score_for_same_span_type(self):
        lower = SpanCandidate(0, 2, "申報", EntityType.PUBLICATION, 0.5, "one")
        higher = SpanCandidate(0, 2, "申報", EntityType.PUBLICATION, 0.9, "two")
        merged = merge_candidates([[lower]], [[higher]])
        self.assertEqual(merged[0], [higher])


if __name__ == "__main__":
    unittest.main()
