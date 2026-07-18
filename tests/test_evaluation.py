from __future__ import annotations

import unittest
from uuid import UUID

from wic_history.evaluation import EvaluationQuestion, QuestionCategory, score_question
from wic_history.evidence import Polygon, RetrievalHit, RetrievalMode, RetrievalResponse, SourcePointer


class EvaluationTests(unittest.TestCase):
    def test_scores_recall_rank_and_citation_pointer(self):
        expected = UUID("00000000-0000-0000-0000-000000000001")
        other = UUID("00000000-0000-0000-0000-000000000002")
        question = EvaluationQuestion(
            question_id="q1",
            query="女學生",
            category=QuestionCategory.EXACT_LOOKUP,
            expected_region_ids=[expected],
            author="historian-a",
        )
        polygon = Polygon(points=[{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}])
        hits = [
            RetrievalHit(
                rank=1,
                score=1,
                source=SourcePointer(source_uri="s3://x", page_number=1, region_id=other, polygon=polygon),
                text="other",
            ),
            RetrievalHit(
                rank=2,
                score=0.5,
                source=SourcePointer(source_uri="s3://x", page_number=1, region_id=expected, polygon=polygon),
                text="expected",
            ),
        ]
        response = RetrievalResponse(query="女學生", mode=RetrievalMode.LEXICAL, hits=hits)

        result = score_question(question, response)

        self.assertEqual(result.recall_at_k, 1.0)
        self.assertEqual(result.reciprocal_rank, 0.5)
        self.assertEqual(result.citation_pointer_rate, 1.0)
        self.assertEqual(result.derivative_pointer_rate, 0.0)
        self.assertEqual(result.historian_gold_evidence_rate, 0.0)

    def test_answerable_question_requires_expected_evidence(self):
        with self.assertRaises(ValueError):
            EvaluationQuestion(
                question_id="q1",
                query="女學生",
                category=QuestionCategory.EXACT_LOOKUP,
                author="historian-a",
            )
