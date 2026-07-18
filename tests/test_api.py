from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from wic_history.api import build_parser, create_app, scenario_context
from wic_history.evidence import RetrievalMode, RetrievalResponse
from wic_history.review_workflow import MentionQueueResponse
from wic_history.insights import EvidenceCounts, InsightReport


class APITests(unittest.TestCase):
    def test_cli_accepts_documented_bind_arguments(self):
        args = build_parser().parse_args(["--host", "127.0.0.1", "--port", "9000"])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 9000)

    def test_scenario_context_does_not_invent_evidence_items(self):
        response = RetrievalResponse(query="女學生", mode=RetrievalMode.LEXICAL, hits=[])
        bundle = scenario_context(response)
        self.assertEqual(bundle.evidence_items, [])
        self.assertTrue(any("No reviewed claims" in warning for warning in bundle.warnings))

    def test_scene_endpoint_abstains_before_loading_generator(self):
        response = RetrievalResponse(query="女學生", mode=RetrievalMode.LEXICAL, hits=[])
        with patch("wic_history.api.lexical_search", return_value=response):
            app = create_app(
                generator_factory=lambda: self.fail("generator should not load without reviewed claims")
            )
            result = TestClient(app).post(
                "/api/generate",
                json={"query": "女學生", "mode": "lexical", "task": "reconstructed_scene"},
            )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "abstained")

    def test_lexical_endpoint_does_not_load_embedding_model(self):
        response = RetrievalResponse(query="女學生", mode=RetrievalMode.LEXICAL, hits=[])
        with patch("wic_history.api.lexical_search", return_value=response) as search:
            app = create_app(embedder_factory=lambda: self.fail("embedder should not load"))
            client = TestClient(app)
            result = client.post(
                "/api/search", json={"query": "女學生", "mode": "lexical", "limit": 3}
            )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["query"], "女學生")
        search.assert_called_once()

    def test_rejects_reversed_year_range(self):
        app = create_app()
        client = TestClient(app)
        response = client.post(
            "/api/search",
            json={"query": "女學生", "mode": "lexical", "year_start": 1930, "year_end": 1920},
        )
        self.assertEqual(response.status_code, 422)

    def test_review_queue_is_exposed_without_mutation(self):
        queue = MentionQueueResponse(
            status="candidate", total=0, offset=0, limit=25, items=[]
        )
        with patch("wic_history.api.list_mention_queue", return_value=queue) as loader:
            app = create_app(database_url="postgresql://example")
            response = TestClient(app).get("/api/review/mentions")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"], [])
        loader.assert_called_once()

    def test_entity_creation_request_requires_name(self):
        app = create_app(database_url="postgresql://example")
        response = TestClient(app).post(
            "/api/review/mentions/00000000-0000-0000-0000-000000000001/entity-resolution",
            json={
                "selected_link_candidate_id": "00000000-0000-0000-0000-000000000002",
                "action": "create_new",
                "reviewer": "historian-a",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_reviewed_insight_report_is_exposed(self):
        report = InsightReport(
            generated_at="2026-07-18T00:00:00Z",
            evidence_counts=EvidenceCounts(),
            items=[],
            warnings=["no reviewed data"],
        )
        with patch("wic_history.api.build_insight_report", return_value=report) as builder:
            app = create_app(database_url="postgresql://example")
            response = TestClient(app).get("/api/insights")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["evidence_counts"]["reviewed_entities"], 0)
        builder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
