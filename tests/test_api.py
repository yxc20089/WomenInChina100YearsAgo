from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from wic_history.api import build_parser, create_app, scenario_context
from wic_history.evidence import RetrievalMode, RetrievalResponse


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


if __name__ == "__main__":
    unittest.main()
