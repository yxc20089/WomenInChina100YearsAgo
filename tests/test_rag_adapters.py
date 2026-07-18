from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from wic_history.rag_adapters import LightRAGClient, prepare_graphrag_workspace


class RAGAdapterTests(unittest.TestCase):
    def test_prepare_graphrag_copies_the_validated_shared_text(self):
        source = Path("artifacts/rag-smoke")
        with tempfile.TemporaryDirectory() as directory:
            result = prepare_graphrag_workspace(source, Path(directory))
            copied = list(Path(directory, "input").glob("*.txt"))
            self.assertEqual(result.documents, 1)
            self.assertEqual(len(copied), 1)
            self.assertIn("graphrag==3.1.1", " ".join(result.next_commands))

    def test_lightrag_insert_uses_document_ids_as_file_sources(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"status": "success", "track_id": "track-1"}
        ).encode()
        with patch("wic_history.rag_adapters.urlopen", return_value=response) as request:
            result = LightRAGClient(
                "http://127.0.0.1:9621", api_key="test-key"
            ).insert_documents([{"id": "page-1", "text": "女學生"}])
        sent = request.call_args.args[0]
        payload = json.loads(sent.data)
        self.assertEqual(result["track_id"], "track-1")
        self.assertEqual(payload["file_sources"], ["wic/page-1.txt"])
        self.assertEqual(sent.headers["X-api-key"], "test-key")
