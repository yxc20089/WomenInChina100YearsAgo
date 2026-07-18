from __future__ import annotations

import unittest

from wic_history.embedding_pipeline import EMBEDDING_DIMENSION, _vector_literal


class EmbeddingPipelineTests(unittest.TestCase):
    def test_vector_literal_is_pgvector_compatible(self):
        self.assertEqual(_vector_literal([0.25, -0.5]), "[0.25,-0.5]")

    def test_schema_dimension_matches_bge_m3(self):
        self.assertEqual(EMBEDDING_DIMENSION, 1024)


if __name__ == "__main__":
    unittest.main()
