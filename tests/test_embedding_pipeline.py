from __future__ import annotations

import unittest
from unittest.mock import patch
from uuid import UUID

from wic_history.article_embedding import ArticleEmbeddingSummary
from wic_history.embedding_pipeline import (
    EMBEDDING_DIMENSION,
    EmbeddingResult,
    _vector_literal,
    build_parser,
    main,
)


class EmbeddingPipelineTests(unittest.TestCase):
    def test_vector_literal_is_pgvector_compatible(self):
        self.assertEqual(_vector_literal([0.25, -0.5]), "[0.25,-0.5]")

    def test_schema_dimension_matches_bge_m3(self):
        self.assertEqual(EMBEDDING_DIMENSION, 1024)

    def test_cli_defaults_to_legacy_region_embedding(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.unit, "region")

    @patch("wic_history.embedding_pipeline.embed_regions")
    def test_main_dispatches_region_without_changing_legacy_arguments(
        self, embed_regions
    ):
        embed_regions.return_value = EmbeddingResult("run", 0, 0, "model", "revision")
        with patch(
            "wic_history.embedding_pipeline.load_pipeline_model_configuration"
        ) as load:
            load.return_value.retrieval.passage_embedding.model_name = "model"
            load.return_value.retrieval.passage_embedding.model_revision = "revision"
            self.assertEqual(main(["--database-url", "postgresql://unused"]), 0)
        embed_regions.assert_called_once_with(
            "postgresql://unused", "model", "revision", 16, None
        )

    def test_main_dispatches_reviewed_revision_target(self):
        revision_id = UUID(int=7)
        with (
            patch(
                "wic_history.embedding_pipeline.load_pipeline_model_configuration"
            ) as load,
            patch("wic_history.article_embedding.embed_reviewed_articles") as embed,
        ):
            load.return_value.retrieval.passage_embedding.model_name = "model"
            load.return_value.retrieval.passage_embedding.model_revision = "revision"
            embed.return_value = ArticleEmbeddingSummary(
                0, 0, 0, (), "model", "revision"
            )
            self.assertEqual(
                main(
                    [
                        "--database-url",
                        "postgresql://unused",
                        "--unit",
                        "reviewed_coherent_unit",
                        "--revision-id",
                        str(revision_id),
                    ]
                ),
                0,
            )
        request = embed.call_args.args[0]
        self.assertEqual(request.revision_id, revision_id)


if __name__ == "__main__":
    unittest.main()
