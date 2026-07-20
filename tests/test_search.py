from __future__ import annotations

import unittest
import json
from uuid import uuid4

from tests.coherent_search_support import manifest
from wic_history.coherent_search import project_coherent_units
from wic_history.coherent_search import CoherentProjectionError
from wic_history.search_manifest import (
    CoherentProjectionPins,
    load_coherent_projection_manifest,
)
from wic_history.search import (
    REGION_PROJECTION_SQL,
    build_parser,
    main,
    region_document,
    region_index_body,
)


class SearchProjectionTests(unittest.TestCase):
    def test_coherent_manifest_rejects_malformed_pin_before_database_access(self):
        with self.assertRaisesRegex(CoherentProjectionError, "lowercase SHA-256"):
            load_coherent_projection_manifest(
                "postgresql://unused",
                CoherentProjectionPins("model", "revision", "bad", "e" * 64),
            )

    def test_cli_defaults_to_region_and_accepts_coherent_unit(self):
        default = build_parser().parse_args(["query", "女學生"])
        coherent = build_parser().parse_args(
            [
                "query",
                "女子教育",
                "--unit",
                "reviewed_coherent_unit",
                "--configuration-sha256",
                "f" * 64,
            ]
        )

        self.assertEqual(default.unit, "region")
        self.assertEqual(coherent.unit, "reviewed_coherent_unit")

    def test_coherent_projection_cli_requires_pinned_snapshot_inputs(self):
        args = build_parser().parse_args(
            [
                "project",
                "--unit",
                "reviewed_coherent_unit",
                "--model",
                "BAAI/bge-m3",
                "--revision",
                "revision",
                "--configuration-sha256",
                "f" * 64,
                "--snapshot-sha256",
                "e" * 64,
            ]
        )

        self.assertEqual(args.snapshot_sha256, "e" * 64)

    def test_coherent_projection_cli_reports_exact_missing_pins_error(self):
        with self.assertRaises(SystemExit) as raised:
            main(
                [
                    "project",
                    "--database-url",
                    "postgresql://unused",
                    "--unit",
                    "reviewed_coherent_unit",
                ]
            )

        self.assertEqual(
            str(raised.exception),
            "coherent projection requires --model, --revision, "
            "--configuration-sha256, and --snapshot-sha256",
        )

    def test_coherent_dense_query_cli_reports_exact_missing_pins_error(self):
        with self.assertRaises(SystemExit) as raised:
            main(
                [
                    "query",
                    "女子教育",
                    "--unit",
                    "reviewed_coherent_unit",
                    "--mode",
                    "dense",
                ]
            )

        self.assertEqual(
            str(raised.exception),
            "coherent dense/hybrid query requires --model, --revision, "
            "and --configuration-sha256",
        )

    def test_mapping_has_cjk_text_and_versioned_dense_vector(self):
        properties = region_index_body()["mappings"]["properties"]
        self.assertEqual(properties["raw_text"]["analyzer"], "cjk")
        self.assertEqual(properties["embedding"]["dimension"], 1024)
        self.assertEqual(region_index_body()["mappings"]["dynamic"], "strict")

    def test_document_keeps_citation_and_provenance(self):
        region_id = uuid4()
        row = {
            "region_id": region_id,
            "page_id": uuid4(),
            "run_id": uuid4(),
            "source_uri": "s3://bucket/volume.pdf",
            "source_sha256": "a" * 64,
            "derivative_id": uuid4(),
            "source_image_uri": "s3://bucket/page.png",
            "source_image_sha256": "b" * 64,
            "evidence_tier": "historian_selected_gold",
            "ocr_selection_basis": "historian_approved",
            "volume_number": 3,
            "publication_year": 1874,
            "page_number": 12,
            "reading_order": 4,
            "region_kind": "text",
            "raw_text": "女子學校",
            "normalized_text": "女子學校",
            "confidence": 0.9,
            "language": "zh-Hant",
            "direction": "vertical",
            "polygon": {"points": [{"x": 1, "y": 2}, {"x": 3, "y": 2}, {"x": 3, "y": 4}]},
            "page_warnings": [],
            "ocr_model": "fixture",
            "ocr_model_revision": "1",
            "embedding_model": None,
            "embedding_model_revision": None,
            "embedding_text": None,
            "entity_ids": [],
            "claim_ids": [],
        }
        document = region_document(row, "2026-01-01T00:00:00Z")
        self.assertEqual(document["region_id"], str(region_id))
        self.assertEqual(document["source_uri"], "s3://bucket/volume.pdf")
        self.assertEqual(document["polygon"], row["polygon"])
        self.assertEqual(document["derivative_id"], str(row["derivative_id"]))

    def test_projection_includes_only_the_active_ocr_selection(self):
        self.assertIn("JOIN evidence.page_ocr_selection", REGION_PROJECTION_SQL)
        self.assertIn("selection.superseded_at IS NULL", REGION_PROJECTION_SQL)
        self.assertIn("JOIN evidence.ocr_run_input", REGION_PROJECTION_SQL)


if __name__ == "__main__":
    unittest.main()


def test_cli_default_and_coherent_query_use_wire_fake(
    search_server: str, capsys
) -> None:
    assert main(["--opensearch-url", search_server, "query", "女子教育"]) == 0
    region = json.loads(capsys.readouterr().out)
    _ = project_coherent_units(search_server, manifest())

    assert (
        main(
            [
                "--opensearch-url",
                search_server,
                "query",
                "女子教育",
                "--unit",
                "reviewed_coherent_unit",
            ]
        )
        == 0
    )
    coherent = json.loads(capsys.readouterr().out)

    assert region["schema_version"] == "1.0"
    assert coherent["schema_version"] == "1.1"
