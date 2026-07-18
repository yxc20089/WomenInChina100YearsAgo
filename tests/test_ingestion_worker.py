from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

from wic_history.evidence import NERArtifact, OCRPageArtifact
from wic_history.ingestion_jobs import DEFAULT_CONFIGURATION
from wic_history.ingestion_worker import (
    PageJobContext,
    _ner_artifact_matches,
    _ocr_artifact_matches,
    _render_manifest_execution,
    resolve_workspace_path,
    stage_output_dir,
)
from wic_history.render_samples import sha256_file


def context(stage: str, configuration: dict | None = None) -> PageJobContext:
    return PageJobContext(
        job_id="00000000-0000-0000-0000-000000000001",
        stage=stage,
        configuration=configuration or DEFAULT_CONFIGURATION[stage],
        source_uri="s3://bucket/prefix/volume.pdf",
        bucket="bucket",
        object_key="prefix/volume.pdf",
        media_type="application/pdf",
        size_bytes=100,
        etag="etag",
        source_sha256="a" * 64,
        integrity_status="ok_fast_checks",
        volume_number=219,
        publication_year=1925,
        page_number=308,
        page_count=599,
        parent_stage=None,
        parent_artifact_uri=None,
        parent_output_sha256=None,
        parent_result=None,
    )


class IngestionWorkerTests(unittest.TestCase):
    def test_artifact_paths_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                resolve_workspace_path(root, "artifacts/page.json"),
                (root / "artifacts/page.json").resolve(),
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_workspace_path(root, "../outside.json")

    def test_stage_output_is_namespaced_by_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = stage_output_dir(context("ocr"), Path(directory))
            self.assertEqual(output.name, "00000000-0000-0000-0000-000000000001")
            self.assertEqual(output.parent.name, "jobs")

    def test_unreviewed_render_manifest_is_adoptable_but_not_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "artifacts" / "page.png"
            image_path.parent.mkdir(parents=True)
            Image.new("L", (20, 30), "white").save(image_path)
            manifest_path = root / "artifacts" / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "rendered",
                        "render_path": image_path.relative_to(root).as_posix(),
                        "render_sha256": sha256_file(image_path),
                        "source_object_sha256": "a" * 64,
                        "source_uri": "s3://bucket/prefix/volume.pdf",
                        "volume_number": 219,
                        "publication_year": 1925,
                        "page_number": 308,
                        "selection": {"gold_status": "unreviewed_ingestion"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            execution = _render_manifest_execution(
                manifest_path, context("render_lossless"), root
            )
            self.assertTrue(execution.adopted)
            self.assertEqual(
                execution.result["render_sha256"], sha256_file(image_path)
            )

    def test_existing_ocr_requires_exact_model_and_image(self):
        artifact = OCRPageArtifact.model_validate_json(
            Path("artifacts/ocr-pilot/v219-p0308.lossless.ppocrv6.json").read_text(
                encoding="utf-8"
            )
        )
        candidate = context(
            "ocr",
            {
                **DEFAULT_CONFIGURATION["ocr"],
                "output_root": "artifacts/test",
            },
        )
        candidate = replace(candidate, source_uri=artifact.source.source_uri)
        self.assertTrue(
            _ocr_artifact_matches(artifact, candidate, artifact.image_sha256)
        )
        self.assertFalse(_ocr_artifact_matches(artifact, candidate, "b" * 64))

    def test_existing_ner_requires_exact_bounded_configuration(self):
        artifact = NERArtifact.model_validate_json(
            Path("artifacts/ner-pilot/v219-p0308.lossless.gliner-x.first50.json").read_text(
                encoding="utf-8"
            )
        )
        configuration = {
            **DEFAULT_CONFIGURATION["ner"],
            "max_regions": 50,
            "batch_size": 2,
        }
        candidate = context("ner", configuration)
        self.assertTrue(
            _ner_artifact_matches(
                artifact, candidate, str(artifact.source_ocr_run_id)
            )
        )
        wrong = context("ner", {**configuration, "max_regions": 25})
        self.assertFalse(
            _ner_artifact_matches(artifact, wrong, str(artifact.source_ocr_run_id))
        )


if __name__ == "__main__":
    unittest.main()
