from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4

from wic_history.evidence import OCRPageArtifact
from wic_history.repository import (
    _confidence_provenance,
    _hunyuan_visual_output_specs,
    _ocr_evidence_tier,
    build_parser,
    read_jsonl,
)


class RepositoryTests(unittest.TestCase):
    def test_cli_exposes_hunyuan_layout_ingestion(self):
        args = build_parser().parse_args(
            ["--database-url", "postgresql://example", "layout", "layout.json"]
        )
        self.assertEqual(args.command, "layout")
        self.assertEqual(args.paths, [Path("layout.json")])

    def test_legacy_smoke_artifact_retains_screening_tier(self):
        artifact = OCRPageArtifact.model_validate_json(
            Path("artifacts/ocr-smoke/v219-p0308.ppocrv6.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(_ocr_evidence_tier(artifact), "screening_derivative")

    def test_jsonl_reader_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text('{"volume_number": 1}\n\n{"volume_number": 2}\n', encoding="utf-8")
            self.assertEqual(
                [item["volume_number"] for item in read_jsonl(path)],
                [1, 2],
            )

    def test_jsonl_reader_reports_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text('{}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"manifest\.jsonl:2"):
                list(read_jsonl(path))

    def test_confidence_is_uncalibrated_unless_artifact_names_calibration(self):
        self.assertEqual(
            _confidence_provenance(0.82, {}),
            ("uncalibrated", None),
        )
        self.assertEqual(
            _confidence_provenance(None, {}),
            ("not_reported", None),
        )
        calibration_id = uuid4()
        self.assertEqual(
            _confidence_provenance(
                0.82,
                {
                    "confidence_status": "calibrated",
                    "calibration_id": str(calibration_id),
                },
            ),
            ("calibrated", calibration_id),
        )

    def test_confidence_provenance_rejects_false_calibration_claims(self):
        with self.assertRaisesRegex(ValueError, "score/calibration"):
            _confidence_provenance(
                0.82,
                {"confidence_status": "calibrated"},
            )
        with self.assertRaisesRegex(ValueError, "score/calibration"):
            _confidence_provenance(
                None,
                {"confidence_status": "uncalibrated"},
            )

    def test_hunyuan_layout_ingestion_builds_both_exact_source_page_outputs(self):
        spotting = '[{"box":[0,0,1000,1000],"text":"霍爾平"}]'
        layout = "霍爾平"
        import hashlib

        run_id = uuid4()
        artifact = SimpleNamespace(
            run=SimpleNamespace(
                run_id=run_id,
                configuration={
                    "raw_task_bundle": {
                        "spotting_task": "spotting_json",
                        "layout_task": "layout_parse",
                        "spotting_raw_output": spotting,
                        "spotting_raw_output_sha256": hashlib.sha256(
                            spotting.encode("utf-8")
                        ).hexdigest(),
                        "layout_raw_output": layout,
                        "layout_raw_output_sha256": hashlib.sha256(
                            layout.encode("utf-8")
                        ).hexdigest(),
                        "confidence_status": "not_emitted_by_model",
                        "confidence_calibration": "not_available",
                    },
                    "assessment_status": "accepted",
                    "review_required": False,
                },
            ),
            source=SimpleNamespace(source_uri="s3://archive/volume.pdf"),
            image_uri="artifacts/pages/v219-p0308.png",
            image_sha256="a" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            path.write_text('{"immutable":true}\n', encoding="utf-8")
            outputs = _hunyuan_visual_output_specs(
                artifact,
                path,
                source_object_id=uuid4(),
                page_id=uuid4(),
                derivative_id=uuid4(),
            )
        self.assertEqual([item.output_kind for item in outputs], ["spotting", "layout"])
        self.assertEqual(outputs[0].raw_output, spotting)
        self.assertEqual(outputs[1].raw_output, layout)
        self.assertEqual(outputs[0].confidence_status, "not_reported")
        self.assertEqual(outputs[1].confidence_status, "not_reported")
        self.assertEqual(outputs[0].evidence_paths[0].path_role, "source_page")
        self.assertEqual(outputs[1].evidence_paths[0].path_role, "source_page")


if __name__ == "__main__":
    unittest.main()
