from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from wic_history.evidence import EntityType, SourcePointer
from wic_history.ner_adapters.base import (
    IssueSplitManifest,
    SnippetSplitAssignment,
    benchmark_dataset_sha256,
)
from wic_history.ner_benchmark import prepare_benchmark_dataset
from wic_history.ner_gold import (
    GoldAdjudication,
    GoldEntitySpan,
    GoldSnippet,
    NERGoldSet,
    ReviewerAnnotation,
)
from wic_history.ner_training import (
    W2NER_IMPLEMENTATION_REVISION,
    AugmentationConfiguration,
    W2NERTrainingExport,
    build_w2ner_training_export,
    main as training_main,
    materialize_w2ner_training_export,
    native_w2ner_record,
    w2ner_training_export_sha256,
)


NOW = datetime(2026, 7, 18, tzinfo=timezone.utc)
CODE_REVISION = "a" * 40
GOLD_SHA256 = "1" * 64
MANIFEST_SHA256 = "2" * 64


def _uuid(number: int) -> UUID:
    return UUID(int=number)


def _snippet(
    number: int,
    *,
    corrected_text: str,
    raw_text: str,
    corrected_entity: str,
    raw_entity: str,
) -> GoldSnippet:
    corrected_start = corrected_text.index(corrected_entity)
    raw_start = raw_text.index(raw_entity)
    entity = GoldEntitySpan(
        entity_type=EntityType.PERSON,
        corrected_start=corrected_start,
        corrected_end=corrected_start + len(corrected_entity),
        corrected_text=corrected_entity,
        raw_start=raw_start,
        raw_end=raw_start + len(raw_entity),
        raw_text=raw_entity,
    )
    annotation = {
        "corrected_text": corrected_text,
        "entities": [entity],
        "annotated_at": NOW,
    }
    return GoldSnippet(
        snippet_id=f"snippet-{number}",
        gold_region_id=_uuid(100 + number),
        source_ocr_run_id=_uuid(200 + number),
        source_ocr_region_id=_uuid(300 + number),
        source=SourcePointer(
            source_uri=f"s3://example/v{number}.pdf",
            volume_number=number,
            publication_year=1900 + (number * 10),
            page_number=1,
            region_id=_uuid(300 + number),
        ),
        raw_ocr_text=raw_text,
        page_genre="news_editorial",
        layout="vertical",
        scan_quality="poor",
        reviews=[
            ReviewerAnnotation(reviewer="reviewer-a", **annotation),
            ReviewerAnnotation(reviewer="reviewer-b", **annotation),
        ],
        adjudication=GoldAdjudication(
            adjudicator="adjudicator-c",
            corrected_text=corrected_text,
            entities=[entity],
            adjudicated_at=NOW,
        ),
    )


def _gold() -> NERGoldSet:
    return NERGoldSet(
        schema_version="1.1",
        dataset_id="gold-training-v1",
        created_at=NOW,
        ontology_version="women-history-zh-v1",
        snippets=[
            _snippet(
                1,
                corrected_text="宋 女士來校",
                raw_text="宋 女土來校",
                corrected_entity="宋 女士",
                raw_entity="宋 女土",
            ),
            _snippet(
                2,
                corrected_text="梁女士訪校",
                raw_text="梁女仕訪校",
                corrected_entity="梁女士",
                raw_entity="梁女仕",
            ),
            _snippet(
                3,
                corrected_text="陳女士任教",
                raw_text="陳女士任教",
                corrected_entity="陳女士",
                raw_entity="陳女士",
            ),
        ],
    )


def _manifest() -> IssueSplitManifest:
    return IssueSplitManifest(
        dataset_id="gold-training-v1",
        created_at=NOW,
        assigned_by="historian",
        assignments=[
            SnippetSplitAssignment(
                snippet_id="snippet-1", issue_id="issue-train", split="train"
            ),
            SnippetSplitAssignment(
                snippet_id="snippet-2",
                issue_id="issue-development",
                split="development",
            ),
            SnippetSplitAssignment(
                snippet_id="snippet-3", issue_id="issue-test", split="test"
            ),
        ],
    )


def _dataset(
    gold: NERGoldSet | None = None, manifest: IssueSplitManifest | None = None
):
    return prepare_benchmark_dataset(
        gold or _gold(),
        GOLD_SHA256,
        manifest or _manifest(),
        MANIFEST_SHA256,
        dataset_id="benchmark-training-v1",
        input_variants=["raw_ocr", "corrected_text"],
        generated_at=NOW,
    )


def _export() -> W2NERTrainingExport:
    return build_w2ner_training_export(
        _gold(),
        _dataset(),
        export_id="w2ner-training-v1",
        project_code_revision=CODE_REVISION,
        maximum_record_characters=16,
        augmentation=AugmentationConfiguration(
            probability=1,
            augmented_copies_per_clean_record=1,
            seed=23,
        ),
        allow_ineligible_technical_export=True,
        generated_at=NOW,
    )


class W2NERTrainingExportTests(unittest.TestCase):
    def test_ineligible_gold_requires_an_explicit_technical_export(self):
        with self.assertRaisesRegex(ValueError, "ineligible for training export"):
            build_w2ner_training_export(
                _gold(),
                _dataset(),
                export_id="blocked",
                project_code_revision=CODE_REVISION,
            )

    def test_export_is_issue_split_safe_exact_and_deterministic(self):
        export = _export()
        repeated = _export()

        self.assertTrue(export.technical_export)
        self.assertEqual(
            export.w2ner_implementation_revision, W2NER_IMPLEMENTATION_REVISION
        )
        self.assertEqual(
            export.source_benchmark_dataset_sha256,
            benchmark_dataset_sha256(_dataset()),
        )
        self.assertEqual(
            w2ner_training_export_sha256(export),
            w2ner_training_export_sha256(repeated),
        )
        self.assertEqual(
            [
                (
                    item.corrected_character,
                    item.raw_ocr_character,
                    item.source_snippet_ids,
                )
                for item in export.empirical_substitutions
            ],
            [("士", "土", ["snippet-1"])],
        )

        augmented = [
            record
            for record in export.records
            if record.augmentation_kind == "empirical_substitution"
        ]
        self.assertEqual(len(augmented), 1)
        self.assertEqual(augmented[0].split, "train")
        self.assertEqual(augmented[0].input_variant, "corrected_text")
        self.assertEqual(augmented[0].source_text, "宋 女士來校")
        self.assertEqual(augmented[0].training_text, "宋 女土來校")
        self.assertEqual(augmented[0].entities[0].training_surface, "宋 女土")
        self.assertEqual(augmented[0].entities[0].index, [0, 1, 2])
        self.assertFalse(
            any(
                record.augmentation_kind != "clean" and record.split != "train"
                for record in export.records
            )
        )

        clean_train = next(
            record
            for record in export.records
            if record.snippet_id == "snippet-1"
            and record.input_variant == "corrected_text"
            and record.augmentation_kind == "clean"
        )
        self.assertEqual(clean_train.sentence, ["宋", "女", "士", "來", "校"])
        self.assertEqual(clean_train.token_character_offsets, [0, 2, 3, 4, 5])
        self.assertEqual(set(native_w2ner_record(clean_train)), {"sentence", "ner"})

    def test_materialization_hashes_native_views_and_refuses_overwrite(self):
        export = _export()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "w2ner"
            paths = materialize_w2ner_training_export(export, output)
            self.assertIn(output / "manifest.json", paths)
            for view in export.views:
                payload = (output / view.filename).read_bytes()
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(), view.native_json_sha256
                )
                native = json.loads(payload)
                self.assertTrue(
                    all(set(record) == {"sentence", "ner"} for record in native)
                )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                materialize_w2ner_training_export(export, output)

    def test_internal_hash_event_and_split_tampering_is_rejected(self):
        export_data = _export().model_dump(mode="json")

        source_tamper = json.loads(json.dumps(export_data))
        source_tamper["records"][0]["source_text_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source-text hash"):
            W2NERTrainingExport.model_validate(source_tamper)

        event_tamper = json.loads(json.dumps(export_data))
        augmented = next(
            record
            for record in event_tamper["records"]
            if record["augmentation_kind"] == "empirical_substitution"
        )
        augmented["substitutions"][0]["raw_ocr_character"] = "仕"
        with self.assertRaisesRegex(ValueError, "do not reconstruct"):
            W2NERTrainingExport.model_validate(event_tamper)

        view_tamper = json.loads(json.dumps(export_data))
        view_tamper["views"][0]["native_json_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "native view hash"):
            W2NERTrainingExport.model_validate(view_tamper)

        leakage_tamper = json.loads(json.dumps(export_data))
        leakage_tamper["empirical_substitutions"][0]["source_snippet_ids"] = [
            "snippet-2"
        ]
        with self.assertRaisesRegex(ValueError, "training snippets only"):
            W2NERTrainingExport.model_validate(leakage_tamper)

    def test_chunking_never_splits_an_entity(self):
        text = ("甲" * 14) + "宋女士" + ("乙" * 10)
        snippet = _snippet(
            1,
            corrected_text=text,
            raw_text=text,
            corrected_entity="宋女士",
            raw_entity="宋女士",
        )
        gold = NERGoldSet(
            schema_version="1.1",
            dataset_id="chunk-gold",
            created_at=NOW,
            ontology_version="women-history-zh-v1",
            snippets=[snippet],
        )
        manifest = IssueSplitManifest(
            dataset_id="chunk-gold",
            created_at=NOW,
            assigned_by="historian",
            assignments=[
                SnippetSplitAssignment(
                    snippet_id="snippet-1", issue_id="issue-train", split="train"
                )
            ],
        )
        dataset = _dataset(gold, manifest)
        export = build_w2ner_training_export(
            gold,
            dataset,
            export_id="chunk-export",
            project_code_revision=CODE_REVISION,
            maximum_record_characters=16,
            augmentation=AugmentationConfiguration(augmented_copies_per_clean_record=0),
            allow_ineligible_technical_export=True,
            generated_at=NOW,
        )
        corrected_records = [
            record
            for record in export.records
            if record.input_variant == "corrected_text"
        ]
        self.assertEqual(
            [(item.snippet_start, item.snippet_end) for item in corrected_records],
            [(0, 14), (14, 27)],
        )
        self.assertEqual(corrected_records[1].entities[0].training_surface, "宋女士")

    def test_w2ner_rejects_two_types_on_the_same_grid_cell(self):
        gold_data = _gold().model_dump(mode="json")
        duplicate = dict(gold_data["snippets"][0]["adjudication"]["entities"][0])
        duplicate["entity_type"] = "organization"
        for section in ("reviews",):
            for review in gold_data["snippets"][0][section]:
                review["entities"].append(dict(duplicate))
        gold_data["snippets"][0]["adjudication"]["entities"].append(duplicate)
        gold = NERGoldSet.model_validate(gold_data)
        with self.assertRaisesRegex(ValueError, "same tail-head grid cell"):
            build_w2ner_training_export(
                gold,
                _dataset(gold),
                export_id="ambiguous-grid",
                project_code_revision=CODE_REVISION,
                allow_ineligible_technical_export=True,
            )

    def test_cli_writes_a_self_describing_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gold_path = root / "gold.json"
            manifest_path = root / "split.json"
            output = root / "output"
            gold_path.write_text(_gold().model_dump_json(indent=2), encoding="utf-8")
            manifest_path.write_text(
                _manifest().model_dump_json(indent=2), encoding="utf-8"
            )
            result = training_main(
                [
                    "--gold",
                    str(gold_path),
                    "--split-manifest",
                    str(manifest_path),
                    "--dataset-id",
                    "cli-dataset",
                    "--export-id",
                    "cli-export",
                    "--project-code-revision",
                    CODE_REVISION,
                    "--allow-ineligible-technical-export",
                    "--output-directory",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["w2ner_implementation_revision"], W2NER_IMPLEMENTATION_REVISION
            )


if __name__ == "__main__":
    unittest.main()
