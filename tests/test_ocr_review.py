from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from pydantic import ValidationError

from wic_history.ocr_review import OCRReviewLedger, load_review_ledger


LEDGER_PATH = Path("artifacts/ocr-challenger/v219-p0308.review-ledger.json")


class OCRReviewTests(unittest.TestCase):
    def test_project_review_ledger_preserves_single_review_boundary(self):
        ledger = load_review_ledger(LEDGER_PATH)

        self.assertEqual(ledger.gold_status, "single_review_not_gold")
        self.assertEqual(
            [case.target.case_id for case in ledger.cases], ["C07", "C08", "C09"]
        )
        self.assertTrue(
            all(
                not review.gold_eligible
                for case in ledger.cases
                for review in case.reviews
            )
        )
        self.assertEqual(
            ledger.cases[0].reviews[0].source_script_transcription,
            "愛寵情人。",
        )
        self.assertEqual(
            ledger.cases[2].reviews[0].source_script_transcription,
            "英皇時召霍臨宮中",
        )

    def test_rejects_changed_raw_model_text(self):
        payload = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        changed = deepcopy(payload)
        changed["cases"][0]["hypotheses"][0]["raw_text"] = "mutated"

        with self.assertRaisesRegex(ValidationError, "raw_text_sha256"):
            OCRReviewLedger.model_validate(changed)

    def test_partial_review_requires_explicit_unresolved_note(self):
        payload = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        changed = deepcopy(payload)
        del changed["cases"][0]["reviews"][0]["unresolved_note"]

        with self.assertRaisesRegex(ValidationError, "unresolved_note"):
            OCRReviewLedger.model_validate(changed)

    def test_single_review_cannot_claim_gold_eligibility(self):
        payload = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        changed = deepcopy(payload)
        changed["cases"][1]["reviews"][0]["gold_eligible"] = True

        with self.assertRaises(ValidationError):
            OCRReviewLedger.model_validate(changed)


if __name__ == "__main__":
    unittest.main()
