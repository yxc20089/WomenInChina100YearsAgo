import unittest

from pydantic import ValidationError

from wic_history.segmentation_review import (
    QUEUE_SQL,
    SegmentationActivationRequest,
    SegmentationReviewRequest,
)


class SegmentationReviewContractTests(unittest.TestCase):
    def test_acceptance_requires_every_unit_checked(self):
        with self.assertRaises(ValidationError):
            SegmentationReviewRequest(
                review_id="00000000-0000-0000-0000-000000000001",
                decision="accept",
                reviewer="historian-a",
                expected_proposal_sha256="a" * 64,
                expected_input_sha256="b" * 64,
                checked_all_units=False,
                confirmation="RECORD_REVIEW_WITHOUT_ACTIVATION",
            )

    def test_activation_requires_explicit_expected_selection_even_when_none(self):
        request = SegmentationActivationRequest(
            selected_by="historian-a",
            expected_previous_selection_id=None,
            expected_proposal_sha256="a" * 64,
            confirmation="ACTIVATE_ACCEPTED_SEGMENTATION",
        )
        self.assertIsNone(request.expected_previous_selection_id)

    def test_queue_binds_exact_derivative_and_reports_staleness(self):
        self.assertIn("segmentation.source_ocr_selection_id", QUEUE_SQL)
        self.assertIn("derivative.image_sha256", QUEUE_SQL)
        self.assertIn("ocr_selection.superseded_at IS NULL", QUEUE_SQL)
        self.assertIn("current_page_active.selection_id", QUEUE_SQL)


if __name__ == "__main__":
    unittest.main()
