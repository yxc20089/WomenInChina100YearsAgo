from __future__ import annotations

import unittest
from uuid import UUID

from wic_history.evidence import (
    RetrievalHit,
    ScenarioContextBundle,
    ScenarioEvidenceItem,
    SourcePointer,
)
from wic_history.generation import GenerationStatus, GenerationTask, generate, prepare_messages


SOURCE = SourcePointer(
    source_uri="s3://example/volume.pdf",
    volume_number=219,
    publication_year=1925,
    page_number=308,
    region_id=UUID("00000000-0000-0000-0000-000000000001"),
)


class FakeGenerator:
    model_identity = "fake-model@test"

    def complete(self, messages):
        self.messages = messages
        return "A cited research brief [region:00000000-0000-0000-0000-000000000001]."


class HallucinatingGenerator(FakeGenerator):
    def complete(self, messages):
        return "Unsupported [region:00000000-0000-0000-0000-000000000099]."


class GenerationTests(unittest.TestCase):
    def _context(self, reviewed: bool = False) -> ScenarioContextBundle:
        hit = RetrievalHit(
            rank=1,
            score=1,
            source=SOURCE,
            text="女學生入學",
            explanation={"retriever": "lexical"},
        )
        items = []
        if reviewed:
            items = [
                ScenarioEvidenceItem(
                    statement="王女士 — attended_school — 務本女塾",
                    epistemic_label="directly_evidenced",
                    sources=[SOURCE],
                    claim_ids=[UUID("00000000-0000-0000-0000-000000000003")],
                )
            ]
        return ScenarioContextBundle(
            research_query="女學生",
            evidence_items=items,
            retrieved_context=[hit],
        )

    def test_scene_abstains_without_reviewed_claims(self):
        result = generate(self._context(), GenerationTask.RECONSTRUCTED_SCENE, FakeGenerator())
        self.assertEqual(result.status, GenerationStatus.ABSTAINED)
        self.assertIsNone(result.model)

    def test_research_brief_uses_generator_and_carries_citation(self):
        generator = FakeGenerator()
        result = generate(self._context(), GenerationTask.RESEARCH_BRIEF, generator)
        self.assertEqual(result.status, GenerationStatus.COMPLETED)
        self.assertEqual(result.model, "fake-model@test")
        self.assertEqual(result.citations, [SOURCE])
        self.assertEqual(len(result.prompt_sha256), 64)

    def test_prompt_marks_archive_text_as_untrusted(self):
        messages, digest = prepare_messages(self._context(True), GenerationTask.RECONSTRUCTED_SCENE)
        self.assertIn("untrusted quoted data", messages[0]["content"])
        self.assertIn("Direct evidence", messages[1]["content"])
        self.assertEqual(len(digest), 64)

    def test_generated_citations_are_checked_against_allowed_context(self):
        result = generate(
            self._context(), GenerationTask.RESEARCH_BRIEF, HallucinatingGenerator()
        )
        self.assertEqual(result.citations, [])
        self.assertEqual(
            result.invalid_citation_ids,
            ["00000000-0000-0000-0000-000000000099"],
        )
