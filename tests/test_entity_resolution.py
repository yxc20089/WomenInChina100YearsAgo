from __future__ import annotations

import json
import unittest
from uuid import UUID, uuid4

from wic_history.entity_resolution import (
    MentionResolutionContext,
    ResolutionDecision,
    ResolutionReason,
    annotate_candidates_with_proposal,
    candidate_set_sha256,
    parse_resolution_content,
    prepare_resolution_messages,
)
from wic_history.evidence import EntityLinkCandidate, EntityType


SUN_ENTITY_ID = UUID("00000000-0000-0000-0000-000000000857")


def context(surface: str = "孫文") -> MentionResolutionContext:
    text = f"總理{surface}在上海演說。"
    start = text.index(surface)
    return MentionResolutionContext(
        mention_id=uuid4(),
        entity_type=EntityType.PERSON,
        mention_text=surface,
        normalized_text=surface,
        region_id=uuid4(),
        source_text=text,
        text_start=start,
        text_end=start + len(surface),
    )


def candidates(item: MentionResolutionContext):
    run_id = uuid4()
    linked = EntityLinkCandidate(
        mention_id=item.mention_id,
        entity_id=SUN_ENTITY_ID,
        authority_uri="https://www.wikidata.org/wiki/Q8573",
        canonical_name="孫中山",
        entity_type=EntityType.PERSON,
        score=0.91,
        features={"retriever": "alias"},
        run_id=run_id,
    )
    nil = EntityLinkCandidate(
        mention_id=item.mention_id,
        canonical_name=item.mention_text,
        entity_type=EntityType.PERSON,
        score=0.09,
        features={"reason": "explicit_nil"},
        nil_candidate=True,
        run_id=run_id,
    )
    return linked, nil


class EntityResolutionTests(unittest.TestCase):
    def test_prompt_binds_exact_mention_aliases_and_candidate_roster(self):
        item = context()
        roster = candidates(item)
        messages, prompt_hash, roster_hash, response_format = prepare_resolution_messages(
            item, roster, {SUN_ENTITY_ID: ["孫文", "孫逸仙"]}
        )
        payload = json.loads(messages[-1]["content"])

        self.assertEqual(payload["mention"]["mention_text"], "孫文")
        entity_candidate = next(
            value
            for value in payload["candidates"]
            if value["candidate_kind"] == "ENTITY"
        )
        self.assertEqual(entity_candidate["aliases"], ["孫文", "孫逸仙"])
        self.assertEqual(payload["candidate_set_sha256"], roster_hash)
        self.assertEqual(len(prompt_hash), 64)
        self.assertIn(
            str(roster[0].link_id),
            response_format["json_schema"]["schema"]["properties"]
            ["selected_link_candidate_id"]["enum"],
        )

    def test_link_selects_only_supplied_non_nil_local_candidate(self):
        item = context()
        linked, nil = candidates(item)
        roster_hash = candidate_set_sha256(item, [linked, nil])
        proposal = parse_resolution_content(
            json.dumps(
                {
                    "decision": "LINK",
                    "selected_link_candidate_id": str(linked.link_id),
                    "diagnostic_score": 0.94,
                    "reason_codes": ["EXACT_ALIAS", "CONTEXT_COMPATIBLE"],
                }
            ),
            item,
            [linked, nil],
            roster_hash=roster_hash,
        )

        self.assertTrue(proposal.valid_model_output)
        self.assertEqual(proposal.decision, ResolutionDecision.LINK)
        self.assertEqual(proposal.selected_link_candidate_id, linked.link_id)

    def test_nil_must_select_the_unique_nil_candidate(self):
        item = context()
        linked, nil = candidates(item)
        roster_hash = candidate_set_sha256(item, [linked, nil])
        proposal = parse_resolution_content(
            json.dumps(
                {
                    "decision": "NIL",
                    "selected_link_candidate_id": str(nil.link_id),
                    "diagnostic_score": 0.8,
                    "reason_codes": ["NO_MATCHING_CANDIDATE"],
                }
            ),
            item,
            [linked, nil],
            roster_hash=roster_hash,
        )

        self.assertEqual(proposal.decision, ResolutionDecision.NIL)
        self.assertEqual(proposal.selected_link_candidate_id, nil.link_id)

    def test_invalid_cross_roster_or_wrong_kind_output_becomes_abstention(self):
        item = context()
        linked, nil = candidates(item)
        roster_hash = candidate_set_sha256(item, [linked, nil])
        cases = [
            {
                "decision": "LINK",
                "selected_link_candidate_id": str(nil.link_id),
                "diagnostic_score": 1,
                "reason_codes": ["EXACT_ALIAS"],
            },
            {
                "decision": "NIL",
                "selected_link_candidate_id": str(uuid4()),
                "diagnostic_score": 1,
                "reason_codes": ["NO_MATCHING_CANDIDATE"],
            },
            {
                "decision": "ABSTAIN",
                "selected_link_candidate_id": str(linked.link_id),
                "diagnostic_score": None,
                "reason_codes": ["INSUFFICIENT_EVIDENCE"],
            },
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                proposal = parse_resolution_content(
                    json.dumps(payload),
                    item,
                    [linked, nil],
                    roster_hash=roster_hash,
                )
                self.assertFalse(proposal.valid_model_output)
                self.assertEqual(proposal.decision, ResolutionDecision.ABSTAIN)
                self.assertIsNone(proposal.selected_link_candidate_id)
                self.assertEqual(
                    proposal.reason_codes, [ResolutionReason.INVALID_MODEL_OUTPUT]
                )

    def test_extra_identity_fields_are_rejected_not_repaired(self):
        item = context()
        linked, nil = candidates(item)
        roster_hash = candidate_set_sha256(item, [linked, nil])
        proposal = parse_resolution_content(
            json.dumps(
                {
                    "decision": "LINK",
                    "selected_link_candidate_id": str(linked.link_id),
                    "diagnostic_score": 1,
                    "reason_codes": ["EXACT_ALIAS"],
                    "entity_id": str(SUN_ENTITY_ID),
                }
            ),
            item,
            [linked, nil],
            roster_hash=roster_hash,
        )

        self.assertFalse(proposal.valid_model_output)
        self.assertIn("unexpected", proposal.validation_error)

    def test_annotation_preserves_roster_and_marks_only_selected_candidate(self):
        item = context()
        linked, nil = candidates(item)
        roster_hash = candidate_set_sha256(item, [linked, nil])
        proposal = parse_resolution_content(
            json.dumps(
                {
                    "decision": "LINK",
                    "selected_link_candidate_id": str(linked.link_id),
                    "diagnostic_score": 0.9,
                    "reason_codes": ["ALIAS_VARIANT"],
                }
            ),
            item,
            [linked, nil],
            roster_hash=roster_hash,
        )
        annotated = annotate_candidates_with_proposal([linked, nil], proposal)

        self.assertEqual([item.link_id for item in annotated], [linked.link_id, nil.link_id])
        self.assertTrue(annotated[0].features["model_proposal_selected"])
        self.assertFalse(annotated[1].features["model_proposal_selected"])
        self.assertEqual(annotated[0].entity_id, SUN_ENTITY_ID)
        self.assertTrue(annotated[1].nil_candidate)

    def test_roster_hash_changes_with_context_or_alias_facts(self):
        first = context("孫文")
        roster = candidates(first)
        base = candidate_set_sha256(first, roster, {SUN_ENTITY_ID: ["孫文"]})
        changed_alias = candidate_set_sha256(
            first, roster, {SUN_ENTITY_ID: ["孫文", "孫逸仙"]}
        )
        changed_context = first.model_copy(
            update={"source_text": "孫文在廣州演說。", "text_start": 0, "text_end": 2}
        )
        changed_text = candidate_set_sha256(
            changed_context, roster, {SUN_ENTITY_ID: ["孫文"]}
        )

        self.assertNotEqual(base, changed_alias)
        self.assertNotEqual(base, changed_text)


if __name__ == "__main__":
    unittest.main()
