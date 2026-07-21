from __future__ import annotations

import unittest
from uuid import uuid4

from wic_history.frontier_extraction import (
    ArticleExtractionV2,
    ExtractionAbstention,
    extract_article,
)
from wic_history.frontier_ocr import FrontierOCRAbstention, validate_transcription


def _minimal_payload(region_id, text="富紳淑女固不貪便宜"):
    return {
        "article_type": "advertisement",
        "modern_paraphrase": "有錢的紳士和淑女並不貪便宜。",
        "topics": [{"label": "商業廣告"}],
        "keywords": ["淑女"],
        "mentions": [
            {
                "mention_key": "m1",
                "region_id": str(region_id),
                "text_start": 2,
                "text_end": 4,
                "surface": text[2:4],
                "entity_type": "person",
                "mention_form": "named",
            }
        ],
        "event_evidence": [],
        "events": [],
        "claims": [],
        "event_dates": [],
    }


class FakeCompletion:
    def __init__(self, content, finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.usage = {}


class FakeGenerator:
    def __init__(self, content, finish_reason="stop"):
        self._content = content
        self._finish = finish_reason

    def complete(self, messages, **_kwargs):
        return FakeCompletion(self._content, self._finish)


class ExtractionValidationTests(unittest.TestCase):
    def test_valid_extraction_round_trips(self):
        import json

        region_id = uuid4()
        text = "富紳淑女固不貪便宜"
        payload = _minimal_payload(region_id, text)
        result = extract_article(
            FakeGenerator(json.dumps(payload, ensure_ascii=False)), text, region_id
        )
        self.assertEqual(result.article_type, "advertisement")
        self.assertEqual(result.topics[0].label, "商業廣告")
        self.assertEqual(result.mentions[0].surface, "淑女")

    def test_wrong_offsets_are_resolved_by_code_from_the_surface(self):
        import json

        region_id = uuid4()
        text = "富紳淑女固不貪便宜"
        payload = _minimal_payload(region_id, text)
        # model claims wrong offsets for a real surface: code relocates it
        payload["mentions"][0]["surface"] = "貪便宜"
        result = extract_article(
            FakeGenerator(json.dumps(payload, ensure_ascii=False)), text, region_id
        )
        mention = result.mentions[0]
        self.assertEqual(
            text[mention.text_start:mention.text_end], "貪便宜"
        )

    def test_fabricated_surface_abstains(self):
        import json

        region_id = uuid4()
        text = "富紳淑女固不貪便宜"
        payload = _minimal_payload(region_id, text)
        payload["mentions"][0]["surface"] = "婦女"  # absent from the text
        with self.assertRaisesRegex(ExtractionAbstention, "absent"):
            _ = extract_article(
                FakeGenerator(json.dumps(payload, ensure_ascii=False)), text, region_id
            )

    def test_out_of_ontology_type_is_schema_invalid(self):
        import json

        region_id = uuid4()
        payload = _minimal_payload(region_id)
        payload["mentions"][0]["entity_type"] = "event"
        with self.assertRaisesRegex(ExtractionAbstention, "schema"):
            _ = extract_article(
                FakeGenerator(json.dumps(payload, ensure_ascii=False)),
                "富紳淑女固不貪便宜",
                region_id,
            )

    def test_foreign_region_id_abstains(self):
        import json

        region_id = uuid4()
        text = "富紳淑女固不貪便宜"
        payload = _minimal_payload(uuid4(), text)  # different region
        with self.assertRaisesRegex(ExtractionAbstention, "foreign region"):
            _ = extract_article(
                FakeGenerator(json.dumps(payload, ensure_ascii=False)), text, region_id
            )

    def test_truncated_response_abstains(self):
        region_id = uuid4()
        with self.assertRaisesRegex(ExtractionAbstention, "finish reason"):
            _ = extract_article(
                FakeGenerator("{}", finish_reason="length"), "文本內容", region_id
            )

    def test_malformed_json_abstains(self):
        region_id = uuid4()
        with self.assertRaisesRegex(ExtractionAbstention, "schema"):
            _ = extract_article(FakeGenerator("not json"), "文本內容", region_id)

    def test_claim_requires_exactly_one_object(self):
        region_id = uuid4()
        base = {
            "claim_key": "c1",
            "subject_mention_key": "m1",
            "predicate": "位於",
            "region_id": str(region_id),
            "text_start": 0,
            "text_end": 2,
            "surface": "富紳",
        }
        payload = _minimal_payload(region_id)
        payload["claims"] = [dict(base)]  # neither object form
        with self.assertRaises(ValueError):
            _ = ArticleExtractionV2.model_validate(payload)
        payload["claims"] = [
            dict(base, object_mention_key="m1", object_value="上海")
        ]  # both
        with self.assertRaises(ValueError):
            _ = ArticleExtractionV2.model_validate(payload)

    def test_event_date_must_reference_known_event(self):
        payload = _minimal_payload(uuid4())
        payload["event_dates"] = [{"event_key": "ghost", "date_iso": "1925-12"}]
        with self.assertRaisesRegex(ValueError, "unknown event"):
            _ = ArticleExtractionV2.model_validate(payload)


class TranscriptionGateTests(unittest.TestCase):
    def test_cjk_transcription_passes(self):
        text = "曰富紳淑女固不貪便宜\n開設閘北新疆路口"
        self.assertEqual(validate_transcription(text), text)

    def test_damaged_characters_marker_allowed(self):
        _ = validate_transcription("富紳□女固不貪便宜")

    def test_off_domain_hallucination_abstains(self):
        with self.assertRaisesRegex(FrontierOCRAbstention, "CJK"):
            _ = validate_transcription("The quick brown fox jumps over the lazy dog")

    def test_empty_abstains(self):
        with self.assertRaisesRegex(FrontierOCRAbstention, "empty"):
            _ = validate_transcription("   \n ")

    def test_repetition_degeneration_abstains(self):
        with self.assertRaisesRegex(FrontierOCRAbstention, "repetition"):
            _ = validate_transcription("注宋注宋注宋\n" * 20)


class ConfigurationTests(unittest.TestCase):
    def test_pinned_configuration_declares_frontier_models(self):
        from wic_history.model_config import load_pipeline_model_configuration

        configuration = load_pipeline_model_configuration()
        self.assertEqual(configuration.frontier_ocr.provider, "openrouter")
        self.assertEqual(configuration.semantic.provider, "local_openai")
        self.assertEqual(configuration.semantic.temperature, 0.0)
        identity = configuration.frontier_ocr.provenance_identity()
        self.assertEqual(identity["model_revision"], "not_available")
        self.assertIn("prompt_sha256", identity)


if __name__ == "__main__":
    unittest.main()
