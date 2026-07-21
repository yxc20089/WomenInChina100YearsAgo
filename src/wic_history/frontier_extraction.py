"""Broad one-pass article extraction (contract v2) via the pinned frontier model.

The second of the two one-time ingestion LLM passes. One call per article
emits everything downstream query surfaces need — named-entity mentions with
exact offsets, events, subject-predicate-object claims, first-class topics,
keywords, article type, and a modern-language paraphrase — so no further
extraction rounds are ever required (schema versioned; a re-run is an
explicit re-ingestion decision). Malformed or offset-inconsistent output
abstains whole; nothing partial is retained.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, final
from uuid import UUID

from pydantic import Field, model_validator

from .semantic_tasks import (
    ExtractedEvent,
    ExtractedEventEvidence,
    StrictModel,
)

EXTRACTION_SCHEMA_VERSION = "wic-article-extraction/v2"

ArticleType = Literal[
    "article", "advertisement", "notice", "listing", "table", "image_caption", "other"
]

# the ontology constraint lives in the schema the model sees, so it cannot
# emit out-of-ontology types that would only fail post-hoc (pilot finding)
NamedEntityType = Literal[
    "person", "place", "address", "organization", "school", "publication", "product"
]
MentionFormV2 = Literal[
    "full_name", "short_name", "title_reference", "kinship_reference", "pronoun", "named"
]


class MentionV2(StrictModel):
    """Named-entity occurrence; offsets are model hints, resolved by code."""

    mention_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")
    region_id: UUID
    text_start: int = Field(ge=0)
    text_end: int = Field(gt=0)
    surface: str = Field(min_length=1, max_length=300)
    entity_type: NamedEntityType
    mention_form: MentionFormV2


class TopicItem(StrictModel):
    label: str = Field(min_length=2, max_length=60)
    note: str = Field(default="", max_length=300)


class ExtractedClaim(StrictModel):
    claim_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")
    subject_mention_key: str
    predicate: str = Field(min_length=1, max_length=120)
    object_mention_key: str | None = None
    object_value: str | None = Field(default=None, max_length=300)
    region_id: UUID
    text_start: int = Field(ge=0)
    text_end: int = Field(gt=0)
    surface: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_claim(self) -> "ExtractedClaim":
        if self.text_end <= self.text_start:
            raise ValueError("claim evidence end must follow start")
        if (self.object_mention_key is None) == (self.object_value is None):
            raise ValueError("claim needs exactly one of object mention or value")
        return self


class EventDate(StrictModel):
    event_key: str
    date_iso: str = Field(pattern=r"^\d{4}(-\d{2}(-\d{2})?)?$")


class ArticleExtractionV2(StrictModel):
    """Everything one article contributes to the graph and search surfaces."""

    article_type: ArticleType
    modern_paraphrase: str = Field(max_length=4000)
    topics: list[TopicItem] = Field(max_length=12)
    keywords: list[str] = Field(max_length=20)
    mentions: list[MentionV2]
    event_evidence: list[ExtractedEventEvidence]
    events: list[ExtractedEvent]
    claims: list[ExtractedClaim]
    event_dates: list[EventDate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cross_references(self) -> "ArticleExtractionV2":
        mention_keys = {mention.mention_key for mention in self.mentions}
        if len(mention_keys) != len(self.mentions):
            raise ValueError("duplicate mention keys")
        evidence_keys = {item.evidence_key for item in self.event_evidence}
        event_keys = set()
        for event in self.events:
            if event.event_key in event_keys:
                raise ValueError("duplicate event keys")
            event_keys.add(event.event_key)
            for participant in event.participant_decisions:
                if participant.mention_key not in mention_keys:
                    raise ValueError("event participant references unknown mention")
            for key in event.evidence_keys:
                if key not in evidence_keys:
                    raise ValueError("event references unknown evidence")
        for claim in self.claims:
            if claim.subject_mention_key not in mention_keys:
                raise ValueError("claim subject references unknown mention")
            if (
                claim.object_mention_key is not None
                and claim.object_mention_key not in mention_keys
            ):
                raise ValueError("claim object references unknown mention")
        for event_date in self.event_dates:
            if event_date.event_key not in event_keys:
                raise ValueError("event date references unknown event")
        for keyword in self.keywords:
            if not 1 <= len(keyword) <= 30:
                raise ValueError("keyword length out of bounds")
        return self


@final
class ExtractionAbstention(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_SYSTEM_PROMPT = """你是一個對1920年代《申報》文本進行結構化抽取的系統。輸入是一個版面區塊的繁體中文轉錄文本（region_id 已給出，offset 以字符計，從 0 開始）。

嚴格按給定 JSON schema 輸出，並遵守：
1. mentions：文中出現的命名實體（person/place/address/organization/school/publication/product），surface 必須與 text[text_start:text_end] 完全一致，保留繁體。
2. events：文中敘述的事件，參與者引用 mention_key，證據引用 evidence_key。
3. claims：主-謂-賓斷言，主語必須是 mention；賓語是 mention 或字面值；surface 為支持該斷言的原文片段（同樣要求 offset 精確）。
4. topics：這個區塊「關於什麼」的主題標籤（現代漢語，2-8字，如「女子教育」「婚姻習俗」「戲院廣告」），供主題檢索使用。
5. keywords：檢索用關鍵詞（可含原文詞與現代對應詞）。
6. article_type：區塊體裁。
7. modern_paraphrase：用現代白話文忠實概述全文內容（供現代詞彙檢索；不是翻譯每一句）。
8. event_dates：能確定的事件日期，ISO 格式（可只到年或月）。
寧缺毋濫：不確定的不要輸出；絕不編造原文沒有的內容。"""


def _grammar_safe(schema: Any) -> Any:
    """Strip constraints grammar compilers reject (llama.cpp: bounded-repeat
    patterns, formats). Structure, required fields and enums — the ontology
    constraint — survive; everything stripped is still enforced by pydantic
    on the response, so nothing is lost except grammar compilability."""
    if isinstance(schema, dict):
        return {
            key: _grammar_safe(value)
            for key, value in schema.items()
            if key not in {"pattern", "format", "minLength", "maxLength",
                           "minItems", "maxItems", "minimum", "maximum",
                           "exclusiveMinimum", "exclusiveMaximum"}
        }
    if isinstance(schema, list):
        return [_grammar_safe(item) for item in schema]
    return schema


def _response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "article_extraction_v2",
            "strict": True,
            "schema": _grammar_safe(ArticleExtractionV2.model_json_schema()),
        },
    }


def extraction_identity(model_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "system_prompt_sha256": hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest(),
        **model_identity,
    }


def _resolved_offsets(kind: str, item, text: str):
    """Authoritative offsets come from code, never the model (v1 lesson:
    models read surfaces correctly but cannot count CJK offsets). The model's
    claimed start only disambiguates repeated surfaces; a surface absent from
    the text is fabrication and abstains the whole article."""
    if text[item.text_start:item.text_end] == item.surface:
        return item
    starts = []
    position = text.find(item.surface)
    while position != -1:
        starts.append(position)
        position = text.find(item.surface, position + 1)
    if not starts:
        raise ExtractionAbstention(f"{kind} surface is absent from the article text")
    start = min(starts, key=lambda value: abs(value - item.text_start))
    return item.model_copy(
        update={"text_start": start, "text_end": start + len(item.surface)}
    )


def _verify_offsets(result: ArticleExtractionV2, text: str) -> ArticleExtractionV2:
    return result.model_copy(
        update={
            "mentions": [
                _resolved_offsets("mention", item, text) for item in result.mentions
            ],
            "event_evidence": [
                _resolved_offsets("event evidence", item, text)
                for item in result.event_evidence
            ],
            "claims": [
                _resolved_offsets("claim", item, text) for item in result.claims
            ],
        }
    )


def extract_article(
    generator: Any,
    article_text: str,
    region_id: UUID,
) -> ArticleExtractionV2:
    """One bounded extraction call; any defect abstains the whole article."""
    if not article_text.strip():
        raise ExtractionAbstention("article text is empty")
    payload = json.dumps(
        {"region_id": str(region_id), "text": article_text},
        ensure_ascii=False,
        sort_keys=True,
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": payload},
    ]
    try:
        completion = generator.complete(messages, response_format=_response_format())
    except RuntimeError as exc:
        raise ExtractionAbstention(f"provider call failed: {exc}") from exc
    finish_reason = getattr(completion, "finish_reason", None)
    if finish_reason not in {None, "stop", "end_turn"}:
        raise ExtractionAbstention(f"non-terminal finish reason {finish_reason!r}")
    content = str(getattr(completion, "content", completion))
    try:
        result = ArticleExtractionV2.model_validate_json(content)
    except ValueError as exc:
        raise ExtractionAbstention(f"response failed schema validation: {exc}") from exc
    for mention in result.mentions:
        if mention.region_id != region_id:
            raise ExtractionAbstention("mention cites a foreign region id")
    return _verify_offsets(result, article_text)
