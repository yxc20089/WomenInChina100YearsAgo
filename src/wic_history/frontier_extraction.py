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

from .semantic_tasks import EVENT_TYPES, StrictModel

EXTRACTION_SCHEMA_VERSION = "wic-article-extraction/v2"

# aligned with evidence.coherent_unit_revision.unit_kind (migration 010) —
# one vocabulary for the concept across extraction, storage and facets
ArticleType = Literal[
    "article", "column", "caption", "advertisement", "classified", "table", "other"
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


class EvidenceSpanV2(StrictModel):
    """A verbatim span of the article text; offsets resolved by code."""

    region_id: UUID
    text_start: int = Field(ge=0)
    text_end: int = Field(gt=0)
    surface: str = Field(min_length=1, max_length=1000)


class EventParticipantV2(StrictModel):
    mention_key: str
    role: str = Field(min_length=1, max_length=100)


class EventV2(StrictModel):
    """One narrated event with its spans inlined — no key indirection.

    The DB event table stores dates as date_start/date_end/date_precision;
    date_iso (YYYY[-MM[-DD]]) maps onto that. Dates come only from the text,
    never from world knowledge.
    """

    event_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")
    event_type: Literal[EVENT_TYPES]  # type: ignore[valid-type]
    trigger: EvidenceSpanV2
    participants: list[EventParticipantV2]
    date_span: EvidenceSpanV2 | None = None
    location_span: EvidenceSpanV2 | None = None
    date_iso: str | None = Field(default=None, pattern=r"^\d{4}(-\d{2}(-\d{2})?)?$")


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


class ArticleExtractionV2(StrictModel):
    """Everything one article contributes to the graph and search surfaces."""

    article_type: ArticleType
    modern_paraphrase: str = Field(max_length=4000)
    topics: list[TopicItem] = Field(max_length=12)
    keywords: list[str] = Field(max_length=20)
    mentions: list[MentionV2]
    events: list[EventV2]
    claims: list[ExtractedClaim]

    @model_validator(mode="after")
    def validate_cross_references(self) -> "ArticleExtractionV2":
        mention_keys = {mention.mention_key for mention in self.mentions}
        if len(mention_keys) != len(self.mentions):
            raise ValueError("duplicate mention keys")
        event_keys = set()
        for event in self.events:
            if event.event_key in event_keys:
                raise ValueError("duplicate event keys")
            event_keys.add(event.event_key)
            for participant in event.participants:
                if participant.mention_key not in mention_keys:
                    raise ValueError("event participant references unknown mention")
        for claim in self.claims:
            if claim.subject_mention_key not in mention_keys:
                raise ValueError("claim subject references unknown mention")
            if (
                claim.object_mention_key is not None
                and claim.object_mention_key not in mention_keys
            ):
                raise ValueError("claim object references unknown mention")
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
2. events：文中敘述的事件；trigger 為觸發詞的原文片段；參與者引用 mention_key；date_span/location_span 為文中的日期、地點片段（若有）。
3. claims：主-謂-賓斷言，主語必須是 mention；賓語是 mention 或字面值；surface 為支持該斷言的原文片段（同樣要求 offset 精確）。
4. topics：這個區塊「關於什麼」的主題標籤（現代漢語，2-8字，如「女子教育」「婚姻習俗」「戲院廣告」），供主題檢索使用。
5. keywords：檢索用關鍵詞（可含原文詞與現代對應詞）。
6. article_type：區塊體裁。
7. modern_paraphrase：用現代白話文忠實概述全文內容（供現代詞彙檢索；不是翻譯每一句）。
8. date_iso：僅當文中明確可定日期時填寫，ISO 格式（可只到年或月）；絕不可依外部知識推測日期。
寧缺毋濫：不確定的不要輸出；絕不編造原文沒有的內容。

邊界規則（必須遵守）：
- 一個名稱只能是一個 mention。名稱中夾有 OCR 雜訊（頓號、空格、換行）時，surface 必須原樣包含這些雜訊、覆蓋整個名稱，絕不可把一個名稱拆成多個 mention，也不可「清理」surface。名稱的規範化屬於後續實體歸併層，不屬於抽取層。
- 不可用現代常識「更正」百年前的譯名或片名；原文寫什麼就是什麼。
- 時間詞（今天、明日、十一月初四）絕不是 person；國別修飾語（如「中國影片」的「中國」）不構成 place mention。
- 廣告的放映時刻表與票價屬於 modern_paraphrase，除非其主語在文中有 mention，否則不構成 claim。

範例（輸入節選：「著名電影\n社會傑作\n傑克葛根\n原作…外加映著名笑林巨子西門拉電…新閘橋堍」）：
mentions 應為 爸爸(product)、傑克葛根(person)、西門拉電(person)、新閘橋堍(place)；
claims 應為 傑克葛根—原作—爸爸、西門拉電—身分—著名笑林巨子；
events 與 event_dates 為空（每日放映時刻表不是可定日期的事件）；
戲院名稱不在本文中，故無戲院 mention，票價不構成 claim，只寫入 modern_paraphrase。"""


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
    def resolved_event(event: EventV2) -> EventV2:
        return event.model_copy(
            update={
                "trigger": _resolved_offsets("event trigger", event.trigger, text),
                "date_span": (
                    _resolved_offsets("event date", event.date_span, text)
                    if event.date_span
                    else None
                ),
                "location_span": (
                    _resolved_offsets("event location", event.location_span, text)
                    if event.location_span
                    else None
                ),
            }
        )

    return result.model_copy(
        update={
            "mentions": [
                _resolved_offsets("mention", item, text) for item in result.mentions
            ],
            "events": [resolved_event(event) for event in result.events],
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
