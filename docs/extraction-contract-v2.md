# Article extraction contract `wic-article-extraction/v2` — worked examples

Machine outputs of the one-time frontier extraction pass. Both examples below
were validated through the actual contract code (`ArticleExtractionV2` +
code-resolved offsets) before being written here.

## Design lineage

- **mention / entity split, offsets into evidence text** — MUC-6/7 → ACE →
  OntoNotes named-entity methodology; matches `evidence.entity_mention` +
  `evidence_span`.
- **events: trigger + typed participants + spans** — ACE event model /
  TAC-KBP event tracks; lands in `evidence.event` (dates map onto
  `date_start/date_end/date_precision`).
- **claims: subject–predicate–object with a justifying span** — TAC-KBP slot
  filling with mandatory provenance; lands in `evidence.claim` +
  `claim_evidence` after mention→entity resolution.
- **topics** — GraphRAG-style theme extraction / subject indexing; lands as
  `(:Topic)` graph nodes + facet rows.
- **modern_paraphrase / keywords** — diachronic-vocabulary recall surface;
  search index only, never the graph.
- `article_type` uses the same vocabulary as
  `evidence.coherent_unit_revision.unit_kind` (migration 010).

## Boundary rules (enforced by prompt + validators)

1. A name is ONE mention. OCR noise inside a name (、 spaces, line breaks)
   stays in the verbatim surface; never split a name, never "clean" it,
   never correct period names from modern canon (探、茶女 stays 探、茶女 —
   normalization belongs to entity resolution, not extraction).
2. Offsets are resolved by code from the surface; the model's offsets only
   disambiguate repeats. A surface absent from the text = fabrication =
   whole-article abstention.
3. Time words are never `person`; nationality modifiers (中國影片) are not
   `place` mentions.
4. Recurring schedules are not events; prices/showtimes without a mentioned
   subject are paraphrase content, not claims.
5. `date_iso` never comes from world knowledge. Month/day come from the
   text; the year comes from the issue's publication date (passed as
   context — pending contract addition), bounded by a validator.

## Example 1 — programme advertisement (real block, p0308 閘北 ad)

Input text (region `8d002218-a3fa-5991-a618-739ce9e9af73`):

```
注意
每逢星期四更換新片

爸爸
六人大本
一次演完

著名電影
社會傑作
傑克葛根
原作

外加映著名笑林巨子西門拉電

滑稽
二大本

時間
每天開映
第一次下午一次半
第二次下午三時半
第三次下午五時半
第四次晚上七時半
第五次晚上九時半
第六次晚上十一時

價目
樓上小洋四角
樓下小洋二角
幼童減半

新閘橋堍
```

Ideal output — note the empty `events`, the claim whose evidence span crosses
a newline, and prices/schedule living only in the paraphrase:

```json
{
 "article_type": "advertisement",
 "modern_paraphrase": "影院節目廣告：放映六本長片《爸爸》，一次演完，為傑克葛根原作的著名社會片；加映著名喜劇明星西門拉電的兩本滑稽短片。每天放映六場，自下午一時半至晚上十一時，每逢星期四更換新片。票價樓上小洋四角、樓下小洋二角，兒童半價。地點在新閘橋堍。",
 "topics": [
  {
   "label": "戲院廣告",
   "note": "影院節目與票價資訊"
  },
  {
   "label": "電影放映",
   "note": "美國電影在滬放映"
  }
 ],
 "keywords": [
  "爸爸",
  "傑克葛根",
  "西門拉電",
  "電影",
  "影戲院",
  "新閘橋",
  "滑稽短片"
 ],
 "mentions": [
  {
   "mention_key": "m_film",
   "region_id": "8d002218-a3fa-5991-a618-739ce9e9af73",
   "text_start": 14,
   "text_end": 16,
   "surface": "爸爸",
   "entity_type": "product",
   "mention_form": "named"
  },
  {
   "mention_key": "m_coogan",
   "region_id": "8d002218-a3fa-5991-a618-739ce9e9af73",
   "text_start": 38,
   "text_end": 42,
   "surface": "傑克葛根",
   "entity_type": "person",
   "mention_form": "full_name"
  },
  {
   "mention_key": "m_semon",
   "region_id": "8d002218-a3fa-5991-a618-739ce9e9af73",
   "text_start": 56,
   "text_end": 60,
   "surface": "西門拉電",
   "entity_type": "person",
   "mention_form": "full_name"
  },
  {
   "mention_key": "m_place",
   "region_id": "8d002218-a3fa-5991-a618-739ce9e9af73",
   "text_start": 156,
   "text_end": 160,
   "surface": "新閘橋堍",
   "entity_type": "place",
   "mention_form": "named"
  }
 ],
 "events": [],
 "claims": [
  {
   "claim_key": "c_authorship",
   "subject_mention_key": "m_coogan",
   "predicate": "原作",
   "object_mention_key": "m_film",
   "object_value": null,
   "region_id": "8d002218-a3fa-5991-a618-739ce9e9af73",
   "text_start": 38,
   "text_end": 45,
   "surface": "傑克葛根\n原作"
  },
  {
   "claim_key": "c_fame",
   "subject_mention_key": "m_semon",
   "predicate": "身分",
   "object_mention_key": null,
   "object_value": "著名笑林巨子",
   "region_id": "8d002218-a3fa-5991-a618-739ce9e9af73",
   "text_start": 50,
   "text_end": 60,
   "surface": "著名笑林巨子西門拉電"
  }
 ]
}
```

## Example 2 — news item with a datable event

Input text (region `00000000-0000-0000-0000-000000000001`):

```
王門李氏女士於十二月十日在南京路青年會禮堂與張君完婚證婚者為李校長
```

Ideal output — one self-contained event; `date_iso`'s year comes from issue
context, not the text (see boundary rule 5):

```json
{
 "article_type": "article",
 "modern_paraphrase": "王家李氏女士12月10日在南京路青年會禮堂與張先生結婚，由李校長證婚。",
 "topics": [
  {
   "label": "婚姻習俗",
   "note": ""
  }
 ],
 "keywords": [
  "完婚",
  "婚禮"
 ],
 "mentions": [
  {
   "mention_key": "m_bride",
   "region_id": "00000000-0000-0000-0000-000000000001",
   "text_start": 0,
   "text_end": 4,
   "surface": "王門李氏",
   "entity_type": "person",
   "mention_form": "kinship_reference"
  },
  {
   "mention_key": "m_road",
   "region_id": "00000000-0000-0000-0000-000000000001",
   "text_start": 13,
   "text_end": 16,
   "surface": "南京路",
   "entity_type": "place",
   "mention_form": "named"
  },
  {
   "mention_key": "m_hall",
   "region_id": "00000000-0000-0000-0000-000000000001",
   "text_start": 16,
   "text_end": 21,
   "surface": "青年會禮堂",
   "entity_type": "place",
   "mention_form": "named"
  },
  {
   "mention_key": "m_groom",
   "region_id": "00000000-0000-0000-0000-000000000001",
   "text_start": 22,
   "text_end": 24,
   "surface": "張君",
   "entity_type": "person",
   "mention_form": "short_name"
  },
  {
   "mention_key": "m_officiant",
   "region_id": "00000000-0000-0000-0000-000000000001",
   "text_start": 30,
   "text_end": 33,
   "surface": "李校長",
   "entity_type": "person",
   "mention_form": "title_reference"
  }
 ],
 "events": [
  {
   "event_key": "e_wedding",
   "event_type": "marriage",
   "trigger": {
    "region_id": "00000000-0000-0000-0000-000000000001",
    "text_start": 24,
    "text_end": 26,
    "surface": "完婚"
   },
   "participants": [
    {
     "mention_key": "m_bride",
     "role": "新娘"
    },
    {
     "mention_key": "m_groom",
     "role": "新郎"
    },
    {
     "mention_key": "m_officiant",
     "role": "證婚人"
    }
   ],
   "date_span": {
    "region_id": "00000000-0000-0000-0000-000000000001",
    "text_start": 7,
    "text_end": 12,
    "surface": "十二月十日"
   },
   "location_span": {
    "region_id": "00000000-0000-0000-0000-000000000001",
    "text_start": 13,
    "text_end": 21,
    "surface": "南京路青年會禮堂"
   },
   "date_iso": "1925-12-10"
  }
 ],
 "claims": []
}
```
