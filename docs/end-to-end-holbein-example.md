# End-to-end checkpoint: Holbein on volume 219, page 308

Status: legacy non-gold Paddle checkpoint; production continuation updated to the 2026-07-19 contract
Date: 2026-07-19

The Paddle measurements below are retained only as provenance for the earlier
experiment. New ingestion uses HunyuanOCR 1.5 alone and does not reproduce the
Paddle stages.

## Result

The source-image-to-PostgreSQL candidate path works with exact reverse
provenance. The current local database contains two model-produced candidate
mentions from the real page:

| Surface | Type | Raw-region offsets | Database verification | Status |
|---|---|---:|---|---|
| `英皇` | role title | `[7, 9)` | `substring(raw_text, 7, 9) = 英皇` | candidate |
| `霍` | person, shortened surname | `[11, 12)` | `substring(raw_text, 11, 12) = 霍` | candidate |

The negative controls also hold: the extraction run stored zero `霍臨`
mentions and zero `宮中` entity mentions.

This does **not** yet constitute a reviewed knowledge-graph fact. The required
text-version, evidence-span, reviewed-text-selection, mention-resolution,
reversible-redirect and event tables now exist. The remaining legitimate
blocker is historical review: the Holbein article still needs an active
historian-approved coherent-unit revision and explicit reviewed text
selections. The implementation refuses to manufacture that approval.

## Executed path

```text
S3 source PDF
  -> registered lossless page derivative and image hash
  -> HunyuanOCR spotting_json line boxes and transcription
  -> HunyuanOCR layout_parse structure and reading order
  -> exact OCR region and polygon
  -> reviewed text selection and coherent-unit activation (human gate)
  -> Qwen3.5-4B multimodal mention/event extraction call
  -> exact span validation and durable mention IDs
  -> separate Qwen3.5-4B ID-bounded local-resolution call
  -> historian mention/local-cluster/event decisions
  -> reviewed-only Neo4j projection and reverse evidence
```

## 1. Source material and image

- Source: `s3://ccaa-us-east-1-504133794192/sb_raw/申报影印本219.pdf`
- Source PDF SHA-256:
  `32f8021750cd0fa3ac961f1835681600e3f730d1b78f35e60947cf3f5e7bdfff`
- Volume/page/year: 219 / 308 / 1925
- Page ID: `d1faa016-c303-4586-a535-3e7a70e0fbea`
- Lossless pilot image:
  `artifacts/lossless-pilot/images/v219/p0308.png`
- Image SHA-256:
  `52ea5e9081bdc7039977670d3c0e77ec49a40f050158aa36bb298ac42a48148e`
- Evidence tier: `non_gold_lossless_pilot`
- Target review crop:
  `artifacts/ocr-challenger/suite-v219-p0308/C09_vertical_clean.png`
- Adjacent context crop:
  `artifacts/ocr-challenger/suite-v219-p0308/C09_holbein_context.png`

The evidence tier is deliberately retained throughout the trace. Running an
accurate model does not promote a pilot image or its transcription to gold.

## 2. OCR evidence and disagreement

The active full-page PaddleOCR run is:

```text
run_id: cc2310a1-c174-4598-8360-1742da5d0262
model:  PP-OCRv6_medium_det + PP-OCRv6_medium_rec
revision: paddleocr-3.7.0-official
```

Its target region is:

```text
region_id: 0d7fdcfe-0a26-4e5f-9c06-60808eff5612
reading_order: 365
confidence: 0.8845100402832031
text: 較上次更受歡迎英皇時召霍臨宮中與之談話並
polygon: [(2946,3240), (3031,3240), (3026,4440), (2940,4440)]
```

The targeted OCR comparison is important:

| Source | Target reading |
|---|---|
| active full-page Paddle run | `英皇時召霍臨宮中` |
| targeted Paddle crop | `英皇時召雷臨宮中` |
| HunyuanOCR 1.5 crop | `英皇時召霍臨宮中` |
| single human review | `英皇時召霍臨宮中` |

Thus the active Paddle region and Hunyuan agree with the reviewed character,
but a different Paddle crop configuration loses the critical `霍`. This is
evidence for retaining OCR alternatives and their exact configurations, not
for trusting one model name globally.

The human-reviewed segmentation is:

```text
英皇 / 時召 / 霍 / 臨 / 宮中
```

It makes `霍` the person reference, `臨` a verb, and `宮中` a generic location.
The existing ledger records this as one human review, not independent
adjudicated gold.

## 3. Deterministic candidate and offset layer

The model does not calculate or return authoritative offsets. Software finds
each supplied surface in the immutable OCR-region string and gives the model a
candidate ID plus disambiguating context:

| ID | Surface | `[start,end)` | Left context | Right context |
|---|---|---:|---|---|
| C1 | `英皇` | `[7,9)` | `較上次更受歡迎` | `時召霍臨宮中與之` |
| C2 | `霍` | `[11,12)` | `更受歡迎英皇時召` | `臨宮中與之談話並` |
| C3 | `霍臨` | `[11,13)` | `更受歡迎英皇時召` | `宮中與之談話並` |
| C4 | `宮中` | `[13,15)` | `歡迎英皇時召霍臨` | `與之談話並` |

The database loader repeats the exact check against the registered OCR row.
It refuses the complete artifact if any cited surface differs from
`raw_text[start:end]`.

## 4. Model selection observed on this real phrase

All three calls used native Ollama JSON schema, thinking disabled,
temperature 0, seed 42 and the same four exact candidates. Expected labels
were used only by the post-response checkpoint, not placed in the model input.

| Model | Cold call | Result | Ingested? |
|---|---:|---|---|
| Qwen3.5-0.8B | 1.71 s | ignored the required object wrapper and returned a bare array labeling all four candidates `PERSON_REFERENCE` | no; whole response rejected |
| Qwen3.5-4B | 12.99 s, including 9.88 s model load | exact object and all four correct labels | yes |
| Qwen3.6-35B-A3B | 14.18 s, including 7.49 s model load | exact object and all four correct labels | eligible, but unnecessary for this row |

This one phrase is not an accuracy benchmark. It is enough to reject using
Qwen3.5-0.8B as the production extractor. The first build uses Qwen3.5-4B
directly. There is no 0.8B attempt and no 35B fallback. Invalid 4B output
abstains and enters review instead of changing models.

## 5. Actual PostgreSQL records

Accepted model artifact:

```text
artifacts/e2e/holbein-v219-p0308.qwen35-4b.json
```

Generated NER artifact:

```text
artifacts/e2e/holbein-v219-p0308.ner.json
```

Registered identities:

```text
NER run:      d9baa32e-910f-53a7-9dde-751efe66179a
NER artifact: 5a2d449c-3fac-51f2-b002-f1e0186b6b25
source OCR:   cc2310a1-c174-4598-8360-1742da5d0262
input hash:   618d8ed80c29aee246e26556a746ea0dc478972443ccb4c8afacf34d0eb76e9d
model blob:   sha256:81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490
prompt/schema:4d97137a2bff223464334609684c824cdd0e89c32a486258cc0696dec98755c9
```

Query-back result:

| Mention ID | Surface | Type | Span | Verified from raw text | Entity ID |
|---|---|---|---:|---|---|
| `0dfd1a67-676a-51de-aa9e-50ac65505bce` | `英皇` | role_title | `[7,9)` | `英皇` | NULL |
| `01fdbd6e-6f82-59ea-9c33-54eeaef65a3b` | `霍` | person | `[11,12)` | `霍` | NULL |

Both rows are `candidate`, retain the page image hash and
`non_gold_lossless_pilot` tier, and point back to the exact OCR region. `霍`
also carries `resolution_scope_required=reviewed_coherent_unit_or_article` and
`do_not_register_as_global_alias=true`.

No model-created entity or identity merge occurs during this load.

## 6. Intended reviewed graph outcome

After coherent-unit review plus local-identity and event review,
the intended result is:

```text
Mention("霍爾平") ---------\
                              > LocalIdentityCluster(article revision)
Mention("霍", article-only) /

Event(SUMMON_TO_COURT)
  participant agent:   unresolved/reviewed English-monarch entity via Mention("英皇")
  participant invitee: LocalIdentityCluster(article revision)
  location literal:    "宮中"
  evidence:             exact span "英皇時召霍臨宮中"
```

`宮中` is retained as an event location literal, not promoted to a named-place
entity. `霍` resolves within the reviewed article but never becomes a global
alias. Cross-document Holbein canonicalization is deferred.

## 7. Current gate after implementing the trace

The schema and code path are now implemented. Production semantics starts only
after a historian approves the coherent unit and selects the reviewed text
version for every member region. At that point `wic-e2e` will use the one model
in `config/pipeline-models.toml`, persist candidate mentions/local clusters/
events, and stop again for mention, identity and event review. Only accepted
records enter Neo4j. The remaining checkpoint is therefore a real review act,
not another automatic migration or inferred identity merge.

## Reproduction

```bash
uv run python experiments/e2e/holbein_v219_p0308.py \
  --output artifacts/e2e/holbein-v219-p0308.qwen35-4b.json

uv run --extra data python experiments/e2e/load_holbein_candidates.py \
  artifacts/e2e/holbein-v219-p0308.qwen35-4b.json \
  --output artifacts/e2e/holbein-v219-p0308.ner.json \
  --database-url "$DATABASE_URL"

# After historian text and coherent-unit approval:
uv run --extra data wic-e2e \
  --database-url "$DATABASE_URL" \
  --coherent-unit-revision-id "$HOLBEIN_REVISION_ID" \
  --output-dir artifacts/e2e/holbein-reviewed-unit

```

Loading the same saved model artifact is idempotent at the
artifact/run/mention level. A new model invocation receives a new immutable run
identity even when its response content happens to be identical.
