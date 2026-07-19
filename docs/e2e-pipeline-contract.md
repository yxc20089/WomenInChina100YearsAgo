# End-to-end first-build contract

Status: normative
Date: 2026-07-19

This document is the fixed implementation contract. Older research notes are
comparative history only. A failed or uncertain stage abstains and enters
review; it never invokes another model.

## Fixed architecture

| Responsibility | Selection | Authority boundary |
|---|---|---|
| Source | `s3://ccaa-us-east-1-504133794192/sb_raw/` | Immutable source URI and SHA-256 |
| Layout and OCR | `tencent/HunyuanOCR@de8f10ad2f00a0cefd790b526de8a65dcfdb3205`, toolkit `a1ce1099db98edceb153710536af23edf4391cf0` | Sole learned page-layout and transcription model |
| Semantic extraction | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Ollama `qwen3.5:4b` | One multimodal article-extraction call |
| Local resolution | The same Qwen3.5-4B deployment | One separate multimodal call over validated mention IDs |
| Evidence authority | PostgreSQL 17 + pgvector | Sole source of evidence, review, and local identity state |
| Retrieval | OpenSearch 3.7 CJK + BGE-M3 + reciprocal-rank fusion | Rebuildable lexical/vector projection |
| Graph exploration | Neo4j Community 2026.06 | Rebuildable reviewed-only projection |

PaddleOCR, deterministic ruling-line segmentation, GLiNER, PP-UIE, and OCR or
semantic model fallbacks are not first-build components. Microsoft GraphRAG,
LazyGraphRAG, LightRAG, MiroFish, and Neptune are not production dependencies.
Global identity resolution and canonical merging are deferred.

## Stage contract

1. **Register and render.** List only the authorized S3 prefix. Register object
   URI, checksum, size, media type, and volume. Render PDF/DjVu pages losslessly
   and store immutable page-image identities.
2. **Hunyuan spotting.** Run official `spotting_json` on the immutable page to
   obtain line text and normalized boxes. Restore boxes to full-page pixels and
   retain the raw response.
3. **Hunyuan layout.** Run official `layout_parse` on the same bytes to obtain
   page structure and proposed reading order. Retain the raw response. Use this
   result, not the spotting prompt's enumerated order, as the ordering proposal.
4. **Validate the page artifact.** Require valid JSON/structure, in-bounds
   geometry, line coverage, unique ordering, identical page hashes, and a
   traceable model/configuration identity. Spotting/layout disagreement or a
   consequential missing character creates a review item. It does not trigger
   another OCR model.
5. **Review and activate an article.** Assemble line regions into a coherent
   article revision, including cross-column/page continuations. A reviewer
   activates the revision and selects reviewed text versions before semantics.
6. **Qwen call 1: extraction.** Send the article image page(s), ordered reviewed
   text, line/box IDs, and closed schema in one multimodal request. The response
   proposes mention occurrences, events, roles, dates, places, relationships,
   and evidence references. Code uniquely locates every surface, validates every
   supplied evidence ID, assigns durable IDs and offsets, and rejects the entire
   call on any invalid child.
7. **Qwen call 2: local resolution.** Send the same visual/text context plus the
   validated durable mention and evidence IDs. The response may only return
   clusters of supplied mention IDs and an explicit unresolved-ID list. It may
   not alter text, create a mention, create a global alias, or merge entities.
8. **Persist evidence.** Store immutable model inputs/outputs, mentions, local
   clusters, events, claims, reviews, and their full reverse provenance in
   PostgreSQL. Every assertion traces through span, text version, line polygon,
   page image, and S3 object.
9. **Project and answer.** Build OpenSearch and Neo4j only from the appropriate
   active/reviewed PostgreSQL records. Qwen answers from supplied evidence IDs
   and cannot promote generated prose to historical fact.

## Confidence and review

Hunyuan does not publish calibrated line/layout probabilities in its official
task output, so no self-reported or fabricated confidence is stored. Evidence
rows use the database confidence vocabulary:

- `not_reported`: the model emitted no numeric score; score and calibration are null;
- `uncalibrated`: a score exists but has no applicable held-out calibration;
- `calibrated`: a score and versioned project calibration record both exist.

Human review is an append-only decision, not a confidence value. A review may
select an evidence version without rewriting the original confidence state.

Calibration uses a double-reviewed, adjudicated 100–200 page set. Report CER,
missing/deleted characters, substitutions, hallucinated/inserted characters,
line detection precision/recall, box IoU, and complete reading-order accuracy.
The calibration record freezes dataset, model, configuration, metric, bins,
and held-out results. Confidence affects review eligibility only and never
changes OCR text.

## Local identity semantics

The Holbein example is stored as two immutable occurrences:

```text
M1: 霍爾平
M2: 霍 in 英皇／時召／霍／臨／宮中
L1: local cluster [M1, M2]
```

`M2 refers_to M1` is scoped to one active article revision. It does not make
`霍` a global alias. Both mentions retain separate offsets, polygons, and
evidence spans. Events may point to `L1` as their provisional local actor.

No first-build command builds cross-document candidate pairs, selects a
canonical entity, or writes a redirect. Local profiles remain available as
future input to a separately designed, more powerful global-resolution stage.

## Failure rules

The pipeline abstains and records a review item when an output is blank,
malformed, truncated, out of bounds, cites an unknown ID, cannot map a surface
uniquely, crosses an article boundary, conflicts on consequential reading
order, or lacks applicable confidence calibration. Raw model output and error
details remain immutable. There is no silent retry with a different model.

## Acceptance requirements

- Every OCR line has page coordinates, raw output, image hash, and model/config
  identity.
- Traditional characters are never silently simplified or normalized in place.
- Vertical layout is validated top-to-bottom within columns and right-to-left
  across columns.
- The semantic runner makes exactly two Qwen calls for an article with local
  resolution: extraction, then ID-bounded coreference.
- `霍 → 霍爾平` can be represented locally while both mentions remain intact.
- No global merge is reachable from the first-build planner or installed CLI.
- Every reviewed graph fact opens the exact source page and highlighted span.
- PostgreSQL can rebuild equivalent OpenSearch and Neo4j projections.
