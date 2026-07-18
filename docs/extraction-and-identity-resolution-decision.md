# Extraction and identity-resolution decision

Status: first-build decision

Date: 2026-07-18

This decision narrows the extraction and entity-deduplication architecture for
the first working system. It is deliberately conservative: models may create
candidates and scored identity assertions, but they may not silently rewrite
source mentions or canonical identities.

## Decision summary

Use a staged model stack, not one model for both mention extraction and global
identity merging.

| Responsibility | First-build selection | Authority boundary |
|---|---|---|
| Named-mention candidates | `paddlenlp/PP-UIE-0.5B` on a Linux/CUDA worker; `knowledgator/gliner-x-large-v0.5@f41e752e4a44883aa840f7d9dae8cb4f0fcb64db` is the offset-native challenger and Mac development fallback | Candidate spans only; every accepted surface must equal the source codepoints at its stored offsets |
| Deterministic candidates | Gazetteers, reviewed aliases, title/name patterns, date/address parsers and OCR-confusion variants | Candidate spans or retrieval keys only; original text is never normalized in place |
| Local semantic extraction | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, locally served as `qwen3.5:4b` by Ollama | May cluster supplied mention IDs and attach supplied evidence/mention IDs to event frames; may not invent spans or global aliases |
| Cross-document retrieval | `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` plus exact/character/OCR-aware blocking | Generates a high-recall roster; absence from top-k is not a negative identity decision |
| Pair reranking | `Qwen/Qwen3-Reranker-0.6B@e61197ed45024b0ed8a2d74b80b4d909f1255473` | Relevance score only, never treated as an identity probability |
| Ambiguous pair proposal | Qwen3.5-4B with short supplied candidate IDs and `SAME`, `DIFFERENT` or `INSUFFICIENT` output | Proposal only; cannot choose a canonical winner or write a redirect |
| Canonical merge | Historian review during the pilot | Append-only, reversible `EntityRedirect`; no model-only global merge |

Qwen3.5-0.8B is retained only as a latency/abstention control. Qwen3.5-2B
is not selected for semantic extraction or identity judgment. The local smoke
tests below found unsafe false positives from both models.

PP-UIE is selected as the first purpose-built extraction worker because its
official task supports Chinese named entities, relations and events and
publishes a zero-shot F1 of 0.773 on modern Chinese news NER. That is useful
prior evidence, not evidence for Traditional Chinese historical OCR. It must
therefore run in candidate mode until project data are reviewed. Its official
examples return strings rather than authoritative offsets, so repeated or
normalized strings that cannot be mapped to exactly one occurrence abstain.

GLiNER-X remains important because it returns source offsets directly and its
official card reports `zh_pud` F1 0.709. Local tests confirm that Chinese
tokenization produces exact-offset candidates quickly, but also show missed
references and false spans. It is not an automatic replacement for PP-UIE or a
review decision.

## Correct ingestion timing

Deduplication is not one continuously mutating operation. Split it into the
following scopes:

```text
approved coherent-unit revision
  -> exact occurrence-level mention candidates
  -> article-local coreference and event extraction
  -> immutable local identity profiles
  -> issue/batch reaches extraction_complete
  -> idempotent cross-document candidate generation
  -> asynchronous pair reranking and SAME/DIFFERENT/INSUFFICIENT proposals
  -> historian identity decision
  -> append-only mention resolution or entity redirect
  -> rebuild reviewed Neo4j/search projections from PostgreSQL
```

The practical batch boundary is one fully extracted issue, or another fixed
cohort of approved coherent units. Candidate generation can run after every
such batch and can update a review queue continuously. Canonical merging does
not run continuously.

Run a complete reconciliation sweep when the extraction model, ontology,
authority catalog, OCR/correction version or redirect catalog changes. Every
candidate batch must freeze those input hashes so an identical retry is
idempotent and a changed input creates a new run.

### Stage 1: local coreference during ingestion

Resolve shortened names, titles and pronouns only inside one exact coherent
unit revision. For example:

```text
霍爾平 ... 英皇時召霍臨宮中
```

may produce an article-scoped assertion from the exact `霍` occurrence to the
earlier `霍爾平` occurrence. It must not create `霍` as a global Holbein alias.
An isolated `霍` remains unresolved.

The model input contains immutable mention IDs, exact offsets, surfaces and the
coherent text. The model output may contain only clusters of those supplied
mention IDs. Unknown IDs, newly generated text, an out-of-unit mention or a
malformed child causes whole-unit abstention.

### Stage 2: cross-document entity resolution after a batch

Build one structured profile per local cluster. A profile may contain:

- exact source name occurrences and reviewed name assertions;
- dates with precision and OCR confidence;
- places, schools, occupations, offices and relationships;
- source issue/article IDs and evidence span IDs;
- external authority candidates and their provenance.

Candidate generation is a union of exact aliases, Traditional/Simplified and
OCR-confusion variants used only for retrieval, character n-grams, authority
IDs and Qwen3 embeddings. Retrieve a broad roster, then rerank structured
profile pairs. The reranker score is an input feature, not a calibrated
probability and not a merge decision.

Qwen3.5-4B receives exactly one bounded pair at a time, with short identifiers
such as `LEFT`, `RIGHT` and evidence IDs. It returns only:

```json
{
  "decision": "SAME | DIFFERENT | INSUFFICIENT",
  "supporting_evidence_ids": [],
  "contradiction_evidence_ids": []
}
```

It cannot introduce a name, fact, entity ID or canonical winner. Invalid
output abstains. Self-reported confidence is ignored.

### Stage 3: reviewed canonicalization

During the pilot, every global entity/entity merge is human-only. An accepted
decision appends a reversible redirect from the losing entity ID to the chosen
canonical ID. It never deletes or rewrites mentions, evidence spans, claims,
old resolution proposals or review records.

Do not construct canonical entities by taking connected components of pairwise
scores. A single bad edge would transitively merge unrelated people. Before a
reviewer can accept a redirect, the system must show all profile evidence and
cluster-wide contradictions.

`孫文`, `孫中山` and `逸仙` can become typed, provenance-bearing name
assertions for one reviewed entity when source or authority evidence supports
that identity. A one-character contextual reference such as `霍` remains
scoped to its article even after its mention resolves to Holbein.

## Extraction contract correction

The current ontology mixes named entities with other graph concepts. That
confuses both generic LLMs and span models. Separate the contracts before
scaling ingestion.

### Mention targets

- person name;
- person reference;
- place name;
- organization or institution name;
- school name;
- publication name;
- named product, only where research questions require it.

Every occurrence is a separate immutable mention, including two identical
`王氏` surfaces at different offsets.

### Mention forms, not entity types

- full name;
- alternate name, style name or pseudonym;
- Latin/transliterated name;
- shortened surname;
- title-only reference;
- anonymous descriptor;
- kinship reference;
- pronoun.

`alias` must not be a standalone entity type. It is a typed name assertion or
a local identity relation between occurrences.

### Separate graph records

- occupations and roles are attributes or evidence-backed relations;
- dates and addresses are typed literals or normalized records;
- events have triggers, participants and evidence spans;
- `advertisement` is document/region genre, not an entity;
- a generic location phrase such as `宮中` may remain an event location
  literal rather than creating a named Place node.

This separation prevents an extractor from treating `畫家`, `作畫` or a
document genre as a canonical entity merely because they appear in the same
label list as people and schools.

## Local smoke evidence

Environment: Ollama 0.24.0 on Apple M4 Pro with 48 GB unified memory,
temperature 0 and seed 42. These are schema regressions, not a historical NER
benchmark.

| Model/runtime identity | Relevant observation |
|---|---|
| `qwen3.5:0.8b`, Ollama ID `f3817196d142` | Fast, normally about 0.3-1.2 seconds on the short fixtures, but returned only `霍爾平` in the Holbein example, omitted alternate names and repeated occurrences, and hallucinated `王女士` for `愛寵情人，不僅悅目娛心。` under the existing prompt. As a bounded resolver it linked isolated `霍` and explicitly distinct `王氏`. |
| `qwen3.5:2b`, Ollama ID `324d162be6ca`, blob `b709d81508a078a686961de6ca07a953b895d9b286c46e17f00fb267f4f2d297` | Higher recall but emitted dangerous surfaces/types including `霍臨宮中`, `英皇時`, `字逸仙` and a false publication for `愛寵情人`; prompt tightening still emitted `霍臨`. Its bounded resolver also linked isolated `霍` and explicitly distinct `王氏`. |
| `qwen3.5:4b`, Ollama ID `2a654d98e6fb`, blob `81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490` | With the occurrence/coreference contract it correctly emitted exact `霍` rather than `霍臨` and linked it locally to `霍爾平`. It nevertheless falsely linked unrelated Huo/Wang examples when asked to combine extraction and coreference. In a short-ID bounded resolver it correctly handled the Holbein antecedent, isolated `霍`, Sun aliases and the explicit distinct-Wang negative at roughly one second per pair after warm-up, but still over-linked a same-name profile. |
| `knowledgator/gliner-x-large@4a4437f439a78d67c87781b42e8c45373d2adcb0`, CPU, GLiNER 0.2.27 | With `zh-hant` Stanza tokenization it produced exact offsets and processed all six short fixtures in about 0.4 seconds. At threshold 0.6 it recovered both `王氏` occurrences, `王女士`, `上海女子學校`, `霍女士`, `霍爾平`, `孫文` and `孫中山`, but missed `逸仙`, `英皇` and the shortened `霍`, and emitted a false place span `召霍臨宮`. This older checkpoint is evidence for the model role, not selection of the new v0.5 checkpoint. |

The first direct UUID-based entity-resolution contract also failed schema
validation for all three Qwen sizes because Ollama returned missing or
unexpected fields. A smaller candidate vocabulary (`C1`, `C2`, `NIL`) plus an
explicit exact JSON template produced parseable output. The implementation
must therefore use short run-local candidate tokens, strict post-validation
and whole-pair abstention. UUIDs remain authoritative internally and are mapped
to short tokens in the frozen request manifest.

## Database responsibilities

| System | Responsibility |
|---|---|
| S3 | Immutable source objects and versioned OCR/transcription derivatives |
| PostgreSQL | Authoritative evidence, occurrence mentions, local clusters, entity profiles, candidates, pair proposals, reviews, entities and append-only redirects |
| `pgvector` in PostgreSQL | First-build embedding index for local profiles and canonical entities; avoids a separate vector authority or synchronization path |
| Neo4j | Rebuildable reviewed graph projection for traversal and graph analytics |
| Search index | Optional rebuildable full-text/reverse-evidence projection; add only when PostgreSQL search is insufficient |

PostgreSQL, not Neo4j, decides which resolution or redirect is active. Every
accepted resolution/redirect appends a durable projection-rebuild request.
Neo4j must resolve entities through the active redirect closure while retaining
the original mention/entity IDs and exact evidence paths. A failed projection
must leave the last completed projection available.

## Promotion policy

For the first build:

1. Accept only exact-offset mention candidates into the review/evidence store.
2. Permit high-precision local coreference proposals, but preserve unresolved
   mentions and all alternatives.
3. Continuously refresh candidate queues after completed ingestion batches.
4. Keep global pair decisions as reviewable assertions.
5. Require historian review for every canonical merge and reversal.
6. Rebuild graph/search projections after accepted identity changes.

Automatic cross-document links may be introduced later only after calibrated
project evidence demonstrates a threshold whose lower confidence bound meets
the required precision. Model or reranker scores alone are not probabilities.

## Immediate implementation consequences

The repository already has candidate-bounded `LINK`/`NIL`/`ABSTAIN` proposals,
but the authoritative schema is missing several pieces needed by this decision:

- mentions are not bound to an approved coherent-unit revision;
- there is no immutable local-coreference run/cluster contract;
- reviewed mention resolution currently mutates a mention rather than
  appending resolution history;
- there is no entity redirect/merge/reversal table or cycle-safe transaction;
- identity candidate idempotency is not based on a semantic pair/input hash;
- graph builds do not bind an identity/redirect snapshot and may omit evidence
  associated with merged entities.

Implement those gaps as a separate identity-resolution plan after coherent-unit
approval. Do not extend the existing page-level entity-link stage into a
continuous canonical merge job.

## Primary sources

- [PaddleNLP PP-UIE documentation](https://paddlenlp.readthedocs.io/zh/latest/llm/application/information_extraction/README.html)
- [GLiNER-X large v0.5 model card](https://huggingface.co/knowledgator/gliner-x-large-v0.5)
- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Qwen3-Reranker-0.6B model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
