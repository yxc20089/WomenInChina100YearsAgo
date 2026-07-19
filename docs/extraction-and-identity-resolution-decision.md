# Extraction and identity-resolution decision

Status: normative first-build decision
Date: 2026-07-19

## Decision

Use Qwen3.5-4B for exactly two article-scoped multimodal calls after reviewed
OCR text and article membership exist:

1. one extraction call proposes mentions, events, roles, literals, and supplied
   evidence references;
2. one local-resolution call clusters only the durable mention IDs validated
   from the first call.

Qwen3.5-0.8B is excluded because the project smoke examples missed mentions
and hallucinated entities. GLiNER, PP-UIE, and other extractors are not
fallbacks. Invalid output is an abstention.

## Mention and event boundary

Named mention targets are people/references, named places, organizations,
schools, publications, and explicitly required products. Roles, occupations,
dates, generic locations, event triggers, and addresses are separate typed
records or literals rather than canonical entities.

Every occurrence is immutable. Code, not the model, assigns an end-exclusive
offset only after exact surface/context validation against the selected text
version. Events may refer only to supplied mention/evidence IDs.

## Local resolution

Local resolution is article-scoped coreference, not merging. A cluster stores:

- the exact active coherent-unit revision;
- its model/configuration/input identities;
- supplied mention members and supporting evidence IDs;
- candidate/reviewed/rejected state;
- an explicit unresolved mention roster.

`霍` may resolve to `霍爾平` when both occur in the same article and image
context. The system retains both occurrences and never registers `霍` as a
corpus-wide alias. Unknown IDs, repeated membership, cross-article membership,
or missing evidence rejects the whole resolution response.

## Global resolution

Cross-document matching, authority linking, canonical winner selection, and
entity redirects are outside the first build. Existing experimental schemas or
modules are retained only as dormant research history and are not scheduled by
the ingestion planner or exposed as an installed CLI.

Article-local profiles preserve all names, roles, events, places, dates,
relationships, and evidence paths so a future stronger model can perform global
resolution without rereading or rewriting source mentions.
