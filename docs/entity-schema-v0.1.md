# Historical knowledge and provenance schema v0.1

Status: design proposal

Date: 2026-07-18

## Purpose

The graph must support historical analysis without separating a conclusion
from the material that produced it. Every mention, entity resolution, event and
claim must be traceable to one or more exact source occurrences. Conversely,
researchers must be able to start from any entity, event or claim and retrieve
all materials that mention or support it.

Canonicalization may merge identities. It must never merge, overwrite or
delete source occurrences.

## Non-negotiable invariants

1. A mention is an immutable occurrence, not an attribute stored on an entity.
2. Every text offset is bound to an immutable transcription version and hash.
3. Every mention or evidence span retains material, page, derivative, region,
   polygon and character-offset provenance when available.
4. An entity may have any number of mentions across any number of materials.
5. A mention may have multiple candidate entity links, but no more than one
   active reviewed resolution at a time.
6. A claim or event may have multiple supporting evidence spans, including
   spans in different materials.
7. Mentioning an entity and supporting a claim about it are distinct relations.
8. Entity merges are append-only redirects. They never rewrite historical
   mention-link or review records.
9. PostgreSQL is the authoritative evidence and review store. Neo4j and search
   reverse indexes are rebuildable projections.
10. Projection or indexing failures must not change authoritative provenance.

## Cardinality

```text
Material 1 -> many Pages
Page 1 -> many Derivatives and Regions
Region 1 -> many immutable TextVersions
TextVersion 1 -> many EvidenceSpans
EvidenceSpan 1 -> zero or many Mentions
Entity 1 <- many reviewed MentionResolutions
Claim 1 -> many ClaimEvidence links -> many EvidenceSpans
Event 1 -> many EventEvidence links -> many EvidenceSpans
Event 1 -> many typed Participants -> Entities or unresolved mentions
```

The derived Entity-to-Material relationship is many-to-many. It is calculated
through mentions and is not stored as a mutable array on the entity.

## Authoritative objects

### Material and text evidence

`Material`

- stable material ID;
- source URI and object SHA-256;
- media type, publication and volume metadata;
- acquisition and integrity provenance.

`PageDerivative`

- page and material IDs;
- exact image URI and SHA-256;
- dimensions, render method and evidence tier;
- immutable relationship to the source object.

`Region`

- page derivative and OCR/layout run IDs;
- stable polygon and reading order;
- region kind and direction.

`TextVersion`

- region ID;
- variant: raw OCR, corrected transcription or approved reconstruction;
- exact text and SHA-256;
- producing run or review decision;
- supersession link without destructive replacement.

`EvidenceSpan`

- text-version ID and exact start/end offsets;
- exact surface text and optional polygon;
- span SHA-256;
- source occurrence ID, even when another occurrence has identical text.

### Mentions and identity

`Mention`

- evidence-span ID;
- entity type and mention form;
- forms include full name, shortened surname, title-only, kinship reference,
  transliteration, pseudonym, anonymous descriptor and pronoun;
- extraction run, score and review state.

`MentionResolution`

- mention ID and proposed entity ID or NIL;
- resolution scope: span, coherent unit, article, issue or corpus;
- candidate features and score;
- review decision and append-only supersession history.

`Entity`

- stable canonical ID and type;
- preferred display name;
- authority identifiers when available;
- reviewed attributes only;
- no embedded list of source IDs or destructive alias list.

`EntityNameAssertion`

- entity ID, name surface, name kind and language/script;
- evidence span or external authority source;
- temporal and contextual scope;
- review state.

A context-scoped shortened mention must not automatically become a global
alias. For example, `霍` may resolve to `霍爾平（Hans Holbein）` within one
article but must not be registered as a globally unique name for that entity.

`EntityRedirect`

- superseded entity ID and canonical entity ID;
- merge decision, reviewer, time and reason;
- reversible status;
- no mutation of the mentions originally linked to either entity.

### Events and claims

`Event`

- event type and reviewed status;
- temporal interval with precision and uncertainty;
- location entity or source literal;
- recurrence/aspect when the text describes habitual activity.

`EventParticipant`

- event ID;
- entity ID or unresolved mention ID;
- controlled participant role;
- evidence and review state.

`Claim`

- subject entity or mention;
- controlled predicate;
- object entity, event or typed literal;
- direct, normalized, inferred or externally supported interpretation level;
- certainty and review state.

`ClaimEvidence` and `EventEvidence`

- independent link IDs;
- claim/event ID and evidence-span ID;
- support role: direct support, context, contradiction or external corroboration;
- review state.

Independent link IDs are required. The same quotation can occur twice in one
region at different offsets and must remain two traceable occurrences.

## Mentioning versus evidencing

Two reverse indexes answer different research questions:

```text
Entity <- REFERS_TO - Mention - ANCHORED_AT -> EvidenceSpan
    "Which materials mention this person?"

Claim/Event - EVIDENCED_BY -> EvidenceSpan
    "Which exact passages support this assertion or event?"
```

Co-occurrence or mention does not by itself support a relationship. Search and
UI responses must label these paths separately.

## Holbein regression example

Source context introduces `霍爾平（Hans Holbein）` and later contains:

```text
英皇時召霍臨宮中
```

Reviewed segmentation:

```text
英皇 / 時召 / 霍 / 臨 / 宮中
```

Required records:

- full-name mention `霍爾平`;
- Latin-name mention `Hans Holbein`;
- reviewed resolution of both full-name mentions to the Holbein entity;
- shortened-surname mention `霍` with article-scoped resolution to Holbein;
- role-title mention `英皇`, with a separate candidate identity link;
- habitual court-summoning event;
- event evidence anchored to the exact target text version and crop;
- negative regression: `霍臨` must never be emitted as a person span.

Reverse traversal from Holbein must return the full-name introduction and the
later shortened mention as distinct occurrences, even though both resolve to
one canonical person.

## PostgreSQL indexes

The authoritative implementation needs indexes equivalent to:

```text
mention(evidence_span_id)
mention_resolution(entity_id, review_status, mention_id)
evidence_span(text_version_id, text_start, text_end)
text_version(region_id, variant, superseded_at)
region(page_id, run_id, reading_order)
page_derivative(page_id, image_sha256)
claim_evidence(claim_id, evidence_span_id)
claim_evidence(evidence_span_id, claim_id)
event_evidence(event_id, evidence_span_id)
event_evidence(evidence_span_id, event_id)
entity_redirect(superseded_entity_id, active)
```

Both directions are indexed because reverse evidence lookup is a primary user
operation, not a rare audit path.

## Neo4j projection

Reviewed projection nodes:

```text
Material, Page, Region, EvidenceSpan, Mention, Entity, Event, Claim
```

Reviewed projection relationships:

```text
Page        -[:PART_OF]-> Material
Region      -[:ON_PAGE]-> Page
EvidenceSpan-[:IN_REGION]-> Region
Mention     -[:ANCHORED_AT]-> EvidenceSpan
Mention     -[:REFERS_TO]-> Entity
Event       -[:EVIDENCED_BY]-> EvidenceSpan
Claim       -[:EVIDENCED_BY]-> EvidenceSpan
Event       -[:PARTICIPANT {role: ...}]-> Entity
Claim       -[:SUBJECT|OBJECT]-> Entity
Entity      -[:REDIRECTS_TO]-> Entity
```

`Entity-[:MENTIONED_IN]->Material` may be materialized as a rebuildable
performance edge with mention count and projection build ID. It cannot replace
the underlying mention path.

## Research API response contract

Entity, event and claim responses must return paginated provenance records:

```text
material ID and source URI
page and derivative IDs
page number and image hash
region and evidence-span IDs
polygon and exact offsets
raw and reviewed text variants
mention surface and resolution decision
claim/event support role
review status
```

The UI should allow a researcher to move from graph insight to the exact scan
crop without another search query.

## Current implementation audit

The existing schema already provides useful foundations:

- multiple `entity_mention` rows can refer to one entity;
- `entity_mention(entity_id)` supports reverse mention lookup;
- `claim_evidence` permits evidence in multiple regions;
- the reviewed Neo4j projection connects entities to regions and pages;
- page nodes retain source and image URIs.

Before schema v0.1 is complete, add:

1. immutable text-version and evidence-span identities;
2. explicit material nodes in the Neo4j projection;
3. event, event-participant and event-evidence tables;
4. append-only mention-resolution and entity-redirect history;
5. independent claim-evidence occurrence IDs;
6. reverse indexes for evidence-to-claim and evidence-to-event queries;
7. regression tests proving that entity merges retain every original material
   occurrence and that repeated identical quotations remain distinct.

## Acceptance tests

The schema is not accepted until automated tests prove:

1. one entity resolves from mentions in multiple materials;
2. reverse lookup returns every distinct material, page and crop;
3. two identical strings at different offsets remain distinct evidence spans;
4. one claim accepts supporting spans from multiple materials;
5. one span can support multiple claims without duplication loss;
6. entity merge and reversal preserve all original mention resolutions;
7. candidate or rejected links never appear as reviewed graph facts;
8. every projected Neo4j evidence path resolves to existing authoritative rows;
9. deleting and rebuilding a projection reproduces the same reverse paths.
