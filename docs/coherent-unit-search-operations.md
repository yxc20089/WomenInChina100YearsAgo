# Reviewed article embedding and search operations

This runbook operates the reviewed coherent-unit article corpus. PostgreSQL is
authoritative; pgvector stores one versioned vector per active reviewed article
revision, and OpenSearch is a rebuildable projection used for retrieval.

The existing region corpus remains the default everywhere:

- `wic-embed` defaults to `--unit region`;
- `wic-search` defaults to `--unit region` and `wic-regions-current`;
- `POST /api/search` defaults to `"corpus": "region"`.

The article path must be selected explicitly as `reviewed_coherent_unit`. It
uses the independent `wic-coherent-units-current` alias and never moves the
region alias.

## Architecture, data flow, and eligibility

The normal flow is:

```text
active reviewed article revisions
  -> canonical selected text + input/content SHA-256
  -> coherent_unit_embedding job per revision
  -> exact BGE-M3 vector in retrieval.embedding
  -> one coherent_unit_search_projection fan-in job
  -> wic-coherent-units-build-<build UUID>
  -> atomic alias move of wic-coherent-units-current
```

A revision is eligible only when all of the following are true:

- `evidence.coherent_unit_revision.unit_kind = 'article'`;
- the revision and its approval segmentation selection are not superseded;
- it has at least one coherent-unit span;
- every span has one active `evidence.region_text_selection` whose selected
  text version has `review_status = 'reviewed'`;
- the selected spans can be materialized without missing or ambiguous corrected
  text alignment, and their immutable OCR page derivatives are available.

Machine-proposed windows are not articles. Changing a reviewed text selection
changes the canonical input/content identity; old embeddings remain provenance
records but are not selected for a new projection. The application paths that
activate reviewed segmentation or accept and select a reviewed text version
call coherent enqueue in the same PostgreSQL transaction. A leased embedding
job that later observes a different active revision identity completes as an
explicit stale no-op and enqueues reconciliation. These are not a universal
database trigger: if a change bypasses those application paths, or no pending
job observes it, an operator must run `wic-batch coherent-backfill` to plan the
current snapshot. A previously completed batch does not monitor future changes.

## Prerequisites

Install the API, PostgreSQL/OpenSearch, model, and test dependencies, start the
local services, and apply all migrations including
`db/migrations/022_coherent_unit_article_search.sql`:

```bash
uv sync --extra api --extra data --extra ner --extra test
docker compose up -d postgres opensearch
uv run wic-migrate --database-url "$DATABASE_URL"
curl --fail --silent "$OPENSEARCH_URL/_cluster/health?pretty"
```

`DATABASE_URL` and `OPENSEARCH_URL` must identify the intended environment.
The pinned retrieval model is read from
`config/pipeline-models.toml`: `BAAI/bge-m3`, revision
`5617a9f61b028005a4858fdac845db406aefb181`, 1,024 dimensions, normalized.
The first embedding run may need network access to populate the Hugging Face
model cache; production workers should use a pre-populated, verified cache.

Before a backfill, use these preliminary candidate and text-selection checks:

```sql
SELECT count(*) AS active_articles
FROM evidence.coherent_unit_revision revision
JOIN evidence.page_article_segmentation_selection approval
  ON approval.selection_id = revision.approval_selection_id
 AND approval.superseded_at IS NULL
WHERE revision.superseded_at IS NULL
  AND revision.unit_kind = 'article';

SELECT revision.revision_id, revision.title,
       count(span.region_id) AS spans,
       count(version.text_version_id) AS reviewed_selected_spans
FROM evidence.coherent_unit_revision revision
JOIN evidence.page_article_segmentation_selection approval
  ON approval.selection_id = revision.approval_selection_id
 AND approval.superseded_at IS NULL
LEFT JOIN evidence.coherent_unit_span span
  ON span.revision_id = revision.revision_id
LEFT JOIN evidence.region_text_selection selection
  ON selection.region_id = span.region_id
 AND selection.superseded_at IS NULL
LEFT JOIN evidence.text_version version
  ON version.text_version_id = selection.text_version_id
 AND version.region_id = span.region_id
 AND version.review_status = 'reviewed'
WHERE revision.superseded_at IS NULL
  AND revision.unit_kind = 'article'
GROUP BY revision.revision_id, revision.title
HAVING count(span.region_id) = 0
    OR count(span.region_id) <> count(version.text_version_id)
ORDER BY revision.revision_id;
```

These queries are triage only, not authoritative eligibility or publication
gates. They do not prove contiguous unique span sequence numbers, valid
non-empty raw intervals, selected-text hash integrity, a usable correction
alignment for partial spans, canonical selected offsets, complete source/image
provenance joins, canonical input/content hashes, or stability across concurrent
review mutations. The real materializer is authoritative: run the guarded
backfill and embedding workers, and treat any materialization/backfill failure
as an eligibility failure to investigate rather than overriding it from these
counts.

## First backfill and worker execution

Create one content-addressed plan for the complete current active snapshot. The
guard defaults to 1,000 revisions; increase it deliberately only after checking
CPU time, model-cache availability, pgvector capacity, and OpenSearch capacity:

```bash
uv run wic-batch --database-url "$DATABASE_URL" coherent-backfill \
  --created-by 'operator-name' --max-revisions 1000
```

Save the returned `batch_id` and `plan_key`. Run bounded embedding workers, then
the projection worker. The projection job remains dependency-blocked until all
per-revision jobs complete:

```bash
uv run wic-worker --database-url "$DATABASE_URL" \
  --opensearch-url "$OPENSEARCH_URL" --worker "$(hostname)-article-embed-1" \
  --batch-id BATCH_UUID --stage coherent_unit_embedding \
  --loop --max-jobs 100 --idle-polls 3 --poll-seconds 5

uv run wic-batch --database-url "$DATABASE_URL" status --batch-id BATCH_UUID

uv run wic-worker --database-url "$DATABASE_URL" \
  --opensearch-url "$OPENSEARCH_URL" --worker "$(hostname)-article-project-1" \
  --batch-id BATCH_UUID --stage coherent_unit_search_projection
```

Run several embedding worker processes when needed; claims use PostgreSQL row
locks and `SKIP LOCKED`. Keep projection concurrency at one operator-controlled
process because it publishes a global alias.

An identical backfill over the same active revisions and pinned configuration
returns the existing batch (`created: false`). Exact completed embeddings are
reused. A failed job can be replayed only after the external cause is fixed:

```bash
uv run wic-batch --database-url "$DATABASE_URL" failures --batch-id BATCH_UUID
uv run wic-batch --database-url "$DATABASE_URL" replay \
  --job-id FAILED_JOB_UUID --requested-by 'operator-name' \
  --reason 'documented external fix'
```

## Direct commands for QA or manual recovery

The scheduler is the preferred first-build path because it freezes the whole
active snapshot. To repair or validate one article vector directly:

```bash
uv run wic-embed --database-url "$DATABASE_URL" \
  --unit reviewed_coherent_unit --revision-id REVISION_UUID --batch-size 16
```

Omit `--revision-id` to process every currently eligible article. The command
uses the complete model configuration; it does not accept individual model
overrides. It prints inserted/reused counts and deterministic run IDs.

Lexical QA needs no embedding pins:

```bash
uv run wic-search --opensearch-url "$OPENSEARCH_URL" query '妇女教育' \
  --unit reviewed_coherent_unit --mode lexical --limit 5
```

Dense and hybrid queries must supply the exact identity recorded in the index:

```bash
uv run wic-search --opensearch-url "$OPENSEARCH_URL" query '妇女教育' \
  --unit reviewed_coherent_unit --mode hybrid --limit 5 \
  --model 'BAAI/bge-m3' \
  --revision '5617a9f61b028005a4858fdac845db406aefb181' \
  --configuration-sha256 "$EMBEDDING_CONFIGURATION_SHA256"
```

Direct projection is intended for controlled re-projection using a previously
validated manifest snapshot identity, not for guessing a new snapshot hash:

```bash
uv run wic-search --opensearch-url "$OPENSEARCH_URL" project \
  --database-url "$DATABASE_URL" --unit reviewed_coherent_unit \
  --model 'BAAI/bge-m3' \
  --revision '5617a9f61b028005a4858fdac845db406aefb181' \
  --configuration-sha256 "$EMBEDDING_CONFIGURATION_SHA256" \
  --snapshot-sha256 "$PROJECTION_MANIFEST_SHA256"
```

Get those values from a completed coherent projection receipt, or inspect them
without modifying state:

```sql
SELECT job.batch_id, job.input_fingerprint AS planned_snapshot_sha256,
       job.configuration->>'embedding_configuration_sha256'
         AS embedding_configuration_sha256,
       job.result->>'projection_manifest_sha256'
         AS projection_manifest_sha256,
       job.result->>'index_name' AS index_name
FROM pipeline.ingestion_job job
WHERE job.stage = 'coherent_unit_search_projection'
  AND job.status = 'completed'
ORDER BY job.completed_at DESC
LIMIT 1;
```

The direct coherent projection always creates a fresh
`wic-coherent-units-build-<UUID without hyphens>` index, validates the strict
mapping, bulk result, refresh, and document count, then atomically moves
`wic-coherent-units-current`. The `--index`, `--alias`, and `--recreate` options
belong to the region path and do not override coherent index naming.

## API selection

Start the local API normally. Coherent lexical search needs only the corpus
selector. Coherent dense/hybrid search additionally reads its pinned identity
from the environment:

```bash
export COHERENT_EMBEDDING_MODEL='BAAI/bge-m3'
export COHERENT_EMBEDDING_REVISION='5617a9f61b028005a4858fdac845db406aefb181'
export COHERENT_EMBEDDING_CONFIGURATION_SHA256="$EMBEDDING_CONFIGURATION_SHA256"
uv run wic-api --host 127.0.0.1 --port 8766 \
  --database-url "$DATABASE_URL" --opensearch-url "$OPENSEARCH_URL"

curl --fail --silent --show-error \
  -H 'content-type: application/json' \
  -d '{"query":"妇女教育","mode":"hybrid","corpus":"reviewed_coherent_unit","limit":5}' \
  http://127.0.0.1:8766/api/search
```

Coherent responses use retrieval schema 1.1 and provide ordered `sources` for
the article; the singular legacy `source` is null. `/api/context`,
`/api/generate`, and `/api/chat` currently reject
`corpus=reviewed_coherent_unit` with HTTP 422. They remain region-only.

## Snapshot identity and safe reruns

Publication is strict about four related identities:

- `input_sha256`: canonical identity of the revision, selected text versions,
  selected intervals, and materializer schema;
- `content_sha256`: SHA-256 of the exact newline-joined article text;
- `configuration_sha256`: deterministic windowing/vector configuration
  (`windowed_mean_v1`, effective token limit, overlap, dimension), not merely a
  model name;
- model name and immutable model revision.

The scheduler `plan_key` also freezes the sorted active revision IDs, their
input/content hashes, and the full coherent embedding/projection configuration.
The projection manifest hash additionally covers the exact articles,
embeddings, and source provenance being indexed. Projection refuses missing,
duplicate, stale, or mixed-identity embeddings.

Long articles are split into overlapping token windows; normalized window
vectors are mean-aggregated and normalized again into one 1,024-dimensional
article vector. Re-running unchanged input reuses the deterministic completed
embedding. A changed selection or configuration produces a different identity
instead of overwriting the prior record.

## Monitoring

Use the scheduler status commands first, then inspect exact coherent jobs:

```sql
SELECT job.job_id, job.stage, job.status, job.attempt_count, job.max_attempts,
       job.coherent_unit_revision_id, job.input_fingerprint,
       job.lease_owner, job.lease_expires_at, job.error_details,
       job.result
FROM pipeline.ingestion_job job
WHERE job.batch_id = 'BATCH_UUID'::uuid
ORDER BY job.stage, job.created_at, job.job_id;

SELECT event.job_id, event.event_type, event.worker_id, event.details,
       event.occurred_at
FROM pipeline.ingestion_job_event event
JOIN pipeline.ingestion_job job USING (job_id)
WHERE job.batch_id = 'BATCH_UUID'::uuid
ORDER BY event.occurred_at, event.event_id;
```

The following vector grouping is historical inventory only. It includes old
revision/configuration identities and must never authorize publication:

```sql
SELECT embedding.model_name, embedding.model_revision,
       embedding.configuration_sha256, count(*) AS vectors
FROM retrieval.embedding embedding
JOIN evidence.processing_run run USING (run_id)
WHERE embedding.target_kind = 'coherent_unit_revision'
  AND run.status = 'completed'
GROUP BY embedding.model_name, embedding.model_revision,
         embedding.configuration_sha256
ORDER BY vectors DESC;
```

Use the successful, exact-snapshot scheduler receipt as the publication gate.
The projection worker rematerializes the active snapshot under the coherent
snapshot lock, verifies the current plan key, exact input/content/config/model
embedding coverage, manifest hash, projected count, and alias publication
before the scheduler accepts completion. Inspect that receipt together with its
durable build row:

```sql
SELECT job.job_id, job.batch_id, job.status,
       job.input_fingerprint AS planned_snapshot_sha256,
       job.configuration->>'planned_revision_count' AS planned_revision_count,
       job.result->>'projection_manifest_sha256'
         AS projection_manifest_sha256,
       job.result->>'documents_indexed' AS documents_indexed,
       job.result->>'index_name' AS index_name,
       build.status AS build_status,
       build.source_snapshot_sha256, build.document_count,
       build.published_at, build.completed_at
FROM pipeline.ingestion_job job
JOIN retrieval.projection_build build
  ON build.build_id = (job.result->>'projection_build_id')::uuid
WHERE job.stage = 'coherent_unit_search_projection'
  AND job.status = 'completed'
  AND build.projection_kind = 'opensearch_coherent_unit'
ORDER BY job.completed_at DESC
LIMIT 10;

SELECT build_id, status, artifact_uri, source_snapshot_sha256,
       document_count, published_at, completed_at
FROM retrieval.projection_build
WHERE projection_kind = 'opensearch_coherent_unit'
ORDER BY completed_at DESC NULLS LAST
LIMIT 10;
```

Check OpenSearch without changing it:

```bash
curl --fail --silent "$OPENSEARCH_URL/_alias/wic-coherent-units-current?pretty"
curl --fail --silent "$OPENSEARCH_URL/wic-coherent-units-current/_count?pretty"
curl --fail --silent \
  "$OPENSEARCH_URL/wic-coherent-units-current/_mapping?pretty"
```

Alert on failed/dead-letter jobs, expired leases, projection document-count
mismatch, an alias pointing to zero or multiple indexes, or an index whose
document embedding identity differs from the worker plan.

## Rollout and rollback

Use this canary sequence:

1. Apply migration 022 and confirm region search is unchanged.
2. Activate and select reviewed text for a small article set.
3. Run a guarded coherent backfill and bounded embedding workers.
4. Inspect the planned per-revision identities and require every embedding
   dependency to complete; do not use historical vector grouping as coverage.
5. Run the projection worker. Require its successful exact-snapshot receipt,
   then compare the durable SQL `document_count` with the OpenSearch alias
   count before serving coherent traffic.
6. Exercise lexical, dense, and hybrid CLI queries, then `/api/search` with an
   explicit coherent corpus.
7. Increase the reviewed article set in bounded batches. Do not switch the
   region defaults during this rollout.

If embedding fails, leave the alias untouched, fix the dependency/data issue,
and replay the failed job. If projection fails before publication, the previous
alias remains searchable. When the alias has moved but the subsequent
PostgreSQL projection receipt fails, the worker attempts to restore the
previous alias only if no other writer has changed it. Compensation is refused
after concurrent alias movement, and an OpenSearch failure during publication
may require operator inspection. Build indexes are not automatically deleted.

To roll back, stop coherent projection workers, identify the last known-good
index from `retrieval.projection_build` and the current alias, then make one
atomic OpenSearch alias update:

```bash
curl --fail --silent --show-error -X POST \
  -H 'content-type: application/json' \
  "$OPENSEARCH_URL/_aliases" \
  -d '{"actions":[
    {"remove":{"index":"CURRENT_BAD_INDEX","alias":"wic-coherent-units-current"}},
    {"add":{"index":"LAST_GOOD_INDEX","alias":"wic-coherent-units-current"}}
  ]}'
```

Confirm the alias and count after rollback. Preserve the failed build index and
PostgreSQL job/event records until the incident is understood. Cancelling a
batch preserves completed artifacts:

```bash
uv run wic-batch --database-url "$DATABASE_URL" cancel \
  --batch-id BATCH_UUID --cancelled-by 'operator-name' \
  --reason 'documented rollback reason'
```

## Live QA checklist

- PostgreSQL migrations finish successfully and migration 022 constraints are
  present.
- Preliminary eligibility SQL reports only understood candidates/incomplete
  selections; the authoritative materializer/backfill completes without an
  eligibility failure.
- The backfill receipt has the expected revision/job counts; an identical
  rerun reports `created: false`.
- Every embedding job completes or explicitly records a stale no-op and
  reconciliation; exact embedding rows have all three lowercase hashes.
- The projection job runs only after its embedding dependencies and records a
  completed `opensearch_coherent_unit` build.
- `wic-coherent-units-current` points to exactly one managed build index and its
  `_count` equals the projection receipt.
- CLI lexical, dense, and hybrid queries return schema 1.1 article hits with
  ordered source spans and no embedding payload.
- `POST /api/search` works with explicit coherent corpus; omitting `corpus`
  still searches regions.
- `/api/context`, `/api/generate`, and `/api/chat` return 422 for the coherent
  corpus until that integration is implemented.
- A failed-job replay and, in a disposable environment, an alias rollback have
  been exercised before a large rollout.

## Current limitations

- Only active reviewed `unit_kind='article'` revisions are supported.
- Region embedding, projection, search, and API behavior remain the defaults.
- BGE-M3 runs on CPU here and needs its pinned model available in the local
  cache or via network on first load.
- Projection and retrieval QA require a live compatible OpenSearch service;
  unit tests or source inspection do not validate cluster behavior.
- Context, generation, and chat do not yet consume coherent article hits.
- Alias compensation is conditional and cannot safely override a concurrent
  publisher; operator reconciliation may be required.
