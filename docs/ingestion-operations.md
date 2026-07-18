# Ingestion orchestration operations

Status: durable scheduler and verified single-page DAG; stage executors are the
next implementation milestone.

## Contract

`wic-batch` treats PostgreSQL as the authoritative control plane. One page plan
currently contains:

```text
render_lossless -> ocr -> embedding
                       -> ner
```

Every plan freezes the manifest source identity, page scope, stage order and
stage configuration. Its canonical SHA-256 `plan_key` makes an identical plan
idempotent even if a caller changes the human-readable name. Each job records a
separate input fingerprint, artifact URI, output checksum and typed result.

The scheduler does not make OCR or NER output historical truth. NER completion
requires `candidate_only=true`, and reviewed entities/claims still pass through
the separate historian workflow.

## Safe planning

Apply migrations and plan a bounded page before starting workers:

```bash
uv run wic-migrate --database-url "$DATABASE_URL"
uv run wic-batch --database-url "$DATABASE_URL" plan \
  --name 'v219 p308 bounded pilot' --created-by researcher \
  --volume 219 --page 308 \
  --configuration '{"ner":{"max_regions":50}}'
```

The default plan guard is 1,000 pages. The loaded corpus contains 340,511 known
pages, so a corpus-wide plan requires `--allow-large-plan`. That flag should be
used only after estimating source-download, rendering, GPU, storage, indexing,
review-queue and retry capacity. Suspect source objects are excluded unless
`--include-suspect` is supplied deliberately.

## Worker lifecycle

A worker claims one dependency-ready job atomically:

```bash
uv run wic-batch --database-url "$DATABASE_URL" claim \
  --worker "$(hostname)-ocr-1" --stage ocr --lease-seconds 900
uv run wic-batch --database-url "$DATABASE_URL" start \
  --worker WORKER_ID --job-id JOB_UUID
uv run wic-batch --database-url "$DATABASE_URL" heartbeat \
  --worker WORKER_ID --job-id JOB_UUID --lease-seconds 900
```

`claim` uses `FOR UPDATE ... SKIP LOCKED`, so several workers can safely poll in
parallel. Expired leases return to `pending` while attempts remain; the final
expired or reported attempt becomes `failed`. A failure records its type and
message and can delay the next attempt.

Completion requires an artifact URI, its lowercase SHA-256, and stage-specific
metadata:

| Stage | Required result fields |
|---|---|
| `render_lossless` | `render_sha256` |
| `ocr` | `ocr_run_id`, nonnegative `regions` |
| `embedding` | `embedding_run_id`, nonnegative `embeddings` |
| `ner` | `ner_run_id`, nonnegative `mentions`, `candidate_only=true`, and `bounded_regions` exactly equal to the planned `max_regions` |

The scheduler refuses a completion after its lease expires or from a different
worker. It also refuses structurally valid but plan-inconsistent NER metadata.

## Progress and audit

```bash
uv run wic-batch --database-url "$DATABASE_URL" status --batch-id BATCH_UUID
curl http://127.0.0.1:8766/api/ingestion/batches/BATCH_UUID
```

The CLI/API report totals by stage and status plus the currently ready count.
Detailed lifecycle evidence is append-only in
`pipeline.ingestion_job_event`; current state is in
`pipeline.ingestion_job`, and immutable batch scope/configuration is in
`pipeline.ingestion_batch`.

## Verified pilot and limits

The semantic pilot for volume 219, page 308 completed four jobs using the
source-resolution render, 1,099-region PP-OCRv6 run, 1,099 BGE-M3 embeddings,
and a 50-region GLiNER-X run with 82 unreviewed candidates. An attempted
completion claiming 25 bounded regions was rejected; the exact planned value of
50 then succeeded. Reusing these verified artifacts tested orchestration without
spending another model run.

The scheduler currently coordinates jobs but does not yet invoke the rendering,
OCR, embedding or NER commands itself. Failed parent jobs also require an
operator decision before their blocked descendants and batch can reach a
terminal state. Automatic stage dispatch, cancellation propagation, resumable
object caching, aggregate search/RAG/graph projection jobs and operational
metrics are subsequent milestones. Do not describe the current code as an
unattended full-corpus ingestion system.
