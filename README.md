# Women in China 100 Years Ago

Evidence-first tools and research notes for reconstructing women's history from the digitized *Shen Bao* archive.

The normative implementation is [the end-to-end first-build contract](docs/e2e-pipeline-contract.md).
The [technical design](docs/technical-design.md) retains earlier comparisons
as research history. The active stack is HunyuanOCR 1.5 for both page tasks,
Qwen3.5-4B for one multimodal extraction call plus one article-local resolution
call, PostgreSQL as authority, OpenSearch for hybrid retrieval, and Neo4j as a
reviewed-only projection. There is no model fallback or first-build global
entity merge.

## Corpus audit

The audit command is read-only with respect to S3. It lists objects and reads only small byte ranges for container validation.

```bash
python -m wic_history.corpus_manifest \
  --bucket ccaa-us-east-1-504133794192 \
  --prefix sb_raw/ \
  --output-dir artifacts/corpus-audit \
  --pdf-page-counts \
  --profile your-read-only-profile
```

For the existing IAM CSV, use `--credentials-csv /path/to/accessKeys.csv`. The file is read in memory; keys are not written to output or logs. Prefer an AWS profile or temporary role for regular use.

Outputs:

- `manifest.jsonl`: canonical machine-readable inventory;
- `manifest.csv`: analyst-friendly inventory;
- `summary.json`: counts, sizes and validation results;
- `potential_duplicates.json`: candidates only, grouped by size and ETag.

`--pdf-page-counts` follows classic PDF cross-reference and page-tree objects with bounded range reads. Unsupported PDFs remain explicitly unresolved; the command never falls back to downloading a whole volume.

Run tests without AWS access:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

For the application/test environment, use
`uv sync --extra api --extra data --extra test`. Hunyuan uses the separate
CUDA environment in `environments/hunyuan-ocr`; the dormant GLiNER research
extra is intentionally not co-installed with its newer Transformers runtime.

Create the deterministic visual-screening page plan after the audit:

```bash
PYTHONPATH=src python -m wic_history.benchmark_sample
```

See [docs/corpus-audit.md](docs/corpus-audit.md) for current findings and limitations.

Render one selected PDF volume into non-authoritative screening JPEGs:

```bash
PYTHONPATH=src python -m wic_history.render_samples \
  --volume 219 \
  --credentials-csv /path/to/accessKeys.csv
```

Source volumes are cached under `/tmp/wic-source-cache` by default. Generated screening images are reproducible and excluded from Git; their hashes and rendering parameters are recorded in `artifacts/benchmark-pages/render_manifest.jsonl`.
The default 120-DPI JPEG is only for visual screening. Gold OCR pages must later be rendered losslessly at source resolution.
DjVu screening requires DjVuLibre (`brew install djvulibre` on macOS); the executable-reported version is recorded in render metadata.

After historians mark complete screening records as `gold_status=include`,
render only those pages from the verified source cache or read-only S3 source:

```bash
uv run wic-gold-render --offline
# Omit --offline to download a selected volume that is absent from the cache.
```

`wic-gold-render` refuses incomplete/anonymous selections, composited or rotated
PDF pages that cannot be extracted without an explicit transform decision, and
source cache mismatches. It directly decodes a single full-page source raster
or native DjVu page, writes PNG without geometric resampling, and records source
object, output-file, and decoded-pixel hashes. An explicitly non-gold plumbing
check is available as `--pilot-sample-id`; its output can never be counted as
gold by the manifest summary.

Run the two official Hunyuan tasks once through the layout command, then
materialize OCR only from that exact paired output. The commands verify the
image, model configuration, raw-output hashes, and layout/OCR agreement:

```bash
wic-layout \
  --image artifacts/lossless-pilot/images/v219/p0308.png \
  --render-manifest artifacts/lossless-pilot/lossless_manifest.jsonl \
  --source-uri 's3://ccaa-us-east-1-504133794192/sb_raw/申报影印本219.pdf' \
  --volume 219 --page 308 --year 1925 \
  --output artifacts/layout/v219-p0308.hunyuan.json
wic-ocr \
  --image artifacts/lossless-pilot/images/v219/p0308.png \
  --layout-artifact artifacts/layout/v219-p0308.hunyuan.json \
  --output artifacts/ocr/v219-p0308.hunyuan.json
```

The manifest selection controls whether the artifact is historian-selected gold
or an explicitly non-gold lossless pilot; the command cannot promote a page on
its own.

Start the local visual review UI:

```bash
PYTHONPATH=src python -m wic_history.review_server
```

Open `http://127.0.0.1:8765`. Reviews are stored atomically in `artifacts/benchmark-review/annotations.json`. The server binds to localhost by default and has no authentication; do not expose it on a public interface.

After lossless pages have active OCR selections in PostgreSQL, build a blinded,
content-addressed NER annotation packet:

```bash
uv run wic-gold-packet build --database-url "$DATABASE_URL" \
  --dataset-id shenbao-ner-pilot-v1 --volume 219 --page 308 \
  --max-units 50 --context-radius 2 \
  --output artifacts/gold-packet-pilot/packet.json \
  --reviewer-view artifacts/gold-packet-pilot/reviewer-view.json \
  --template artifacts/gold-packet-pilot/annotations-template.json
```

The administrative packet records balanced sampling reasons; give reviewers
only the blinded view plus their own copy of the annotation template. The
builder verifies the registered lossless image hash and labels the result
`annotation_candidate`. It reports explicit eligibility failures rather than
calling a small pilot gold. After two independent passes and adjudication,
validate and freeze NER gold schema 1.1 with:

```bash
uv run wic-gold-packet finalize \
  --packet artifacts/gold-packet-pilot/packet.json \
  --annotations artifacts/gold-packet-pilot/completed-annotations.json \
  --output artifacts/gold/ner-v1.json
```

Finalization rejects incomplete units, duplicate reviewers, mismatched text
offsets/surfaces, a changed packet hash, and reuse of a model OCR region UUID as
the independent gold identity. See
[docs/gold-annotation-packets.md](docs/gold-annotation-packets.md).

Generate immutable coherent-unit candidates only after OCR selection. Machine
windows are not articles and do not enter reviewed retrieval:

```bash
uv run wic-segment --database-url "$DATABASE_URL" propose \
  --max-regions 24 --max-characters 600 \
  --proposed-by deterministic-baseline-v1
```

An accepted named review and a separate activation copy approved content into
revisioned coherent units with exact OCR spans. See
[docs/segmentation-operations.md](docs/segmentation-operations.md) before using
the export/import/review/activate workflow.

Run semantics only on an active coherent-unit revision whose regions have
historian-selected reviewed text versions:

```bash
uv run wic-e2e \
  --database-url "$DATABASE_URL" \
  --coherent-unit-revision-id REVISION_UUID \
  --output-dir artifacts/e2e/REVISION_UUID
```

For a unit with extracted mentions this makes exactly two Qwen3.5-4B calls:
combined mention/event extraction, then local ID-bounded resolution. Both calls
receive hash-verified page images. Local clusters remain candidate evidence;
there is no canonical entity merge.

## Local evidence and retrieval stack

Copy `.env.example` to an untracked `.env` and replace its development passwords, then start the selected databases:

```bash
docker compose up -d
uv run wic-migrate --database-url "$DATABASE_URL"
```

Load the audited archive catalog and paired Hunyuan artifacts:

```bash
uv run wic-ingest --database-url "$DATABASE_URL" manifest artifacts/corpus-audit/manifest.jsonl
uv run wic-ingest --database-url "$DATABASE_URL" layout artifacts/layout/v219-p0308.hunyuan.json
uv run wic-ingest --database-url "$DATABASE_URL" ocr artifacts/ocr/v219-p0308.hunyuan.json
```

Create an idempotent, dependency-gated ingestion plan before processing pages:

```bash
uv run wic-batch --database-url "$DATABASE_URL" plan \
  --name 'volume 219 page 308 evidence ingestion' --created-by researcher \
  --volume 219 --page 308 \
  --aggregate-stages search_projection,rag_export,graph_projection
uv run wic-batch --database-url "$DATABASE_URL" status --batch-id BATCH_UUID
```

The page DAG is `render_lossless -> layout -> OCR -> embedding`. Optional batch
fan-in jobs add `embedding -> search_projection`, `OCR -> rag_export`, and
reviewed PostgreSQL evidence to `graph_projection`. Article semantics is not a
page job: after reviewed text and coherent-unit activation, `wic-e2e` performs
the two Qwen calls. PostgreSQL records
immutable plan/input fingerprints, dependencies, bounded stage configuration,
leases, retries, artifact checksums, typed completion metadata, and an
append-only event history. Planning is guarded at 1,000 pages by default; the
current manifest has 340,511 known pages, so `--allow-large-plan` must follow an
explicit cost and capacity review. See
[docs/ingestion-operations.md](docs/ingestion-operations.md) for the worker
contract and current limitations.

Terminal failures cancel only dependency-blocked descendants; independent
branches may finish before the batch becomes `failed`. Inspect, explicitly
replay a dead-letter root, or cancel a batch with:

```bash
uv run wic-batch --database-url "$DATABASE_URL" failures --batch-id BATCH_UUID
uv run wic-batch --database-url "$DATABASE_URL" replay --job-id FAILED_JOB_UUID \
  --requested-by operator-name --reason 'documented recovery reason'
uv run wic-batch --database-url "$DATABASE_URL" cancel --batch-id BATCH_UUID \
  --cancelled-by operator-name --reason 'documented operational reason'
```

Run one ready job, with automatic lease heartbeats and safe retry recording:

```bash
uv run wic-worker --database-url "$DATABASE_URL" \
  --worker "$(hostname)-worker-1" --batch-id BATCH_UUID
```

For a bounded polling process, opt into loop mode with both work and idle stop
limits:

```bash
uv run wic-worker --database-url "$DATABASE_URL" \
  --worker "$(hostname)-ocr-1" --batch-id BATCH_UUID --stage ocr \
  --loop --max-jobs 100 --idle-polls 3 --poll-seconds 5
```

The loop summary reports attempts by status plus adopted/fresh artifact counts
and whether it stopped on the job or idle bound. Loop mode never defaults to an
unbounded daemon.

Use `--offline` only when the size-verified source object is already in the
local source cache. A worker first validates and adopts an exact existing
artifact when possible; otherwise it invokes the pinned renderer, Hunyuan
layout/OCR, or embedding stage. Per-job outputs under `artifacts/ingestion-*` are
generated data and excluded from Git.

Aggregate workers build a batch-specific OpenSearch index before atomically
moving `wic-regions-current`, export the batch OCR scope with exact citation
sidecars, and rebuild Neo4j from reviewed claims only. OpenSearch and Neo4j are
global rebuildable views of the current PostgreSQL state; the RAG export is
limited to the plan's volume/page scope.

OCR ingestion retains every byte-distinct page image in
`archive.page_derivative` and chooses the preferred derivative monotonically by
reviewed evidence tier, then resolution. Screening images are never overwritten
when a lossless pilot or gold render arrives.

Each OCR run is also bound to its exact derivative. Retrieval projects only the
one active page/run selection; changing models or choosing a benchmark winner is
an explicit, audited operation:

```bash
uv run wic-ingest --database-url "$DATABASE_URL" ocr-select \
  --volume 219 --page 308 \
  --run-id cc2310a1-c174-4598-8360-1742da5d0262 \
  --basis technical_default --selected-by 'researcher-name' \
  --note 'Source-resolution non-gold pipeline selection'
```

Generate BGE-M3 embeddings, rebuild the OpenSearch projection, and issue an evidence-citing hybrid query:

```bash
uv run wic-embed --database-url "$DATABASE_URL" --source-ocr-run-id cc2310a1-c174-4598-8360-1742da5d0262
uv run wic-search --opensearch-url "$OPENSEARCH_URL" project --database-url "$DATABASE_URL" --recreate
uv run wic-search --opensearch-url "$OPENSEARCH_URL" query '士女' --mode hybrid --limit 5
```

OpenSearch v2 indexes only active OCR selections. Every hit carries the source
object hash, derivative UUID/hash/tier, OCR run, exact region polygon, and the
selection basis used to admit that run.

Project reviewed claims/entities to Neo4j, export an identical citation-mapped
corpus for the isolated RAG comparisons, and start the local researcher API:

```bash
uv run wic-graph --database-url "$DATABASE_URL" --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" --neo4j-password "$NEO4J_PASSWORD"
uv run wic-rag-export --database-url "$DATABASE_URL" \
  --output artifacts/rag-three-year
uv run wic-api --host 127.0.0.1 --port 8766 \
  --database-url "$DATABASE_URL" --opensearch-url "$OPENSEARCH_URL" \
  --neo4j-uri "$NEO4J_URI" --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD"
```

Open `http://127.0.0.1:8766` for lexical, dense, or hybrid search. Scenario
context returned by the API contains only reviewed claims; with the current
smoke data it abstains explicitly. `LLM_BASE_URL`, `LLM_MODEL`, and the
immutable `LLM_MODEL_REVISION` optionally enable a local OpenAI-compatible chat
endpoint. Remote endpoints additionally require HTTPS and explicit
`LLM_ALLOW_REMOTE=true` data-egress consent. Research briefs must label OCR as
unreviewed leads; reconstructed scenes hard-abstain until reviewed claims
exist. See [`docs/generation-operations.md`](docs/generation-operations.md) for
the provider, privacy, provenance and output-validation contract.
The frozen generation-quality protocol, executable runner, objective scorer,
model-blind two-review/adjudication workflow, and paired-bootstrap comparator
are under [`experiments/generation/`](experiments/generation/). It contains no
scores or winner because no approved model or historian-authored generation
set exists yet.

After a search, `Discuss evidence` opens a browser-held multi-turn research
conversation. Every follow-up performs fresh retrieval under the selected mode
and year filters. At most 12 prior user/assistant turns are passed inside an
untrusted context envelope—not as system instructions and never as evidence.
Archive citations are accepted only when their region UUID occurs in the
current retrieval or a reviewed claim. Conversations are not persisted by the
server. Without an LLM configuration the endpoint returns an explicit
`unavailable` response while preserving the retrieved evidence bundle.
Outputs with missing, malformed or foreign citations are returned as
`rejected`, with the unsafe model text withheld and hashed. Scene outputs must
also contain the three epistemic sections in order and cite a reviewed claim
inside `Direct evidence`. The UI displays model, prompt/context/output hashes,
resolved scan links, validation errors and warnings.

The same interface exposes a historian review queue and reviewed-only insight
signals. A reviewer first accepts or rejects the exact NER span, then makes a
separate entity-resolution decision: link to a reviewed candidate, create a new
reviewed entity from the explicit NIL option, or keep it unresolved. Each action
is transactionally audited and idempotent by review UUID. The current 492
machine candidates are unreviewed screening/lossless pilot/three-year outputs; opening the
queue does not promote them. Dataset and run filters isolate exact experiment
cohorts. Re-run `wic-graph` after genuine reviews before using the graph
insight view. Insight cards are analytical leads and never become historical
claims automatically.

`Explore machine leads` is a deliberately separate pre-review workspace. It
summarizes the active OCR scope, applies a small versioned set of
women-centered theme patterns, links every example to its registered scan
derivative and region, and exposes NER candidate counts and pairwise exact
agreement. Its labels and warnings make clear that these are triage signals,
not frequency evidence or historical findings. The current three-page scope
(1924–1926) is useful for prioritizing review but remains far too small for
frequency or corpus-level interpretation.

The bounded three-year expansion contains active source-resolution pages 202/338
(1924), 219/308 (1925), and 230/367 (1926): 2,498 OCR regions and matching
BGE-M3 embeddings. The global RAG export contains three documents, 2,471 exact
region citations, 27 accounted empty regions, and 16,834 characters. Exact
`否認廣州今戒嚴` (1924), hybrid `士女` (1925), and hybrid `西裝` (1926) live
queries all return the expected page with derivative/image hashes and polygons.
These are retrieval plumbing checks over unreviewed OCR, not historical findings.

Candidate claims have a second queue showing their subject, predicate, object,
model revision, and every cited scan passage. Acceptance is rejected unless at
least one evidence passage is attached and all referenced entities are already
reviewed. The insights view reports whether review-authoritative PostgreSQL is
newer than the derived Neo4j projection; when it says `STALE`, run `wic-graph`
before interpreting graph patterns.

`wic-relations` is deliberately reviewed-input-only. Its v2 rules require exact
linked mention spans, an ontology-compatible argument pair, an intervening cue
without clause crossing, and complete scan provenance; otherwise they abstain.
The independent `wic-relation-benchmark` runner verifies the byte-exact source
NER gold set, freezes every prediction (including negatives), and scores exact
relations/evidence without inserting claims. The live archive currently has no
reviewed linked mentions and therefore correctly produces no relation claims.

The researcher API binds to localhost by default and currently has no
authentication or authorization layer. Do not expose it outside a trusted local
development environment.

Run the scored citation-retrieval smoke comparison and validate the common RAG
input with:

```bash
for mode in lexical dense hybrid; do
  uv run wic-eval --questions experiments/retrieval/lossless-pilot-questions.jsonl \
    --output "artifacts/eval-pilot/${mode}.json" --mode "$mode" --limit 5
done
uv run wic-rag-adapter validate --export artifacts/rag-pilot
```

The single smoke question is not a quality claim; the real gate requires
historian-authored/adjudicated questions. Pinned NER candidates and the paired
corrected-text/raw-OCR protocol are under `experiments/ner/`; the relation/event
shortlist, gold contract, rule adapter and scorer are under `experiments/relation/`; isolated
GraphRAG/LightRAG requirements and the fair-comparison protocol are under
`experiments/rag/`; retrieval judgments and metrics are under
`experiments/retrieval/`; grounded assistant/scene evaluation is under
`experiments/generation/`.

Gold transcription/NER policy is in
[`docs/annotation-guidelines.md`](docs/annotation-guidelines.md). Once two
independent annotations have been adjudicated, `wic-ner-score` validates their
offsets and produces exact/relaxed, evidence-validity, OCR-loss, per-type and
decade/genre/layout/quality model reports. No current smoke artifact is gold data.

The same policy defines model-independent OCR/layout polygons. `wic-ocr-score`
compares byte-identical page artifacts using detection F1/IoU, CER,
reading-order, region-kind/direction, geometry, throughput and stratified
metrics. Benchmark commands and refusal conditions are under `experiments/ocr/`.

The committed OCR/NER files include technical smoke artifacts from a lossy
screening derivative and source-resolution, explicitly non-gold lossless
OCR/NER pilots. They demonstrate provenance, coordinates, persistence,
retrieval and benchmark isolation; they are not gold transcriptions, accuracy
results, or reviewed historical assertions.
