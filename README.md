# Women in China 100 Years Ago

Evidence-first tools and research notes for reconstructing women's history from the digitized *Shen Bao* archive.

The current technical design is in [docs/technical-design.md](docs/technical-design.md).

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

For the fully locked development environment, use `uv sync --all-extras` and `uv run` instead of setting `PYTHONPATH`.

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

Start the local visual review UI:

```bash
PYTHONPATH=src python -m wic_history.review_server
```

Open `http://127.0.0.1:8765`. Reviews are stored atomically in `artifacts/benchmark-review/annotations.json`. The server binds to localhost by default and has no authentication; do not expose it on a public interface.

## Local evidence and retrieval stack

Copy `.env.example` to an untracked `.env` and replace its development passwords, then start the selected databases:

```bash
docker compose up -d
uv run wic-migrate --database-url "$DATABASE_URL"
```

Load the audited archive catalog and versioned OCR/NER artifacts:

```bash
uv run wic-ingest --database-url "$DATABASE_URL" manifest artifacts/corpus-audit/manifest.jsonl
uv run wic-ingest --database-url "$DATABASE_URL" ocr artifacts/ocr-smoke/v219-p0308.ppocrv6.json
uv run wic-ingest --database-url "$DATABASE_URL" ner artifacts/ner-smoke/v219-p0308.gliner-multi-v2.1.json
```

Generate BGE-M3 embeddings, rebuild the OpenSearch projection, and issue an evidence-citing hybrid query:

```bash
uv run wic-embed --database-url "$DATABASE_URL" --source-ocr-run-id 213e0078-59d5-4a56-8811-a59e40ed0800
uv run wic-search --opensearch-url "$OPENSEARCH_URL" project --database-url "$DATABASE_URL" --recreate
uv run wic-search --opensearch-url "$OPENSEARCH_URL" query '富紳淑女' --mode hybrid --limit 5
```

Project reviewed claims/entities to Neo4j, export an identical citation-mapped
corpus for the isolated RAG comparisons, and start the local researcher API:

```bash
uv run wic-graph --database-url "$DATABASE_URL" --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" --neo4j-password "$NEO4J_PASSWORD"
uv run wic-rag-export --database-url "$DATABASE_URL" \
  --output artifacts/rag-smoke --volume 219 --page 308
uv run wic-api --host 127.0.0.1 --port 8766 \
  --database-url "$DATABASE_URL" --opensearch-url "$OPENSEARCH_URL" \
  --neo4j-uri "$NEO4J_URI" --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD"
```

Open `http://127.0.0.1:8766` for lexical, dense, or hybrid search. Scenario
context returned by the API contains only reviewed claims; with the current
smoke data it abstains explicitly. `LLM_BASE_URL` and `LLM_MODEL` optionally
enable a local or hosted OpenAI-compatible chat endpoint. Research briefs must
label OCR as unreviewed leads; reconstructed scenes hard-abstain until reviewed
claims exist.

The same interface exposes a historian review queue and reviewed-only insight
signals. A reviewer first accepts or rejects the exact NER span, then makes a
separate entity-resolution decision: link to a reviewed candidate, create a new
reviewed entity from the explicit NIL option, or keep it unresolved. Each action
is transactionally audited and idempotent by review UUID. The current 187
machine candidates are unreviewed smoke outputs; opening the queue does not
promote them. Re-run `wic-graph` after genuine reviews before using the graph
insight view. Insight cards are analytical leads and never become historical
claims automatically.

Candidate claims have a second queue showing their subject, predicate, object,
model revision, and every cited scan passage. Acceptance is rejected unless at
least one evidence passage is attached and all referenced entities are already
reviewed. The insights view reports whether review-authoritative PostgreSQL is
newer than the derived Neo4j projection; when it says `STALE`, run `wic-graph`
before interpreting graph patterns.

The researcher API binds to localhost by default and currently has no
authentication or authorization layer. Do not expose it outside a trusted local
development environment.

Run the scored citation-retrieval smoke comparison and validate the common RAG
input with:

```bash
for mode in lexical dense hybrid; do
  uv run wic-eval --questions experiments/retrieval/smoke-questions.jsonl \
    --output "artifacts/eval-smoke/${mode}.json" --mode "$mode" --limit 5
done
uv run wic-rag-adapter validate --export artifacts/rag-smoke
```

The single smoke question is not a quality claim; the real gate requires
historian-authored/adjudicated questions. Pinned NER candidates and the paired
corrected-text/raw-OCR protocol are under `experiments/ner/`; isolated
GraphRAG/LightRAG requirements and the fair-comparison protocol are under
`experiments/rag/`; retrieval judgments and metrics are under
`experiments/retrieval/`.

The committed OCR/NER files are technical smoke artifacts from a lossy screening derivative. They demonstrate provenance, coordinates, persistence, and retrieval; they are not gold transcriptions or reviewed historical assertions.
