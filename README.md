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

The committed OCR/NER files are technical smoke artifacts from a lossy screening derivative. They demonstrate provenance, coordinates, persistence, and retrieval; they are not gold transcriptions or reviewed historical assertions.
