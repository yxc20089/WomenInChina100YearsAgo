# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Evidence-first tools for reconstructing women's history from the digitized *Shen Bao* (申報) newspaper archive (1920s Traditional Chinese, vertical right-to-left layout). The normative implementation contract is `docs/e2e-pipeline-contract.md`; `docs/objective-status.md` tracks what is actually implemented and what remains unproven. `docs/technical-design.md` is research history, not the current design.

## Commands

```bash
# Environment (Python 3.12, uv-managed)
uv sync --extra api --extra data --extra test

# All tests (no AWS/database/docker needed)
PYTHONPATH=src python -m unittest discover -s tests -v

# Single test module / case (pytest also works)
PYTHONPATH=src python -m unittest tests.test_evidence -v
uv run pytest tests/test_evidence.py -k test_name

# Lint (ruff, default settings)
ruff check src tests

# Local evidence/retrieval stack (PostgreSQL+pgvector, OpenSearch, Neo4j)
cp .env.example .env   # replace dev passwords first
docker compose up -d
uv run wic-migrate --database-url "$DATABASE_URL"
```

Every CLI command is a `wic-*` console script defined in `pyproject.toml` `[project.scripts]`, one module per command in `src/wic_history/`. The README documents the full operational sequence (corpus audit → render → layout/OCR → ingest → batch plan → worker → embed → search/graph/API).

### Environment isolation constraints

- The `ner` (GLiNER) and `ocr` (HunyuanOCR) extras are declared conflicting in `[tool.uv]` and must never be installed together (incompatible Transformers ranges).
- HunyuanOCR's native runtime lives in a separate CUDA environment (`environments/hunyuan-ocr/`), not in the repo venv.
- Qwen3.5-4B runs via a pinned local Ollama; BGE-M3 via sentence-transformers.

## Architecture

### Fixed single-model stack (first build)

`config/pipeline-models.toml` is the sole source of model identities: names, immutable revisions, official prompts, decoding parameters, and output-schema hashes. Do not hardcode model names/revisions elsewhere or "upgrade" a model without changing this file.

- **Layout + OCR**: pinned HunyuanOCR 1.5, two official tasks (`spotting_json` for line text/boxes, `layout_parse` for structure/reading order) on the same immutable page bytes.
- **Semantics**: Qwen3.5-4B makes exactly two calls per article via `wic-e2e` — one multimodal mention/event extraction call, then one ID-bounded local-resolution call. Resolution may only cluster supplied mention IDs; it can never create mentions, aliases, or global merges.
- **Authority**: PostgreSQL 17 + pgvector is the only source of truth. OpenSearch (CJK + BGE-M3 + RRF hybrid) and Neo4j (reviewed-claims-only) are rebuildable projections, never authorities.

There is **no model fallback and no global entity merge**. A failed, malformed, truncated, or ambiguous model output abstains and creates a review item; it never silently retries with another model. Preserve this pattern in any new stage.

### Page DAG and semantics boundary

Ingestion (`wic-batch` plan / `wic-worker` execute) runs the page DAG `render_lossless → layout → ocr → embedding`, with optional batch fan-in stages `search_projection`, `rag_export`, `graph_projection`. Jobs are lease-based, idempotent, fingerprinted, and append-only-evented in PostgreSQL. Planning is guarded at 1,000 pages (the corpus has 340,511); `--allow-large-plan` requires an explicit cost review.

Article semantics is deliberately **not** a page job: `wic-e2e` runs only on an active coherent-unit revision whose regions have historian-selected reviewed text versions. Segmentation (`wic-segment`), review, and activation are separate audited steps.

### Provenance and review invariants

These invariants shape most of the code; violating them is a design regression, not a style issue:

- Every assertion traces back through evidence span → text version → region polygon → page-image hash → S3 source object. Raw model outputs are retained immutably with model/config identity.
- Machine outputs (OCR, NER candidates, segmentation windows, insight signals) are never automatically promoted to reviewed/gold/historical status. Gold status comes only from historian selection plus two independent reviews and adjudication; commands hard-refuse to relabel pilots as gold.
- OCR run selection, segmentation activation, and entity/claim review are explicit, audited operations keyed by UUID; changing an active selection is never implicit.
- No fabricated confidence: Hunyuan emits none, so evidence rows use `not_reported`/`uncalibrated`/`calibrated` — never invent scores.
- Traditional characters are never silently simplified or normalized in place.

### Layout of supporting directories

- `db/migrations/` — numbered SQL migrations, applied by `wic-migrate` and auto-applied on fresh docker Postgres init.
- `tests/` — mirrors `src/wic_history/` one-to-one (`test_<module>.py`), `unittest.TestCase` style, designed to run without external services.
- `artifacts/` — small committed provenance/smoke artifacts; large or derived outputs (`ingestion-*`, rendered images, gold packets) are gitignored. Committed OCR/NER artifacts are explicitly non-gold technical demonstrations.
- `experiments/` — frozen benchmark protocols per research axis (ocr, ner, relation, retrieval, rag, generation), each with its own README. They define gates and refusal conditions; none currently contain scores or winners.

### Security posture

The review server (`wic-review`, port 8765) and researcher API (`wic-api`, port 8766) bind to localhost with no authentication — never expose them. The corpus audit is read-only against S3 (list + bounded range reads only). Remote LLM endpoints require HTTPS plus explicit `LLM_ALLOW_REMOTE=true` data-egress consent.

## Writing style in docs and commits

Commit subjects are short imperative phrases ("Qualify Hunyuan-driven region discovery"). Documentation is deliberately precise about epistemic status — keep the distinction between smoke/pilot/plumbing checks and gold/reviewed/historical results in anything you write here.
