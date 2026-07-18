# RAG comparison protocol

All candidate systems must index the same `wic-rag-export` output. The JSONL
file is convenient for programmatic loaders; `documents/` contains the same
plain text for GraphRAG's input directory. `citations.jsonl` maps character
spans to evidence-region UUIDs and scan polygons.

Validate the shared input and create GraphRAG's isolated input workspace:

```bash
uv run wic-rag-adapter validate --export artifacts/rag-smoke
uv run wic-rag-adapter prepare-graphrag --export artifacts/rag-smoke \
  --workspace /tmp/wic-graphrag-smoke
```

The second command prints the pinned `init`, Traditional-Chinese `prompt-tune`,
`index`, Global-query, and DRIFT-query commands. Supply the same recorded LLM
and embedding models for every compared run; generated `settings.yaml` and
prompts must be archived with the score report.

Pinned isolated environments:

```bash
python -m venv .venv-graphrag
.venv-graphrag/bin/pip install -r experiments/rag/requirements-graphrag.txt

python -m venv .venv-lightrag
.venv-lightrag/bin/pip install -r experiments/rag/requirements-lightrag.txt
```

Do not install either framework into the ingestion environment. Their generated
graphs and summaries are disposable experimental indexes. Run indexing only
after supplying a separately managed LLM endpoint and recording model, prompt,
token, latency, and monetary-cost metadata. No LLM credential is committed here.

The initial page export is a pipeline smoke test. A fair historian-facing
comparison starts only after reviewed article segmentation and a 30–50-question
evaluation set exist.

For LightRAG, start the pinned server on localhost with authentication and a
separately recorded LLM/embedding configuration. Then load and query the same
documents through its official REST routes:

```bash
uv run wic-rag-adapter load-lightrag --export artifacts/rag-smoke \
  --base-url http://127.0.0.1:9621 --api-key "$LIGHTRAG_API_KEY"
uv run wic-rag-adapter query-lightrag '女學生的教育網絡如何變化？' \
  --mode mix --base-url http://127.0.0.1:9621 --api-key "$LIGHTRAG_API_KEY"
```

The adapter assigns `wic/<page-uuid>.txt` as LightRAG's `file_source`, which can
be resolved through the export sidecar to exact OCR regions. Do not expose the
LightRAG server without `--key`/`LIGHTRAG_API_KEY`; its default bind address is
not localhost.
