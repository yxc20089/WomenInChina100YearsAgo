# RAG comparison protocol

All candidate systems must index the same `wic-rag-export` output. The JSONL
file is convenient for programmatic loaders; `documents/` contains the same
plain text for GraphRAG's input directory. `citations.jsonl` maps character
spans to evidence-region UUIDs and scan polygons.

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
