# Historical-Chinese OCR and layout benchmark

The selected learned OCR model is HunyuanOCR 1.5. BF16 llama.cpp/Metal is the
locally qualified candidate for source-derived crop `spotting_json` requests;
four independent slots increase batch throughput without sharing request
history. Whole-page `layout_parse` failed at source and proxy resolution, so it
is not an ingestion authority. Earlier Paddle/PP-OCR artifacts remain
comparison evidence, not an inference fallback.

Launch a crop server with four split-KV slots. `--ctx-size` is the total across
slots, so 40,960 provides 10,240 tokens per request:

```bash
.cache/hunyuan-llamacpp/llama.cpp/build/bin/llama-server \
  --model .cache/hunyuan-llamacpp/gguf/hyocr-bf16.gguf \
  --mmproj .cache/hunyuan-llamacpp/gguf/mmproj-hyocr-f16.gguf \
  --alias HYVL-BF16 --host 127.0.0.1 --port 18080 \
  --ctx-size 40960 --parallel 4 --no-kv-unified \
  --cache-ram 0 --n-gpu-layers 99 --no-webui
```

Run the provenance-capturing client with concurrent, cache-free requests:

```bash
.venv/bin/python experiments/ocr/llamacpp_hunyuan_smoke.py \
  --task spotting_json \
  --model HYVL-BF16 \
  --concurrency 4 \
  --no-cache-prompt \
  --acceleration metal \
  --model-gguf .cache/hunyuan-llamacpp/gguf/hyocr-bf16.gguf \
  --mmproj-gguf .cache/hunyuan-llamacpp/gguf/mmproj-hyocr-f16.gguf \
  --llama-server .cache/hunyuan-llamacpp/llama.cpp/build/bin/llama-server \
  --output artifacts/ocr-challenger/llamacpp-metal/smoke.json \
  artifacts/ocr-challenger/suite-v219-p0308/C09_vertical_clean.png
```

The client records image/model hashes, exact project prompt, decoding
parameters, concurrency, prompt-cache policy, finish reason, usage, timings,
runtime build, and raw output. See
[`hunyuanocr-1.5-hardware.md`](hunyuanocr-1.5-hardware.md) for the verified
Metal placement, CPU control, conversion blocker, and deployment decision.

The gold policy is
[`docs/annotation-guidelines.md`](../../docs/annotation-guidelines.md). The
`wic_history.ocr_gold.OCRGoldSet` contract requires two distinct independent
page annotations plus adjudication, source-image SHA-256, page dimensions,
positive convex in-bounds polygons, unique region IDs and reading orders, and
NFC evidence transcriptions.

Run every candidate on byte-identical lossless pages and score one immutable
model revision per report:

```bash
uv run wic-ocr-score --gold artifacts/gold/ocr-layout-v1.json \
  --predictions artifacts/ocr-benchmark/ppocrv6/*.json \
  --iou-threshold 0.5 \
  --output artifacts/ocr-benchmark/ppocrv6.score.json
```

The scorer refuses image-hash/dimension mismatches, unknown pages, duplicate
page artifacts, non-OCR runs, and mixed model revisions. Missing prediction
pages remain failures rather than disappearing from the denominator. It reports
region detection F1, invalid geometry, matched IoU/area coverage,
matched-region CER, reading-order CER and pair accuracy, kind/direction
accuracy, throughput, recorded peak memory, per-page diagnostics, and
genre/layout/quality/decade strata.

Do not use the committed lossy screening JPEG or its one-page smoke OCR as gold.
The first real comparison starts only after lossless render hashes and
adjudicated page regions are frozen.

The selected-page renderer is ready:

```bash
uv run wic-gold-render --offline
```

It reads `artifacts/benchmark-review/annotations.json`, requires complete named
screening decisions, and emits `artifacts/gold-pages/lossless_manifest.jsonl`.
For the non-gold page-308 pipeline pilot, it directly decoded the embedded
6176×8960 JBIG2 raster, performed no geometric transform, and verified identical
decoded-pixel hashes across PNG writing. The committed pilot manifest is
provenance evidence only, not a benchmark judgment.

The same page has also completed a source-resolution PP-OCRv6 plumbing run:

```bash
uv run wic-ocr \
  --image artifacts/lossless-pilot/images/v219/p0308.png \
  --render-manifest artifacts/lossless-pilot/lossless_manifest.jsonl \
  --source-uri 's3://ccaa-us-east-1-504133794192/sb_raw/申报影印本219.pdf' \
  --volume 219 --page 308 --year 1925 --language ch \
  --tile-size 1200 --overlap 120 --worker-batch-size 5 \
  --output artifacts/ocr-pilot/v219-p0308.lossless.ppocrv6.json
```

It produced 1,099 coordinate-preserving regions over 54 tiles. Its artifact and
database derivative retain the render manifest and full source-object hash, but
its `non_gold_lossless_pilot` tier excludes it from OCR quality claims.
