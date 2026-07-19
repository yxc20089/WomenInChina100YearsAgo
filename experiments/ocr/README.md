# Historical-Chinese OCR and layout benchmark

The selected learned OCR model is HunyuanOCR 1.5. F16 llama.cpp/Metal — an
officially documented deployment path with a byte-reproducible official-recipe
conversion — is the locally qualified candidate for source-derived crop
`spotting_json` requests; four independent slots increase batch throughput
without sharing request history. Whole-page `layout_parse` failed at source,
in-spec 4K proxy, and smaller proxy resolution even with a 16,384-token
budget, so it is not an ingestion authority. Earlier Paddle/PP-OCR artifacts remain
comparison evidence, not an inference fallback.

Launch a crop server with four split-KV slots. `--ctx-size` is the total across
slots, so 40,960 provides 10,240 tokens per request:

```bash
.cache/hunyuan-llamacpp/llama.cpp/build/bin/llama-server \
  --model .cache/hunyuan-llamacpp/gguf/hyocr-f16.gguf \
  --mmproj .cache/hunyuan-llamacpp/gguf/mmproj-hyocr-f16.gguf \
  --alias HYVL-F16 --host 127.0.0.1 --port 18080 \
  --ctx-size 40960 --parallel 4 --no-kv-unified \
  --cache-ram 0 --n-gpu-layers 99 --no-webui
```

Run the provenance-capturing client with concurrent, cache-free requests:

```bash
.venv/bin/python experiments/ocr/llamacpp_hunyuan_smoke.py \
  --task spotting_json \
  --model HYVL-F16 \
  --concurrency 4 \
  --no-cache-prompt \
  --acceleration metal \
  --model-gguf .cache/hunyuan-llamacpp/gguf/hyocr-f16.gguf \
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

## Region proposals and crop validation

Region discovery is decomposed instead of asked of the OCR model whole-page
(which fails; see the hardware note). Two diagnostic tools produce reviewable
proposals on one page:

```bash
.venv/bin/python experiments/ocr/region_crop_validation.py \
  --artifact artifacts/ocr-pilot/v219-p0308.lossless.ppocrv6.json \
  --image artifacts/lossless-pilot/images/v219/p0308.png \
  --model-gguf .cache/hunyuan-llamacpp/gguf/hyocr-f16.gguf \
  --mmproj-gguf .cache/hunyuan-llamacpp/gguf/mmproj-hyocr-f16.gguf \
  --output artifacts/ocr-challenger/region-proposal/p0308-ppocrv6-crop-validation.json
.venv/bin/python experiments/ocr/region_cell_geometry.py \
  --artifact artifacts/ocr-pilot/v219-p0308.lossless.ppocrv6.json \
  --image artifacts/lossless-pilot/images/v219/p0308.png \
  --proxy artifacts/ocr-challenger/layout-proxy/v219-p0308.long-side-2240.normal-polarity.png \
  --overlay-out artifacts/ocr-challenger/region-proposal/p0308-cell-overlay.png \
  --output artifacts/ocr-challenger/region-proposal/p0308-cell-geometry.json
```

`region_crop_validation.py` sends every text-detection box (padded, polarity
inverted) through the qualified crop-level `spotting_json` path and records
validity and fuzzy agreement per crop. `region_cell_geometry.py` extracts
ruling-line separators from text-free ink, decomposes cells, clusters
detector boxes into within-cell column proposals, and emits
`display_or_figure` proposals from large missed-ink components — the display
glyphs and illustrations that line detectors under-segment.
`display_crop_validation.py` runs those display/column proposals through the
same crop path. Its p0308 result fixes two boundaries: transcription must stay
at line/display-crop scale (oversized column crops reproduce the whole-page
repetition failure), and near-blank sliver proposals can elicit fluent
off-domain hallucination that passes JSON/termination gates (one top-edge
sliver returned a mathematics exercise), so proposal shape/ink filters and a
CJK-content plausibility gate are required before any crop output is trusted.
All outputs are `diagnostic_not_qualified` proposals for human review and
never assert semantic article boundaries. Line-detector boxes come from the committed
PP-OCRv6 pilot; a learned proposal model would require the same explicit
contract amendment, pinned revision, and gold detection benchmark as any
other component.

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
