# HunyuanOCR 1.5 hardware paths

Date: 2026-07-19

This note distinguishes locally verified execution from documented or
prospective deployment paths. Version 1.0 conversions must not be reported as
HunyuanOCR 1.5 results.

## Apple Silicon result

Native Transformers inference is the current working local path.

- Host: Apple M4 Pro, 48 GB unified memory, macOS 15.6.1 arm64
- Runtime: Torch 2.13.0, Transformers 5.13.0, Accelerate 1.14.0
- Device check: Torch built with MPS; `torch.backends.mps.is_available()` true
- Model: `tencent/HunyuanOCR` revision
  `de8f10ad2f00a0cefd790b526de8a65dcfdb3205`
- Result: the model loaded and generated successfully on MPS; tight-crop
  inference was approximately 0.5–2.4 seconds in the diagnostic runs
- Qualification: Tencent documents NVIDIA environments, not MPS. This is a
  locally demonstrated engineering path, not an official parity claim.

This path is sufficient for crop-level development on the current Mac. It
should remain in an isolated environment because the model's required
Transformers version differs from the project lock.

## MLX status

There is no Tencent or Apple HunyuanOCR 1.5 MLX release. Community
`mlx-vlm` supports the HunyuanOCR/HunyuanVL architecture, but the published
HunyuanOCR MLX weights found in this review are version 1.0. Static inspection
also found a likely strict-loading issue around 1.5's redundant tied
`lm_head.weight`. Treat native MLX as experimental until a pinned 1.5 conversion
can reproduce the fixed crop suite.

Useful upstream evidence:

- [`mlx-vlm` Hunyuan support commit](https://github.com/Blaizzy/mlx-vlm/commit/9c8e9373a83b5b0ed090ad4ae457842343f4eab8)
- [community 1.0 4-bit conversion](https://huggingface.co/hadeseus/HunyuanOCR-mlx-4bit)

## llama.cpp / Metal status

Update 2026-07-19: the `tencent/HunyuanOCR` model card documents llama.cpp as
an official deployment path for CPU/consumer-GPU/laptop environments, with a
DFlash-adapted fork for PC-side speculative decoding. Tencent ships no
pre-converted GGUF files, so self-conversion is the official recipe. The
conversion has been re-derived and is byte-reproducible: with the local
checkpoint verified against the pinned revision (`model.safetensors` SHA-256
`632a1e082c4dd5a3284cf1ffcdba2fdaa06f435762c58c2f34aff0f3bd6c0249` matches the
Hugging Face LFS record) and llama.cpp master `571d0d54` under its own pinned
converter requirements (`transformers==4.57.6`), `convert_hf_to_gguf.py
--outtype f16` reproduced `hyocr-f16.gguf`
(`f3b4e8e2c7db5c7346b9f28059f7813a7c9b1ff3273434cf79be468a82c551ed`) and
`mmproj-hyocr-f16.gguf`
(`4cfd75fc001e7d03e36fdd26a0b36b2d5a2af6922d7800f99ca1673a104559a2`)
byte-identically to the pair behind every recorded run. The F16 pair is now
the canonical local serving artifact; the BF16 language GGUF was removed after
the recorded byte-identical crop results. A spot-check on the reconverted pair
reproduced C09 and the blank C11 behavior byte-for-byte
([`f16v2-regression-c09-c11.json`](../../artifacts/ocr-challenger/llamacpp-metal/f16v2-regression-c09-c11.json)).

BF16 language-model execution on Metal is locally demonstrated. Runtime and
hardware do not decide OCR accuracy; qualification is based on the fixed
historical crops, negative controls, and page outputs below.

The run used:

- host: Apple M4 Pro, 48 GB unified memory, arm64;
- source checkpoint: `tencent/HunyuanOCR` revision
  `de8f10ad2f00a0cefd790b526de8a65dcfdb3205`;
- upstream llama.cpp commit
  `571d0d540df04f25298d0e159e520d9fc62ed121`, build 10068;
- build: Metal and embedded Metal library enabled, Accelerate enabled, and
  OpenMP disabled to avoid the local Anaconda CMake/AppleClang OpenMP linker
  mismatch;
- BF16 language GGUF SHA-256
  `1728022dbbfefc9e0e1017a0246ebceacdb97351fb191c95e20176ff25708b42`;
- F16 vision-projector GGUF SHA-256
  `4cfd75fc001e7d03e36fdd26a0b36b2d5a2af6922d7800f99ca1673a104559a2`.

The projector remains F16 because the current Hunyuan llama.cpp path requires
it for Metal's unsupported BF16 `IM2COL` operation. This is BF16 language
inference with an F16 vision projector, not an all-BF16 graph.

Verbose server logs proved hardware placement: the server selected the Apple
M4 Pro Metal device, offloaded all 25/25 language layers, allocated the KV
cache on `MTL0`, and reported `CLIP using MTL0 backend` for the vision encoder.
This is real GPU acceleration rather than CPU execution in a Metal-capable
binary.

An uncached control on C09 produced byte-identical text on CPU and Metal. CPU
took 2.712 seconds and Metal 0.848 seconds, a 3.2x end-to-end speedup. Prompt
processing increased from 174 to 1,081 tokens/second (6.2x); generation
increased from 147 to 170 tokens/second (1.16x). The raw artifacts are
[`cpu-c09-structured-parse.json`](../../artifacts/ocr-challenger/llamacpp-metal/cpu-c09-structured-parse.json)
and
[`metal-c09-structured-parse.json`](../../artifacts/ocr-challenger/llamacpp-metal/metal-c09-structured-parse.json).

### Accuracy and serving boundary

The crop-level path is promising:

- C01 through C09 were byte-identical between the F16 and BF16 language GGUF
  runs, so changing the language tensor dtype did not repair or cause the
  consequential crop errors;
- all ten text-bearing fixed crops stopped normally;
- C01 returned `中央大戲院`;
- C07 and C08 reproduced the already known model errors `愛能情人` and
  `不僅悅且娛心`, so Metal did not repair the missing/incorrect important
  glyphs;
- C09 preserved the crucial `英皇時召霍臨宮中`, but changed the later
  reference transcript's `殷待` to `般待`;
- the official `spotting_json` prompt returned parseable boxes and text for
  C01/C07/C08/C09.

The negative-control and whole-page layout gates nevertheless failed:

- the blank C11 control deterministically repeated `“我”说` until the
  2,048-token limit, whereas the reference Transformers run abstained with
  `图中没有文字`;
- crop-level `layout_parse` returned boxes without the requested full text;
- the 6,176 x 8,960 lossless page requires 16,524 prompt tokens, exceeding
  Tencent's documented `--ctx-size 10240` example;
- with BF16, four independent 32,768-token slots, and prompt caching disabled,
  the full page fit in memory but took 261.3 seconds, hit the 4,096-token
  generation limit, and degenerated into repeated layout boxes. Visual/prompt
  prefill consumed 205.6 seconds for 16,524 tokens; generation consumed 55.1
  seconds;
- a 1,544 x 2,240 page proxy reduced prompt prefill to 8.5 seconds and 3,476
  tokens, but still repeated the date/header until the 512-token cap. Resizing
  fixes visual-token cost, not whole-page layout quality.

The crop-region architecture still assigns region determination to Hunyuan;
deterministic ruling-line detection is not the selected substitute. Follow-up
tests separated Hunyuan's three relevant task/runtime cases:

- the source scan is white-on-black, so a reversible black-on-white polarity
  transform is required for the layout proxy and must be recorded alongside
  the proxy-to-source coordinate transform;
- the official Transformers BF16 checkpoint completed the dedicated `layout`
  task on a 772 x 1,120 normalized proxy, but its boxes covered only about 22%
  of the visibly dense lower-left half of the page;
- the same checkpoint's `spotting_json` task returned valid JSON for that proxy
  but covered only the rightmost portion of the page;
- on a readable 900 x 1,248 viewport, `layout` emitted inverted and out-of-range
  boxes, while `spotting_json` emitted one inverted box after otherwise valid
  items;
- a custom region-only prompt was ignored and collapsed into repetitive OCR,
  so prompt editing is not an accepted repair;
- llama.cpp/Metal truncated all four 900 x 1,248 viewport `layout` calls at
  2,048 tokens, and its `spotting_json` response on a 480 x 700 dense viewport
  was malformed and repetitive.

A 2026-07-19 in-spec follow-up removed the resolution and budget confounds.
The model card specifies a 4K maximum image resolution and a 128K context, so
the 6,176 x 8,960 full-page run was out-of-specification input, and the
512–4,096-token caps confounded several earlier truncation verdicts. With the
reconverted F16 pair, a single 32,768-token slot, and a 16,384-token budget:

- `layout_parse` on a 2,647 x 3,840 normal-polarity proxy consumed 10,126
  prompt tokens and produced 584 well-formed items whose union covers about
  88% of the page — far above the earlier 22% — but the items are mostly
  full-width horizontal bands rather than column-aware regions, the tail
  degenerated into repeated invalid boxes, and the run hit the 16,384-token
  cap without terminating;
- `spotting_json` on the same proxy returned full-height column boxes with
  garbled repeated text and also hit the cap;
- `layout_parse` on the 1,544 x 2,240 proxy with the same 16,384 budget
  stopped normally after only 190 tokens with zero well-formed items, so its
  earlier failure was never a budget problem.

Raw runs:
[`f16v2-inspec-layout-parse-3840.json`](../../artifacts/ocr-challenger/llamacpp-metal/f16v2-inspec-layout-parse-3840.json),
[`f16v2-inspec-spotting-3840.json`](../../artifacts/ocr-challenger/llamacpp-metal/f16v2-inspec-spotting-3840.json),
[`f16v2-inspec-layout-parse-2240.json`](../../artifacts/ocr-challenger/llamacpp-metal/f16v2-inspec-layout-parse-2240.json).

These are coverage and output-contract failures, not evidence that a geometric
detector should silently replace Hunyuan. Strict JSON, termination,
coordinate, and coverage gates fail at every in-spec configuration tested on
this officially supported path. Until a Hunyuan region run satisfies
strict JSON, coordinate, coverage, and visual-review gates, page ingestion must
abstain at region discovery.

The exact diagnostic inputs and transforms are recorded in the
[`normal-polarity viewport manifest`](../../artifacts/ocr-challenger/layout-proxy/v219-p0308.normal-polarity-viewports.manifest.json).
Representative raw outputs are the
[`Transformers 4,096-token layout run`](../../artifacts/ocr-challenger/transformers-mps/bf16-layout-proxy-1120-normal-polarity-4096.json),
[`Transformers spotting run`](../../artifacts/ocr-challenger/transformers-mps/bf16-spotting-proxy-1120-normal-polarity.json),
[`Transformers readable-viewport spotting run`](../../artifacts/ocr-challenger/transformers-mps/bf16-spotting-viewport-q00-normal-4096.json), and
[`llama.cpp four-viewport layout run`](../../artifacts/ocr-challenger/llamacpp-metal/bf16-layout-viewports2240-normal.json).

See the immutable raw runs in
[`fixed-suite-structured-parse.json`](../../artifacts/ocr-challenger/llamacpp-metal/fixed-suite-structured-parse.json),
[`exact-crops-spotting-json.json`](../../artifacts/ocr-challenger/llamacpp-metal/exact-crops-spotting-json.json),
[`bf16-fixed-suite-structured-parse.json`](../../artifacts/ocr-challenger/llamacpp-metal/bf16-fixed-suite-structured-parse.json),
[`bf16-no-cache-full-page-layout-parse.json`](../../artifacts/ocr-challenger/llamacpp-metal/bf16-no-cache-full-page-layout-parse.json),
and
[`bf16-no-cache-layout-proxy-2240.json`](../../artifacts/ocr-challenger/llamacpp-metal/bf16-no-cache-layout-proxy-2240.json).

### Concurrent serving and isolation

The production-shaped crop server uses four slots, continuous batching, split
per-sequence KV buffers (`--no-kv-unified`), and 10,240 tokens per crop slot.
The client sends a complete one-image request every time and explicitly sets
`cache_prompt=false`. Thus old per-slot KV prefixes are not reused; recorded
timings reported `cache_n=0` for every request.

A 16-request mixed-image burst produced exactly one stable output hash for each
of four source crops, all 16 outputs matched sequential execution, and every
request stopped normally. Sequential execution took 13.65 seconds (1.17
requests/s); four-way execution took 9.35 seconds (1.71 requests/s), a 1.46x
throughput gain on this M4 Pro. Individual latency rises under GPU contention,
so four slots are a batch-ingestion setting rather than a latency claim.

This build already uses the split-KV high-throughput design merged in
[llama.cpp PR #14363](https://github.com/ggml-org/llama.cpp/pull/14363). That
change removes cross-sequence attention for independent requests. It improves
multi-request decoding but does not accelerate the vision encoder for one
large page.

Raw load-test artifacts are
[`bf16-no-cache-sequential16.json`](../../artifacts/ocr-challenger/llamacpp-metal/bf16-no-cache-sequential16.json)
and
[`bf16-no-cache-concurrency4-isolation16.json`](../../artifacts/ocr-challenger/llamacpp-metal/bf16-no-cache-concurrency4-isolation16.json).

### Conversion reproducibility

An earlier session described the runnable GGUF pair as an unreviewed
Transformers-fallback artifact. That interpretation is retracted. llama.cpp's
converter pins `transformers==4.57.6` in its own
`requirements-convert_hf_to_gguf.txt`; under that supported configuration the
converter reads the checkpoint's raw `config.json`, takes the `xdrope` branch
added with upstream HunyuanOCR support, and emits a byte-reproducible GGUF.
Installing Transformers 5.13 instead lets `AutoConfig` normalize `xdrope` to
dynamic RoPE, which trips the converter's vanilla-Hunyuan assertion (`HunYuan
dynamic RoPE scaling assumptions changed`); this was reproduced live on
2026-07-19 and is a converter environment mismatch, not a provenance defect of
the produced GGUF. The observed whole-page and blank-input failures are
output-quality findings on an officially supported deployment path; they are
not conversion artifacts, and they still should not be described as a
CUDA-versus-Metal accuracy effect.

References:

- [Tencent's current llama.cpp guide](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/a1ce1099db98edceb153710536af23edf4391cf0/docs/llama_cpp.md)
- [llama.cpp HunyuanOCR support](https://github.com/ggml-org/llama.cpp/pull/21395)
- [Metal conversion follow-up](https://github.com/ggml-org/llama.cpp/commit/7bfe60fdf929ae569b81bbbce7ff7be5a1f8e354)
- [pre-1.5 GGUF repository](https://huggingface.co/ggml-org/HunyuanOCR-GGUF)

LM Studio and Ollama could package the qualified GGUF pair, but they are front
ends, not independent accuracy or acceleration backends; crop-level parity
with native Transformers beyond the fixed suite remains unmeasured.

## CUDA reference

Keep one official CUDA/BF16 environment as a cross-backend reproducibility
reference, not as an assumption that CUDA has better OCR quality.
Tencent documents H20-class NVIDIA execution with at least 24 GB VRAM for its
vLLM and native Transformers paths. The practical choices are:

| Path | Role |
|---|---|
| native Transformers 5.13 | accuracy and cross-backend reference |
| vLLM 0.18.1 autoregressive | production throughput baseline |
| vLLM nightly with DFlash | later long-output optimization, isolated from the baseline |

See Tencent's [inference selection guide](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/a1ce1099db98edceb153710536af23edf4391cf0/inference/README.md).

## Deployment decision

For now:

1. use the F16 Metal server (`hyocr-f16.gguf` + `mmproj-hyocr-f16.gguf`, the
   byte-reproducible official-recipe conversion) with four independent slots as
   the qualified execution candidate for padded crop-level `spotting_json`
   requests; the BF16 language GGUF was removed after recorded byte-identical
   crop results;
2. explicitly disable prompt-prefix caching for reproducible, history-free OCR
   requests;
3. do not use whole-page Hunyuan `layout_parse`: it fails at both source and
   proxy resolution;
4. retain Hunyuan as the only learned crop-region authority; exhaustive
   overlapping viewports may bound input size but do not themselves claim
   semantic boundaries, and the current Hunyuan region path remains
   unqualified because the tested outputs fail coverage or coordinate gates;
5. keep official CUDA BF16 only as a cross-backend reference; the GGUF
   conversion itself is byte-reproducible under llama.cpp's pinned converter
   requirements and is no longer an open caveat;
6. do not try DFlash until the crop pipeline is frozen, because speculative
   decoding cannot establish OCR correctness;
7. never merge OCR text based on backend speed: retain source image/crop, raw
   transcript, prompt, model revision, runtime, and normalization separately.
