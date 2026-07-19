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

### Conversion blocker

The runnable GGUF pair was emitted only when Transformers 5.6.2 did not know
the `hunyuan_vl` class and llama.cpp fell back to the raw `config.json`.
Conversion then preserved the checkpoint's XD-RoPE metadata, but this fallback
is not a sufficiently reviewed production provenance path.

Repeating conversion with Tencent's required Transformers 5.13.0 recognized
`HunYuanVLForConditionalGeneration`, normalized the checkpoint's `xdrope`
configuration to dynamic RoPE, and failed llama.cpp's own assertion:
`HunYuan dynamic RoPE scaling assumptions changed`. No 5.13 GGUF was produced.
That unresolved conversion incompatibility remains a provenance risk. The
observed whole-page and blank-input failures are separate output-quality
risks; neither should be described as a CUDA-versus-Metal accuracy effect.

References:

- [Tencent's current llama.cpp guide](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/a1ce1099db98edceb153710536af23edf4391cf0/docs/llama_cpp.md)
- [llama.cpp HunyuanOCR support](https://github.com/ggml-org/llama.cpp/pull/21395)
- [Metal conversion follow-up](https://github.com/ggml-org/llama.cpp/commit/7bfe60fdf929ae569b81bbbce7ff7be5a1f8e354)
- [pre-1.5 GGUF repository](https://huggingface.co/ggml-org/HunyuanOCR-GGUF)

LM Studio and Ollama could package a qualified GGUF pair, but neither resolves
the current 1.5 conversion or parity failure. They are front ends, not
independent accuracy or acceleration backends.

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

1. use the BF16 Metal server with four independent slots as the qualified
   execution candidate for padded crop-level `spotting_json` requests;
2. explicitly disable prompt-prefix caching for reproducible, history-free OCR
   requests;
3. do not use whole-page Hunyuan `layout_parse`: it fails at both source and
   proxy resolution;
4. retain the immutable lossless page, use a downsampled proxy only for
   confidence-bearing ruling-line/column proposals, and map padded crop boxes
   back to source pixels before Hunyuan OCR;
5. keep official CUDA BF16 only as a cross-backend reference and keep the GGUF
   conversion caveat attached to every run;
6. do not try DFlash until the crop pipeline is frozen, because speculative
   decoding cannot establish OCR correctness;
7. never merge OCR text based on backend speed: retain source image/crop, raw
   transcript, prompt, model revision, runtime, and normalization separately.
