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

Metal execution is now locally demonstrated, but the exact HunyuanOCR 1.5
GGUF path is **experimental and not production-qualified**.

The run used:

- host: Apple M4 Pro, 48 GB unified memory, arm64;
- source checkpoint: `tencent/HunyuanOCR` revision
  `de8f10ad2f00a0cefd790b526de8a65dcfdb3205`;
- upstream llama.cpp commit
  `571d0d540df04f25298d0e159e520d9fc62ed121`, build 10068;
- build: Metal and embedded Metal library enabled, Accelerate enabled, and
  OpenMP disabled to avoid the local Anaconda CMake/AppleClang OpenMP linker
  mismatch;
- F16 language GGUF SHA-256
  `f3b4e8e2c7db5c7346b9f28059f7813a7c9b1ff3273434cf79be468a82c551ed`;
- F16 vision-projector GGUF SHA-256
  `4cfd75fc001e7d03e36fdd26a0b36b2d5a2af6922d7800f99ca1673a104559a2`.

Verbose server logs proved hardware placement: the server selected the Apple
M4 Pro Metal device, offloaded all 25/25 language layers, allocated the KV
cache on `MTL0`, and reported `CLIP using MTL0 backend` for the vision encoder.
At 32,768 context it projected about 4.2 GiB of Metal memory. This is real GPU
acceleration rather than CPU execution in a Metal-capable binary.

An uncached control on C09 produced byte-identical text on CPU and Metal. CPU
took 2.712 seconds and Metal 0.848 seconds, a 3.2x end-to-end speedup. Prompt
processing increased from 174 to 1,081 tokens/second (6.2x); generation
increased from 147 to 170 tokens/second (1.16x). The raw artifacts are
[`cpu-c09-structured-parse.json`](../../artifacts/ocr-challenger/llamacpp-metal/cpu-c09-structured-parse.json)
and
[`metal-c09-structured-parse.json`](../../artifacts/ocr-challenger/llamacpp-metal/metal-c09-structured-parse.json).

### Accuracy and serving boundary

The crop-level path is promising:

- all ten text-bearing fixed crops stopped normally;
- C01 returned `中央大戲院`;
- C07 and C08 reproduced the already known model errors `愛能情人` and
  `不僅悅且娛心`, so Metal did not repair the missing/incorrect important
  glyphs;
- C09 preserved the crucial `英皇時召霍臨宮中`, but changed the later
  reference transcript's `殷待` to `般待`;
- the official `spotting_json` prompt returned parseable boxes and text for
  C01/C07/C08/C09.

The parity and layout gates nevertheless failed:

- the blank C11 control deterministically repeated `“我”说` until the
  2,048-token limit, whereas the reference Transformers run abstained with
  `图中没有文字`;
- crop-level `layout_parse` returned boxes without the requested full text;
- the 6,176 x 8,960 lossless page requires 16,524 prompt tokens, exceeding
  Tencent's documented `--ctx-size 10240` example;
- at 32,768 context the full page fit in memory but took 307.3 seconds, hit the
  4,096-token generation limit, and degenerated into repeated layout boxes.

See the immutable raw runs in
[`fixed-suite-structured-parse.json`](../../artifacts/ocr-challenger/llamacpp-metal/fixed-suite-structured-parse.json),
[`exact-crops-spotting-json.json`](../../artifacts/ocr-challenger/llamacpp-metal/exact-crops-spotting-json.json),
and
[`full-page-layout-parse.json`](../../artifacts/ocr-challenger/llamacpp-metal/full-page-layout-parse.json).

### Conversion blocker

The runnable GGUF pair was emitted only when Transformers 5.6.2 did not know
the `hunyuan_vl` class and llama.cpp fell back to the raw `config.json`.
Conversion then preserved the checkpoint's XD-RoPE metadata, but this fallback
is not a sufficiently reviewed production provenance path.

Repeating conversion with Tencent's required Transformers 5.13.0 recognized
`HunYuanVLForConditionalGeneration`, normalized the checkpoint's `xdrope`
configuration to dynamic RoPE, and failed llama.cpp's own assertion:
`HunYuan dynamic RoPE scaling assumptions changed`. No 5.13 GGUF was produced.
That unresolved conversion incompatibility, together with the observed parity
failures, prevents promotion of this backend.

References:

- [Tencent's current llama.cpp guide](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/a1ce1099db98edceb153710536af23edf4391cf0/docs/llama_cpp.md)
- [llama.cpp HunyuanOCR support](https://github.com/ggml-org/llama.cpp/pull/21395)
- [Metal conversion follow-up](https://github.com/ggml-org/llama.cpp/commit/7bfe60fdf929ae569b81bbbce7ff7be5a1f8e354)
- [pre-1.5 GGUF repository](https://huggingface.co/ggml-org/HunyuanOCR-GGUF)

LM Studio and Ollama could package a qualified GGUF pair, but neither resolves
the current 1.5 conversion or parity failure. They are front ends, not
independent accuracy or acceleration backends.

## CUDA recommendation

Keep one official CUDA/BF16 environment as the reproducibility reference.
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

1. keep native Transformers MPS for local crop-level development;
2. keep official CUDA BF16 as the batch and production reference;
3. use this GGUF/Metal build only for reproducible research and upstream
   debugging, not ingestion;
4. require an upstream-reviewed 1.5 conversion plus clean blank, fixed-crop,
   spotting, and full-page layout gates before promotion;
5. do not try DFlash until the base llama.cpp path passes parity, because
   speculative decoding cannot establish base-model correctness;
6. never merge OCR text based on backend speed: retain source image/crop, raw
   transcript, prompt, model revision, runtime, and normalization separately.
