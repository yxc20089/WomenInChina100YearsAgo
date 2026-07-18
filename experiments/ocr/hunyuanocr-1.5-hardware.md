# HunyuanOCR 1.5 hardware paths

Date: 2026-07-18

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

Current llama.cpp has HunyuanOCR and Metal support, and Tencent provides a
llama.cpp guide. However, local conversion of the exact 1.5 revision with
llama.cpp commit `571d0d540df04f25298d0e159e520d9fc62ed121` failed before
writing GGUF. The converter asserted that Hunyuan's dynamic RoPE assumptions
had changed after it read the 1.5 context and frequency metadata.

Therefore:

- do not claim the present 1.5 checkpoint works through llama.cpp on this Mac;
- do not substitute `ggml-org/HunyuanOCR-GGUF`, which predates 1.5;
- revisit after an upstream converter update or a reviewed conversion patch;
- if conversion succeeds, compare F16 Metal against the MPS/CUDA transcript
  before evaluating Q8_0.

References:

- [Tencent's current llama.cpp guide](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/a1ce1099db98edceb153710536af23edf4391cf0/docs/llama_cpp.md)
- [llama.cpp HunyuanOCR support](https://github.com/ggml-org/llama.cpp/pull/21395)
- [Metal conversion follow-up](https://github.com/ggml-org/llama.cpp/commit/7bfe60fdf929ae569b81bbbce7ff7be5a1f8e354)
- [pre-1.5 GGUF repository](https://huggingface.co/ggml-org/HunyuanOCR-GGUF)

LM Studio and Ollama may eventually host the converted GGUF pair, but neither
solves the current 1.5 conversion blocker. They are packaging options, not
independent acceleration backends.

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

1. use native Transformers MPS for local, crop-level development;
2. use official CUDA BF16 for batch benchmarking and the production reference;
3. defer MLX and 1.5 GGUF/Metal until they pass the identical frozen crops;
4. never merge OCR text based on backend speed: retain the source crop, raw
   transcript, model revision, runtime, and any normalization separately.
