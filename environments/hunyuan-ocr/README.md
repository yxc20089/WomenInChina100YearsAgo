# HunyuanOCR 1.5 CUDA worker environment

HunyuanOCR 1.5's official native runtime requires Transformers 5.13+ and an
NVIDIA CUDA device. Keep it separate from the repository environment because
the optional GLiNER research dependency currently constrains Transformers to
an older range.

Create this environment on the CUDA worker with a PyTorch build matching its
driver, then install `requirements.txt`. The application loads the sole OCR
model name, immutable revision, official task prompts, decoding parameters,
dtype, and device from `config/pipeline-models.toml`.

Run:

```bash
wic-layout \
  --image "$PAGE_IMAGE" \
  --source-uri "$SOURCE_URI" \
  --render-manifest "$LOSSLESS_MANIFEST" \
  --page "$PAGE_NUMBER" \
  --output "$LAYOUT_ARTIFACT"
wic-ocr \
  --image "$PAGE_IMAGE" \
  --layout-artifact "$LAYOUT_ARTIFACT" \
  --output "$OCR_ARTIFACT"
```

The worker fails if CUDA or the official HunyuanVL Transformers class is
absent. It never switches to another OCR model, an older GGUF, or a community
MLX checkpoint. A failure is retained as a review/operations failure.

Primary runtime reference: [Tencent HunyuanOCR Transformers setup](https://github.com/Tencent-Hunyuan/HunyuanOCR/tree/a1ce1099db98edceb153710536af23edf4391cf0/inference/transformers).
