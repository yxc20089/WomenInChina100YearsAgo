# HunyuanOCR 1.5 difficult-glyph pilot

Date: 2026-07-18

Case: volume 219, page 308, central theatre advertisement heading

Status: candidate-only diagnostic; not historian gold

## Outcome

HunyuanOCR 1.5 is the first tested model to return the provisional visual
transcription `中央大戲院` exactly, including both the stylized `戲` and the
historical right-to-left reading order.

| Input/model | Raw output | Direction-derived output |
|---|---|---|
| Stored polarity-inverted tiled PP-OCRv6 | `院大央中` | `中央大院` |
| Expanded polarity-inverted PP-OCRv6 | `院武大央中` | `中央大武院` |
| Expanded polarity-inverted PP-OCRv5 server | `院武大央中` | `中央大武院` |
| Expanded polarity-inverted PaddleOCR-VL 1.6 | `院壽大中央` | not safely normalizable |
| Correct-polarity tight crop, PP-OCRv6 | `院默大央中` | `中央大默院` |
| Correct-polarity tight crop, HunyuanOCR 1.5 | `中央大戲院` (three identical greedy runs) | already logical order |

The frozen comparison record is
[`artifacts/ocr-challenger/v219-p0308.central-theatre.comparison.json`](../../artifacts/ocr-challenger/v219-p0308.central-theatre.comparison.json).
The byte-preserved input crop is
[`artifacts/ocr-challenger/crops/v219-p0308.central-theatre.corrected.png`](../../artifacts/ocr-challenger/crops/v219-p0308.central-theatre.corrected.png).

## Rendering defect found

The PDF page contains one 6176×8960, one-bit JBIG2 `/ImageMask`. Direct image
extraction returned the stencil samples but discarded the PDF painting
semantics, producing white text on a black page. A normal Poppler page render
was the exact inverse of that PNG at every pixel.

The lossless renderer now composites `/ImageMask` pages at the exact embedded
raster dimensions. It records the decoded source-mask hash separately from the
painted-page hash and performs no geometric transform. Existing pilot artifacts
remain immutable evidence of the superseded run; corrected renders require new
run identities and new downstream OCR artifacts.

## Exact Hunyuan system

- Model: `tencent/HunyuanOCR` at revision
  `de8f10ad2f00a0cefd790b526de8a65dcfdb3205`
- Toolkit: `Tencent-Hunyuan/HunyuanOCR` at
  `a1ce1099db98edceb153710536af23edf4391cf0`
- Current family: HunyuanOCR 1.5, 1B BF16
- Task: `structured_parse`
- Frozen official prompt: `提取图中的文字。`
- Generation: greedy, BF16, eager attention, repetition penalty 1.08,
  128-token cap
- Local diagnostic runtime: Transformers 5.13.0, Torch 2.13.0, Apple Metal
- Output SHA-256: `1017a1aaca2e0588910e7139fc22963e8f03653966587bae65d409113ed5602b`

Tencent documents HunyuanOCR 1.5 as an end-to-end OCR-specialized VLM and says
its upgraded training includes historical OCR data, rare/ancient script work,
4K images and a 128K context. Those claims make it relevant, but they are not a
substitute for this archive-specific comparison. The official NVIDIA runtime
requires a separate environment; the Metal run here is an unvalidated but
repeatable local diagnostic. The Tencent Hunyuan Community License is recorded
for deployment review but is not used as a model-quality criterion.

Official sources:

- [HunyuanOCR repository](https://github.com/Tencent-Hunyuan/HunyuanOCR/tree/a1ce1099db98edceb153710536af23edf4391cf0)
- [HunyuanOCR model revision](https://huggingface.co/tencent/HunyuanOCR/tree/de8f10ad2f00a0cefd790b526de8a65dcfdb3205)
- [Official task prompts](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/a1ce1099db98edceb153710536af23edf4391cf0/inference/utils/hunyuan_tasks.py)
- [Official Transformers setup](https://github.com/Tencent-Hunyuan/HunyuanOCR/tree/a1ce1099db98edceb153710536af23edf4391cf0/inference/transformers)

## Architecture decision

Paddle remains useful as the fast detector and coordinate-preserving baseline,
but its recognition confidence cannot select authoritative text. HunyuanOCR 1.5
becomes the leading difficult-region recognition candidate. Its generated text
must remain a separate OCR hypothesis attached to the source crop and model
run; it must not overwrite raw Paddle regions.

The next independent Chinese-lab comparison should use the same corrected crop:

1. `rednote-hilab/dots.mocr` for dense layout and Traditional Chinese examples;
2. `baidu/Qianfan-OCR` for its explicit newspaper mode;
3. `zai-org/GLM-OCR` as a compact 0.9B deployment control;
4. `deepseek-ai/DeepSeek-OCR-2` only after the more structured candidates.

The production flow remains detector-backed:

```text
correct-polarity source render
  -> fast detector + immutable polygons
  -> baseline recognizer
  -> quality triggers (tile edge, disagreement, rare/stylized glyph, order)
  -> HunyuanOCR/independent-lab crop hypotheses
  -> reversible text reconstruction with insertion/substitution/order ledger
  -> historian review for consequential disagreement
  -> NER and entity resolution in reconstructed-text coordinates
```

`中央大戲院` is therefore a successful machine hypothesis and a regression
target, not yet reviewed historical gold.

## Broader targeted suite

The follow-up eleven-crop suite found three exact theatre headings and a strong
clean vertical-text result, but also a missing character in a clean date, an
unreordered RTL sentence, and ungrounded prose on a severely degraded line.
HunyuanOCR 1.5 is therefore a leading difficult-region candidate, not a
universal page-wide authority. See the full
[targeted-suite report](hunyuanocr-1.5-targeted-suite.md).

The locally working Apple Silicon route is native Transformers on MPS. Current
HunyuanOCR 1.5 GGUF conversion failed on changed dynamic-RoPE metadata, and the
available MLX/GGUF packages found in this review are version 1.0. See the
[hardware-path report](hunyuanocr-1.5-hardware.md).
