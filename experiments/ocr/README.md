# Historical-Chinese OCR and layout benchmark

The provisional evidence-extraction candidate is PP-StructureV3 + PP-OCRv6;
PaddleOCR-VL-1.6 is the difficult-region challenger. MinerU remains an
end-to-end parsing challenger subject to license review. These are benchmark
roles, not quality conclusions.

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
