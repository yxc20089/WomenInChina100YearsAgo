# Corpus audit report

Generated: 2026-07-18  
Source: `s3://ccaa-us-east-1-504133794192/sb_raw/`  
Machine-readable outputs: [`artifacts/corpus-audit`](../artifacts/corpus-audit/)

## Result

The prefix contains the complete numbered sequence of 400 *Shen Bao* volumes plus one year/volume index image. No audio, video or born-digital text is present.

| Container | Objects | Bytes | Resolved pages |
|---|---:|---:|---:|
| PDF | 395 | 167,284,981,412 | 336,112 across 391 readable PDFs |
| DjVu | 5 | 1,772,878,220 | 4,399 |
| JPEG index | 1 | 708,760 | Not applicable |
| **Total** | **401** | **169,058,568,392** | **340,511 known pages** |

The page total is a lower bound because four malformed PDFs cannot yet expose their page trees. The readable PDFs range from 501 to 2,265 pages, with a median of 815 pages.

The proposed 1924–1926 pilot comprises 32 volumes and 24,218 pages:

| Year | Volumes | Pages |
|---|---:|---:|
| 1924 | 10 | 8,502 |
| 1925 | 11 | 7,147 |
| 1926 | 11 | 8,569 |

## Integrity findings

Fast bounded checks passed for 397 objects. These checks validate signatures and PDF/JPEG trailer markers, not every byte or page stream.

Four PDFs have valid PDF headers but no `startxref`/`%%EOF` trailer and require recovery assessment:

| Volume | Year | Size | Finding |
|---:|---:|---:|---|
| 73 | 1903 | 160,403,456 | Truncated/missing PDF trailer |
| 256 | 1929 | 562,966,528 | Truncated/missing PDF trailer |
| 262 | 1929 | 406,482,944 | Truncated/missing PDF trailer |
| 327 | 1935 | 395,276,288 | Truncated/missing PDF trailer |

All volume numbers 1–400 are represented exactly once. No potential duplicate groups share both size and ETag. This is not proof that no duplicate pages exist; image-level duplicate analysis remains outstanding.

The five bundled DjVu directory counts are:

| Volume | Year | Pages |
|---:|---:|---:|
| 93 | 1908 | 841 |
| 94 | 1908 | 805 |
| 95 | 1908 | 859 |
| 96 | 1908 | 891 |
| 106 | 1910 | 1,003 |

## Validation method and limits

- S3 listing supplies object key, size, modification time, storage class and ETag.
- Header/trailer validation uses bounded `GetObject` byte ranges.
- PDF page counts follow classic xref, catalog and page-tree objects using bounded reads; whole volumes are not downloaded.
- DjVu counts come from the bundled `DIRM` directory.
- ETags are retained as source metadata but are not represented as SHA-256 checksums.
- Page-image decodability, full SHA-256, raster consistency, blank/duplicate pages and text-layer state still require deeper inspection.
- A complete downloaded sample (volume 2) contains 599 300-DPI, 1-bit JBIG2 page images and effectively no embedded text.

## Benchmark sampling

[`artifacts/benchmark-sample`](../artifacts/benchmark-sample/) contains a deterministic 500-page candidate plan: 50 evenly distributed pages from one median-sized readable volume in each of ten year/format strata. It is a screening pool—not the gold set.

Rendering status:

- 450 pages from the nine PDF strata have been rendered, hashed and recorded in `artifacts/benchmark-pages/render_manifest.jsonl`;
- the screening derivatives occupy 807,563,185 bytes;
- 50 pages from DjVu volume 93 are explicitly marked `unsupported_renderer` pending a pinned DjVuLibre-compatible renderer;
- source PDFs are cached outside the repository under `/tmp/wic-source-cache` and may be safely regenerated from S3.

After rendering, reviewers must assign page genre and quality labels, then select 150–250 pages with deliberate coverage of vertical/mixed layout, advertisements, classifieds, photographs/captions, tables, rare glyphs, bleed-through, skew, gutters and cropping.

## Next actions

1. Obtain clean replacements or test non-destructive recovery for the four malformed PDFs.
2. Download only the ten selected screening volumes into temporary/cache storage.
3. Render the 500 candidate pages with stable identifiers and checksums.
4. Visually stratify the pages and select the 150–250 page gold set.
5. Write transcription/layout annotation guidelines before producing ground truth.

Screening renders use 120-DPI grayscale JPEG at quality 72 to limit local storage. This is not an OCR input format; selected gold pages will be rendered losslessly at their source resolution.
