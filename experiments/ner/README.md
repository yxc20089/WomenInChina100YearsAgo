# Historical-Chinese NER benchmark

The pinned candidate registry is `candidates.json`; the implementation-ready
comparison, hardware estimates and frozen stop/go gates are in
[`decision-memo.md`](decision-memo.md). This is a shortlist, not a production
selection.

The annotation policy and adjudication rules are in
[`docs/annotation-guidelines.md`](../../docs/annotation-guidelines.md). The
machine-readable contract is implemented by `wic_history.ner_gold.NERGoldSet`.
It rejects fewer than two distinct reviewers, bad corrected/raw offsets,
surface mismatches, duplicate spans, and duplicate snippet/region identities.
Gold schema 1.1 assigns a model-independent gold region UUID and explicitly maps
it to a source OCR run/region. Prediction schema 1.1 records raw versus
corrected/multimodal input, canonical input SHA-256, dataset/split, ontology,
adapter and prompt/schema revision. Legacy 1.0 artifacts remain readable but do
not satisfy the scored benchmark provenance gate.

Evaluate each applicable model on paired double-corrected text, corresponding
raw OCR, and observed-confusion noise augmentation. Split by issue/date rather
than random snippets so near-duplicate newspaper language cannot leak across
sets. Report exact and relaxed span F1 by type, hallucinated-span rate,
evidence/offset validity, throughput, peak memory, and degradation as OCR CER
rises.

The production hypothesis is a cascade: gazetteers and rules; the winner of an
identical-head MacBERT/GujiRoBERTa/SIKU supervised tournament; Otter CE and/or
GLiNER-X only if a raw-recoverable recall-union gate passes; then NuExtract3
only for disagreements, rare types, implicit relations, or difficult page
crops. Every stage must preserve exact source offsets and may abstain. Entity
linking and claim review remain separate gates.

Use W2NER as the primary overlap/discontinuous-capable supervised head, compare
it to GlobalPointer on one frozen backbone, and retain a single-label BIO/CRF
head only as a flat control. The policy permits nested spans and the same
surface to carry distinct defensible types. Scores from different extractors
are not comparable until calibrated, so artifacts retain every extractor's raw
support instead of discarding disagreements.

## Pinned compatibility comparison

GLiNER-X uses a Stanza word splitter. Automatic per-region language detection
is unsafe for short noisy newspaper OCR: the technical run misclassified
regions as many unrelated languages and began downloading their tokenizers.
For this known corpus, explicitly download and force Traditional Chinese:

```bash
uv run python -c "import stanza; stanza.download('zh-hant', processors='tokenize')"
uv run wic-ner --ocr-artifact artifacts/ocr-smoke/v219-p0308.ppocrv6.json \
  --output artifacts/ner-smoke/v219-p0308.gliner-x-large.first50.json \
  --model knowledgator/gliner-x-large \
  --revision 4a4437f439a78d67c87781b42e8c45373d2adcb0 \
  --word-splitter-language zh-hant --max-regions 50 --batch-size 2
uv run wic-ner-compare \
  --left artifacts/ner-smoke/v219-p0308.gliner-multi-v2.1.first50.json \
  --right artifacts/ner-smoke/v219-p0308.gliner-x-large.first50.json \
  --output artifacts/ner-smoke/comparison-first50.json
```

The identical first-50-region smoke comparison produced 67 GLiNER-X candidates
and 5 GLiNER multi-v2.1 candidates. Only two agreed on exact span and type; four
shared a span regardless of type; candidate Jaccard was 0.029. High-confidence
GLiNER-X candidates include suspicious OCR fragments. This is evidence of
greater candidate volume and major disagreement—not better accuracy. The
reproducible report is `artifacts/ner-smoke/comparison-first50.json`; only the
paired gold benchmark can select the model and threshold.

## Source-resolution non-gold comparison

The comparison was repeated on the 6176×8960 lossless pipeline pilot using all
14 ontology labels, nested spans and multi-label output. On the identical first
50 eligible regions (259 Unicode characters), GLiNER multi-v2.1 emitted 7
candidates and GLiNER-X emitted 82. They shared only 3 exact span/type candidates
and 5 spans regardless of type; candidate Jaccard was 0.035. Rules found zero
mentions on the complete 1,099-region page. Many high-confidence GLiNER-X
outputs are visibly implausible OCR fragments, so neither model is selected or
promoted.

The artifacts and report are under `artifacts/ner-pilot/`. Their identical input
SHA-256 is `a354383d42b824d6586fed9a916aec35f4c600083ae1d37ff4d6d3537f121571`.

Score an artifact on both paired inputs after the adjudicated gold set is
frozen:

```bash
uv run wic-ner-score --gold artifacts/gold/ner-v1.json \
  --predictions artifacts/ner-benchmark/gliner-x.corrected.json \
  --input-text corrected --output artifacts/ner-benchmark/gliner-x.corrected.score.json
uv run wic-ner-score --gold artifacts/gold/ner-v1.json \
  --predictions artifacts/ner-benchmark/gliner-x.raw-ocr.json \
  --input-text raw_ocr --output artifacts/ner-benchmark/gliner-x.raw-ocr.score.json
```

The scorer penalizes invalid/out-of-gold predictions as false positives and
also reports them separately. Raw recoverable recall isolates NER behavior;
end-to-end raw recall includes entities destroyed by OCR. It additionally emits
per-type, decade, page-genre, layout and scan-quality exact scores, aggregate OCR CER,
duration/throughput, and recorded peak memory when the model run supplies it.
