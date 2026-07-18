# Historical-Chinese NER benchmark

The pinned candidate registry is `candidates.json`. It is a shortlist, not a
production selection.

The annotation policy and adjudication rules are in
[`docs/annotation-guidelines.md`](../../docs/annotation-guidelines.md). The
machine-readable contract is implemented by `wic_history.ner_gold.NERGoldSet`.
It rejects fewer than two distinct reviewers, bad corrected/raw offsets,
surface mismatches, duplicate spans, and duplicate snippet/region identities.

Evaluate each applicable model on two paired inputs: double-corrected text and
the corresponding raw OCR. Split by issue/date rather than random snippets so
near-duplicate newspaper language cannot leak across sets. Report exact and
relaxed span F1 by type, hallucinated-span rate, evidence/offset validity,
throughput, peak memory, and degradation as OCR CER rises.

The production hypothesis is a cascade: gazetteers and rules; a SIKU-BERT-based
project-specific span/token model plus GLiNER-X candidates; then NuExtract3 only
for disagreements, rare types, implicit relations, or difficult page crops.
Every stage must preserve exact source offsets and may abstain. Entity linking
and claim review remain separate gates.

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
