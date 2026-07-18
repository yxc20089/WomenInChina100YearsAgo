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
identical-head MacBERT/MacBERT-DAPT/mmBERT/GujiRoBERTa/SIKU supervised tournament;
GLiNER-X only if a raw-recoverable recall-union gate passes; then NuExtract3
only for disagreements, rare types, implicit relations, or difficult page
crops. Otter remains research-only until its checkpoint-weight license is
explicit. Every stage must preserve exact source offsets and may abstain.
Entity linking and claim review remain separate gates.

The 2026 frontier review found no model with target evidence strong enough to
replace MacBERT-W2NER before scoring. It adds pinned mmBERT-W2NER only as a
same-head challenger and requires a Chinese tokenizer/character-offset
round-trip test. PP-UIE-0.5B is tracked separately for post-independent-review
annotation suggestions; its mutable weight path and string-only demonstrated
output keep it outside the executable tournament until exact bytes, rights,
and offset recovery are frozen. Pinned GLiNER large v2.5 remains a low-priority
control because its card provides no Chinese, historical, or OCR result.

### Tokenizer offset qualification

Before training mmBERT, run the committed six-case Unicode fixture through the
pinned tokenizer snapshot:

```bash
uv run wic-ner-tokenizer-check \
  --fixture experiments/ner/tokenizer-offset-fixture-v1.json \
  --model jhu-clsp/mmBERT-base \
  --revision c5955035435e2bf121cde7f3c8863ef52ff35d82 \
  --code-revision 0000000000000000000000000000000000000000 \
  --output artifacts/ner-benchmark/mmbert-tokenizer-offsets.json
```

Replace the zero revision with the exact commit containing the qualification
code. The artifact hashes every downloaded tokenizer/configuration file and
fails on a moving revision, a slow tokenizer, Unicode normalization drift,
missing or multiply covered non-whitespace characters, or any probe that does
not round-trip to the exact source span. The fixture covers uninterrupted
Traditional Chinese, variant forms, a supplementary-plane character, line
breaks/full-width punctuation, OCR-confusion stress strings, `□`, and `�`.
The sole virtual-token exception is
`standalone_sentencepiece_prefix_duplicate_v1`: an initial standalone `▁`
may be excluded from alignment only when it duplicates the real next token's
exact `(0, 1)` source offset. The artifact retains and flags that token, and
downstream adapters must apply the same policy. Other overlap remains a hard
failure.
Passing proves offset plumbing only; unknown-token counts are reported, and no
result is evidence of NER accuracy or historical validity.

Use W2NER at official implementation commit
`a34ff841891919001080edefb50e14fa9dc15e1c` as the primary
overlap/discontinuous-capable supervised head, compare it to GlobalPointer on
one frozen backbone, and retain a single-label BIO/CRF head only as a flat
control. The policy permits nested spans and the same surface to carry distinct
defensible types. The current evidence schema stores contiguous spans, so
discontinuous scoring remains blocked on an explicit contract extension.
Scores from different extractors are not comparable until calibrated, so
artifacts retain every extractor's raw support instead of discarding
disagreements.

## Executable benchmark boundary

`benchmark-spec.json` freezes the arms, eligibility rules, metrics, provenance
requirements and stop/go gates. Its `benchmark_results` array is intentionally
empty: there is no eligible historical gold set and therefore no model winner.

The benchmark artifact is deliberately different from the production
`NERArtifact`. A scientific test split can contain snippets drawn from many OCR
runs, so `BenchmarkPredictionArtifact` records every source OCR run and, for
every input including zero-mention inputs, the snippet ID, model-independent
gold-region UUID, source OCR run/region UUID and exact input-text hash. The
production artifact still represents one OCR run and remains the only artifact
accepted by ingestion.

Create an issue-level manifest only after historians assign the snippets to
real issues. One issue may appear in exactly one split:

```json
{
  "schema_version": "1.0",
  "dataset_id": "ner-gold-v1",
  "created_at": "2026-07-18T00:00:00Z",
  "assigned_by": "historian-id",
  "assignments": [
    {"snippet_id": "snippet-001", "issue_id": "issue-001", "split": "train"}
  ],
  "notes": []
}
```

Freeze the paired raw/corrected inputs:

```bash
uv run wic-ner-benchmark prepare \
  --gold artifacts/gold/ner-v1.json \
  --split-manifest artifacts/gold/ner-v1.issue-splits.json \
  --dataset-id ner-benchmark-v1 \
  --input-variant raw_ocr --input-variant corrected_text \
  --output artifacts/ner-benchmark/dataset-v1.json
```

The command may prepare a small technical fixture but marks it ineligible until
it has at least 500 snippets, 30 issues, all three issue-isolated splits and
three publication decades. `run` refuses such a dataset unless the caller adds
the conspicuous `--allow-ineligible-technical-run` flag. That flag never makes
the output scientific evidence.

Rules and both pinned GLiNER arms use the common executable adapter today:

```bash
uv run wic-ner-benchmark run \
  --dataset artifacts/ner-benchmark/dataset-v1.json \
  --split development --input-variant raw_ocr \
  --adapter gliner --model knowledgator/gliner-x-large \
  --revision 4a4437f439a78d67c87781b42e8c45373d2adcb0 \
  --license Apache-2.0 --word-splitter-language zh-hant \
  --code-revision 0000000000000000000000000000000000000000 \
  --output artifacts/ner-benchmark/gliner-x.development.raw.json
```

Replace the zero code revision with the exact 40-character project commit. The
adapter rejects moving model labels and abbreviated code revisions. The
MacBERT/mmBERT/historical W2NER, Otter and structured-generation arms are
specified but hard-blocked on their
listed training, license, prompt/schema, offset-resolution or hardware
requirements; they are not silently approximated by another implementation.

Score the separate prediction artifact with the existing scorer. Its CLI
verifies the exact gold file SHA-256 and scores only the frozen split inputs,
including inputs for which the model emitted no mention:

```bash
uv run wic-ner-score --gold artifacts/gold/ner-v1.json \
  --predictions artifacts/ner-benchmark/gliner-x.development.raw.json \
  --benchmark-dataset artifacts/ner-benchmark/dataset-v1.json \
  --input-text raw_ocr \
  --output artifacts/ner-benchmark/gliner-x.development.raw.score.json
```

Use Unicode characters/second and regions/second for throughput comparisons.
Candidate mentions/second is reported only as an output-density diagnostic and
must not favor a model merely for emitting more candidates.

After scoring the same frozen input for two adapters, calculate the selection
interval by resampling complete issues rather than individual snippets:

```bash
uv run wic-ner-benchmark-compare \
  --baseline-score artifacts/ner-benchmark/macbert.test.raw.score.json \
  --challenger-score artifacts/ner-benchmark/macbert-dapt.test.raw.score.json \
  --bootstrap-samples 10000 --seed 1729 \
  --output artifacts/ner-benchmark/macbert-dapt-vs-macbert.test.raw.json
```

The comparator refuses scores whose exact gold hash, benchmark dataset, split,
input hash, ontology or confidence threshold differ. A positive interval still
does not waive the absolute quality, cost, evidence-integrity or historian
review gates.

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

Legacy single-OCR-run artifacts can also be scored on both paired inputs after
the adjudicated gold set is frozen:

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
