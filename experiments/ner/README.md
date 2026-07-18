# Late-Qing/Republican Traditional-Chinese NER benchmark

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

The corpus is printed Traditional Chinese from roughly 100 years ago, not
ancient/Classical Chinese. Treat its grammar and discourse as modern or
transitional newspaper Chinese unless historians label a particular passage
otherwise. The main modeling differences are character forms, period names and
titles, typography/layout, punctuation and OCR corruption—not a blanket ancient
language structure. Original characters remain immutable evidence; any
Simplified or normalized form is an auxiliary search feature only.

The production hypothesis is a cascade: gazetteers and rules; a frozen-backbone
head comparison followed by a MacBERT/MacBERT-DAPT/mmBERT/qualified
Chinese-ModernBERT supervised tournament, with SIKU retained as a directly
target-period-evaluated control and GujiRoBERTa only as a license-gated ancient
domain-mismatch control; GLiNER-X only if a raw-recoverable recall-union gate
passes; then locally served Qwen3.5-0.8B or NuExtract3 only for disagreements,
rare types, implicit relations, or difficult page crops. Chinese ModernBERT
cannot execute until its pinned
custom tokenizer passes the raw-offset gate with destructive preprocessing
disabled. Otter remains research-only until its checkpoint-weight license is
explicit. Every stage must preserve exact source offsets and may abstain.
Entity linking and claim review remain separate gates.

The 2026 frontier review found no model with target evidence strong enough to
win before scoring. It adds pinned mmBERT only as a
same-head challenger and requires a Chinese tokenizer/character-offset
round-trip test. PP-UIE-0.5B is tracked separately for post-independent-review
annotation suggestions; its mutable weight path and string-only demonstrated
output keep it outside the executable tournament until exact bytes, rights,
and offset recovery are frozen. Pinned GLiNER large v2.5 remains a low-priority
control because its card provides no Chinese or target OCR result and its
whitespace-style splitter is unsafe for uninterrupted Chinese.

A second independent frontier search adds
`feynmanzhao/chinese-modernbert-large-wwm@b00f1ff1901161f68339890fe48e4dbb6ee76f4d`
as a tokenizer-gated, same-head challenger—not as a new primary. Its 377M
parameters, Chinese-only pretraining and long context are interesting, but its
paper reports no NER or target-domain result. More importantly, its custom
tokenizer defaults to preprocessing that changes source text. Execution is
blocked until the pinned code is audited in isolation, `do_text_preprocessing`
is false, and an expanded exact-offset fixture passes.

The higher-priority new experiment is a clean versus empirical OCR-noise
factorial for each finalist encoder. Begin with training-only, length-preserving
character substitutions sampled from adjudicated training-issue confusions.
Insertions and deletions require deterministic source-to-augmented edit maps.
The 2026 historical-German VET study reported 77.9 entity micro-F1 for its
synthetic-noise arm, but that external score is not transferable to this
archive. Its reference repository has no declared license, so the committed
protocol requires a clean-room implementation rather than copied code.

That exporter is now implemented and unit-tested. It creates one-character
W2NER tokens while omitting whitespace that the official loader cannot map to
subwords; the manifest retains an exact character-offset map. It derives only
length-preserving confusions from adjudicated training-issue entities, augments
corrected training records only, and preserves held-out raw/corrected views.
Eligible gold and model training remain pending.

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

The committed mmBERT run at
`experiments/ner/mmbert-tokenizer-offset-qualification.json` passed all six
cases with no unknown tokens. It records two audited virtual prefixes, model
revision `c5955035435e2bf121cde7f3c8863ef52ff35d82`, code revision
`3ae5fb5ad8e93c901dbd249e3e7d3f50fc88499a`, tokenizer-file manifest
`66367f39282358dd81ab317b563d283dcecaff20cd70ac97c0bffa7da222314e`,
and artifact SHA-256
`20a09b9b750faed9dc744233a33cf3378961e3d7fb224d3ad9267e1b984b38b8`.
This clears only the tokenizer gate; eligible gold, the trained W2NER head,
and the model-quality tournament remain absent.

Use W2NER at official implementation commit
`a34ff841891919001080edefb50e14fa9dc15e1c` as the first implemented
overlap/discontinuous-capable head candidate, not as a winner. Compare it to
GlobalPointer and a single-label BIO/CRF control on one frozen MacBERT backbone
with identical issue splits and search budgets. The EvaHan model-selection
table favors W2NER by 1.15 F1 over GlobalPointer, but its authors explicitly
warn that augmented samples may leak across those random folds; its public-test
comparison changes encoders and cannot isolate the head. The current evidence
schema stores contiguous spans, so discontinuous scoring remains blocked on an
explicit contract extension.
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

Export the paired data into the pinned official W2NER loader format:

```bash
uv run wic-ner-training-export \
  --gold artifacts/gold/ner-v1.json \
  --split-manifest artifacts/gold/ner-v1.issue-splits.json \
  --dataset-id ner-benchmark-v1 --export-id w2ner-training-v1 \
  --project-code-revision 0000000000000000000000000000000000000000 \
  --maximum-record-characters 256 --augmentation-probability 0.15 \
  --augmented-copies 1 --seed 17 \
  --output-directory artifacts/ner-benchmark/w2ner-training-v1
```

The output directory contains a self-validating `manifest.json` and native
`sentence`/`ner` JSON views for each split and input variant, plus a distinct
training corrected-text view containing empirical substitutions. Existing
files are never overwritten. Small ineligible fixtures require the explicit
`--allow-ineligible-technical-export` flag and cannot become scientific runs.

## Local Qwen structured-NER arm

Qwen3.5-0.8B is a benchmark challenger, not an exclusion. The canonical arm is
Ollama 0.32.0 with the official `qwen3.5:0.8b` Q8 registry manifest
`sha256:f3817196d142eaf72ce79dfebe53dcb20bd21da87ce13e138a8f8e10a866b3a4`.
Abort if Ollama's tags API reports different bytes, then copy the verified model
to a frozen project-local name. LM Studio 0.4.19 build 2 with pinned
`Qwen3.5-0.8B-Q8_0.gguf` SHA-256
`0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c` is an
optional, separately scored runtime. Both expose an OpenAI-compatible chat
endpoint through the implemented backend-neutral adapter. Its prompt/schema
SHA-256 is
`ee8fafd34d2b5c3039b46b231eacd491c0d5bfb9b33760381fc59adb8a457505` and
its response-format SHA-256 is
`cdc6d4c736bae55ec47ef2795e22a4435262f9f8d6bb9d95643498a46a8f33a6`.
Freeze the Hugging Face source revision, local artifact SHA-256, server
version, model configuration, prompt/JSON-schema SHA-256 and all decoding
controls. The first arm is text-only with `reasoning_effort=none`, temperature
zero and a fixed seed. Repeat a schema canary and report nondeterminism rather than assuming the
same seed is equivalent across runtimes. Image input is a separate page-crop
ablation.

The response contract is an `entities` array of `type`, `surface`, `start` and
exclusive `end`. A validator must reject unknown ontology values, ranges outside
the Unicode source string, `source[start:end] != surface`, duplicate entities,
and a surface without unique offsets when the model omits or contradicts them.
This is how the project can benefit from Qwen's Chinese competence without
letting a generative model rewrite historical evidence.

Run the verified Ollama condition:

```bash
uv run wic-ner-benchmark run \
  --dataset artifacts/ner-benchmark/dataset-v1.json \
  --split development --input-variant raw_ocr \
  --adapter structured-generation \
  --model Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --license Apache-2.0 \
  --base-url http://127.0.0.1:11434/v1 \
  --served-model project/qwen35-08b-q8-ner-v1 \
  --runtime-name ollama --runtime-version 0.32.0 \
  --local-artifact-sha256 f3817196d142eaf72ce79dfebe53dcb20bd21da87ce13e138a8f8e10a866b3a4 \
  --ollama-manifest-digest sha256:f3817196d142eaf72ce79dfebe53dcb20bd21da87ce13e138a8f8e10a866b3a4 \
  --quantization Q8_0 --seed 42 --schema-canary-repetitions 3 \
  --code-revision 0000000000000000000000000000000000000000 \
  --output artifacts/ner-benchmark/qwen35-08b.ollama.development.raw.json
```

The command checks the live Ollama version and requested model digest before
running the canary. For LM Studio, use `--runtime-name lm_studio`, its exact
runtime version, `--local-model-artifact /path/to/model.gguf`, and the file's
`--local-artifact-sha256`; that file is hashed before inference. The optional
`NER_LLM_API_KEY` environment variable supplies a bearer token without placing
it in CLI arguments or artifacts. Non-loopback endpoints require HTTPS and
`--allow-remote-model-endpoint`; redirects are not followed. Replace the zero
code revision with the exact project commit.

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
MacBERT/mmBERT supervised arms and Otter remain blocked on their listed training
or license requirements. The Qwen structured-generation adapter is implemented
but remains unscored until a real content-verified local model and eligible gold
exist; it is not silently approximated by another model.

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

The official [GLiNER-X collection](https://huggingface.co/collections/knowledgator/gliner-x)
contains three current Apache-2.0 checkpoints and three legacy v0.5 checkpoints.
For the current family, published `zh_pud` F1 is 0.5792 for small `d51a098…`,
0.6152 for base `a6c7a8f…`, and 0.6794 for large `4a4437f…`; the same table gives
GLiNER multi-v2.1 0.6410. The underlying [Chinese PUD
corpus](https://universaldependencies.org/treebanks/zh_pud/index.html) is modern,
professionally translated news/Wikipedia and its published inventory visibly
contains Traditional characters. That makes the score relevant modern-script
evidence, but not evidence for Republican-era vocabulary, OCR corruption,
historical typography, or this project's ontology. Keep current large as the
current-release/Stanza arm, base only as its efficiency control after large
qualifies or under a clear hardware constraint, and small only as diagnostic.

The legacy `gliner-x-large-v0.5@f41e752…` card reports a higher `zh_pud` F1 of
0.709. Licensing is not a blocker for this project, so run it as an active
accuracy challenger, separately from the current large checkpoint. Record its
CC-BY-NC-SA-4.0 license as provenance, and freeze its universal/Jieba Chinese
splitter because it differs from the current Stanza path. The legacy base
reports 0.623; the legacy small reports 0.269 and has a whitespace splitter
unsafe for uninterrupted Chinese, so neither warrants another main arm.

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
