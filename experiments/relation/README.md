# Relation and event extraction evaluation

This directory freezes the relation-extraction shortlist and the minimum
project-specific evaluation contract. It contains no historical gold scores
and names no winner. The live database currently has zero reviewed linked
mentions and zero claims, so a nonempty historical relation artifact would be
false evidence at this stage.

## Selection

The required baseline is `reviewed-co-mention-rules-v2`. It emits a candidate
only when two exact adjudicated mention spans have ontology-compatible types
and an explicit cue occurs strictly between them without crossing a clause
boundary. It cites the minimal argument/cue span and carries the source-object,
derivative, image, region and polygon provenance. It is deliberately
high-precision and low-recall.

The first supervised hypothesis is a MacBERT pair classifier trained through
the pinned DeepKE framework. It conditions on adjudicated entity mentions,
which separates relation classification from NER and entity-linking errors.
PRGC is the joint overlapping-triple challenger. NuExtract3 is routed only to
difficult/image-context cases after valid JSON, verbatim argument, exact offset
and evidence recovery. OneKE is a noncommercial research control. None has a
historical-Chinese OCR relation result that settles this project.

The candidate rationale, rights and exact revisions are in `candidates.json`.
The most directly relevant external Shen Bao corpus is retyped rather than
OCR-derived and does not provide this claim ontology; its repository also lacks
an explicit dataset license. HistRED is Hanja/Korean and CC-BY-NC-ND. These are
research context, not substitutes for project gold.

## Gold contract

`RelationBenchmarkDataset` depends on a byte-exact adjudicated NER gold file.
Every relation unit freezes:

- its issue-level split and approved coherent-unit revision;
- one or more complete scan/region source pointers;
- corrected and raw-OCR text;
- exact mappings into the byte-frozen source NER snippets, plus
  model-independent relation mention IDs copied from their adjudicated
  corrected/raw spans and types;
- two independent relation annotations and an independent adjudication;
- exact evidence spans containing both relation arguments, including negative
  units where no schema relation is asserted.

Eligibility is derived from the contents and cannot be toggled manually. The
minimum is 300 unique units from 30 issues and two decades, with at least 50
units in every issue-isolated split, 50 positive and 10 negative test examples,
and at least 10 test relations for every frozen predicate. Every source pointer
must include source/image SHA-256, derivative, evidence tier, region and
polygon. Technical fixtures may run only with the explicit ineligible flag and
are never scientific scores.

## Commands

Run the deterministic baseline after the relation dataset and its exact source
NER gold file have been finalized:

```bash
uv run wic-relation-benchmark run-rules \
  --dataset artifacts/relation-benchmark/dataset-v1.json \
  --ner-gold artifacts/ner-gold/gold-v1.json \
  --split test --input-variant raw_ocr \
  --code-revision 0000000000000000000000000000000000000000 \
  --output artifacts/relation-benchmark/rules.raw.predictions.json
```

Replace the zero revision with the exact 40-character project commit. Score a
pinned prediction artifact:

```bash
uv run wic-relation-benchmark score \
  --dataset artifacts/relation-benchmark/dataset-v1.json \
  --ner-gold artifacts/ner-gold/gold-v1.json \
  --predictions artifacts/relation-benchmark/model-a.raw.predictions.json \
  --output artifacts/relation-benchmark/model-a.raw.report.json
```

The scorer rejects dataset/hash/split/input drift, arguments outside the
adjudicated NER spans, unknown predicates, invalid evidence offsets/surfaces and
incomplete result coverage. It reports exact relation F1, evidence-span F1,
negative false positives, raw recoverability, annotation agreement, strata,
raw-recoverable and end-to-end raw relation quality, throughput, latency and
provider usage/cost when applicable. Structural
evidence overlap is not semantic entailment.

Compare two reports on identical issue clusters:

```bash
uv run wic-relation-benchmark compare \
  --report-a artifacts/relation-benchmark/model-a.raw.report.json \
  --label-a candidate-a \
  --report-b artifacts/relation-benchmark/model-b.raw.report.json \
  --label-b candidate-b \
  --bootstrap-seed 17 --bootstrap-resamples 5000 \
  --output artifacts/relation-benchmark/a-vs-b.raw.paired.json
```

The confidence interval resamples whole issues, not individual relations. No
score promotes a candidate to a reviewed claim. Historians must separately
inspect entailment, approve or reject the claim, and only then rebuild Neo4j.

## Primary sources

- [DeepKE official repository](https://github.com/zjunlp/DeepKE)
- [PRGC paper](https://aclanthology.org/2021.acl-long.486/)
- [NuExtract3 official model card](https://huggingface.co/numind/NuExtract3)
- [OneKE official repository](https://github.com/zjunlp/OneKE)
- [KnowCoder-X paper](https://aclanthology.org/2025.findings-acl.748/)
- [TRUE-UIE paper](https://aclanthology.org/2024.naacl-long.103/)
- [HistRED paper and license statement](https://aclanthology.org/2023.acl-long.180/)
- [Historical Chinese newspaper dataset paper](https://aclanthology.org/2024.lrec-main.35/)
