# NER candidate decision memo

Research cutoff: 2026-07-18  
Decision status: benchmark shortlist; no production winner

## Recommendation

Benchmark a project-trained **SIKU-BERT overlap-capable span head** as the
likely high-throughput primary, **GLiNER-X** as an open-label recall challenger,
and **NuExtract3** only as a routed difficult-case stage. Keep **Qwen3.5-9B** as
a constrained high-compute control and stop that arm unless it materially beats
NuExtract3 on identical routed cases.

No candidate may write accepted entities or graph facts. Exact Unicode
surface/offset validation and historian review remain mandatory.

| Candidate | Target role | Direct evidence and gap | Approximate deployment envelope |
|---|---|---|---|
| `knowledgator/gliner-x-large@4a4437f439a78d67c87781b42e8c45373d2adcb0` | Zero-shot/open-label recall challenger | [Official card](https://huggingface.co/knowledgator/gliner-x-large) reports `zh_pud` F1 0.6794, but no historical, Traditional-Chinese, or OCR benchmark | 864.6M parameters; roughly 6–10 GB CPU RAM or 4–6 GB GPU VRAM |
| `SIKU-BERT/sikubert@fc656de2d6bde33919102dd3abe31c843f42226a` + trained span head | Supervised primary | Historical Traditional-Chinese pretraining; the [GujiBERT project](https://github.com/hsc748NLP/GujiBERT-and-GujiGPT) reports *Shiji* NER F1 90.68, but newspaper/OCR transfer is unknown | BERT-base; roughly 2–4 GB inference memory and 12–16 GB VRAM for full fine-tuning |
| `numind/NuExtract3@2e9fca82ee641e6bb6e1f5d905241e994be27a07` | Routed difficult cases, image/context ablation, later relation extraction | [Official card](https://huggingface.co/numind/NuExtract3) supports text/images, JSON templates and verbatim strings; its extraction benchmark is not an auditable historical-Chinese result | About 9.3 GB BF16 weights; 24 GB GPU recommended |
| `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Structured-output control only | [Official card](https://huggingface.co/Qwen/Qwen3.5-9B) covers Chinese and vision, but provides no target NER evidence | About 19.3 GB BF16 weights; 48 GB GPU for an honest BF16 control |

All four repositories declare Apache-2.0. That does not settle training-data or
scan redistribution rights. Verify Siku digitization provenance before
redistributing a fine-tuned checkpoint; do not use CHisIEC until its repository
license is clarified; and do not confuse AWS read permission with permission to
publish scans, OCR, or annotations.

## Frozen benchmark protocol

- At least 500 adjudicated coherent snippets from at least 30 issues; continue
  sampling until the locked test set has at least 100 mentions per core type and
  30 per reported rare type.
- Split 60/20/20 by issue/date, never random snippets. Freeze the gold JSON,
  split manifest, ontology, prompts and SHA-256 before test inference.
- Run paired corrected-text and raw-OCR arms. Image-assisted arms are separate
  ablations and must still resolve every output to a numbered OCR block and
  exact character offsets.
- Primary metric: exact span-and-type micro F1. Also report macro/per-type and
  relaxed F1, raw recoverability, end-to-end raw recall, invalid/duplicate
  output, calibration/risk coverage, throughput, p50/p95 latency, memory and
  cost, with issue-cluster bootstrap confidence intervals.
- Require pre-adjudication reviewer agreement F1 of at least 0.85 overall and
  0.75 on core types; otherwise revise the ontology before model comparison.

## Stop/go gates

- Basic model: corrected exact F1 ≥0.85; raw-recoverable exact F1 ≥0.75; no
  stratum with `n ≥ 50` below 0.60; zero persisted invalid spans or duplicates.
- Statistical: at least +3 absolute F1 over rules + GLiNER-multi with paired 95%
  confidence interval above zero, or within one point at ≥3× throughput.
- SIKU primary: beat GLiNER-X by ≥3 points, or remain within one point at ≥3×
  throughput.
- GLiNER-X safety net: add ≥3 raw-recoverable recall points to rules + SIKU
  while reducing precision by no more than five points.
- Stage-one union: raw-recoverable recall ≥0.90 overall and ≥0.85 on core types,
  with precision ≥0.75.
- NuExtract3 routing: route ≤20% of snippets and add ≥5 F1 on routed cases or ≥3
  overall without reducing overall precision by more than two points.
- Qwen control: retain only if it beats NuExtract3 by ≥3 points on the identical
  routed cases at no more than 2× latency/cost.

## Production hypothesis

```text
raw OCR + immutable provenance
  -> rules/gazetteers + SIKU span model + GLiNER-X
  -> retain per-model evidence, calibration and disagreements
  -> NuExtract3 on low-confidence/disagreement/rare/image-context cases only
  -> exact schema/surface/offset validation
  -> historian mention review -> entity linking -> reviewed claims -> graph
```

