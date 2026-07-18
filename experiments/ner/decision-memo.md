# NER candidate decision memo

Research cutoff: 2026-07-18  
Decision status: benchmark shortlist; no production winner

## Recommendation

Run a supervised fixed-ontology tournament using the identical **W2NER** head
on **MacBERT**, **GujiRoBERTa-jian-fan** (license-gated), and **SIKU-BERT**.
MacBERT is the provisional first arm because the closest directly relevant
retyped *Shen Bao* comparison reports 58.26 F1 for MacBERT versus 56.83 for
SIKU. This is a small difference and neither result measures OCR robustness.

Use **Otter CE** and **GLiNER-X** as open-type recall challengers, and
**NuExtract3** only as a routed difficult/image-context stage. Use
**Qwen3.6-27B** as the high-compute multimodal ceiling when 80 GB-class hardware
is available; otherwise retain Qwen3.5-9B only as a cheaper control.

No candidate may write accepted entities or graph facts. Exact Unicode
surface/offset validation and historian review remain mandatory.

| Candidate | Target role | Direct evidence and gap | Approximate deployment envelope |
|---|---|---|---|
| `hfl/chinese-macbert-base@a986e004d2a7f2a1c2f5a3edef4e20604a974ed1` + W2NER | First supervised arm | A directly relevant [1872–1947 historical-newspaper study](https://aclanthology.org/2024.lrec-main.35/) reports 58.26 F1 on retyped *Shen Bao* versus 56.83 for SIKU; no OCR test | BERT-base; roughly 2–4 GB inference memory and 16–24 GB training VRAM, with snippet length bounded for W2NER's grid |
| `hsc748NLP/GujiRoBERTa_jian_fan@8e755704c4ae91eded4ebcabe17fecedb42d324f` + W2NER | Historical-encoder supervised arm, license-gated | A controlled [EvaHan study](https://aclanthology.org/2025.alp-1.24/) reports W2NER 88.48 average F1 versus GlobalPointer 87.33 and CRF 86.33; the GujiRoBERTa-W2NER test result was 86.34, but this is clean ancient text | BERT-base class; roughly 2–4 GB inference memory and 16–24 GB training VRAM |
| `SIKU-BERT/sikubert@fc656de2d6bde33919102dd3abe31c843f42226a` + W2NER | Historical-encoder supervised arm | Historical Traditional-Chinese pretraining; 56.83 F1 in the retyped *Shen Bao* comparison. The Guji project reports 90.68 on Traditional *Shiji*, while its mixed-script GujiBERT arm reports 93.76; none is an OCR result | BERT-base class; roughly 2–4 GB inference memory and 16–24 GB training VRAM |
| `whoisjones/otter-ce-mmbert@aed019f74647c225e14bc6d0792afdd458dfdb2d` | Compact open-type challenger, license-gated | The [2026 preprint](https://arxiv.org/abs/2601.06347) reports 100+ languages and gains over GLiNER-X-base, but no historical, Traditional-Chinese, or OCR result | About 309M parameters/1.24 GB F32 weights; benchmark before estimating production memory |
| `knowledgator/gliner-x-large@4a4437f439a78d67c87781b42e8c45373d2adcb0` | Existing zero-shot/open-label recall challenger | [Official card](https://huggingface.co/knowledgator/gliner-x-large) reports `zh_pud` F1 0.6794, but no historical, Traditional-Chinese, or OCR benchmark | 864.6M parameters; roughly 6–10 GB CPU RAM or 4–6 GB GPU VRAM |
| `numind/NuExtract3@2e9fca82ee641e6bb6e1f5d905241e994be27a07` | Routed difficult cases, image/context ablation, later relation extraction | [Official card](https://huggingface.co/numind/NuExtract3) supports text/images, JSON templates and verbatim strings; its extraction benchmark is not an auditable historical-Chinese result | About 9.3 GB BF16 weights; 24 GB GPU recommended |
| `Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | Multimodal structured-output ceiling only | [Official card](https://huggingface.co/Qwen/Qwen3.6-27B) covers Chinese and vision but provides no target NER evidence | About 55.6 GB BF16 weights; 80 GB GPU minimum for the short-context BF16 arm |

MacBERT, SIKU-BERT, GLiNER-X, NuExtract3 and Qwen3.6 declare Apache-2.0. The
GujiRoBERTa and Otter checkpoint cards do not declare a license even though
their associated code repositories do; they remain experiment-only until the
weights are cleared. Verify Siku digitization provenance before redistributing
a fine-tuned checkpoint. Do not use CHisIEC or the public ENP-*Shen Bao* corpus
until their repository/data rights are clarified, and do not confuse AWS read
permission with permission to publish scans, OCR, or annotations.

## Frozen benchmark protocol

- At least 500 adjudicated coherent snippets from at least 30 issues; continue
  sampling until the locked test set has at least 100 mentions per core type and
  30 per reported rare type.
- Split 60/20/20 by issue/date, never random snippets. Freeze the gold JSON,
  split manifest, ontology, prompts and SHA-256 before test inference.
- Train MacBERT, GujiRoBERTa and SIKU with the identical W2NER head and frozen
  hyperparameter-search budget. Compare W2NER, GlobalPointer and flat CRF on
  one frozen encoder; do not attribute head gains to a different backbone.
- Run paired corrected-text, raw-OCR and observed-confusion noise-augmentation
  arms. Image-assisted arms are separate
  ablations and must still resolve every output to a numbered OCR block and
  exact character offsets.
- Primary metric: exact span-and-type micro F1. Also report macro/per-type and
  relaxed F1, raw recoverability, end-to-end raw recall, invalid/duplicate
  output, calibration/risk coverage, throughput, p50/p95 latency, memory and
  cost, with issue-cluster bootstrap confidence intervals.
- Record pre-adjudication span/type agreement and revise unclear guidelines
  before model comparison. Do not impose an unsupported 0.85 initial threshold:
  the closest retyped-newspaper baselines reach only 43.89–58.26 F1 and report
  newspaper annotation agreement κ=0.72.

## Stop/go gates

- Evidence integrity: zero persisted invalid offsets, missing exact surfaces,
  duplicates or unpinned artifacts for every arm.
- Supervised selection: choose a winner when the issue-cluster bootstrap lower
  95% bound of its exact-F1 delta is above zero, or select a model within one
  point of the best if it provides at least 3× throughput or at most one-third
  the cost. Report the achieved absolute quality; do not hide it behind a
  relative win.
- Open-type safety net: retain Otter or GLiNER-X only if it adds at least three
  raw-recoverable recall points to rules + the supervised winner while reducing
  precision by no more than five points.
- NuExtract/Qwen routing: route at most 20% of snippets and retain a stage only
  if routed exact F1 rises by at least five points or overall F1 by at least two,
  with ambiguous/absent exact surfaces below 0.5%.
- Human workflow: no accuracy score permits automatic acceptance. Measure
  historian correction time and set a corpus-scale deployment threshold only
  after the blinded review pilot establishes the current manual baseline.

## Production hypothesis

```text
raw OCR + immutable provenance
  -> rules/gazetteers + winner of MacBERT/GujiRoBERTa/SIKU W2NER tournament
  -> Otter and/or GLiNER-X only if the recall-union gate passes
  -> retain per-model evidence, calibration and disagreements
  -> NuExtract3 on low-confidence/disagreement/rare/image-context cases only
  -> exact schema/surface/offset validation
  -> historian mention review -> entity linking -> reviewed claims -> graph
```
