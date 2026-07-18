# NER candidate decision memo

Research cutoff: 2026-07-18  
Decision status: benchmark shortlist; no production winner

## Recommendation

Run a supervised fixed-ontology tournament using the identical **W2NER** head
on **MacBERT**, **GujiRoBERTa-jian-fan** (license-gated), and **SIKU-BERT**.
MacBERT is the provisional first arm because the closest directly relevant
retyped *Shen Bao* comparison reports 58.26 F1 for MacBERT versus 56.83 for
SIKU. This is a small difference and neither result measures OCR robustness.

Add **MacBERT-DAPT-W2NER** as the principal domain challenger: continue the
same MacBERT checkpoint on unlabeled project newspaper text from training
issues only, then use the same W2NER head and tuning budget. This is an
experiment design, not a released model or a claimed improvement. Pin the
[official W2NER implementation](https://github.com/ljynlp/W2NER/tree/a34ff841891919001080edefb50e14fa9dc15e1c)
at `a34ff841891919001080edefb50e14fa9dc15e1c`; its MIT-licensed code supports
flat, nested and discontinuous NER, although the current project evidence
contract represents contiguous spans and cannot yet score a discontinuous
slice.

Add **mmBERT-base-W2NER** as one controlled supervised challenger, not as the
new primary. The [official mmBERT paper](https://arxiv.org/abs/2509.06888)
reports broad multilingual coverage and stronger aggregate
classification/retrieval results, but explicitly says mmBERT ties XLM-R on NER
and identifies a prefix-tokenization weakness. It has no historical Chinese or
OCR evidence. Require a Traditional-Chinese character-offset round-trip test
and give it the identical W2NER head, split, and training budget.

That offset-only gate now passes on the pinned tokenizer: the committed
six-case artifact has SHA-256
`20a09b9b750faed9dc744233a33cf3378961e3d7fb224d3ad9267e1b984b38b8`, no
unknown tokens, and two explicitly recorded standalone prefix markers handled
under `standalone_sentencepiece_prefix_duplicate_v1`. This does not change the
candidate ranking or supply NER-quality evidence; it only permits the mmBERT
arm to enter the future same-head tournament.

Use **GLiNER-X** as the executable open-type recall challenger. Retain **Otter
CE** as research-only until its checkpoint-weight license is explicit. Use
**NuExtract3** only as a routed difficult/image-context stage. Use
**Qwen3.6-27B** as the high-compute multimodal ceiling when 80 GB-class hardware
is available; otherwise retain Qwen3.5-9B only as a cheaper control.

Track **PP-UIE-0.5B** only as post-independent-annotation assistance. Official
[PaddleNLP documentation](https://paddlenlp.readthedocs.io/zh/latest/llm/application/information_extraction/README.html)
reports Chinese/English open-schema IE and 0.773 zero-shot F1 on modern clean
CCIR2021 news NER, not historical OCR. Its demonstrated output is strings, not
authoritative offsets, and an immutable public weight revision was not exposed
through the model API. Do not execute it until exact bytes and rights are
frozen; accept only uniquely recoverable verbatim spans. GLiNER large v2.5 is
now pinned at `3d6d176…`, but remains a low-priority control because its card
provides no Chinese score or target evidence beyond GLiNER-X.

No candidate may write accepted entities or graph facts. Exact Unicode
surface/offset validation and historian review remain mandatory.

| Candidate | Target role | Direct evidence and gap | Approximate deployment envelope |
|---|---|---|---|
| `hfl/chinese-macbert-base@a986e004d2a7f2a1c2f5a3edef4e20604a974ed1` + W2NER | First supervised arm | A directly relevant [1872–1947 historical-newspaper study](https://aclanthology.org/2024.lrec-main.35/) reports 58.26 F1 on retyped *Shen Bao* versus 56.83 for SIKU; no OCR test | BERT-base; roughly 2–4 GB inference memory and 16–24 GB training VRAM, with snippet length bounded for W2NER's grid |
| MacBERT-DAPT + W2NER | Proposed target-domain challenger | No released project checkpoint or score. Continue the pinned MacBERT only on training-issue newspaper text to test whether domain pretraining helps; freeze the DAPT corpus/checkpoint hashes | Additional pretraining plus the same W2NER training envelope; GPU estimate must be measured in an isolated pilot |
| `jhu-clsp/mmBERT-base@c5955035435e2bf121cde7f3c8863ef52ff35d82` + W2NER | New multilingual supervised challenger | MIT weights and Chinese coverage; the paper reports no historical/OCR result and says it ties XLM-R on NER, so it cannot replace target-supported MacBERT a priori | About 1.23 GB weights; estimated 4–8 GB inference and 16–24 GB fine-tuning VRAM, with the same bounded W2NER grid |
| `hsc748NLP/GujiRoBERTa_jian_fan@8e755704c4ae91eded4ebcabe17fecedb42d324f` + W2NER | Historical-encoder supervised arm, license-gated | A controlled [EvaHan study](https://aclanthology.org/2025.alp-1.24/) reports W2NER 88.48 average F1 versus GlobalPointer 87.33 and CRF 86.33; the GujiRoBERTa-W2NER test result was 86.34, but this is clean ancient text | BERT-base class; roughly 2–4 GB inference memory and 16–24 GB training VRAM |
| `SIKU-BERT/sikubert@fc656de2d6bde33919102dd3abe31c843f42226a` + W2NER | Historical-encoder supervised arm | Historical Traditional-Chinese pretraining; 56.83 F1 in the retyped *Shen Bao* comparison. The Guji project reports 90.68 on Traditional *Shiji*, while its mixed-script GujiBERT arm reports 93.76; none is an OCR result | BERT-base class; roughly 2–4 GB inference memory and 16–24 GB training VRAM |
| `whoisjones/otter-ce-mmbert@aed019f74647c225e14bc6d0792afdd458dfdb2d` | Research-only, excluded pending weight-license clarity | The [2026 preprint](https://arxiv.org/abs/2601.06347) reports broad multilingual gains over GLiNER-X-base, but no Chinese-specific, historical, Traditional-Chinese, OCR or nested result; the checkpoint declares no license | About 309M parameters/1.24 GB F32 weights; do not execute on project data until rights are cleared |
| `knowledgator/gliner-x-large@4a4437f439a78d67c87781b42e8c45373d2adcb0` | Existing zero-shot/open-label recall challenger | [Official card](https://huggingface.co/knowledgator/gliner-x-large) reports `zh_pud` F1 0.6794, but no historical, Traditional-Chinese, or OCR benchmark | 864.6M parameters; roughly 6–10 GB CPU RAM or 4–6 GB GPU VRAM |
| `numind/NuExtract3@2e9fca82ee641e6bb6e1f5d905241e994be27a07` | Routed difficult cases, image/context ablation, later relation extraction | [Official card](https://huggingface.co/numind/NuExtract3) supports text/images, JSON templates and verbatim strings; its extraction benchmark is not an auditable historical-Chinese result | About 9.3 GB BF16 weights; 24 GB GPU recommended |
| `Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | Multimodal structured-output ceiling only | [Official card](https://huggingface.co/Qwen/Qwen3.6-27B) covers Chinese and vision but provides no target NER evidence | About 55.6 GB BF16 weights; 80 GB GPU minimum for the short-context BF16 arm |

MacBERT, SIKU-BERT, GLiNER-X, NuExtract3 and Qwen3.6 declare Apache-2.0;
mmBERT declares MIT. The
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
- Train MacBERT, MacBERT-DAPT, mmBERT, GujiRoBERTa and SIKU with the identical W2NER head and frozen
  hyperparameter-search budget. Compare W2NER, GlobalPointer and flat CRF on
  one frozen encoder; do not attribute head gains to a different backbone.
- Before the mmBERT arm, prove tokenizer-to-Unicode-character round trips on
  uninterrupted Traditional Chinese, variants, punctuation and observed OCR
  confusions; any invalid offset is a hard failure. A standalone initial
  SentencePiece `▁` may be excluded only under the committed duplicate-`(0, 1)`
  virtual-prefix policy, must remain visible in the audit artifact, and must be
  handled identically by the training/inference adapter.
- Build the MacBERT-DAPT corpus only from training issues after splitting;
  freeze its document list, bytes and SHA-256 so development/test language
  cannot leak into continued pretraining.
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
- Supervised replacement: require the issue-cluster bootstrap lower 95% bound
  of the raw exact-F1 delta to exceed zero, at least two absolute F1 points over
  the current supervised baseline, zero invalid offsets, and no core entity
  type declining by more than three points. A separately documented model
  within one point of the best may be selected for at least 3× throughput or at
  most one-third the cost after non-inferiority is established.
- Open-type safety net: retain GLiNER-X only if it adds at least three
  raw-recoverable recall points to rules + the supervised winner while reducing
  precision by no more than five points. Reconsider Otter only after checkpoint
  rights are explicit, then subject it to the same gate.
- NuExtract/Qwen routing: route at most 20% of snippets and retain a stage only
  if routed exact F1 rises by at least five points or overall F1 by at least two,
  with ambiguous/absent exact surfaces below 0.5%.
- Human workflow: no accuracy score permits automatic acceptance. Measure
  historian correction time and set a corpus-scale deployment threshold only
  after the blinded review pilot establishes the current manual baseline.

## Production hypothesis

```text
raw OCR + immutable provenance
  -> rules/gazetteers + winner of MacBERT/MacBERT-DAPT/mmBERT/GujiRoBERTa/SIKU W2NER tournament
  -> GLiNER-X only if the recall-union gate passes; Otter stays rights-blocked
  -> retain per-model evidence, calibration and disagreements
  -> NuExtract3 on low-confidence/disagreement/rare/image-context cases only
  -> exact schema/surface/offset validation
  -> historian mention review -> entity linking -> reviewed claims -> graph
```
