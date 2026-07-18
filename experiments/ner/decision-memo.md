# NER candidate decision memo

Research cutoff: 2026-07-18  
Decision status: benchmark shortlist; no production winner

## Recommendation

Do not preselect a supervised head. First run a frozen-**MacBERT** head
comparison among **W2NER**, **GlobalPointer**, and a flat **BIO/CRF** control
using identical data, search budget and issue splits. Then use the selected
overlap-capable head in the main encoder tournament with MacBERT-DAPT, mmBERT
and tokenizer-qualified Chinese ModernBERT. Retain **SIKU-BERT** as a
target-period-evaluated control despite its ancient-book pretraining, and
**GujiRoBERTa-jian-fan** only as a license-gated ancient-domain mismatch
control. MacBERT is the first backbone because the closest directly relevant
retyped *Shen Bao* comparison
reports 58.26 F1 for MacBERT versus 56.83 for SIKU. This is a small difference
and neither result measures OCR robustness.

The target language is late-Qing/Republican-era printed Traditional Chinese,
not ancient/Classical Chinese. The design assumes largely modern/transitional
newspaper syntax and context. Period character forms, names, titles,
typography/layout, punctuation and OCR errors are the main adaptation concerns;
ancient-text pretraining is not treated as inherently closer. Never normalize
the evidence string to Simplified Chinese; normalized variants may exist only as
separate retrieval/linking features.

Add **MacBERT-DAPT-W2NER** as the principal domain challenger: continue the
same MacBERT checkpoint on unlabeled project newspaper text from training
issues only, then use the same W2NER head and tuning budget. This is an
experiment design, not a released model or a claimed improvement. W2NER is the
first implemented data path, not the selected head. Pin its
[official implementation](https://github.com/ljynlp/W2NER/tree/a34ff841891919001080edefb50e14fa9dc15e1c)
at `a34ff841891919001080edefb50e14fa9dc15e1c`; its MIT-licensed code supports
flat, nested and discontinuous NER, although the current project evidence
contract represents contiguous spans and cannot yet score a discontinuous
slice.

The only Chinese head-comparison evidence found is insufficient to declare a
winner. The [EvaHan study](https://aclanthology.org/2025.alp-1.24/) reports, with
one SikuBERT encoder, average model-selection F1 of 88.48 for W2NER, 87.33 for
GlobalPointer, 86.33 for BiLSTM-CRF and 83.40 for MRC. The W2NER margin over
GlobalPointer is 1.15 points. More importantly, the authors explicitly warn
that randomly split augmented samples can leak information across those folds.
Their later public-test W2NER result uses GujiRoBERTa while the CRF baseline
uses SikuRoBERTa, so it does not isolate the head. This motivates the project
comparison; it does not establish W2NER as the project winner.

Add **mmBERT-base-W2NER** as one controlled supervised challenger, not as the
new primary. The [official mmBERT paper](https://arxiv.org/abs/2509.06888)
reports broad multilingual coverage and stronger aggregate
classification/retrieval results, but explicitly says mmBERT ties XLM-R on NER
and identifies a prefix-tokenization weakness. It has no target-period Traditional-Chinese or
OCR evidence. Require a Traditional-Chinese character-offset round-trip test
and give it the identical W2NER head, split, and training budget.

That offset-only gate now passes on the pinned tokenizer: the committed
six-case artifact has SHA-256
`20a09b9b750faed9dc744233a33cf3378961e3d7fb224d3ad9267e1b984b38b8`, no
unknown tokens, and two explicitly recorded standalone prefix markers handled
under `standalone_sentencepiece_prefix_duplicate_v1`. This does not change the
candidate ranking or supply NER-quality evidence; it only permits the mmBERT
arm to enter the future same-head tournament.

Add **Chinese ModernBERT-large-WWM + W2NER** as one further tokenizer-gated
encoder arm. The pinned
[Apache-2.0 checkpoint](https://huggingface.co/feynmanzhao/chinese-modernbert-large-wwm/tree/b00f1ff1901161f68339890fe48e4dbb6ee76f4d)
has about 377M parameters, modern Chinese-only pretraining and an 8,192-token
context. Its [paper](https://arxiv.org/abs/2510.12285) reports no NER,
Traditional-Chinese, historical or OCR evaluation, and its CLUE comparison is
mixed rather than a uniform win over RoBERTa-WWM-large. The custom tokenizer's
default preprocessing changes fullwidth text, replaces URLs/emails, removes
HTML/control characters, normalizes whitespace and strips boundaries. Do not
execute it on evidence text until pinned custom code is isolated and audited,
`do_text_preprocessing=False` is enforced, and the expanded exact-offset gate
passes. This is a benchmark challenger, not evidence of superiority.

Prioritize **OCR-noise-aware training** above adding further encoders. A
[2026 historical-document study](https://arxiv.org/abs/2601.00488) reports
77.9 entity micro-F1 for synthetic-noise training on German VET documents,
ahead of its clean/noisy arms. That result is from another language, domain and
ontology. Our factorial therefore holds encoder, selected head, split and search
budget fixed; derives confusion statistics only from training issues; starts
with length-preserving substitutions; and requires explicit edit maps for
insertions/deletions. Its public reference code has no declared license, so
implementation must be clean-room.

The clean-room exporter is now implemented as `wic-ner-training-export`. It
pins the official W2NER revision, refuses ineligible data unless the output is
explicitly marked technical, derives same-length character confusions from
training issues only, never augments development/test records, and writes
native W2NER views plus a manifest containing exact source, split, record,
offset and payload hashes. This clears the data-conversion implementation
blocker; eligible gold, trained heads and scored runs still do not exist.

Use the Apache-2.0 [**GLiNER** framework pinned at `2f10b62…`](https://github.com/urchade/GLiNER/commit/2f10b62f801880560e0d35734c9c7ee44d0c37b3)
and the pinned **GLiNER-X** checkpoint as the executable open-type recall
challenger. Its native span output and multi-label/nested mode fit candidate
generation well, but its Chinese splitter must be frozen and its contiguous
spans do not solve discontinuous project evidence. Current GLiNER2 checkpoints
do not officially list Chinese, and GLiNER v2.5's whitespace-style splitting
can group uninterrupted Chinese into one unit, so neither displaces GLiNER-X.
The official [GLiNER-X collection](https://huggingface.co/collections/knowledgator/gliner-x)
contains small `d51a098…`, base `a6c7a8f…`, and large `4a4437f…` checkpoints.
Its card reports `zh_pud` F1 of 0.5792, 0.6152, and 0.6794 respectively, versus
0.6410 for GLiNER multi-v2.1. Run large first as the accuracy candidate; run base
only as a separately scored efficiency control if large qualifies or hardware
requires it; keep small diagnostic-only. None of those scores identifies
Traditional versus Simplified text or measures newspaper OCR.
Retain **Otter CE** as
research-only until its checkpoint-weight license is explicit. Use
**NuExtract3** only as a routed difficult/image-context stage.

Add [**Qwen3.5-0.8B**](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)
as a real low-cost structured-NER challenger. Chinese-capable pretraining makes
Traditional-Chinese success plausible enough to test, while the absence of a
target NER/OCR result prevents assuming it. Use the official
[Ollama `qwen3.5:0.8b` Q8 package](https://ollama.com/library/qwen3.5%3A0.8b),
verified at registry manifest
`sha256:f3817196d142eaf72ce79dfebe53dcb20bd21da87ce13e138a8f8e10a866b3a4`,
as the canonical local arm. The pinned
[Unsloth Q8 GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/tree/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0)
in LM Studio is an optional, separately scored system—not an equivalent runtime.
Both use one backend-neutral OpenAI-compatible adapter. Start text-only with
non-thinking, schema-constrained output and separately score any page-image
ablation. Reject unknown labels, non-verbatim surfaces, invalid/ambiguous
offsets and duplicates. Keep **Qwen3.6-27B** only as the high-compute
multimodal ceiling.

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
| `feynmanzhao/chinese-modernbert-large-wwm@b00f1ff1901161f68339890fe48e4dbb6ee76f4d` + W2NER | Chinese-specific tokenizer-gated challenger | Apache-2.0 and 1.2T modern-Chinese-token pretraining, but no NER/Traditional/historical/OCR result; default custom preprocessing is incompatible with source offsets | 754 MB BF16 weights; estimated 24 GB fine-tuning with aggressive checkpointing and safer at 48 GB because the 377M encoder is combined with W2NER's quadratic grid |
| `hsc748NLP/GujiRoBERTa_jian_fan@8e755704c4ae91eded4ebcabe17fecedb42d324f` + W2NER | Secondary ancient-domain mismatch control, license-gated | The project is not ancient Chinese. The [EvaHan study](https://aclanthology.org/2025.alp-1.24/) reports W2NER 88.48 average model-selection F1 versus GlobalPointer 87.33 and CRF 86.33 on ancient text, but warns of augmented-sample leakage across those folds; its 86.34 public-test result does not isolate the head from the encoder | BERT-base class; defer training until rights are clear and target-relevant arms are complete |
| `SIKU-BERT/sikubert@fc656de2d6bde33919102dd3abe31c843f42226a` + W2NER | Target-period-evaluated control despite ancient-book pretraining | Its pretraining domain is not the project's newspaper language, but the retyped *Shen Bao* comparison directly reports 56.83 F1 versus MacBERT 58.26; neither is an OCR result | BERT-base class; roughly 2–4 GB inference memory and 16–24 GB training VRAM |
| `whoisjones/otter-ce-mmbert@aed019f74647c225e14bc6d0792afdd458dfdb2d` | Research-only, excluded pending weight-license clarity | The [2026 preprint](https://arxiv.org/abs/2601.06347) reports broad multilingual gains over GLiNER-X-base, but no Chinese-specific, historical, Traditional-Chinese, OCR or nested result; the checkpoint declares no license | About 309M parameters/1.24 GB F32 weights; do not execute on project data until rights are cleared |
| `knowledgator/gliner-x-large@4a4437f439a78d67c87781b42e8c45373d2adcb0` | Existing zero-shot/open-label recall challenger | [Official card](https://huggingface.co/knowledgator/gliner-x-large) reports `zh_pud` F1 0.6794, but no historical, Traditional-Chinese, or OCR benchmark | 864.6M parameters; roughly 6–10 GB CPU RAM or 4–6 GB GPU VRAM |
| `numind/NuExtract3@2e9fca82ee641e6bb6e1f5d905241e994be27a07` | Routed difficult cases, image/context ablation, later relation extraction | [Official card](https://huggingface.co/numind/NuExtract3) supports text/images, JSON templates and verbatim strings; its extraction benchmark is not an auditable historical-Chinese result | About 9.3 GB BF16 weights; 24 GB GPU recommended |
| `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17` | Low-cost local generative NER challenger | [Official card](https://huggingface.co/Qwen/Qwen3.5-0.8B) describes Chinese/multilingual and vision capability, but no NER, historical, Traditional-Chinese, OCR-noise or exact-offset result; measure it on the same frozen gold | About 1.75 GB published safetensors before runtime/quantization overhead; record the actual local artifact hash and measured memory |
| `Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | Multimodal structured-output ceiling only | [Official card](https://huggingface.co/Qwen/Qwen3.6-27B) covers Chinese and vision but provides no target NER evidence | About 55.6 GB BF16 weights; 80 GB GPU minimum for the short-context BF16 arm |

MacBERT, SIKU-BERT, GLiNER-X, NuExtract3, Qwen3.5 and Qwen3.6 declare Apache-2.0;
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
- First compare W2NER, GlobalPointer and flat CRF on frozen MacBERT; no head is
  selected in advance. Then train MacBERT, MacBERT-DAPT, mmBERT,
  tokenizer-qualified Chinese ModernBERT, GujiRoBERTa and SIKU with the selected
  overlap-capable head and frozen hyperparameter-search budget. Do not attribute
  head gains to a different backbone.
- Before the mmBERT arm, prove tokenizer-to-Unicode-character round trips on
  uninterrupted Traditional Chinese, variants, punctuation and observed OCR
  confusions; any invalid offset is a hard failure. A standalone initial
  SentencePiece `▁` may be excluded only under the committed duplicate-`(0, 1)`
  virtual-prefix policy, must remain visible in the audit artifact, and must be
  handled identically by the training/inference adapter.
- Build the MacBERT-DAPT corpus only from training issues after splitting;
  freeze its document list, bytes and SHA-256 so development/test language
  cannot leak into continued pretraining.
- For each finalist encoder, run clean versus training-only empirical
  OCR-substitution augmentation with the same head, split, search budget and at
  least three seeds. Derive confusions from training issues only; require an
  explicit edit map before allowing any length-changing augmentation.
- Run paired corrected-text, raw-OCR and observed-confusion noise-augmentation
  arms. Image-assisted arms are separate
  ablations and must still resolve every output to a numbered OCR block and
  exact character offsets.
- Run Qwen3.5-0.8B first as a text-only, deterministic structured-output arm
  through the same OpenAI-compatible client. Ollama Q8 is the canonical runtime;
  a pinned LM Studio GGUF is a distinct optional arm because the same seed is
  not deterministic across runtime, quantization or hardware. Freeze the
  artifact SHA-256, server version, served configuration, prompt/schema hash,
  decoding controls and repeated schema-canary response hashes.
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
- NuExtract/Qwen routing: benchmark Qwen3.5 on the complete bounded test arm,
  but route at most 20% in a proposed production cascade; retain a generative stage only
  if routed exact F1 rises by at least five points or overall F1 by at least two,
  with ambiguous/absent exact surfaces below 0.5%.
- Human workflow: no accuracy score permits automatic acceptance. Measure
  historian correction time and set a corpus-scale deployment threshold only
  after the blinded review pilot establishes the current manual baseline.

## Production hypothesis

```text
raw OCR + immutable provenance
  -> frozen-backbone W2NER/GlobalPointer/CRF head comparison
  -> rules/gazetteers + winner of the supervised encoder tournament using the selected head
  -> GLiNER-X only if the recall-union gate passes; Otter stays rights-blocked
  -> retain per-model evidence, calibration and disagreements
  -> Qwen3.5-0.8B or NuExtract3 on low-confidence/disagreement/rare/image-context cases only
  -> exact schema/surface/offset validation
  -> historian mention review -> entity linking -> reviewed claims -> graph
```
