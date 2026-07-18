# Gold transcription and NER annotation guidelines

Status: version 1.0 draft for historian pilot
Applies to: lossless, source-resolution *Shen Bao* gold pages only

## 1. Scholarly boundary

Gold data is a human judgment record, not a model output. Each snippet requires
two independent annotations and a separately recorded adjudication. Model
suggestions may be shown only after the independent passes are saved, and must
never be copied into gold without a named reviewer decision.

The unit is an article, column, caption, advertisement, or other coherent region
with stable page and polygon provenance. Do not concatenate a whole newspaper
page for NER: reading order is uncertain and unrelated columns create false
context. Split train/dev/test by issue date, not by random region.

## 2. Transcription

- Transcribe the printed character, not a modernized reading. Preserve
  Traditional, historical, variant, rare, and erroneous printed forms.
- Apply Unicode NFC only. Do not silently convert Traditional/Simplified forms,
  names, punctuation, currencies, dates, or units.
- Preserve printed punctuation. Do not add modern punctuation to an
  unpunctuated passage in the evidence transcription.
- Use the actual logical reading order selected by the reviewer. Record layout
  as `vertical`, `horizontal`, `mixed`, or `unknown`.
- Use one `□` for each character position that is genuinely unreadable. If the
  number of missing characters is unknowable, isolate a smaller snippet or mark
  it unusable instead of inventing text.
- Put uncertain readings in the reviewer note. The adjudicator chooses the best
  evidenced reading or `□`; uncertainty is not resolved by model confidence.
- Keep the raw OCR string unchanged. The corrected transcription is a parallel
  human layer and never overwrites OCR provenance.

Every annotated span must satisfy exact substring equality. Character offsets
are zero-based, end-exclusive Unicode-character offsets; they are not UTF-8
byte offsets.

## 3. OCR and layout regions

- Annotate only the frozen lossless render identified by its SHA-256. A model
  run on a rescaled, recompressed, deskewed, or otherwise different image is a
  different benchmark input.
- Draw a tight positive-area convex polygon around each coherent text line,
  caption, headline, table cell/row, or other evaluable region. Polygons must
  stay inside the image and must not self-intersect.
- Use the most specific supported region kind. Photographs and illustrations
  may have empty transcription; their separately printed captions are distinct
  `caption` regions.
- Assign each region a unique integer reading order. For conventional vertical
  Chinese, read down a column and proceed from the rightmost column to the
  left, unless the printed composition clearly indicates another sequence.
- In mixed pages, follow the semantic sequence visible in the composition and
  record uncertainty. Do not infer article membership solely from geometric
  proximity.
- Bound tables and advertisements consistently across reviewers. When one
  polygon would mix unrelated reading sequences, split it into defensible
  subregions.
- Region UUIDs in the adjudication are model-independent gold identities. Never
  reuse a PaddleOCR or other model region UUID as the gold identity.

OCR scoring uses one-to-one polygon matches at a declared IoU threshold. Report
region detection precision/recall/F1, mean matched IoU, covered gold area,
matched-region CER, full-page reading-order CER, pairwise reading-order
accuracy, region-kind and text-direction accuracy, invalid geometry,
pages/second, peak memory, and page-level failures.

## 4. Entity boundary rules

Annotate the longest semantically complete printed mention. Include surname and
name, `氏` when it is part of the referential form, and an attached disambiguator
only when dropping it changes the identity expressed by the text. Exclude
surrounding punctuation and purely grammatical particles.

Overlapping spans are permitted only when each span has a distinct, separately
referential type. Do not create duplicate span/type annotations. OCR garbage,
page furniture, and generic classes such as an unreferential `女子` are not
entities. A named or contextually individuated `某女士` may be a `person`; record
the uncertainty in the note.

Entity linking is a later task. NER records what the text mentions; it must not
guess which real-world person it denotes. Gender, marital status, ethnicity,
reputation, class, and similar attributes are separately evidenced claims, not
NER labels.

## 5. Ontology 1.0

| Label | Include | Exclude / distinguish |
|---|---|---|
| `person` | Named or individuated people and complete referential name forms | Generic groups; link identity later |
| `alias` | A name explicitly introduced as an alias, courtesy name, art name, or former name | The relation to a person is a later claim |
| `kinship_term` | Referential kinship expressions such as a specific person's mother or sister | Generic discussion of mothers/daughters |
| `place` | Geographic areas, settlements, districts, countries | Street-level locations (`address`) and institutions |
| `address` | Street, lane, number, or other locating expression | General place names |
| `organization` | Associations, companies, agencies, charities, clubs | Schools use `school`; publications use `publication` |
| `school` | Schools, colleges, academies, training institutes | A generic phrase such as “女子教育” |
| `occupation` | Livelihood or profession used referentially | Institutional post/title (`role_title`) |
| `role_title` | Office, rank, elected/appointed position, honorific institutional role | Generic occupation |
| `publication` | Named newspapers, journals, books, columns, or works | Publisher organization when separately mentioned |
| `event` | Named or textually bounded events, meetings, ceremonies, strikes, exhibitions | Mere verbs without an event-denoting span |
| `date` | Printed dates, years, eras, and bounded temporal expressions | Publication year inferred only from metadata |
| `product` | Named goods, brands, medicines, and commodities | The advertisement unit itself |
| `advertisement` | A named advertisement/campaign when the text refers to it as an object | Page-genre classification, which belongs in layout metadata |

Annotators must flag recurring cases that do not fit these rules. Change the
ontology only through a versioned decision applied consistently to the full
gold set.

## 6. Raw-OCR alignment

Each adjudicated entity always has corrected-text offsets. Also provide
`raw_start`, `raw_end`, and `raw_text` when a human can identify one contiguous
raw-OCR span corresponding to the printed mention—even if OCR substituted some
characters. Leave all three raw fields null when OCR deleted, fragmented, or
merged the mention beyond a defensible contiguous alignment.

This distinction supports two honest raw-input measures:

- NER recall over raw-recoverable entities, which isolates extraction quality;
- end-to-end recall over all adjudicated entities, which also counts entities
  lost by OCR.

## 7. Review and adjudication

1. Reviewer A transcribes and annotates without seeing Reviewer B or model output.
2. Reviewer B does the same independently.
3. The adjudicator sees the scan and both passes, resolves every transcription,
   boundary, type, and raw-alignment disagreement, and records a note for policy
   decisions or irreducible uncertainty.
4. A validator rejects duplicate reviewers, bad offsets, mismatched surfaces,
   duplicate spans, missing region UUIDs, and non-unique snippet/region IDs.
5. Freeze a dataset version and hash before model scoring. Never revise gold in
   response to one model's errors without a documented, model-blind review.

`wic-gold-packet` may be used to prepare candidate work. Its administrative
packet contains selection strata, while its blinded reviewer view removes those
signals and all model predictions. A packet target is an OCR-region pilot unit
with adjacent regions supplied only as reading context. Packet schema 1.1
bounds that context to an active historian-approved coherent-unit revision when
one exists. Page-wide context is retained only for an explicitly ineligible
proposal packet. The builder must report the packet ineligible while any
sampled unit lacks an approved bound, issue IDs are missing, or the frozen
sample requirements are unmet. The finalizer requires a newly assigned gold
region UUID; it rejects reuse of the source model's OCR region UUID.

Report agreement before adjudication, exact and relaxed span F1 by type, raw
recoverability, OCR CER, invalid-evidence rate, throughput, peak memory, and
scores by decade, layout, scan quality, and genre. Publish both micro totals and
small-stratum sample counts.
