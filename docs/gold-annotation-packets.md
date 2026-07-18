# Gold annotation packet operations

`wic-gold-packet` bridges active OCR in PostgreSQL to independent NER review. It
does not create gold automatically.

## Boundary

The builder reads only `page_ocr_selection` rows whose `superseded_at` is null.
Every annotation target retains the source-object SHA-256, registered derivative
UUID and image SHA-256, OCR run/region UUID, polygon, raw text, confidence and
adjacent reading-order context. Empty OCR regions are excluded.

Sampling deliberately covers overlapping strata:

- women-centered theme matches;
- exact-candidate disagreement between learned NER runs;
- low, medium, high or unknown OCR confidence;
- regions with NER candidates and a no-candidate baseline.

Within a stratum, ordering is SHA-256 of dataset ID plus source OCR region UUID.
An identical database selection and configuration therefore produces the same
packet ID even when `generated_at` changes. Counts overlap because one target
may satisfy several strata.

The administrative packet exposes sampling reasons for audit. The reviewer
view removes those reasons and contains no model predictions. Reviewer A and B
must receive separate copies of the blank template and must not see each
other's work. The adjacent regions are context only; annotations and offsets
target the designated raw-OCR string.

## Build and verify

```bash
uv run wic-gold-packet build --database-url "$DATABASE_URL" \
  --dataset-id shenbao-ner-pilot-v1 --ontology-version women-history-zh-v1 \
  --volume 219 --page 308 --max-units 50 --context-radius 2 \
  --output artifacts/gold-packet-pilot/packet.json \
  --reviewer-view artifacts/gold-packet-pilot/reviewer-view.json \
  --template artifacts/gold-packet-pilot/annotations-template.json
```

The command fails if a registered image is absent, outside local `artifacts/`,
or differs from its database SHA-256. It also refuses a page filter without a
volume and refuses scopes without active text-bearing OCR.

The verified 2026-07-18 pilot produced packet
`5d241438dc434f7f8ebf9dfbf06918786ce61e7184a8303de9a65ebf70035260`:

| Measure | Value |
|---|---:|
| Units | 50 |
| Pages / volumes / decades | 1 / 1 / 1 |
| Women-theme units | 8 |
| NER-disagreement units | 10 |
| Low / medium / high OCR-confidence units | 10 / 13 / 27 |
| NER-candidate / no-candidate units | 11 / 39 |

The packet correctly has `benchmark_eligible=false`. It lacks the frozen
minimum 500 units, issue/article identifiers for a 30-issue split, and coverage
beyond the 1920s. It is suitable for testing the annotation instructions and
measuring reviewer effort—not for choosing a model.

After the bounded 1924 and 1926 ingestion pages were added, an unfiltered
three-page packet produced stable ID
`fae8be96fdd8e7ad95aa47ca451cd5f13692c5c676d5c671483023c0d8eacdba`.
It contains 150 units across three volumes, including 9 women-theme, 27
NER-disagreement, 25 low-confidence and 122 no-candidate-baseline selections
(overlapping counts). It still correctly fails benchmark eligibility: all three
pages are from the 1920s, issue/article IDs are absent, and it has fewer than
500 units.

## Complete and finalize

For every unit, each reviewer supplies a name, corrected text, exact spans,
timestamp and optional notes. The adjudicator additionally supplies page genre,
layout, scan quality and a newly assigned `gold_region_id`.

```bash
uv run wic-gold-packet finalize \
  --packet artifacts/gold-packet-pilot/packet.json \
  --annotations artifacts/gold-packet-pilot/completed-annotations.json \
  --output artifacts/gold/ner-v1.json
```

Finalization is all-or-nothing. It rejects a changed packet identity, missing or
extra units, duplicate reviewers, incorrect corrected/raw offsets, surface
mismatches, duplicate entities, and any gold UUID copied from a source OCR
region. The output is NER gold schema 1.1 and remains subject to the project
data-rights policy before redistribution.

## What remains human work

The current `artifacts/benchmark-review/annotations.json` contains zero visual
screening decisions. Historians must still select 150–250 source-resolution
pages, define coherent article/column units and complete two blinded passes plus
adjudication. The tool makes those decisions reproducible; it does not make
them.
