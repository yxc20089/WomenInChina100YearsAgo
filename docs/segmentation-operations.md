# Issue and coherent-unit segmentation

Segmentation has two deliberately separate planes:

1. `article_segmentation`, `article`, and `article_region` contain immutable
   machine or historian-authored **proposals** over one exact OCR selection.
   They are never retrieval evidence.
2. An accepted named review plus an explicit activation copies the proposal
   into distinct, revisioned `coherent_unit_revision` and
   `coherent_unit_span` rows. Only these copied revisions are eligible for the
   reviewed-unit RAG export and bounded annotation context.

Issue identity is also a human assertion. `archive.issue` and the append-
preserving `page_issue_assignment` history support issue-level benchmark
splits; model suggestions must not create active assignments automatically.

## Generate candidates

Apply migrations, then build deterministic reading-order windows over active
OCR selections:

```bash
uv run wic-migrate --database-url "$DATABASE_URL"
uv run wic-segment --database-url "$DATABASE_URL" propose \
  --max-regions 24 --max-characters 600 \
  --proposed-by deterministic-baseline-v1
```

The baseline is an annotation-sized window proposal, not an article-boundary
model. Its identity hashes the exact region IDs, OCR text, polygons,
configuration, page, OCR run, and OCR-selection UUID. Repeating the command
reuses the same proposal.

On the current three-page evidence slice it produced 106 candidates covering
all 2,498 OCR regions: 41 units for volume 202/page 338, 46 for volume 219/page
308, and 19 for volume 230/page 367. These counts are workflow diagnostics,
not historical article counts.

Export one proposal for scan-based correction, edit only the unit metadata and
span membership, then import it as a new unapproved proposal:

```bash
uv run wic-segment --database-url "$DATABASE_URL" export \
  --run-id SEGMENTATION_RUN_UUID --output artifacts/segmentation-edit.json

uv run wic-segment --database-url "$DATABASE_URL" import \
  --input artifacts/segmentation-edit.json --proposed-by HISTORIAN_ID
```

The importer requires every OCR character to appear exactly once. It rejects
foreign/missing regions, gaps, overlaps, invalid offsets, stale OCR selections,
duplicate unit ordinals, and changed input hashes. A region may be split across
two units by adjacent end-exclusive spans. Import never records a review.

After checking the issue header or other bibliographic evidence, create a
stable issue and assign its pages in issue order:

```bash
uv run wic-segment --database-url "$DATABASE_URL" create-issue \
  --publication-date 1925-01-01 --issue-number VERIFIED_NUMBER \
  --created-by HISTORIAN_ID

uv run wic-segment --database-url "$DATABASE_URL" assign-page \
  --issue-id ISSUE_UUID --volume 219 --page 308 --sequence 0 \
  --assigned-by HISTORIAN_ID
```

The example date/number are placeholders: never run it without verification.
Reassignment supersedes the old page assignment rather than overwriting it.

## Review and activate

Open the main researcher application at `http://127.0.0.1:8766`, choose
**Review machine candidates**, then open **Coherent-unit proposals**. The detail
view verifies the exact registered scan hash and displays every unit, raw text
span, offset and polygon. Its JSON editor imports changes as a new unapproved
proposal; it never changes the proposal being viewed.

The UI and CLI record whole-proposal review. Use either only after a historian
has checked every boundary, order, unit type, title, and exact span against the
scan:

```bash
uv run wic-segment --database-url "$DATABASE_URL" review \
  --run-id SEGMENTATION_RUN_UUID --decision accept \
  --reviewer HISTORIAN_ID --note 'Checked against source scan'

uv run wic-segment --database-url "$DATABASE_URL" activate \
  --review-id ACCEPTED_REVIEW_UUID --selected-by HISTORIAN_ID \
  --expected-previous-selection-id CURRENT_SELECTION_UUID
```

Omit `--expected-previous-selection-id` only when the page has no active
segmentation. The UI always sends the value it loaded, including explicit
`null`; activation fails if another reviewer changed it in the meantime.

`reject` and `needs_revision` reviews cannot be activated; PostgreSQL enforces
that rule independently of the CLI. Activation also fails if the exact source
OCR selection has been superseded or if any source region is missing or
duplicated, if the proposal/input hash changed, or if a later review requested
revision/rejection. Network retries reuse the original review and activation
UUIDs. Successful activation creates new approved coherent-unit revisions and exact
end-exclusive OCR-region spans; it never mutates proposal rows.

The current live proposals have **not** been reviewed or activated. One
unchanged export/import round trip was stored with proposer
`workflow-roundtrip-not-reviewed` solely to exercise validation; it is not a
historian decision. The
reviewed-unit export therefore hard-fails, as intended:

```bash
uv run wic-rag-export --database-url "$DATABASE_URL" \
  --unit reviewed_coherent_unit --output artifacts/rag-reviewed
```

## Remaining review-tool work

Before corpus-scale segmentation, add a scan-based editor around the JSON
contract for merge, split, reorder, type/title correction, cross-page
continuation, and issue assignment.
Edits must create a new `historian_authored` proposal or coherent-unit revision;
the database rejects updates/deletes of proposal and review rows. A reviewed
unit may span pages, and its member offsets allow a reviewer to split one OCR
region without changing OCR evidence.

The researcher API has no authentication or CSRF protection. Reviewer names
are audit labels, not verified identities. Keep it bound to loopback until an
authenticated deployment is designed.
