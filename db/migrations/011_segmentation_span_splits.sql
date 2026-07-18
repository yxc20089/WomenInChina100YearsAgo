BEGIN;

-- A historian may split one OCR region across two coherent units.  Full source
-- coverage and non-overlap are validated transactionally by the importer and
-- activation path; the exact offsets remain stored on every membership.
ALTER TABLE evidence.article_region
    DROP CONSTRAINT article_region_run_region_unique;

CREATE INDEX article_region_run_region_idx
    ON evidence.article_region(run_id, region_id, text_start, text_end);

COMMIT;
