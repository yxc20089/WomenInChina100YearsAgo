BEGIN;

-- The legacy tables were intentionally unused placeholders.  Stop rather than
-- silently assigning new semantics if another deployment already populated them.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM evidence.article LIMIT 1)
       OR EXISTS (SELECT 1 FROM evidence.article_region LIMIT 1) THEN
        RAISE EXCEPTION 'migration 010 requires empty legacy article tables';
    END IF;
END
$$;

CREATE TABLE archive.issue (
    issue_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    publication_date date,
    issue_number text,
    edition_label text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_by text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (publication_date IS NOT NULL OR issue_number IS NOT NULL)
);

CREATE TABLE archive.page_issue_assignment (
    assignment_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id uuid NOT NULL REFERENCES archive.page(page_id),
    issue_id uuid NOT NULL REFERENCES archive.issue(issue_id),
    sequence_number integer NOT NULL CHECK (sequence_number >= 0),
    assigned_by text NOT NULL,
    note text,
    assigned_at timestamptz NOT NULL DEFAULT now(),
    superseded_at timestamptz,
    CHECK (superseded_at IS NULL OR superseded_at >= assigned_at)
);

CREATE UNIQUE INDEX page_issue_assignment_active_page_idx
    ON archive.page_issue_assignment(page_id) WHERE superseded_at IS NULL;
CREATE UNIQUE INDEX page_issue_assignment_active_issue_sequence_idx
    ON archive.page_issue_assignment(issue_id, sequence_number) WHERE superseded_at IS NULL;

-- Machine or historian-authored proposals are immutable processing output over
-- one exact OCR selection.  They never become approved evidence in place.
ALTER TABLE evidence.page_ocr_selection
    ADD CONSTRAINT page_ocr_selection_id_page_run_unique
    UNIQUE (selection_id, page_id, run_id);

CREATE TABLE evidence.article_segmentation (
    run_id uuid PRIMARY KEY REFERENCES evidence.processing_run(run_id),
    page_id uuid NOT NULL REFERENCES archive.page(page_id),
    source_ocr_run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    source_ocr_selection_id uuid NOT NULL,
    input_sha256 text NOT NULL CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    proposal_sha256 text NOT NULL UNIQUE CHECK (proposal_sha256 ~ '^[0-9a-f]{64}$'),
    method text NOT NULL,
    method_version text NOT NULL,
    proposal_kind text NOT NULL CHECK (proposal_kind IN ('machine', 'historian_authored')),
    configuration jsonb NOT NULL DEFAULT '{}'::jsonb,
    proposed_by text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (source_ocr_selection_id, page_id, source_ocr_run_id)
        REFERENCES evidence.page_ocr_selection(selection_id, page_id, run_id),
    UNIQUE (run_id, page_id),
    UNIQUE (run_id, page_id, source_ocr_run_id)
);

ALTER TABLE evidence.ocr_region
    ADD CONSTRAINT ocr_region_id_page_run_unique
    UNIQUE (region_id, page_id, run_id);

ALTER TABLE evidence.article
    ADD COLUMN page_id uuid NOT NULL REFERENCES archive.page(page_id),
    ADD COLUMN ordinal integer NOT NULL CHECK (ordinal >= 0),
    ADD COLUMN confidence double precision CHECK (confidence BETWEEN 0 AND 1),
    ADD CONSTRAINT article_id_page_run_unique UNIQUE (article_id, page_id, run_id),
    ADD CONSTRAINT article_run_ordinal_unique UNIQUE (run_id, ordinal),
    ADD CONSTRAINT article_segmentation_fk
        FOREIGN KEY (run_id, page_id)
        REFERENCES evidence.article_segmentation(run_id, page_id);

ALTER TABLE evidence.article_region
    ADD COLUMN page_id uuid NOT NULL,
    ADD COLUMN run_id uuid NOT NULL,
    ADD COLUMN source_ocr_run_id uuid NOT NULL,
    ADD COLUMN text_start integer,
    ADD COLUMN text_end integer,
    ADD COLUMN role text NOT NULL DEFAULT 'body',
    ADD CONSTRAINT article_region_offsets_check CHECK (
        (text_start IS NULL) = (text_end IS NULL)
        AND (text_start IS NULL OR (text_start >= 0 AND text_end >= text_start))
    ),
    ADD CONSTRAINT article_region_article_provenance_fk
        FOREIGN KEY (article_id, page_id, run_id)
        REFERENCES evidence.article(article_id, page_id, run_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT article_region_segmentation_provenance_fk
        FOREIGN KEY (run_id, page_id, source_ocr_run_id)
        REFERENCES evidence.article_segmentation(run_id, page_id, source_ocr_run_id),
    ADD CONSTRAINT article_region_ocr_provenance_fk
        FOREIGN KEY (region_id, page_id, source_ocr_run_id)
        REFERENCES evidence.ocr_region(region_id, page_id, run_id),
    ADD CONSTRAINT article_region_run_region_unique UNIQUE (run_id, region_id);

CREATE INDEX article_segmentation_page_idx
    ON evidence.article_segmentation(page_id, created_at DESC);
CREATE INDEX article_region_source_idx
    ON evidence.article_region(source_ocr_run_id, region_id);

CREATE TABLE evidence.article_segmentation_review (
    review_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES evidence.article_segmentation(run_id),
    decision text NOT NULL CHECK (decision IN ('accept', 'reject', 'needs_revision')),
    reviewer text NOT NULL,
    note text,
    reviewed_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (review_id, run_id)
);

CREATE INDEX article_segmentation_review_run_idx
    ON evidence.article_segmentation_review(run_id, reviewed_at DESC);

-- An accepted review creates this selection event, then the application copies
-- proposal membership into distinct approved coherent-unit revisions below.
CREATE TABLE evidence.page_article_segmentation_selection (
    selection_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id uuid NOT NULL REFERENCES archive.page(page_id),
    run_id uuid NOT NULL,
    review_id uuid NOT NULL,
    selection_basis text NOT NULL CHECK (selection_basis = 'historian_approved'),
    selected_by text NOT NULL,
    selected_at timestamptz NOT NULL DEFAULT now(),
    superseded_at timestamptz,
    FOREIGN KEY (run_id, page_id)
        REFERENCES evidence.article_segmentation(run_id, page_id),
    FOREIGN KEY (review_id, run_id)
        REFERENCES evidence.article_segmentation_review(review_id, run_id),
    CHECK (superseded_at IS NULL OR superseded_at >= selected_at)
);

CREATE UNIQUE INDEX page_article_segmentation_active_page_idx
    ON evidence.page_article_segmentation_selection(page_id)
    WHERE superseded_at IS NULL;
CREATE UNIQUE INDEX page_article_segmentation_active_run_idx
    ON evidence.page_article_segmentation_selection(run_id)
    WHERE superseded_at IS NULL;

CREATE TABLE evidence.coherent_unit (
    unit_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE evidence.coherent_unit_revision (
    revision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    unit_id uuid NOT NULL REFERENCES evidence.coherent_unit(unit_id),
    revision_number integer NOT NULL CHECK (revision_number > 0),
    issue_id uuid REFERENCES archive.issue(issue_id),
    unit_kind text NOT NULL CHECK (
        unit_kind IN ('article', 'column', 'caption', 'advertisement', 'classified', 'table', 'other')
    ),
    title text,
    source_proposal_article_id uuid REFERENCES evidence.article(article_id),
    approval_selection_id uuid NOT NULL
        REFERENCES evidence.page_article_segmentation_selection(selection_id),
    content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    approved_by text NOT NULL,
    approved_at timestamptz NOT NULL DEFAULT now(),
    note text,
    superseded_at timestamptz,
    UNIQUE (unit_id, revision_number),
    CHECK (superseded_at IS NULL OR superseded_at >= approved_at)
);

CREATE UNIQUE INDEX coherent_unit_revision_active_idx
    ON evidence.coherent_unit_revision(unit_id) WHERE superseded_at IS NULL;
CREATE INDEX coherent_unit_revision_issue_idx
    ON evidence.coherent_unit_revision(issue_id) WHERE superseded_at IS NULL;

CREATE TABLE evidence.coherent_unit_span (
    revision_id uuid NOT NULL
        REFERENCES evidence.coherent_unit_revision(revision_id) ON DELETE CASCADE,
    region_id uuid NOT NULL REFERENCES evidence.ocr_region(region_id),
    sequence_number integer NOT NULL CHECK (sequence_number >= 0),
    text_start integer NOT NULL CHECK (text_start >= 0),
    text_end integer NOT NULL CHECK (text_end >= text_start),
    polygon jsonb,
    role text NOT NULL DEFAULT 'body',
    PRIMARY KEY (revision_id, region_id, text_start, text_end),
    UNIQUE (revision_id, sequence_number)
);

CREATE INDEX coherent_unit_span_region_idx ON evidence.coherent_unit_span(region_id);

CREATE FUNCTION evidence.enforce_accepted_article_segmentation_selection()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    review_decision text;
    reviewed_run_id uuid;
BEGIN
    SELECT decision, run_id INTO review_decision, reviewed_run_id
    FROM evidence.article_segmentation_review
    WHERE review_id = NEW.review_id;
    IF review_decision IS DISTINCT FROM 'accept' OR reviewed_run_id IS DISTINCT FROM NEW.run_id THEN
        RAISE EXCEPTION 'article segmentation selection requires an accepted review for the same run';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER page_article_segmentation_acceptance_trigger
BEFORE INSERT OR UPDATE OF run_id, review_id
ON evidence.page_article_segmentation_selection
FOR EACH ROW
EXECUTE FUNCTION evidence.enforce_accepted_article_segmentation_selection();

CREATE FUNCTION evidence.reject_proposal_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'segmentation proposals and reviews are immutable; create a new revision';
END;
$$;

CREATE TRIGGER article_segmentation_immutable_trigger
BEFORE UPDATE OR DELETE ON evidence.article_segmentation
FOR EACH ROW EXECUTE FUNCTION evidence.reject_proposal_mutation();
CREATE TRIGGER article_proposal_immutable_trigger
BEFORE UPDATE OR DELETE ON evidence.article
FOR EACH ROW EXECUTE FUNCTION evidence.reject_proposal_mutation();
CREATE TRIGGER article_region_proposal_immutable_trigger
BEFORE UPDATE OR DELETE ON evidence.article_region
FOR EACH ROW EXECUTE FUNCTION evidence.reject_proposal_mutation();
CREATE TRIGGER article_segmentation_review_immutable_trigger
BEFORE UPDATE OR DELETE ON evidence.article_segmentation_review
FOR EACH ROW EXECUTE FUNCTION evidence.reject_proposal_mutation();

COMMIT;
