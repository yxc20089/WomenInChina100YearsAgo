BEGIN;

-- A reviewed text version is not automatically the version used downstream.
-- This append-only selection makes the active transcription explicit while
-- retaining every raw OCR and corrected hypothesis.
CREATE TABLE evidence.region_text_selection (
    selection_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id uuid NOT NULL REFERENCES evidence.ocr_region(region_id),
    text_version_id uuid NOT NULL REFERENCES evidence.text_version(text_version_id),
    review_id uuid NOT NULL REFERENCES evidence.review_decision(review_id),
    selection_basis text NOT NULL CHECK (selection_basis = 'historian_approved'),
    selected_by text NOT NULL,
    note text,
    selected_at timestamptz NOT NULL DEFAULT now(),
    superseded_at timestamptz,
    CHECK (superseded_at IS NULL OR superseded_at >= selected_at)
);

CREATE UNIQUE INDEX region_text_selection_active_idx
    ON evidence.region_text_selection(region_id)
    WHERE superseded_at IS NULL;

CREATE FUNCTION evidence.enforce_reviewed_region_text_selection()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM evidence.text_version version
        JOIN evidence.review_decision review
          ON review.review_id = NEW.review_id
         AND review.target_kind = 'text_version'
         AND review.target_id = version.text_version_id
         AND review.decision = 'accept'
        WHERE version.text_version_id = NEW.text_version_id
          AND version.region_id = NEW.region_id
          AND version.review_status = 'reviewed'
    ) THEN
        RAISE EXCEPTION 'active text selection requires an accepted reviewed version for the same region';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER region_text_selection_review_trigger
BEFORE INSERT OR UPDATE OF region_id, text_version_id, review_id
ON evidence.region_text_selection
FOR EACH ROW
EXECUTE FUNCTION evidence.enforce_reviewed_region_text_selection();

-- All article-scoped semantic tasks bind their immutable input to one active
-- historian-approved coherent-unit revision and the central model config.
CREATE TABLE evidence.semantic_run_input (
    run_id uuid PRIMARY KEY REFERENCES evidence.processing_run(run_id),
    coherent_unit_revision_id uuid NOT NULL
        REFERENCES evidence.coherent_unit_revision(revision_id),
    task_kind text NOT NULL CHECK (
        task_kind IN ('mention_classification', 'local_coreference', 'event_frames')
    ),
    input_sha256 text NOT NULL CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    configuration_sha256 text NOT NULL CHECK (configuration_sha256 ~ '^[0-9a-f]{64}$'),
    prompt_schema_sha256 text NOT NULL CHECK (prompt_schema_sha256 ~ '^[0-9a-f]{64}$'),
    artifact_uri text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (
        coherent_unit_revision_id, task_kind, input_sha256,
        configuration_sha256, prompt_schema_sha256
    )
);

CREATE INDEX semantic_run_input_unit_idx
    ON evidence.semantic_run_input(coherent_unit_revision_id, task_kind, created_at);

COMMIT;
