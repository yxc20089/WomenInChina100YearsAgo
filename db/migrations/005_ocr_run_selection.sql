BEGIN;

ALTER TABLE archive.page_derivative
    ADD CONSTRAINT page_derivative_id_page_unique
    UNIQUE (derivative_id, page_id);

CREATE TABLE evidence.ocr_run_input (
    run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    page_id uuid NOT NULL REFERENCES archive.page(page_id),
    derivative_id uuid NOT NULL,
    artifact_id uuid NOT NULL UNIQUE,
    artifact_uri text NOT NULL,
    evidence_tier text NOT NULL CHECK (
        evidence_tier IN (
            'screening_derivative',
            'unreviewed_input',
            'non_gold_lossless_pilot',
            'historian_selected_gold'
        )
    ),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, page_id),
    UNIQUE (run_id, page_id, derivative_id),
    FOREIGN KEY (derivative_id, page_id)
        REFERENCES archive.page_derivative(derivative_id, page_id)
);

CREATE INDEX ocr_run_input_derivative_idx
    ON evidence.ocr_run_input(derivative_id, run_id);

CREATE TABLE evidence.page_ocr_selection (
    selection_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id uuid NOT NULL REFERENCES archive.page(page_id),
    run_id uuid NOT NULL,
    derivative_id uuid NOT NULL,
    selection_basis text NOT NULL CHECK (
        selection_basis IN (
            'technical_default',
            'benchmark_winner',
            'historian_approved'
        )
    ),
    selected_by text NOT NULL CHECK (length(btrim(selected_by)) > 0),
    note text,
    selected_at timestamptz NOT NULL DEFAULT now(),
    superseded_at timestamptz,
    FOREIGN KEY (run_id, page_id, derivative_id)
        REFERENCES evidence.ocr_run_input(run_id, page_id, derivative_id),
    CHECK (superseded_at IS NULL OR superseded_at >= selected_at)
);

CREATE UNIQUE INDEX page_ocr_one_active_selection_idx
    ON evidence.page_ocr_selection(page_id)
    WHERE superseded_at IS NULL;

CREATE INDEX page_ocr_selection_history_idx
    ON evidence.page_ocr_selection(page_id, selected_at DESC, selection_id);

COMMIT;
