BEGIN;

CREATE TABLE evidence.ner_run_input (
    run_id uuid PRIMARY KEY REFERENCES evidence.processing_run(run_id),
    source_ocr_run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    artifact_id uuid NOT NULL UNIQUE,
    artifact_uri text NOT NULL,
    artifact_schema_version text NOT NULL CHECK (
        artifact_schema_version IN ('1.0', '1.1')
    ),
    input_variant text CHECK (
        input_variant IS NULL OR input_variant IN (
            'raw_ocr', 'corrected_text', 'multimodal_transcript'
        )
    ),
    input_sha256 text CHECK (
        input_sha256 IS NULL OR input_sha256 ~ '^[0-9a-f]{64}$'
    ),
    dataset_id text,
    split_id text,
    ontology_version text,
    adapter_id text,
    prompt_schema_revision text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (
        artifact_schema_version = '1.0'
        OR (
            input_variant IS NOT NULL
            AND input_sha256 IS NOT NULL
            AND dataset_id IS NOT NULL
            AND split_id IS NOT NULL
            AND ontology_version IS NOT NULL
            AND adapter_id IS NOT NULL
        )
    )
);

CREATE INDEX ner_run_input_source_idx
    ON evidence.ner_run_input(source_ocr_run_id, run_id);

COMMIT;
