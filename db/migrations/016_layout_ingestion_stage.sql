BEGIN;

ALTER TABLE pipeline.ingestion_job
    DROP CONSTRAINT ingestion_job_stage_check;

ALTER TABLE pipeline.ingestion_job
    ADD CONSTRAINT ingestion_job_stage_check CHECK (
        stage IN (
            'render_lossless',
            'layout',
            'ocr',
            'embedding',
            'ner',
            'entity_link',
            'search_projection',
            'rag_export',
            'graph_projection'
        )
    ) NOT VALID;

ALTER TABLE pipeline.ingestion_job
    VALIDATE CONSTRAINT ingestion_job_stage_check;

COMMIT;
