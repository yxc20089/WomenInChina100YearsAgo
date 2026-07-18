BEGIN;

ALTER TABLE pipeline.ingestion_batch
    DROP CONSTRAINT ingestion_batch_status_check;

ALTER TABLE pipeline.ingestion_batch
    ADD CONSTRAINT ingestion_batch_status_check CHECK (
        status IN ('active', 'completed', 'failed', 'cancelled')
    );

CREATE INDEX ingestion_job_dead_letter_idx
    ON pipeline.ingestion_job(batch_id, stage, completed_at, job_id)
    WHERE status = 'failed';

COMMIT;
