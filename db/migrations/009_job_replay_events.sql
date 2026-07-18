BEGIN;

ALTER TABLE pipeline.ingestion_job_event
    DROP CONSTRAINT ingestion_job_event_event_type_check;

ALTER TABLE pipeline.ingestion_job_event
    ADD CONSTRAINT ingestion_job_event_event_type_check CHECK (
        event_type IN (
            'planned', 'leased', 'started', 'heartbeat', 'completed',
            'retry_scheduled', 'failed', 'cancelled', 'lease_expired',
            'reopened'
        )
    );

COMMIT;
