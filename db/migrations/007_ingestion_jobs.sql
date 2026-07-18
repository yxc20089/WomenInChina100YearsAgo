BEGIN;

CREATE SCHEMA IF NOT EXISTS pipeline;

CREATE TABLE pipeline.ingestion_batch (
    batch_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_key text NOT NULL UNIQUE CHECK (plan_key ~ '^[0-9a-f]{64}$'),
    name text NOT NULL CHECK (length(btrim(name)) > 0),
    scope jsonb NOT NULL,
    configuration jsonb NOT NULL,
    status text NOT NULL DEFAULT 'active' CHECK (
        status IN ('active', 'completed', 'cancelled')
    ),
    created_by text NOT NULL CHECK (length(btrim(created_by)) > 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    CHECK (completed_at IS NULL OR completed_at >= created_at)
);

CREATE TABLE pipeline.ingestion_job (
    job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id uuid NOT NULL REFERENCES pipeline.ingestion_batch(batch_id)
        ON DELETE CASCADE,
    job_key text NOT NULL CHECK (job_key ~ '^[0-9a-f]{64}$'),
    stage text NOT NULL CHECK (
        stage IN (
            'render_lossless',
            'ocr',
            'embedding',
            'ner',
            'search_projection',
            'rag_export',
            'graph_projection'
        )
    ),
    scope_kind text NOT NULL CHECK (scope_kind IN ('page', 'batch')),
    source_object_id uuid REFERENCES archive.source_object(source_object_id),
    volume_id uuid REFERENCES archive.volume(volume_id),
    page_number integer CHECK (page_number > 0),
    input_fingerprint text NOT NULL CHECK (
        input_fingerprint ~ '^[0-9a-f]{64}$'
    ),
    configuration jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL DEFAULT 'pending' CHECK (
        status IN (
            'pending', 'leased', 'running', 'completed',
            'failed', 'cancelled'
        )
    ),
    priority integer NOT NULL DEFAULT 0,
    attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    available_at timestamptz NOT NULL DEFAULT now(),
    lease_owner text,
    lease_expires_at timestamptz,
    artifact_uri text,
    output_sha256 text CHECK (
        output_sha256 IS NULL OR output_sha256 ~ '^[0-9a-f]{64}$'
    ),
    result jsonb,
    error_details jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz,
    UNIQUE (batch_id, job_key),
    CHECK (
        (scope_kind = 'page' AND source_object_id IS NOT NULL
         AND volume_id IS NOT NULL AND page_number IS NOT NULL)
        OR
        (scope_kind = 'batch' AND source_object_id IS NULL
         AND volume_id IS NULL AND page_number IS NULL)
    ),
    CHECK (
        (status IN ('leased', 'running')
         AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR
        (status NOT IN ('leased', 'running'))
    ),
    CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at)
);

CREATE TABLE pipeline.ingestion_job_dependency (
    job_id uuid NOT NULL REFERENCES pipeline.ingestion_job(job_id)
        ON DELETE CASCADE,
    depends_on_job_id uuid NOT NULL REFERENCES pipeline.ingestion_job(job_id)
        ON DELETE CASCADE,
    PRIMARY KEY (job_id, depends_on_job_id),
    CHECK (job_id <> depends_on_job_id)
);

CREATE TABLE pipeline.ingestion_job_event (
    event_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES pipeline.ingestion_job(job_id)
        ON DELETE CASCADE,
    event_type text NOT NULL CHECK (
        event_type IN (
            'planned', 'leased', 'started', 'heartbeat', 'completed',
            'retry_scheduled', 'failed', 'cancelled', 'lease_expired'
        )
    ),
    worker_id text,
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    occurred_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX ingestion_job_ready_idx
    ON pipeline.ingestion_job(status, available_at, priority DESC, created_at)
    WHERE status = 'pending';

CREATE INDEX ingestion_job_lease_idx
    ON pipeline.ingestion_job(lease_expires_at)
    WHERE status IN ('leased', 'running');

CREATE INDEX ingestion_job_page_idx
    ON pipeline.ingestion_job(volume_id, page_number, stage);

CREATE INDEX ingestion_dependency_parent_idx
    ON pipeline.ingestion_job_dependency(depends_on_job_id, job_id);

CREATE INDEX ingestion_event_job_idx
    ON pipeline.ingestion_job_event(job_id, occurred_at, event_id);

COMMIT;
