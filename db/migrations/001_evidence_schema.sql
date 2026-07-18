BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS archive;
CREATE SCHEMA IF NOT EXISTS evidence;
CREATE SCHEMA IF NOT EXISTS retrieval;

CREATE TABLE IF NOT EXISTS archive.source_object (
    source_object_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_uri text NOT NULL UNIQUE,
    bucket text,
    object_key text,
    media_type text NOT NULL,
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    etag text,
    sha256 text CHECK (sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$'),
    integrity_status text NOT NULL,
    integrity_details jsonb NOT NULL DEFAULT '{}'::jsonb,
    observed_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS archive.volume (
    volume_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_object_id uuid NOT NULL UNIQUE REFERENCES archive.source_object(source_object_id),
    volume_number integer NOT NULL UNIQUE CHECK (volume_number > 0),
    publication_year integer NOT NULL CHECK (publication_year BETWEEN 1800 AND 2100),
    edition_label text,
    page_count integer CHECK (page_count > 0),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS archive.page (
    page_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    volume_id uuid NOT NULL REFERENCES archive.volume(volume_id),
    page_number integer NOT NULL CHECK (page_number > 0),
    source_image_uri text,
    source_image_sha256 text CHECK (source_image_sha256 IS NULL OR source_image_sha256 ~ '^[0-9a-f]{64}$'),
    width integer CHECK (width > 0),
    height integer CHECK (height > 0),
    dpi integer CHECK (dpi > 0),
    page_status text NOT NULL DEFAULT 'available'
        CHECK (page_status IN ('available', 'corrupt', 'missing', 'quarantined')),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (volume_id, page_number)
);

CREATE TABLE IF NOT EXISTS evidence.processing_run (
    run_id uuid PRIMARY KEY,
    kind text NOT NULL,
    engine text NOT NULL,
    model_name text NOT NULL,
    model_revision text NOT NULL,
    software_version text,
    configuration jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    started_at timestamptz NOT NULL,
    completed_at timestamptz,
    error_details jsonb,
    CHECK (completed_at IS NULL OR completed_at >= started_at)
);

CREATE TABLE IF NOT EXISTS evidence.ocr_region (
    region_id uuid PRIMARY KEY,
    page_id uuid NOT NULL REFERENCES archive.page(page_id),
    parent_region_id uuid REFERENCES evidence.ocr_region(region_id),
    run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    region_kind text NOT NULL,
    reading_order integer NOT NULL CHECK (reading_order >= 0),
    polygon jsonb NOT NULL,
    raw_text text NOT NULL,
    normalized_text text,
    confidence double precision CHECK (confidence BETWEEN 0 AND 1),
    language text NOT NULL DEFAULT 'zh-Hant',
    direction text NOT NULL DEFAULT 'unknown'
        CHECK (direction IN ('vertical', 'horizontal', 'mixed', 'unknown')),
    engine_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (page_id, run_id, reading_order)
);

CREATE INDEX IF NOT EXISTS ocr_region_page_order_idx
    ON evidence.ocr_region(page_id, reading_order);
CREATE INDEX IF NOT EXISTS ocr_region_normalized_trgm_placeholder_idx
    ON evidence.ocr_region USING btree (left(normalized_text, 256));

CREATE TABLE IF NOT EXISTS evidence.article (
    article_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    title text,
    article_type text NOT NULL DEFAULT 'unknown',
    publication_date date,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evidence.article_region (
    article_id uuid NOT NULL REFERENCES evidence.article(article_id) ON DELETE CASCADE,
    region_id uuid NOT NULL REFERENCES evidence.ocr_region(region_id),
    sequence_number integer NOT NULL CHECK (sequence_number >= 0),
    PRIMARY KEY (article_id, region_id),
    UNIQUE (article_id, sequence_number)
);

CREATE TABLE IF NOT EXISTS evidence.entity (
    entity_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type text NOT NULL,
    canonical_name text NOT NULL,
    normalized_name text,
    authority_uri text,
    entity_status text NOT NULL DEFAULT 'candidate'
        CHECK (entity_status IN ('candidate', 'reviewed', 'merged', 'rejected')),
    attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS entity_authority_uri_unique_idx
    ON evidence.entity(authority_uri) WHERE authority_uri IS NOT NULL;
CREATE INDEX IF NOT EXISTS entity_normalized_name_idx
    ON evidence.entity(normalized_name, entity_type);

CREATE TABLE IF NOT EXISTS evidence.entity_mention (
    mention_id uuid PRIMARY KEY,
    region_id uuid NOT NULL REFERENCES evidence.ocr_region(region_id),
    run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    entity_id uuid REFERENCES evidence.entity(entity_id),
    entity_type text NOT NULL,
    mention_text text NOT NULL,
    normalized_text text,
    text_start integer NOT NULL CHECK (text_start >= 0),
    text_end integer NOT NULL CHECK (text_end >= text_start),
    polygon jsonb,
    confidence double precision CHECK (confidence BETWEEN 0 AND 1),
    mention_status text NOT NULL DEFAULT 'candidate'
        CHECK (mention_status IN ('candidate', 'reviewed', 'rejected')),
    attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS entity_mention_region_idx ON evidence.entity_mention(region_id);
CREATE INDEX IF NOT EXISTS entity_mention_entity_idx ON evidence.entity_mention(entity_id);

CREATE TABLE IF NOT EXISTS evidence.entity_link_candidate (
    link_candidate_id uuid PRIMARY KEY,
    mention_id uuid NOT NULL REFERENCES evidence.entity_mention(mention_id) ON DELETE CASCADE,
    run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    proposed_entity_id uuid REFERENCES evidence.entity(entity_id),
    proposed_authority_uri text,
    proposed_canonical_name text NOT NULL,
    score double precision NOT NULL CHECK (score BETWEEN 0 AND 1),
    is_nil boolean NOT NULL DEFAULT false,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evidence.claim (
    claim_id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    subject_entity_id uuid NOT NULL REFERENCES evidence.entity(entity_id),
    predicate text NOT NULL,
    object_entity_id uuid REFERENCES evidence.entity(entity_id),
    object_literal jsonb,
    event_date_start date,
    event_date_end date,
    claim_status text NOT NULL DEFAULT 'candidate'
        CHECK (claim_status IN ('candidate', 'reviewed', 'disputed', 'rejected', 'superseded')),
    confidence double precision CHECK (confidence BETWEEN 0 AND 1),
    supporting_quote text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (num_nonnulls(object_entity_id, object_literal) = 1),
    CHECK (event_date_end IS NULL OR event_date_start IS NULL OR event_date_end >= event_date_start)
);

CREATE INDEX IF NOT EXISTS claim_subject_predicate_idx
    ON evidence.claim(subject_entity_id, predicate);
CREATE INDEX IF NOT EXISTS claim_object_entity_idx
    ON evidence.claim(object_entity_id) WHERE object_entity_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS claim_status_idx ON evidence.claim(claim_status);

CREATE TABLE IF NOT EXISTS evidence.claim_evidence (
    claim_id uuid NOT NULL REFERENCES evidence.claim(claim_id) ON DELETE CASCADE,
    region_id uuid NOT NULL REFERENCES evidence.ocr_region(region_id),
    text_start integer CHECK (text_start >= 0),
    text_end integer CHECK (text_end >= text_start),
    evidence_quote text NOT NULL,
    polygon jsonb,
    PRIMARY KEY (claim_id, region_id, evidence_quote),
    CHECK ((text_start IS NULL) = (text_end IS NULL))
);

CREATE TABLE IF NOT EXISTS evidence.review_decision (
    review_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    target_kind text NOT NULL CHECK (target_kind IN ('ocr_region', 'mention', 'entity_link', 'claim')),
    target_id uuid NOT NULL,
    decision text NOT NULL CHECK (decision IN ('accept', 'reject', 'dispute', 'supersede', 'needs_review')),
    reviewer text NOT NULL,
    note text,
    previous_value jsonb,
    new_value jsonb,
    reviewed_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS review_target_idx
    ON evidence.review_decision(target_kind, target_id, reviewed_at DESC);

CREATE TABLE IF NOT EXISTS retrieval.embedding (
    embedding_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    target_kind text NOT NULL CHECK (target_kind IN ('region', 'article', 'entity', 'claim')),
    target_id uuid NOT NULL,
    run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    model_name text NOT NULL,
    model_revision text NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (target_kind, target_id, model_name, model_revision)
);

CREATE INDEX IF NOT EXISTS embedding_hnsw_cosine_idx
    ON retrieval.embedding USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS retrieval.projection_build (
    build_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    projection_kind text NOT NULL CHECK (projection_kind IN ('opensearch', 'neo4j', 'lightrag', 'graphrag')),
    source_schema_version text NOT NULL,
    configuration jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'superseded')),
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    artifact_uri text,
    error_details jsonb
);

COMMIT;

