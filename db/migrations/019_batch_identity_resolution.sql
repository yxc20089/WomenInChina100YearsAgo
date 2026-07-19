BEGIN;

CREATE TABLE evidence.identity_resolution_cohort (
    cohort_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    scope jsonb NOT NULL,
    snapshot_sha256 text NOT NULL CHECK (snapshot_sha256 ~ '^[0-9a-f]{64}$'),
    configuration_sha256 text NOT NULL CHECK (configuration_sha256 ~ '^[0-9a-f]{64}$'),
    created_by text NOT NULL,
    status text NOT NULL DEFAULT 'frozen' CHECK (
        status IN ('frozen', 'deciding', 'reviewing', 'completed', 'superseded')
    ),
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    UNIQUE (snapshot_sha256, configuration_sha256),
    CHECK (completed_at IS NULL OR completed_at >= created_at)
);

CREATE TABLE evidence.identity_profile (
    identity_profile_id uuid PRIMARY KEY,
    cohort_id uuid NOT NULL
        REFERENCES evidence.identity_resolution_cohort(cohort_id) ON DELETE CASCADE,
    profile_kind text NOT NULL CHECK (profile_kind IN ('mention', 'local_cluster', 'entity')),
    entity_type text NOT NULL,
    entity_id uuid REFERENCES evidence.entity(entity_id),
    mention_ids uuid[] NOT NULL DEFAULT '{}',
    evidence_span_ids uuid[] NOT NULL DEFAULT '{}',
    name_surfaces text[] NOT NULL,
    profile_sha256 text NOT NULL CHECK (profile_sha256 ~ '^[0-9a-f]{64}$'),
    attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (cohort_id, profile_sha256),
    CHECK ((profile_kind = 'entity') = (entity_id IS NOT NULL)),
    CHECK (cardinality(name_surfaces) > 0),
    CHECK (profile_kind = 'entity' OR cardinality(mention_ids) > 0)
);

CREATE INDEX identity_profile_entity_idx
    ON evidence.identity_profile(entity_id) WHERE entity_id IS NOT NULL;

CREATE TABLE evidence.identity_pair_candidate (
    identity_pair_candidate_id uuid PRIMARY KEY,
    cohort_id uuid NOT NULL
        REFERENCES evidence.identity_resolution_cohort(cohort_id) ON DELETE CASCADE,
    left_profile_id uuid NOT NULL REFERENCES evidence.identity_profile(identity_profile_id),
    right_profile_id uuid NOT NULL REFERENCES evidence.identity_profile(identity_profile_id),
    blocking_methods text[] NOT NULL,
    embedding_score double precision,
    reranker_score double precision,
    candidate_rank integer NOT NULL CHECK (candidate_rank > 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (left_profile_id <> right_profile_id),
    CHECK (embedding_score IS NULL OR embedding_score BETWEEN -1 AND 1),
    CHECK (reranker_score IS NULL OR reranker_score BETWEEN 0 AND 1),
    UNIQUE (cohort_id, left_profile_id, right_profile_id)
);

CREATE TABLE evidence.identity_pair_decision (
    identity_pair_decision_id uuid PRIMARY KEY,
    identity_pair_candidate_id uuid NOT NULL
        REFERENCES evidence.identity_pair_candidate(identity_pair_candidate_id),
    run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    decision text NOT NULL CHECK (
        decision IN ('SAME', 'DIFFERENT', 'INSUFFICIENT')
    ),
    supporting_evidence_ids uuid[] NOT NULL DEFAULT '{}',
    contradiction_evidence_ids uuid[] NOT NULL DEFAULT '{}',
    prompt_sha256 text NOT NULL CHECK (prompt_sha256 ~ '^[0-9a-f]{64}$'),
    raw_output_sha256 text NOT NULL CHECK (raw_output_sha256 ~ '^[0-9a-f]{64}$'),
    review_status text NOT NULL DEFAULT 'candidate' CHECK (
        review_status IN ('candidate', 'reviewed', 'rejected', 'superseded')
    ),
    review_id uuid REFERENCES evidence.review_decision(review_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (identity_pair_candidate_id, run_id)
);

ALTER TABLE evidence.review_decision
    DROP CONSTRAINT review_decision_target_kind_check;
ALTER TABLE evidence.review_decision
    ADD CONSTRAINT review_decision_target_kind_check CHECK (
        target_kind IN (
            'ocr_region', 'text_version', 'evidence_span', 'mention',
            'mention_resolution', 'entity_link', 'entity_name_assertion',
            'entity_redirect', 'identity_pair_decision',
            'event', 'event_participant', 'event_evidence', 'claim'
        )
    ) NOT VALID;
ALTER TABLE evidence.review_decision
    VALIDATE CONSTRAINT review_decision_target_kind_check;

COMMIT;
