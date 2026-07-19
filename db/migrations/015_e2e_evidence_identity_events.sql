BEGIN;

-- Physical layout is versioned separately from OCR. A crop is a virtual view
-- over the immutable derivative and all geometry remains in page coordinates.
CREATE TABLE evidence.layout_run_input (
    run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    page_id uuid NOT NULL REFERENCES archive.page(page_id),
    derivative_id uuid NOT NULL,
    artifact_uri text,
    configuration_sha256 text NOT NULL CHECK (configuration_sha256 ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, page_id),
    FOREIGN KEY (derivative_id, page_id)
        REFERENCES archive.page_derivative(derivative_id, page_id)
);

CREATE TABLE evidence.layout_region (
    layout_region_id uuid PRIMARY KEY,
    run_id uuid NOT NULL,
    page_id uuid NOT NULL,
    derivative_id uuid NOT NULL,
    parent_layout_region_id uuid REFERENCES evidence.layout_region(layout_region_id),
    region_kind text NOT NULL CHECK (
        region_kind IN ('page', 'panel', 'column', 'text_group', 'image', 'table', 'rule', 'other')
    ),
    polygon jsonb NOT NULL,
    proposed_reading_order integer CHECK (proposed_reading_order >= 0),
    direction text NOT NULL DEFAULT 'unknown' CHECK (
        direction IN ('vertical', 'horizontal', 'mixed', 'unknown')
    ),
    source_method text NOT NULL,
    confidence double precision CHECK (confidence BETWEEN 0 AND 1),
    boundary_evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    engine_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (run_id, page_id)
        REFERENCES evidence.layout_run_input(run_id, page_id),
    FOREIGN KEY (derivative_id, page_id)
        REFERENCES archive.page_derivative(derivative_id, page_id),
    UNIQUE (run_id, layout_region_id)
);

CREATE INDEX layout_region_page_order_idx
    ON evidence.layout_region(page_id, proposed_reading_order);

ALTER TABLE evidence.ocr_region
    ADD COLUMN layout_region_id uuid REFERENCES evidence.layout_region(layout_region_id);

-- Every string used as evidence belongs to an immutable, hashed text version.
CREATE TABLE evidence.text_version (
    text_version_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id uuid NOT NULL REFERENCES evidence.ocr_region(region_id),
    parent_text_version_id uuid REFERENCES evidence.text_version(text_version_id),
    producing_run_id uuid REFERENCES evidence.processing_run(run_id),
    variant text NOT NULL CHECK (
        variant IN (
            'raw_ocr', 'ocr_hypothesis', 'corrected_transcription',
            'approved_reconstruction', 'normalized_search'
        )
    ),
    text_content text NOT NULL,
    text_sha256 text NOT NULL CHECK (text_sha256 ~ '^[0-9a-f]{64}$'),
    language text NOT NULL DEFAULT 'zh-Hant',
    review_status text NOT NULL DEFAULT 'candidate' CHECK (
        review_status IN ('candidate', 'reviewed', 'rejected', 'superseded')
    ),
    configuration jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (region_id, variant, text_sha256)
);

CREATE INDEX text_version_region_variant_idx
    ON evidence.text_version(region_id, variant, created_at);

CREATE TABLE evidence.text_version_alignment (
    alignment_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_text_version_id uuid NOT NULL REFERENCES evidence.text_version(text_version_id),
    target_text_version_id uuid NOT NULL REFERENCES evidence.text_version(text_version_id),
    operations jsonb NOT NULL,
    alignment_sha256 text NOT NULL CHECK (alignment_sha256 ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (source_text_version_id <> target_text_version_id),
    UNIQUE (source_text_version_id, target_text_version_id, alignment_sha256)
);

CREATE TABLE evidence.evidence_span (
    evidence_span_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    text_version_id uuid NOT NULL REFERENCES evidence.text_version(text_version_id),
    text_start integer NOT NULL CHECK (text_start >= 0),
    text_end integer NOT NULL CHECK (text_end > text_start),
    surface_text text NOT NULL,
    surface_sha256 text NOT NULL CHECK (surface_sha256 ~ '^[0-9a-f]{64}$'),
    polygon jsonb,
    span_role text NOT NULL DEFAULT 'evidence',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (text_version_id, text_start, text_end, surface_sha256)
);

CREATE INDEX evidence_span_text_offsets_idx
    ON evidence.evidence_span(text_version_id, text_start, text_end);

CREATE FUNCTION evidence.enforce_evidence_span_surface()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    authoritative_text text;
BEGIN
    SELECT text_content INTO authoritative_text
    FROM evidence.text_version
    WHERE text_version_id = NEW.text_version_id;
    IF substring(
        authoritative_text FROM NEW.text_start + 1 FOR NEW.text_end - NEW.text_start
    ) IS DISTINCT FROM NEW.surface_text THEN
        RAISE EXCEPTION 'evidence span surface does not match its immutable text version';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER evidence_span_surface_trigger
BEFORE INSERT OR UPDATE OF text_version_id, text_start, text_end, surface_text
ON evidence.evidence_span
FOR EACH ROW EXECUTE FUNCTION evidence.enforce_evidence_span_surface();

-- Backfill a raw text version for every existing OCR occurrence.
INSERT INTO evidence.text_version (
    region_id, producing_run_id, variant, text_content, text_sha256,
    language, review_status, configuration
)
SELECT region_id, run_id, 'raw_ocr', raw_text,
       encode(digest(convert_to(raw_text, 'UTF8'), 'sha256'), 'hex'),
       language, 'candidate',
       jsonb_build_object('backfilled_from_ocr_region', true)
FROM evidence.ocr_region
ON CONFLICT (region_id, variant, text_sha256) DO NOTHING;

ALTER TABLE evidence.entity_mention
    ADD COLUMN evidence_span_id uuid REFERENCES evidence.evidence_span(evidence_span_id),
    ADD COLUMN coherent_unit_revision_id uuid
        REFERENCES evidence.coherent_unit_revision(revision_id),
    ADD COLUMN mention_form text;

INSERT INTO evidence.evidence_span (
    text_version_id, text_start, text_end, surface_text, surface_sha256,
    polygon, span_role
)
SELECT version.text_version_id, mention.text_start, mention.text_end,
       mention.mention_text,
       encode(digest(convert_to(mention.mention_text, 'UTF8'), 'sha256'), 'hex'),
       mention.polygon, 'mention'
FROM evidence.entity_mention mention
JOIN evidence.text_version version
  ON version.region_id = mention.region_id AND version.variant = 'raw_ocr'
WHERE mention.text_end > mention.text_start
  AND substring(
      version.text_content FROM mention.text_start + 1
      FOR mention.text_end - mention.text_start
  ) = mention.mention_text
ON CONFLICT (text_version_id, text_start, text_end, surface_sha256) DO NOTHING;

UPDATE evidence.entity_mention mention
SET evidence_span_id = span.evidence_span_id
FROM evidence.text_version version
JOIN evidence.evidence_span span USING (text_version_id)
WHERE version.region_id = mention.region_id
  AND version.variant = 'raw_ocr'
  AND span.text_start = mention.text_start
  AND span.text_end = mention.text_end
  AND span.surface_text = mention.mention_text
  AND mention.evidence_span_id IS NULL;

CREATE INDEX entity_mention_evidence_span_idx
    ON evidence.entity_mention(evidence_span_id);
CREATE INDEX entity_mention_coherent_unit_idx
    ON evidence.entity_mention(coherent_unit_revision_id);

-- Article-local coreference is immutable and cannot silently become a global alias.
CREATE TABLE evidence.local_coreference_run (
    local_coreference_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    processing_run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    coherent_unit_revision_id uuid NOT NULL
        REFERENCES evidence.coherent_unit_revision(revision_id),
    input_sha256 text NOT NULL CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    configuration_sha256 text NOT NULL CHECK (configuration_sha256 ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (coherent_unit_revision_id, input_sha256, configuration_sha256)
);

CREATE TABLE evidence.local_coreference_cluster (
    local_cluster_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    local_coreference_run_id uuid NOT NULL
        REFERENCES evidence.local_coreference_run(local_coreference_run_id),
    review_status text NOT NULL DEFAULT 'candidate' CHECK (
        review_status IN ('candidate', 'reviewed', 'rejected')
    ),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE evidence.local_coreference_member (
    local_cluster_id uuid NOT NULL
        REFERENCES evidence.local_coreference_cluster(local_cluster_id),
    local_coreference_run_id uuid NOT NULL
        REFERENCES evidence.local_coreference_run(local_coreference_run_id),
    mention_id uuid NOT NULL REFERENCES evidence.entity_mention(mention_id),
    PRIMARY KEY (local_cluster_id, mention_id),
    UNIQUE (local_coreference_run_id, mention_id)
);

CREATE FUNCTION evidence.enforce_local_cluster_run()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM evidence.local_coreference_cluster cluster
        WHERE cluster.local_cluster_id = NEW.local_cluster_id
          AND cluster.local_coreference_run_id = NEW.local_coreference_run_id
    ) THEN
        RAISE EXCEPTION 'local coreference member run differs from its cluster';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER local_coreference_member_run_trigger
BEFORE INSERT OR UPDATE ON evidence.local_coreference_member
FOR EACH ROW EXECUTE FUNCTION evidence.enforce_local_cluster_run();

CREATE TABLE evidence.mention_resolution (
    mention_resolution_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    mention_id uuid NOT NULL REFERENCES evidence.entity_mention(mention_id),
    proposed_entity_id uuid REFERENCES evidence.entity(entity_id),
    is_nil boolean NOT NULL DEFAULT false,
    resolution_scope text NOT NULL CHECK (
        resolution_scope IN ('span', 'coherent_unit', 'article', 'issue', 'corpus')
    ),
    coherent_unit_revision_id uuid
        REFERENCES evidence.coherent_unit_revision(revision_id),
    run_id uuid REFERENCES evidence.processing_run(run_id),
    proposal text NOT NULL CHECK (
        proposal IN ('SAME', 'DIFFERENT', 'INSUFFICIENT', 'NIL')
    ),
    supporting_evidence_ids uuid[] NOT NULL DEFAULT '{}',
    contradiction_evidence_ids uuid[] NOT NULL DEFAULT '{}',
    review_status text NOT NULL DEFAULT 'candidate' CHECK (
        review_status IN ('candidate', 'reviewed', 'rejected', 'superseded')
    ),
    review_id uuid REFERENCES evidence.review_decision(review_id),
    supersedes_resolution_id uuid
        REFERENCES evidence.mention_resolution(mention_resolution_id),
    superseded_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((is_nil AND proposed_entity_id IS NULL) OR (NOT is_nil AND proposed_entity_id IS NOT NULL)),
    CHECK (
        resolution_scope NOT IN ('coherent_unit', 'article')
        OR coherent_unit_revision_id IS NOT NULL
    )
);

CREATE UNIQUE INDEX mention_resolution_one_active_reviewed_idx
    ON evidence.mention_resolution(mention_id)
    WHERE review_status = 'reviewed' AND superseded_at IS NULL;
CREATE INDEX mention_resolution_entity_idx
    ON evidence.mention_resolution(proposed_entity_id, review_status);

CREATE TABLE evidence.entity_name_assertion (
    entity_name_assertion_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id uuid NOT NULL REFERENCES evidence.entity(entity_id),
    evidence_span_id uuid REFERENCES evidence.evidence_span(evidence_span_id),
    external_authority_uri text,
    name_surface text NOT NULL,
    name_kind text NOT NULL CHECK (
        name_kind IN ('preferred', 'full', 'alternate', 'style', 'pseudonym', 'transliteration')
    ),
    language text NOT NULL DEFAULT 'zh-Hant',
    assertion_scope text NOT NULL DEFAULT 'corpus',
    review_status text NOT NULL DEFAULT 'candidate' CHECK (
        review_status IN ('candidate', 'reviewed', 'rejected', 'superseded')
    ),
    review_id uuid REFERENCES evidence.review_decision(review_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (num_nonnulls(evidence_span_id, external_authority_uri) >= 1)
);

CREATE INDEX entity_name_assertion_entity_idx
    ON evidence.entity_name_assertion(entity_id, review_status);

CREATE TABLE evidence.entity_redirect (
    entity_redirect_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    superseded_entity_id uuid NOT NULL REFERENCES evidence.entity(entity_id),
    canonical_entity_id uuid NOT NULL REFERENCES evidence.entity(entity_id),
    review_id uuid NOT NULL REFERENCES evidence.review_decision(review_id),
    reason text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    reversed_at timestamptz,
    reversal_review_id uuid REFERENCES evidence.review_decision(review_id),
    CHECK (superseded_entity_id <> canonical_entity_id),
    CHECK ((reversed_at IS NULL) = (reversal_review_id IS NULL))
);

CREATE UNIQUE INDEX entity_redirect_one_active_source_idx
    ON evidence.entity_redirect(superseded_entity_id)
    WHERE reversed_at IS NULL;

CREATE FUNCTION evidence.reject_entity_redirect_cycle()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF EXISTS (
        WITH RECURSIVE redirect_chain(entity_id) AS (
            SELECT NEW.canonical_entity_id
            UNION ALL
            SELECT redirect.canonical_entity_id
            FROM evidence.entity_redirect redirect
            JOIN redirect_chain chain
              ON redirect.superseded_entity_id = chain.entity_id
            WHERE redirect.reversed_at IS NULL
        )
        SELECT 1 FROM redirect_chain
        WHERE entity_id = NEW.superseded_entity_id
    ) THEN
        RAISE EXCEPTION 'entity redirect would create a cycle';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER entity_redirect_cycle_trigger
BEFORE INSERT OR UPDATE OF superseded_entity_id, canonical_entity_id, reversed_at
ON evidence.entity_redirect
FOR EACH ROW
WHEN (NEW.reversed_at IS NULL)
EXECUTE FUNCTION evidence.reject_entity_redirect_cycle();

-- Events are first-class evidence-backed records, not generic co-occurrence edges.
CREATE TABLE evidence.event (
    event_id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    event_type text NOT NULL,
    trigger_evidence_span_id uuid REFERENCES evidence.evidence_span(evidence_span_id),
    date_start date,
    date_end date,
    date_precision text,
    date_uncertainty text,
    location_entity_id uuid REFERENCES evidence.entity(entity_id),
    location_literal text,
    aspect text,
    event_status text NOT NULL DEFAULT 'candidate' CHECK (
        event_status IN ('candidate', 'reviewed', 'disputed', 'rejected', 'superseded')
    ),
    confidence double precision CHECK (confidence BETWEEN 0 AND 1),
    attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (date_end IS NULL OR date_start IS NULL OR date_end >= date_start),
    CHECK (num_nonnulls(location_entity_id, location_literal) <= 1)
);

CREATE INDEX event_type_status_idx ON evidence.event(event_type, event_status);

CREATE TABLE evidence.event_participant (
    event_participant_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id uuid NOT NULL REFERENCES evidence.event(event_id) ON DELETE CASCADE,
    entity_id uuid REFERENCES evidence.entity(entity_id),
    mention_id uuid REFERENCES evidence.entity_mention(mention_id),
    participant_role text NOT NULL,
    evidence_span_id uuid REFERENCES evidence.evidence_span(evidence_span_id),
    review_status text NOT NULL DEFAULT 'candidate' CHECK (
        review_status IN ('candidate', 'reviewed', 'rejected', 'superseded')
    ),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (num_nonnulls(entity_id, mention_id) = 1)
);

CREATE INDEX event_participant_entity_idx ON evidence.event_participant(entity_id);
CREATE INDEX event_participant_mention_idx ON evidence.event_participant(mention_id);

CREATE TABLE evidence.event_evidence (
    event_evidence_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id uuid NOT NULL REFERENCES evidence.event(event_id) ON DELETE CASCADE,
    evidence_span_id uuid NOT NULL REFERENCES evidence.evidence_span(evidence_span_id),
    support_role text NOT NULL CHECK (
        support_role IN ('direct_support', 'context', 'contradiction', 'external_corroboration')
    ),
    review_status text NOT NULL DEFAULT 'candidate' CHECK (
        review_status IN ('candidate', 'reviewed', 'rejected', 'superseded')
    ),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (event_id, evidence_span_id, support_role)
);

CREATE INDEX event_evidence_span_idx
    ON evidence.event_evidence(evidence_span_id, event_id);

-- Claim evidence receives an independent occurrence identity and an optional
-- exact evidence-span reference while retaining all legacy columns.
ALTER TABLE evidence.claim_evidence
    ADD COLUMN claim_evidence_id uuid DEFAULT gen_random_uuid(),
    ADD COLUMN evidence_span_id uuid REFERENCES evidence.evidence_span(evidence_span_id),
    ADD COLUMN support_role text NOT NULL DEFAULT 'direct_support' CHECK (
        support_role IN ('direct_support', 'context', 'contradiction', 'external_corroboration')
    );

ALTER TABLE evidence.claim_evidence
    ALTER COLUMN claim_evidence_id SET NOT NULL,
    DROP CONSTRAINT claim_evidence_pkey,
    ADD CONSTRAINT claim_evidence_pkey PRIMARY KEY (claim_evidence_id);

CREATE UNIQUE INDEX claim_evidence_occurrence_idx
    ON evidence.claim_evidence(
        claim_id, region_id, COALESCE(text_start, -1), COALESCE(text_end, -1),
        evidence_quote, support_role
    );
CREATE INDEX claim_evidence_span_idx
    ON evidence.claim_evidence(evidence_span_id, claim_id);

-- Widen controlled target lists for the new authoritative layers.
ALTER TABLE evidence.review_decision
    DROP CONSTRAINT review_decision_target_kind_check;
ALTER TABLE evidence.review_decision
    ADD CONSTRAINT review_decision_target_kind_check CHECK (
        target_kind IN (
            'ocr_region', 'text_version', 'evidence_span', 'mention',
            'mention_resolution', 'entity_link', 'entity_name_assertion',
            'entity_redirect', 'event', 'event_participant', 'event_evidence',
            'claim'
        )
    ) NOT VALID;
ALTER TABLE evidence.review_decision
    VALIDATE CONSTRAINT review_decision_target_kind_check;

ALTER TABLE retrieval.embedding
    DROP CONSTRAINT embedding_target_kind_check;
ALTER TABLE retrieval.embedding
    ADD CONSTRAINT embedding_target_kind_check CHECK (
        target_kind IN (
            'region', 'evidence_span', 'coherent_unit', 'article',
            'identity_profile', 'entity', 'claim'
        )
    ) NOT VALID;
ALTER TABLE retrieval.embedding
    VALIDATE CONSTRAINT embedding_target_kind_check;

COMMIT;
