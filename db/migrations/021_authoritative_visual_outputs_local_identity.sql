BEGIN;

-- Confidence values are not probabilities until a named calibration artifact
-- says how they were measured.  Calibration records are immutable evidence.
CREATE TABLE evidence.confidence_calibration (
    calibration_id uuid PRIMARY KEY,
    task_kind text NOT NULL,
    model_name text NOT NULL,
    model_revision text NOT NULL,
    method text NOT NULL,
    dataset_id text NOT NULL,
    dataset_sha256 text NOT NULL CHECK (dataset_sha256 ~ '^[0-9a-f]{64}$'),
    artifact_uri text NOT NULL,
    artifact_sha256 text NOT NULL CHECK (artifact_sha256 ~ '^[0-9a-f]{64}$'),
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (
        task_kind, model_name, model_revision, method,
        dataset_sha256, artifact_sha256
    )
);

-- Preserve exact visual-model responses separately from parsed text/layout
-- projections.  In particular, a Hunyuan spotting or layout response remains
-- recoverable byte-for-byte even if a later parser changes.
CREATE TABLE evidence.visual_model_output (
    visual_output_id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES evidence.processing_run(run_id),
    output_kind text NOT NULL CHECK (
        output_kind IN ('spotting', 'layout', 'recognition')
    ),
    artifact_uri text NOT NULL,
    artifact_sha256 text NOT NULL CHECK (artifact_sha256 ~ '^[0-9a-f]{64}$'),
    raw_output text NOT NULL,
    raw_output_sha256 text NOT NULL CHECK (raw_output_sha256 ~ '^[0-9a-f]{64}$'),
    structured_output jsonb NOT NULL DEFAULT '{}'::jsonb,
    confidence double precision CHECK (confidence BETWEEN 0 AND 1),
    confidence_status text NOT NULL CHECK (
        confidence_status IN ('not_reported', 'uncalibrated', 'calibrated')
    ),
    calibration_id uuid REFERENCES evidence.confidence_calibration(calibration_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (
        (confidence_status = 'not_reported'
            AND confidence IS NULL AND calibration_id IS NULL)
        OR (confidence_status = 'uncalibrated'
            AND confidence IS NOT NULL AND calibration_id IS NULL)
        OR (confidence_status = 'calibrated'
            AND confidence IS NOT NULL AND calibration_id IS NOT NULL)
    ),
    UNIQUE (run_id, visual_output_id)
);

CREATE INDEX visual_model_output_run_kind_idx
    ON evidence.visual_model_output(run_id, output_kind, created_at);

-- This is the normalized, exact reverse path from an output to the archive
-- bytes that produced it.  Redundant URIs and hashes are intentional: the
-- trigger verifies them at insert time and immutability preserves the observed
-- path even if a rebuild changes a projection.
CREATE TABLE evidence.visual_model_evidence_path (
    evidence_path_id uuid PRIMARY KEY,
    visual_output_id uuid NOT NULL
        REFERENCES evidence.visual_model_output(visual_output_id),
    path_role text NOT NULL CHECK (
        path_role IN ('source_page', 'input_crop', 'output_geometry')
    ),
    source_object_id uuid NOT NULL
        REFERENCES archive.source_object(source_object_id),
    page_id uuid NOT NULL REFERENCES archive.page(page_id),
    derivative_id uuid NOT NULL
        REFERENCES archive.page_derivative(derivative_id),
    layout_region_id uuid REFERENCES evidence.layout_region(layout_region_id),
    region_id uuid REFERENCES evidence.ocr_region(region_id),
    text_version_id uuid REFERENCES evidence.text_version(text_version_id),
    evidence_span_id uuid REFERENCES evidence.evidence_span(evidence_span_id),
    source_uri text NOT NULL,
    image_uri text NOT NULL,
    image_sha256 text NOT NULL CHECK (image_sha256 ~ '^[0-9a-f]{64}$'),
    crop_uri text,
    crop_sha256 text CHECK (crop_sha256 IS NULL OR crop_sha256 ~ '^[0-9a-f]{64}$'),
    crop_bounds jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((crop_uri IS NULL) = (crop_sha256 IS NULL)),
    CHECK (path_role <> 'input_crop' OR crop_uri IS NOT NULL),
    CHECK (evidence_span_id IS NULL OR text_version_id IS NOT NULL),
    CHECK (text_version_id IS NULL OR region_id IS NOT NULL),
    UNIQUE (visual_output_id, evidence_path_id)
);

CREATE INDEX visual_model_evidence_path_page_idx
    ON evidence.visual_model_evidence_path(page_id, derivative_id);
CREATE INDEX visual_model_evidence_path_span_idx
    ON evidence.visual_model_evidence_path(evidence_span_id)
    WHERE evidence_span_id IS NOT NULL;

CREATE FUNCTION evidence.enforce_visual_output_sha256()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF encode(digest(convert_to(NEW.raw_output, 'UTF8'), 'sha256'), 'hex')
       IS DISTINCT FROM NEW.raw_output_sha256 THEN
        RAISE EXCEPTION 'visual-model raw output SHA-256 does not match exact output text';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER visual_model_output_sha256_trigger
BEFORE INSERT OR UPDATE OF raw_output, raw_output_sha256
ON evidence.visual_model_output
FOR EACH ROW EXECUTE FUNCTION evidence.enforce_visual_output_sha256();

CREATE FUNCTION evidence.enforce_visual_evidence_path()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM archive.page page
        JOIN archive.volume volume USING (volume_id)
        JOIN archive.source_object source USING (source_object_id)
        JOIN archive.page_derivative derivative USING (page_id)
        WHERE source.source_object_id = NEW.source_object_id
          AND page.page_id = NEW.page_id
          AND derivative.derivative_id = NEW.derivative_id
          AND source.source_uri = NEW.source_uri
          AND derivative.image_uri = NEW.image_uri
          AND derivative.image_sha256 = NEW.image_sha256
    ) THEN
        RAISE EXCEPTION 'visual-model evidence path does not match archive source bytes';
    END IF;
    IF NEW.layout_region_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM evidence.layout_region layout
        WHERE layout.layout_region_id = NEW.layout_region_id
          AND layout.page_id = NEW.page_id
          AND layout.derivative_id = NEW.derivative_id
    ) THEN
        RAISE EXCEPTION 'visual-model evidence path has a mismatched layout occurrence';
    END IF;
    IF NEW.region_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM evidence.ocr_region region
        WHERE region.region_id = NEW.region_id
          AND region.page_id = NEW.page_id
          AND (
              NEW.layout_region_id IS NULL
              OR region.layout_region_id = NEW.layout_region_id
          )
    ) THEN
        RAISE EXCEPTION 'visual-model evidence path has a mismatched OCR occurrence';
    END IF;
    IF NEW.text_version_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM evidence.text_version version
        WHERE version.text_version_id = NEW.text_version_id
          AND version.region_id = NEW.region_id
    ) THEN
        RAISE EXCEPTION 'visual-model evidence path has a mismatched text version';
    END IF;
    IF NEW.evidence_span_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM evidence.evidence_span span
        WHERE span.evidence_span_id = NEW.evidence_span_id
          AND span.text_version_id = NEW.text_version_id
    ) THEN
        RAISE EXCEPTION 'visual-model evidence path has a mismatched evidence span';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER visual_model_evidence_path_trigger
BEFORE INSERT OR UPDATE ON evidence.visual_model_evidence_path
FOR EACH ROW EXECUTE FUNCTION evidence.enforce_visual_evidence_path();

CREATE FUNCTION evidence.reject_authoritative_output_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'authoritative model outputs and calibrations are immutable';
END;
$$;

CREATE TRIGGER confidence_calibration_immutable_trigger
BEFORE UPDATE OR DELETE ON evidence.confidence_calibration
FOR EACH ROW EXECUTE FUNCTION evidence.reject_authoritative_output_mutation();
CREATE TRIGGER visual_model_output_immutable_trigger
BEFORE UPDATE OR DELETE ON evidence.visual_model_output
FOR EACH ROW EXECUTE FUNCTION evidence.reject_authoritative_output_mutation();
CREATE TRIGGER visual_model_evidence_path_immutable_trigger
BEFORE UPDATE OR DELETE ON evidence.visual_model_evidence_path
FOR EACH ROW EXECUTE FUNCTION evidence.reject_authoritative_output_mutation();

-- Semantic task artifacts also retain the exact response in PostgreSQL.  The
-- nullable pair preserves legacy rows from migrations 017--020; every new
-- first-build semantic run supplies both values.
ALTER TABLE evidence.semantic_run_input
    ADD COLUMN raw_output text,
    ADD COLUMN raw_output_sha256 text
        CHECK (raw_output_sha256 IS NULL OR raw_output_sha256 ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT semantic_run_raw_output_pair_check CHECK (
        (raw_output IS NULL AND raw_output_sha256 IS NULL)
        OR (raw_output IS NOT NULL AND raw_output_sha256 IS NOT NULL)
    );

CREATE TRIGGER semantic_run_output_sha256_trigger
BEFORE INSERT OR UPDATE OF raw_output, raw_output_sha256
ON evidence.semantic_run_input
FOR EACH ROW EXECUTE FUNCTION evidence.enforce_visual_output_sha256();

CREATE TRIGGER semantic_run_input_immutable_trigger
BEFORE UPDATE OR DELETE ON evidence.semantic_run_input
FOR EACH ROW EXECUTE FUNCTION evidence.reject_authoritative_output_mutation();

-- Existing score-bearing occurrence tables now state whether each score is
-- absent, uncalibrated, or tied to an immutable calibration artifact.
ALTER TABLE evidence.layout_region
    ADD COLUMN confidence_status text,
    ADD COLUMN calibration_id uuid
        REFERENCES evidence.confidence_calibration(calibration_id);
UPDATE evidence.layout_region
SET confidence_status = CASE
    WHEN confidence IS NULL THEN 'not_reported' ELSE 'uncalibrated'
END;
ALTER TABLE evidence.layout_region
    ALTER COLUMN confidence_status SET NOT NULL,
    ALTER COLUMN confidence_status SET DEFAULT 'not_reported',
    ADD CONSTRAINT layout_region_confidence_provenance_check CHECK (
        (confidence_status = 'not_reported'
            AND confidence IS NULL AND calibration_id IS NULL)
        OR (confidence_status = 'uncalibrated'
            AND confidence IS NOT NULL AND calibration_id IS NULL)
        OR (confidence_status = 'calibrated'
            AND confidence IS NOT NULL AND calibration_id IS NOT NULL)
    );

ALTER TABLE evidence.ocr_region
    ADD COLUMN confidence_status text,
    ADD COLUMN calibration_id uuid
        REFERENCES evidence.confidence_calibration(calibration_id);
UPDATE evidence.ocr_region
SET confidence_status = CASE
    WHEN confidence IS NULL THEN 'not_reported' ELSE 'uncalibrated'
END;
ALTER TABLE evidence.ocr_region
    ALTER COLUMN confidence_status SET NOT NULL,
    ALTER COLUMN confidence_status SET DEFAULT 'not_reported',
    ADD CONSTRAINT ocr_region_confidence_provenance_check CHECK (
        (confidence_status = 'not_reported'
            AND confidence IS NULL AND calibration_id IS NULL)
        OR (confidence_status = 'uncalibrated'
            AND confidence IS NOT NULL AND calibration_id IS NULL)
        OR (confidence_status = 'calibrated'
            AND confidence IS NOT NULL AND calibration_id IS NOT NULL)
    );

ALTER TABLE evidence.entity_mention
    ADD COLUMN confidence_status text,
    ADD COLUMN calibration_id uuid
        REFERENCES evidence.confidence_calibration(calibration_id);
UPDATE evidence.entity_mention
SET confidence_status = CASE
    WHEN confidence IS NULL THEN 'not_reported' ELSE 'uncalibrated'
END;
ALTER TABLE evidence.entity_mention
    ALTER COLUMN confidence_status SET NOT NULL,
    ALTER COLUMN confidence_status SET DEFAULT 'not_reported',
    ADD CONSTRAINT entity_mention_confidence_provenance_check CHECK (
        (confidence_status = 'not_reported'
            AND confidence IS NULL AND calibration_id IS NULL)
        OR (confidence_status = 'uncalibrated'
            AND confidence IS NOT NULL AND calibration_id IS NULL)
        OR (confidence_status = 'calibrated'
            AND confidence IS NOT NULL AND calibration_id IS NOT NULL)
    );

-- Article-local identity remains a clustering of mention occurrences.  It has
-- no canonical entity target and no merge side effect.  These redundant scope
-- columns let PostgreSQL enforce one coherent-unit revision end to end.
ALTER TABLE evidence.local_coreference_run
    ADD CONSTRAINT local_coreference_run_scope_unique
        UNIQUE (local_coreference_run_id, coherent_unit_revision_id);

ALTER TABLE evidence.local_coreference_cluster
    ADD COLUMN coherent_unit_revision_id uuid;
UPDATE evidence.local_coreference_cluster cluster
SET coherent_unit_revision_id = run.coherent_unit_revision_id
FROM evidence.local_coreference_run run
WHERE run.local_coreference_run_id = cluster.local_coreference_run_id;
ALTER TABLE evidence.local_coreference_cluster
    ALTER COLUMN coherent_unit_revision_id SET NOT NULL,
    ADD CONSTRAINT local_coreference_cluster_run_scope_fk
        FOREIGN KEY (local_coreference_run_id, coherent_unit_revision_id)
        REFERENCES evidence.local_coreference_run(
            local_coreference_run_id, coherent_unit_revision_id
        ),
    ADD CONSTRAINT local_coreference_cluster_scope_unique
        UNIQUE (
            local_cluster_id, local_coreference_run_id,
            coherent_unit_revision_id
        );

ALTER TABLE evidence.local_coreference_member
    ADD COLUMN coherent_unit_revision_id uuid;
UPDATE evidence.local_coreference_member member
SET coherent_unit_revision_id = run.coherent_unit_revision_id
FROM evidence.local_coreference_run run
WHERE run.local_coreference_run_id = member.local_coreference_run_id;
ALTER TABLE evidence.local_coreference_member
    ALTER COLUMN coherent_unit_revision_id SET NOT NULL,
    ADD CONSTRAINT local_coreference_member_cluster_scope_fk
        FOREIGN KEY (
            local_cluster_id, local_coreference_run_id,
            coherent_unit_revision_id
        ) REFERENCES evidence.local_coreference_cluster(
            local_cluster_id, local_coreference_run_id,
            coherent_unit_revision_id
        );

CREATE FUNCTION evidence.enforce_active_local_identity_run()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM evidence.coherent_unit_revision revision
        WHERE revision.revision_id = NEW.coherent_unit_revision_id
          AND revision.superseded_at IS NULL
    ) THEN
        RAISE EXCEPTION 'local identity requires one active coherent-unit revision';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER local_coreference_run_active_revision_trigger
BEFORE INSERT OR UPDATE OF coherent_unit_revision_id
ON evidence.local_coreference_run
FOR EACH ROW EXECUTE FUNCTION evidence.enforce_active_local_identity_run();

CREATE FUNCTION evidence.enforce_local_identity_cluster_scope()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    run_revision_id uuid;
BEGIN
    SELECT run.coherent_unit_revision_id INTO run_revision_id
    FROM evidence.local_coreference_run run
    WHERE run.local_coreference_run_id = NEW.local_coreference_run_id;
    IF run_revision_id IS NULL THEN
        RAISE EXCEPTION 'local identity cluster requires a registered local run';
    END IF;
    IF NEW.coherent_unit_revision_id IS NULL THEN
        NEW.coherent_unit_revision_id := run_revision_id;
    ELSIF NEW.coherent_unit_revision_id IS DISTINCT FROM run_revision_id THEN
        RAISE EXCEPTION 'local identity cluster differs from its run revision';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER local_coreference_cluster_scope_trigger
BEFORE INSERT OR UPDATE OF local_coreference_run_id, coherent_unit_revision_id
ON evidence.local_coreference_cluster
FOR EACH ROW EXECUTE FUNCTION evidence.enforce_local_identity_cluster_scope();

CREATE FUNCTION evidence.enforce_local_identity_member_scope()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    run_revision_id uuid;
BEGIN
    SELECT run.coherent_unit_revision_id INTO run_revision_id
    FROM evidence.local_coreference_run run
    WHERE run.local_coreference_run_id = NEW.local_coreference_run_id;
    IF run_revision_id IS NULL THEN
        RAISE EXCEPTION 'local identity member requires a registered local run';
    END IF;
    IF NEW.coherent_unit_revision_id IS NULL THEN
        NEW.coherent_unit_revision_id := run_revision_id;
    ELSIF NEW.coherent_unit_revision_id IS DISTINCT FROM run_revision_id THEN
        RAISE EXCEPTION 'local identity member differs from its run revision';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM evidence.entity_mention mention
        JOIN evidence.coherent_unit_span unit_span
          ON unit_span.revision_id = NEW.coherent_unit_revision_id
         AND unit_span.region_id = mention.region_id
         AND mention.text_start >= unit_span.text_start
         AND mention.text_end <= unit_span.text_end
        WHERE mention.mention_id = NEW.mention_id
          AND mention.coherent_unit_revision_id = NEW.coherent_unit_revision_id
          AND mention.evidence_span_id IS NOT NULL
    ) THEN
        RAISE EXCEPTION 'local identity member must be an exact mention occurrence in the same revision';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER local_coreference_member_scope_trigger
BEFORE INSERT OR UPDATE OF mention_id, coherent_unit_revision_id
ON evidence.local_coreference_member
FOR EACH ROW EXECUTE FUNCTION evidence.enforce_local_identity_member_scope();

CREATE TRIGGER local_coreference_run_immutable_trigger
BEFORE UPDATE OR DELETE ON evidence.local_coreference_run
FOR EACH ROW EXECUTE FUNCTION evidence.reject_authoritative_output_mutation();

CREATE FUNCTION evidence.enforce_local_cluster_review_transition()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'article-local identity clusters cannot be deleted';
    END IF;
    IF NEW.local_cluster_id IS DISTINCT FROM OLD.local_cluster_id
       OR NEW.local_coreference_run_id IS DISTINCT FROM OLD.local_coreference_run_id
       OR NEW.coherent_unit_revision_id IS DISTINCT FROM OLD.coherent_unit_revision_id
       OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION 'article-local identity cluster evidence is immutable';
    END IF;
    IF NEW.review_status IS DISTINCT FROM OLD.review_status
       AND OLD.review_status <> 'candidate' THEN
        RAISE EXCEPTION 'a completed local identity review cannot be rewritten';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER local_coreference_cluster_review_transition_trigger
BEFORE UPDATE OR DELETE ON evidence.local_coreference_cluster
FOR EACH ROW EXECUTE FUNCTION evidence.enforce_local_cluster_review_transition();
CREATE TRIGGER local_coreference_member_immutable_trigger
BEFORE UPDATE OR DELETE ON evidence.local_coreference_member
FOR EACH ROW EXECUTE FUNCTION evidence.reject_authoritative_output_mutation();

ALTER TABLE evidence.review_decision
    DROP CONSTRAINT review_decision_target_kind_check;
ALTER TABLE evidence.review_decision
    ADD CONSTRAINT review_decision_target_kind_check CHECK (
        target_kind IN (
            'ocr_region', 'text_version', 'evidence_span', 'mention',
            'mention_resolution', 'local_coreference_cluster',
            'entity_link', 'entity_name_assertion', 'entity_redirect',
            'identity_pair_decision', 'event', 'event_participant',
            'event_evidence', 'claim'
        )
    ) NOT VALID;
ALTER TABLE evidence.review_decision
    VALIDATE CONSTRAINT review_decision_target_kind_check;

COMMIT;
