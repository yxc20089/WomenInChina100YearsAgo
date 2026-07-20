BEGIN;

ALTER TABLE retrieval.embedding
    ADD COLUMN input_sha256 text CHECK (
        input_sha256 IS NULL OR input_sha256 ~ '^[0-9a-f]{64}$'
    ),
    ADD COLUMN content_sha256 text CHECK (
        content_sha256 IS NULL OR content_sha256 ~ '^[0-9a-f]{64}$'
    ),
    ADD COLUMN configuration_sha256 text CHECK (
        configuration_sha256 IS NULL OR configuration_sha256 ~ '^[0-9a-f]{64}$'
    ),
    DROP CONSTRAINT embedding_target_kind_check,
    DROP CONSTRAINT embedding_target_kind_target_id_model_name_model_revision_key,
    ADD CONSTRAINT embedding_hash_bundle_check CHECK (
        num_nonnulls(input_sha256, content_sha256, configuration_sha256) IN (0, 3)
    ),
    ADD CONSTRAINT embedding_coherent_unit_revision_hashes_check CHECK (
        target_kind <> 'coherent_unit_revision'
        OR num_nonnulls(input_sha256, content_sha256, configuration_sha256) = 3
    );

ALTER TABLE retrieval.embedding
    ADD CONSTRAINT embedding_target_kind_check CHECK (
        target_kind IN (
            'region', 'evidence_span', 'coherent_unit',
            'coherent_unit_revision', 'article', 'identity_profile',
            'entity', 'claim'
        )
    ) NOT VALID;

ALTER TABLE retrieval.embedding
    VALIDATE CONSTRAINT embedding_target_kind_check;

CREATE UNIQUE INDEX embedding_legacy_identity_idx
    ON retrieval.embedding(target_kind, target_id, model_name, model_revision)
    WHERE input_sha256 IS NULL
      AND content_sha256 IS NULL
      AND configuration_sha256 IS NULL;

CREATE UNIQUE INDEX embedding_versioned_identity_idx
    ON retrieval.embedding(
        target_kind, target_id, model_name, model_revision, input_sha256,
        content_sha256, configuration_sha256
    )
    WHERE input_sha256 IS NOT NULL
      AND content_sha256 IS NOT NULL
      AND configuration_sha256 IS NOT NULL;

ALTER TABLE pipeline.ingestion_job
    ADD COLUMN coherent_unit_revision_id uuid
        REFERENCES evidence.coherent_unit_revision(revision_id),
    DROP CONSTRAINT ingestion_job_stage_check,
    DROP CONSTRAINT ingestion_job_scope_kind_check,
    DROP CONSTRAINT ingestion_job_check;

ALTER TABLE pipeline.ingestion_job
    ADD CONSTRAINT ingestion_job_stage_check CHECK (
        stage IN (
            'render_lossless',
            'layout',
            'ocr',
            'embedding',
            'coherent_unit_embedding',
            'ner',
            'entity_link',
            'search_projection',
            'coherent_unit_search_projection',
            'rag_export',
            'graph_projection'
        )
    ) NOT VALID,
    ADD CONSTRAINT ingestion_job_scope_kind_check CHECK (
        scope_kind IN ('page', 'batch', 'coherent_unit_revision')
    ) NOT VALID,
    ADD CONSTRAINT ingestion_job_scope_check CHECK (
        (scope_kind = 'page'
         AND source_object_id IS NOT NULL
         AND volume_id IS NOT NULL
         AND page_number IS NOT NULL
         AND coherent_unit_revision_id IS NULL)
        OR
        (scope_kind = 'batch'
         AND source_object_id IS NULL
         AND volume_id IS NULL
         AND page_number IS NULL
         AND coherent_unit_revision_id IS NULL)
        OR
        (scope_kind = 'coherent_unit_revision'
         AND source_object_id IS NULL
         AND volume_id IS NULL
         AND page_number IS NULL
         AND coherent_unit_revision_id IS NOT NULL)
    ) NOT VALID,
    ADD CONSTRAINT ingestion_job_stage_scope_check CHECK (
        (stage <> 'coherent_unit_embedding'
         OR scope_kind = 'coherent_unit_revision')
        AND (scope_kind <> 'coherent_unit_revision'
             OR stage = 'coherent_unit_embedding')
        AND (stage <> 'coherent_unit_search_projection'
             OR scope_kind = 'batch')
    ) NOT VALID;

ALTER TABLE pipeline.ingestion_job
    VALIDATE CONSTRAINT ingestion_job_stage_check,
    VALIDATE CONSTRAINT ingestion_job_scope_kind_check,
    VALIDATE CONSTRAINT ingestion_job_scope_check,
    VALIDATE CONSTRAINT ingestion_job_stage_scope_check;

ALTER TABLE retrieval.projection_build
    ADD COLUMN source_snapshot_sha256 text CHECK (
        source_snapshot_sha256 IS NULL
        OR source_snapshot_sha256 ~ '^[0-9a-f]{64}$'
    ),
    ADD COLUMN document_count integer CHECK (
        document_count IS NULL OR document_count >= 0
    ),
    ADD COLUMN published_at timestamptz,
    DROP CONSTRAINT projection_build_projection_kind_check;

ALTER TABLE retrieval.projection_build
    ADD CONSTRAINT projection_build_projection_kind_check CHECK (
        projection_kind IN (
            'opensearch', 'opensearch_coherent_unit', 'neo4j',
            'lightrag', 'graphrag'
        )
    ) NOT VALID,
    ADD CONSTRAINT projection_build_coherent_snapshot_check CHECK (
        projection_kind <> 'opensearch_coherent_unit'
        OR (source_snapshot_sha256 IS NOT NULL
            AND document_count IS NOT NULL
            AND published_at IS NOT NULL)
    ) NOT VALID;

ALTER TABLE retrieval.projection_build
    VALIDATE CONSTRAINT projection_build_projection_kind_check,
    VALIDATE CONSTRAINT projection_build_coherent_snapshot_check;

COMMIT;
