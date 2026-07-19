BEGIN;

ALTER TABLE evidence.semantic_run_input
    DROP CONSTRAINT semantic_run_input_task_kind_check;

ALTER TABLE evidence.semantic_run_input
    ADD CONSTRAINT semantic_run_input_task_kind_check CHECK (
        task_kind IN (
            'mention_discovery', 'mention_classification',
            'local_coreference', 'event_frames'
        )
    ) NOT VALID;

ALTER TABLE evidence.semantic_run_input
    VALIDATE CONSTRAINT semantic_run_input_task_kind_check;

COMMIT;
