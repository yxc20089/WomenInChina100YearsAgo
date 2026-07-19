BEGIN;

ALTER TABLE evidence.identity_pair_candidate
    ADD COLUMN embedding_run_id uuid REFERENCES evidence.processing_run(run_id),
    ADD COLUMN reranker_run_id uuid REFERENCES evidence.processing_run(run_id);

CREATE INDEX identity_pair_candidate_embedding_run_idx
    ON evidence.identity_pair_candidate(embedding_run_id)
    WHERE embedding_run_id IS NOT NULL;
CREATE INDEX identity_pair_candidate_reranker_run_idx
    ON evidence.identity_pair_candidate(reranker_run_id)
    WHERE reranker_run_id IS NOT NULL;

COMMIT;
