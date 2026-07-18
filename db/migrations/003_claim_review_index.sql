BEGIN;

CREATE INDEX IF NOT EXISTS claim_review_queue_idx
    ON evidence.claim(claim_status, created_at, confidence DESC);

COMMIT;
