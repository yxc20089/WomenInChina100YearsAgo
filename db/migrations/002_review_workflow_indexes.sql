BEGIN;

CREATE INDEX IF NOT EXISTS entity_mention_review_queue_idx
    ON evidence.entity_mention(mention_status, created_at, confidence DESC);

CREATE INDEX IF NOT EXISTS entity_link_candidate_mention_score_idx
    ON evidence.entity_link_candidate(mention_id, is_nil, score DESC);

COMMIT;
