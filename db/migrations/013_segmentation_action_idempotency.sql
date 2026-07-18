BEGIN;

-- A single accepted review may create at most one activation event.  Retrying
-- an API request returns that event rather than superseding/recreating units.
ALTER TABLE evidence.page_article_segmentation_selection
    ADD CONSTRAINT page_article_segmentation_review_unique UNIQUE (review_id);

COMMIT;
