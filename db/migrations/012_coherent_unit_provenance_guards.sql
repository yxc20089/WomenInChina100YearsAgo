BEGIN;

CREATE FUNCTION evidence.enforce_coherent_revision_proposal_selection()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    proposal_run_id uuid;
    selected_run_id uuid;
BEGIN
    IF NEW.source_proposal_article_id IS NULL THEN
        RETURN NEW;
    END IF;
    SELECT run_id INTO proposal_run_id
    FROM evidence.article
    WHERE article_id = NEW.source_proposal_article_id;
    SELECT run_id INTO selected_run_id
    FROM evidence.page_article_segmentation_selection
    WHERE selection_id = NEW.approval_selection_id;
    IF proposal_run_id IS DISTINCT FROM selected_run_id THEN
        RAISE EXCEPTION 'coherent-unit revision proposal must belong to its approved selection';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER coherent_revision_proposal_selection_trigger
BEFORE INSERT OR UPDATE OF source_proposal_article_id, approval_selection_id
ON evidence.coherent_unit_revision
FOR EACH ROW
EXECUTE FUNCTION evidence.enforce_coherent_revision_proposal_selection();

CREATE FUNCTION evidence.enforce_coherent_span_proposal_membership()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    proposal_article_id uuid;
BEGIN
    SELECT source_proposal_article_id INTO proposal_article_id
    FROM evidence.coherent_unit_revision
    WHERE revision_id = NEW.revision_id;
    IF proposal_article_id IS NULL THEN
        RETURN NEW;
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM evidence.article_region member
        WHERE member.article_id = proposal_article_id
          AND member.region_id = NEW.region_id
          AND member.text_start = NEW.text_start
          AND member.text_end = NEW.text_end
          AND member.role = NEW.role
    ) THEN
        RAISE EXCEPTION 'approved coherent-unit span must match its reviewed proposal';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER coherent_span_proposal_membership_trigger
BEFORE INSERT OR UPDATE OF revision_id, region_id, text_start, text_end, role
ON evidence.coherent_unit_span
FOR EACH ROW
EXECUTE FUNCTION evidence.enforce_coherent_span_proposal_membership();

COMMIT;
