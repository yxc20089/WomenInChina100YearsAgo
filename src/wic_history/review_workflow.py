"""Transactional historian review for NER spans and entity resolution."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from .evidence import StrictModel
from .link_pipeline import normalize_name


class ReviewNotFoundError(ValueError):
    pass


class ReviewConflictError(ValueError):
    pass


class LinkCandidateView(StrictModel):
    link_candidate_id: UUID
    proposed_entity_id: UUID | None = None
    proposed_authority_uri: str | None = None
    proposed_canonical_name: str
    score: float = Field(ge=0, le=1)
    is_nil: bool
    features: dict[str, Any] = Field(default_factory=dict)


class MentionQueueItem(StrictModel):
    mention_id: UUID
    mention_status: Literal["candidate", "reviewed", "rejected"]
    entity_id: UUID | None = None
    entity_type: str
    mention_text: str
    normalized_text: str | None = None
    text_start: int
    text_end: int
    confidence: float | None = None
    polygon: dict[str, Any] | None = None
    region_id: UUID
    region_text: str
    source_uri: str
    source_image_uri: str | None = None
    volume_number: int
    publication_year: int
    page_number: int
    model_name: str
    model_revision: str
    extractor: str | None = None
    link_candidates: list[LinkCandidateView] = Field(default_factory=list)


class MentionQueueResponse(StrictModel):
    status: Literal["candidate", "reviewed", "rejected"]
    total: int
    offset: int
    limit: int
    items: list[MentionQueueItem]


class MentionReviewRequest(StrictModel):
    review_id: UUID = Field(default_factory=uuid4)
    decision: Literal["accept", "reject", "needs_review"]
    reviewer: str = Field(min_length=1, max_length=200)
    note: str | None = Field(default=None, max_length=4000)


class EntityResolutionRequest(StrictModel):
    review_id: UUID = Field(default_factory=uuid4)
    selected_link_candidate_id: UUID
    action: Literal["link_existing", "create_new", "keep_nil"]
    reviewer: str = Field(min_length=1, max_length=200)
    note: str | None = Field(default=None, max_length=4000)
    canonical_name: str | None = Field(default=None, max_length=500)
    authority_uri: str | None = Field(default=None, max_length=2000)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "EntityResolutionRequest":
        if self.action == "create_new" and not (self.canonical_name or "").strip():
            raise ValueError("create_new requires canonical_name")
        if self.action != "create_new" and (
            self.canonical_name is not None or self.authority_uri is not None or self.attributes
        ):
            raise ValueError(
                "canonical_name, authority_uri and attributes apply only to create_new"
            )
        return self


class ReviewResult(StrictModel):
    review_id: UUID
    mention_id: UUID
    mention_status: Literal["candidate", "reviewed", "rejected"]
    entity_id: UUID | None = None
    action: str


class ClaimEvidenceView(StrictModel):
    region_id: UUID
    region_text: str
    evidence_quote: str
    text_start: int | None = None
    text_end: int | None = None
    polygon: dict[str, Any] | None = None
    source_uri: str
    source_image_uri: str | None = None
    volume_number: int
    publication_year: int
    page_id: UUID
    page_number: int


class ClaimQueueItem(StrictModel):
    claim_id: UUID
    claim_status: Literal["candidate", "reviewed", "disputed", "rejected", "superseded"]
    subject_entity_id: UUID
    subject_canonical_name: str
    predicate: str
    object_entity_id: UUID | None = None
    object_canonical_name: str | None = None
    object_literal: Any | None = None
    event_date_start: date | None = None
    event_date_end: date | None = None
    confidence: float | None = None
    supporting_quote: str
    model_name: str
    model_revision: str
    evidence: list[ClaimEvidenceView] = Field(default_factory=list)


class ClaimQueueResponse(StrictModel):
    status: Literal["candidate", "reviewed", "disputed", "rejected", "superseded"]
    total: int
    offset: int
    limit: int
    items: list[ClaimQueueItem]


class ClaimReviewRequest(StrictModel):
    review_id: UUID = Field(default_factory=uuid4)
    decision: Literal["accept", "reject", "dispute", "needs_review"]
    reviewer: str = Field(min_length=1, max_length=200)
    note: str | None = Field(default=None, max_length=4000)


class ClaimReviewResult(StrictModel):
    review_id: UUID
    claim_id: UUID
    claim_status: Literal["candidate", "reviewed", "disputed", "rejected", "superseded"]
    action: str


MENTION_QUEUE_SQL = """
    SELECT m.mention_id, m.mention_status, m.entity_id, m.entity_type,
           m.mention_text, m.normalized_text, m.text_start, m.text_end,
           m.confidence, m.polygon, m.attributes,
           r.region_id, r.raw_text AS region_text,
           s.source_uri, p.source_image_uri, v.volume_number,
           v.publication_year, p.page_number,
           pr.model_name, pr.model_revision
    FROM evidence.entity_mention m
    JOIN evidence.ocr_region r USING (region_id)
    JOIN archive.page p USING (page_id)
    JOIN archive.volume v USING (volume_id)
    JOIN archive.source_object s USING (source_object_id)
    JOIN evidence.processing_run pr ON pr.run_id = m.run_id
    WHERE m.mention_status = %(status)s
      AND (%(model_name)s::text IS NULL OR pr.model_name = %(model_name)s::text)
    ORDER BY m.created_at, m.confidence DESC NULLS LAST, m.mention_id
    LIMIT %(limit)s OFFSET %(offset)s
"""


CLAIM_QUEUE_SQL = """
    SELECT c.claim_id, c.claim_status, c.subject_entity_id,
           subject.canonical_name AS subject_canonical_name,
           c.predicate, c.object_entity_id,
           object.canonical_name AS object_canonical_name,
           c.object_literal, c.event_date_start, c.event_date_end,
           c.confidence, c.supporting_quote,
           pr.model_name, pr.model_revision
    FROM evidence.claim c
    JOIN evidence.entity subject ON subject.entity_id = c.subject_entity_id
    LEFT JOIN evidence.entity object ON object.entity_id = c.object_entity_id
    JOIN evidence.processing_run pr ON pr.run_id = c.run_id
    WHERE c.claim_status = %(status)s
      AND (%(model_name)s::text IS NULL OR pr.model_name = %(model_name)s::text)
    ORDER BY c.created_at, c.confidence DESC NULLS LAST, c.claim_id
    LIMIT %(limit)s OFFSET %(offset)s
"""


CLAIM_EVIDENCE_VALIDATION_SQL = """
    SELECT count(*) AS evidence_count,
           count(*) FILTER (
             WHERE ce.text_start IS NULL
                OR ce.evidence_quote = ''
                OR substring(
                     r.raw_text FROM ce.text_start + 1
                     FOR ce.text_end - ce.text_start
                   ) IS DISTINCT FROM ce.evidence_quote
           ) AS invalid_evidence_count,
           count(*) FILTER (
             WHERE ce.evidence_quote = %(supporting_quote)s
           ) AS supporting_quote_count
    FROM evidence.claim_evidence ce
    JOIN evidence.ocr_region r USING (region_id)
    WHERE ce.claim_id = %(claim_id)s
"""


def _clients() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    return psycopg, dict_row


def list_mention_queue(
    database_url: str,
    status: Literal["candidate", "reviewed", "rejected"] = "candidate",
    *,
    limit: int = 25,
    offset: int = 0,
    model_name: str | None = None,
) -> MentionQueueResponse:
    psycopg, dict_row = _clients()
    parameters = {
        "status": status,
        "model_name": model_name,
        "limit": limit,
        "offset": offset,
    }
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        total = connection.execute(
            """
            SELECT count(*)
            FROM evidence.entity_mention m
            JOIN evidence.processing_run pr ON pr.run_id = m.run_id
            WHERE m.mention_status = %(status)s
              AND (%(model_name)s::text IS NULL OR pr.model_name = %(model_name)s::text)
            """,
            parameters,
        ).fetchone()["count"]
        rows = connection.execute(MENTION_QUEUE_SQL, parameters).fetchall()
        mention_ids = [row["mention_id"] for row in rows]
        link_rows = (
            connection.execute(
                """
                SELECT link_candidate_id, mention_id, proposed_entity_id,
                       proposed_authority_uri, proposed_canonical_name,
                       score, is_nil, features
                FROM evidence.entity_link_candidate
                WHERE mention_id = ANY(%s)
                ORDER BY mention_id, is_nil, score DESC, link_candidate_id
                """,
                (mention_ids,),
            ).fetchall()
            if mention_ids
            else []
        )
    links_by_mention: dict[UUID, list[LinkCandidateView]] = {
        mention_id: [] for mention_id in mention_ids
    }
    for row in link_rows:
        links_by_mention[row["mention_id"]].append(
            LinkCandidateView.model_validate(
                {key: value for key, value in row.items() if key != "mention_id"}
            )
        )
    items = []
    for row in rows:
        attributes = row.pop("attributes") or {}
        items.append(
            MentionQueueItem.model_validate(
                {
                    **row,
                    "extractor": attributes.get("extractor"),
                    "link_candidates": links_by_mention[row["mention_id"]],
                }
            )
        )
    return MentionQueueResponse(
        status=status, total=total, offset=offset, limit=limit, items=items
    )


def list_claim_queue(
    database_url: str,
    status: Literal["candidate", "reviewed", "disputed", "rejected", "superseded"] = "candidate",
    *,
    limit: int = 25,
    offset: int = 0,
    model_name: str | None = None,
) -> ClaimQueueResponse:
    psycopg, dict_row = _clients()
    parameters = {
        "status": status,
        "model_name": model_name,
        "limit": limit,
        "offset": offset,
    }
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        total = connection.execute(
            """
            SELECT count(*)
            FROM evidence.claim c
            JOIN evidence.processing_run pr ON pr.run_id = c.run_id
            WHERE c.claim_status = %(status)s
              AND (%(model_name)s::text IS NULL OR pr.model_name = %(model_name)s::text)
            """,
            parameters,
        ).fetchone()["count"]
        rows = connection.execute(CLAIM_QUEUE_SQL, parameters).fetchall()
        claim_ids = [row["claim_id"] for row in rows]
        evidence_rows = (
            connection.execute(
                """
                SELECT ce.claim_id, ce.region_id, r.raw_text AS region_text,
                       ce.evidence_quote, ce.text_start, ce.text_end, ce.polygon,
                       s.source_uri, p.source_image_uri, v.volume_number,
                       v.publication_year, p.page_id, p.page_number
                FROM evidence.claim_evidence ce
                JOIN evidence.ocr_region r USING (region_id)
                JOIN archive.page p USING (page_id)
                JOIN archive.volume v USING (volume_id)
                JOIN archive.source_object s USING (source_object_id)
                WHERE ce.claim_id = ANY(%s)
                ORDER BY ce.claim_id, v.volume_number, p.page_number,
                         r.reading_order, ce.region_id
                """,
                (claim_ids,),
            ).fetchall()
            if claim_ids
            else []
        )
    evidence_by_claim: dict[UUID, list[ClaimEvidenceView]] = {
        claim_id: [] for claim_id in claim_ids
    }
    for row in evidence_rows:
        evidence_by_claim[row["claim_id"]].append(
            ClaimEvidenceView.model_validate(
                {key: value for key, value in row.items() if key != "claim_id"}
            )
        )
    return ClaimQueueResponse(
        status=status,
        total=total,
        offset=offset,
        limit=limit,
        items=[
            ClaimQueueItem.model_validate(
                {**row, "evidence": evidence_by_claim[row["claim_id"]]}
            )
            for row in rows
        ],
    )


def review_mention(
    database_url: str, mention_id: UUID, request: MentionReviewRequest
) -> ReviewResult:
    psycopg, dict_row = _clients()
    desired_status = {
        "accept": "reviewed",
        "reject": "rejected",
        "needs_review": "candidate",
    }[request.decision]
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        existing_review = connection.execute(
            """SELECT target_kind, target_id, decision, reviewer, note, new_value
               FROM evidence.review_decision WHERE review_id = %s""",
            (request.review_id,),
        ).fetchone()
        if existing_review:
            if (
                existing_review["target_kind"] != "mention"
                or existing_review["target_id"] != mention_id
            ):
                raise ReviewConflictError("review_id already belongs to another target")
            expected = existing_review["new_value"] or {}
            if (
                existing_review["decision"] != request.decision
                or existing_review["reviewer"] != request.reviewer
                or existing_review["note"] != request.note
                or expected.get("mention_status") != desired_status
                or expected.get("reviewer") != request.reviewer
            ):
                raise ReviewConflictError("review_id retry payload differs from the stored decision")
            current = connection.execute(
                "SELECT mention_status, entity_id FROM evidence.entity_mention WHERE mention_id = %s",
                (mention_id,),
            ).fetchone()
            if current is None:
                raise ReviewNotFoundError("mention does not exist")
            return ReviewResult(
                review_id=request.review_id,
                mention_id=mention_id,
                mention_status=current["mention_status"],
                entity_id=current["entity_id"],
                action=request.decision,
            )
        mention = connection.execute(
            """
            SELECT mention_id, mention_status, entity_id, entity_type,
                   mention_text, normalized_text
            FROM evidence.entity_mention WHERE mention_id = %s FOR UPDATE
            """,
            (mention_id,),
        ).fetchone()
        if mention is None:
            raise ReviewNotFoundError("mention does not exist")
        if mention["mention_status"] != "candidate":
            raise ReviewConflictError(
                f"mention is already {mention['mention_status']}; review reversal requires a new workflow"
            )
        previous = dict(mention)
        if request.decision != "needs_review":
            connection.execute(
                "UPDATE evidence.entity_mention SET mention_status = %s WHERE mention_id = %s",
                (desired_status, mention_id),
            )
        new_value = {
            "mention_status": desired_status,
            "reviewer": request.reviewer,
        }
        connection.execute(
            """
            INSERT INTO evidence.review_decision (
                review_id, target_kind, target_id, decision, reviewer, note,
                previous_value, new_value
            ) VALUES (%s, 'mention', %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            """,
            (
                request.review_id,
                mention_id,
                request.decision,
                request.reviewer,
                request.note,
                json.dumps(previous, ensure_ascii=False, default=str),
                json.dumps(new_value, ensure_ascii=False),
            ),
        )
    return ReviewResult(
        review_id=request.review_id,
        mention_id=mention_id,
        mention_status=desired_status,
        entity_id=mention["entity_id"],
        action=request.decision,
    )


def resolve_entity(
    database_url: str, mention_id: UUID, request: EntityResolutionRequest
) -> ReviewResult:
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        existing_review = connection.execute(
            """SELECT target_kind, target_id, decision, reviewer, note, new_value
               FROM evidence.review_decision WHERE review_id = %s""",
            (request.review_id,),
        ).fetchone()
        if existing_review:
            if (
                existing_review["target_kind"] != "entity_link"
                or existing_review["target_id"] != request.selected_link_candidate_id
            ):
                raise ReviewConflictError("review_id already belongs to another target")
            expected = existing_review["new_value"] or {}
            if (
                existing_review["decision"] != "accept"
                or existing_review["reviewer"] != request.reviewer
                or existing_review["note"] != request.note
                or expected.get("action") != request.action
                or expected.get("reviewer") != request.reviewer
                or expected.get("canonical_name") != request.canonical_name
                or expected.get("authority_uri") != request.authority_uri
                or expected.get("attributes") != request.attributes
            ):
                raise ReviewConflictError("review_id retry payload differs from the stored decision")
            current = connection.execute(
                "SELECT mention_status, entity_id FROM evidence.entity_mention WHERE mention_id = %s",
                (mention_id,),
            ).fetchone()
            if current is None:
                raise ReviewNotFoundError("mention does not exist")
            return ReviewResult(
                review_id=request.review_id,
                mention_id=mention_id,
                mention_status=current["mention_status"],
                entity_id=current["entity_id"],
                action=request.action,
            )
        mention = connection.execute(
            """
            SELECT mention_id, mention_status, entity_id, entity_type,
                   mention_text, normalized_text
            FROM evidence.entity_mention WHERE mention_id = %s FOR UPDATE
            """,
            (mention_id,),
        ).fetchone()
        if mention is None:
            raise ReviewNotFoundError("mention does not exist")
        if mention["mention_status"] != "reviewed":
            raise ReviewConflictError("entity resolution requires an accepted mention span")
        if mention["entity_id"] is not None:
            raise ReviewConflictError("mention already resolves to an entity")
        candidate = connection.execute(
            """
            SELECT link_candidate_id, mention_id, proposed_entity_id,
                   proposed_authority_uri, proposed_canonical_name,
                   score, is_nil, features
            FROM evidence.entity_link_candidate
            WHERE link_candidate_id = %s FOR UPDATE
            """,
            (request.selected_link_candidate_id,),
        ).fetchone()
        if candidate is None or candidate["mention_id"] != mention_id:
            raise ReviewNotFoundError("link candidate does not exist for this mention")

        entity_id: UUID | None = None
        if request.action == "link_existing":
            if candidate["is_nil"] or candidate["proposed_entity_id"] is None:
                raise ReviewConflictError("link_existing requires a non-NIL candidate")
            entity = connection.execute(
                """
                SELECT entity_id, entity_type, entity_status
                FROM evidence.entity WHERE entity_id = %s
                """,
                (candidate["proposed_entity_id"],),
            ).fetchone()
            if (
                entity is None
                or entity["entity_status"] != "reviewed"
                or entity["entity_type"] != mention["entity_type"]
            ):
                raise ReviewConflictError("target entity must be reviewed and type-compatible")
            entity_id = entity["entity_id"]
        elif request.action == "create_new":
            if not candidate["is_nil"]:
                raise ReviewConflictError("create_new requires selection of the NIL/new candidate")
            canonical_name = request.canonical_name.strip()
            attributes = {
                **request.attributes,
                "created_by_review": str(request.review_id),
                "source_mention_id": str(mention_id),
            }
            try:
                entity_id = connection.execute(
                    """
                    INSERT INTO evidence.entity (
                        entity_type, canonical_name, normalized_name, authority_uri,
                        entity_status, attributes
                    ) VALUES (%s, %s, %s, %s, 'reviewed', %s::jsonb)
                    RETURNING entity_id
                    """,
                    (
                        mention["entity_type"],
                        canonical_name,
                        normalize_name(canonical_name),
                        request.authority_uri,
                        json.dumps(attributes, ensure_ascii=False),
                    ),
                ).fetchone()["entity_id"]
            except psycopg.errors.UniqueViolation as exc:
                raise ReviewConflictError(
                    "authority URI already belongs to another entity"
                ) from exc
        elif not candidate["is_nil"]:
            raise ReviewConflictError("keep_nil requires selection of the NIL candidate")

        if entity_id is not None:
            connection.execute(
                "UPDATE evidence.entity_mention SET entity_id = %s WHERE mention_id = %s",
                (entity_id, mention_id),
            )
        previous_value = {key: value for key, value in candidate.items() if key != "features"}
        previous_value["features"] = candidate["features"]
        new_value = {
            "action": request.action,
            "mention_id": str(mention_id),
            "entity_id": str(entity_id) if entity_id else None,
            "reviewer": request.reviewer,
            "canonical_name": request.canonical_name,
            "authority_uri": request.authority_uri,
            "attributes": request.attributes,
        }
        connection.execute(
            """
            INSERT INTO evidence.review_decision (
                review_id, target_kind, target_id, decision, reviewer, note,
                previous_value, new_value
            ) VALUES (%s, 'entity_link', %s, 'accept', %s, %s, %s::jsonb, %s::jsonb)
            """,
            (
                request.review_id,
                request.selected_link_candidate_id,
                request.reviewer,
                request.note,
                json.dumps(previous_value, ensure_ascii=False, default=str),
                json.dumps(new_value, ensure_ascii=False),
            ),
        )
    return ReviewResult(
        review_id=request.review_id,
        mention_id=mention_id,
        mention_status="reviewed",
        entity_id=entity_id,
        action=request.action,
    )


def review_claim(
    database_url: str, claim_id: UUID, request: ClaimReviewRequest
) -> ClaimReviewResult:
    psycopg, dict_row = _clients()
    desired_status = {
        "accept": "reviewed",
        "reject": "rejected",
        "dispute": "disputed",
        "needs_review": "candidate",
    }[request.decision]
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        existing_review = connection.execute(
            """SELECT target_kind, target_id, decision, reviewer, note, new_value
               FROM evidence.review_decision WHERE review_id = %s""",
            (request.review_id,),
        ).fetchone()
        if existing_review:
            if (
                existing_review["target_kind"] != "claim"
                or existing_review["target_id"] != claim_id
            ):
                raise ReviewConflictError("review_id already belongs to another target")
            expected = existing_review["new_value"] or {}
            if (
                existing_review["decision"] != request.decision
                or existing_review["reviewer"] != request.reviewer
                or existing_review["note"] != request.note
                or expected.get("claim_status") != desired_status
                or expected.get("reviewer") != request.reviewer
            ):
                raise ReviewConflictError(
                    "review_id retry payload differs from the stored decision"
                )
            current = connection.execute(
                "SELECT claim_status FROM evidence.claim WHERE claim_id = %s",
                (claim_id,),
            ).fetchone()
            if current is None:
                raise ReviewNotFoundError("claim does not exist")
            return ClaimReviewResult(
                review_id=request.review_id,
                claim_id=claim_id,
                claim_status=current["claim_status"],
                action=request.decision,
            )

        claim = connection.execute(
            """
            SELECT c.claim_id, c.claim_status, c.subject_entity_id, c.predicate,
                   c.object_entity_id, c.object_literal, c.event_date_start,
                   c.event_date_end, c.confidence, c.supporting_quote,
                   subject.entity_status AS subject_status,
                   object.entity_status AS object_status
            FROM evidence.claim c
            JOIN evidence.entity subject ON subject.entity_id = c.subject_entity_id
            LEFT JOIN evidence.entity object ON object.entity_id = c.object_entity_id
            WHERE c.claim_id = %s FOR UPDATE OF c
            """,
            (claim_id,),
        ).fetchone()
        if claim is None:
            raise ReviewNotFoundError("claim does not exist")
        if claim["claim_status"] != "candidate":
            raise ReviewConflictError(
                f"claim is already {claim['claim_status']}; review reversal requires a new workflow"
            )
        evidence_validation = connection.execute(
            CLAIM_EVIDENCE_VALIDATION_SQL,
            {"claim_id": claim_id, "supporting_quote": claim["supporting_quote"]},
        ).fetchone()
        evidence_count = evidence_validation["evidence_count"]
        if request.decision == "accept" and evidence_count == 0:
            raise ReviewConflictError("a claim cannot be accepted without cited evidence")
        if request.decision == "accept" and (
            evidence_validation["invalid_evidence_count"]
            or evidence_validation["supporting_quote_count"] == 0
        ):
            raise ReviewConflictError(
                "claim evidence must match exact non-empty OCR offsets and the supporting quote"
            )
        if request.decision == "accept" and (
            claim["subject_status"] != "reviewed"
            or (
                claim["object_entity_id"] is not None
                and claim["object_status"] != "reviewed"
            )
        ):
            raise ReviewConflictError(
                "a claim can be accepted only when its referenced entities are reviewed"
            )
        if request.decision != "needs_review":
            connection.execute(
                "UPDATE evidence.claim SET claim_status = %s, updated_at = now() WHERE claim_id = %s",
                (desired_status, claim_id),
            )
        new_value = {
            "claim_status": desired_status,
            "reviewer": request.reviewer,
            "evidence_count": evidence_count,
        }
        connection.execute(
            """
            INSERT INTO evidence.review_decision (
                review_id, target_kind, target_id, decision, reviewer, note,
                previous_value, new_value
            ) VALUES (%s, 'claim', %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            """,
            (
                request.review_id,
                claim_id,
                request.decision,
                request.reviewer,
                request.note,
                json.dumps(dict(claim), ensure_ascii=False, default=str),
                json.dumps(new_value, ensure_ascii=False),
            ),
        )
    return ClaimReviewResult(
        review_id=request.review_id,
        claim_id=claim_id,
        claim_status=desired_status,
        action=request.decision,
    )
