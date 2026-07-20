"""Authoritative E2E persistence and reverse-evidence operations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable, Literal
from uuid import UUID, uuid4


TextVariant = Literal[
    "raw_ocr",
    "ocr_hypothesis",
    "corrected_transcription",
    "approved_reconstruction",
    "normalized_search",
]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clients() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    return psycopg, dict_row


@dataclass(frozen=True, slots=True)
class LocatedSpan:
    text_start: int
    text_end: int
    surface_text: str
    left_context: str
    right_context: str


@dataclass(frozen=True, slots=True)
class TextVersionResult:
    text_version_id: str
    text_sha256: str
    alignment_id: str | None


@dataclass(frozen=True, slots=True)
class TextReviewResult:
    review_id: str
    text_version_id: str
    review_status: str
    selection_id: str | None


@dataclass(frozen=True, slots=True)
class EvidenceSpanResult:
    evidence_span_id: str
    text_version_id: str
    text_start: int
    text_end: int
    surface_text: str


@dataclass(frozen=True, slots=True)
class EventParticipantSpec:
    participant_role: str
    entity_id: UUID | None = None
    mention_id: UUID | None = None
    evidence_span_id: UUID | None = None

    def __post_init__(self) -> None:
        if (self.entity_id is None) == (self.mention_id is None):
            raise ValueError("participant requires exactly one entity_id or mention_id")
        if not self.participant_role.strip():
            raise ValueError("participant_role must not be blank")


@dataclass(frozen=True, slots=True)
class EventEvidenceSpec:
    evidence_span_id: UUID
    support_role: Literal[
        "direct_support", "context", "contradiction", "external_corroboration"
    ] = "direct_support"


@dataclass(frozen=True, slots=True)
class EventResult:
    event_id: str
    participants: int
    evidence: int
    status: str


@dataclass(frozen=True, slots=True)
class EntityRedirectResult:
    entity_redirect_id: str
    superseded_entity_id: str
    canonical_entity_id: str
    review_id: str
    active: bool


ConfidenceStatus = Literal["not_reported", "uncalibrated", "calibrated"]
VisualOutputKind = Literal["spotting", "layout", "recognition"]
VisualEvidencePathRole = Literal["source_page", "input_crop", "output_geometry"]


def _validate_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ConfidenceCalibrationSpec:
    calibration_id: UUID
    task_kind: str
    model_name: str
    model_revision: str
    method: str
    dataset_id: str
    dataset_sha256: str
    artifact_uri: str
    artifact_sha256: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "task_kind",
            "model_name",
            "model_revision",
            "method",
            "dataset_id",
            "artifact_uri",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        _validate_sha256(self.dataset_sha256, "dataset_sha256")
        _validate_sha256(self.artifact_sha256, "artifact_sha256")


@dataclass(frozen=True, slots=True)
class VisualEvidencePathSpec:
    evidence_path_id: UUID
    path_role: VisualEvidencePathRole
    source_object_id: UUID
    page_id: UUID
    derivative_id: UUID
    source_uri: str
    image_uri: str
    image_sha256: str
    layout_region_id: UUID | None = None
    region_id: UUID | None = None
    text_version_id: UUID | None = None
    evidence_span_id: UUID | None = None
    crop_uri: str | None = None
    crop_sha256: str | None = None
    crop_bounds: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.source_uri.strip() or not self.image_uri.strip():
            raise ValueError("source_uri and image_uri must not be blank")
        _validate_sha256(self.image_sha256, "image_sha256")
        if (self.crop_uri is None) != (self.crop_sha256 is None):
            raise ValueError("crop_uri and crop_sha256 must be supplied together")
        if self.path_role == "input_crop" and self.crop_uri is None:
            raise ValueError("input_crop evidence requires exact crop bytes")
        if self.crop_sha256 is not None:
            _validate_sha256(self.crop_sha256, "crop_sha256")
        if self.evidence_span_id is not None and self.text_version_id is None:
            raise ValueError("evidence_span_id requires text_version_id")
        if self.text_version_id is not None and self.region_id is None:
            raise ValueError("text_version_id requires region_id")


@dataclass(frozen=True, slots=True)
class VisualModelOutputSpec:
    visual_output_id: UUID
    output_kind: VisualOutputKind
    artifact_uri: str
    artifact_sha256: str
    raw_output: str
    evidence_paths: tuple[VisualEvidencePathSpec, ...]
    structured_output: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    confidence_status: ConfidenceStatus = "not_reported"
    calibration_id: UUID | None = None

    def __post_init__(self) -> None:
        if not self.artifact_uri.strip():
            raise ValueError("artifact_uri must not be blank")
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        if not self.evidence_paths:
            raise ValueError("visual output requires at least one exact evidence path")
        if len({item.evidence_path_id for item in self.evidence_paths}) != len(
            self.evidence_paths
        ):
            raise ValueError("visual output evidence-path IDs must be unique")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        valid = (
            self.confidence_status == "not_reported"
            and self.confidence is None
            and self.calibration_id is None
        ) or (
            self.confidence_status == "uncalibrated"
            and self.confidence is not None
            and self.calibration_id is None
        ) or (
            self.confidence_status == "calibrated"
            and self.confidence is not None
            and self.calibration_id is not None
        )
        if not valid:
            raise ValueError("confidence_status does not match score/calibration provenance")

    @property
    def raw_output_sha256(self) -> str:
        return _sha256_text(self.raw_output)


@dataclass(frozen=True, slots=True)
class VisualOutputPersistResult:
    run_id: str
    outputs: int
    evidence_paths: int
    reused: bool


@dataclass(frozen=True, slots=True)
class LocalIdentityClusterSpec:
    local_cluster_id: UUID
    mention_ids: tuple[UUID, ...]

    def __post_init__(self) -> None:
        if not self.mention_ids:
            raise ValueError("local identity cluster must contain a mention occurrence")
        if len(set(self.mention_ids)) != len(self.mention_ids):
            raise ValueError("a mention occurrence may appear only once in a cluster")


@dataclass(frozen=True, slots=True)
class LocalIdentityPersistResult:
    local_coreference_run_id: str
    clusters: int
    memberships: int
    reused: bool


@dataclass(frozen=True, slots=True)
class LocalIdentityReviewResult:
    local_cluster_id: str
    review_id: str
    review_status: str
    memberships: int
    reused: bool


def locate_unique_surface(
    text: str,
    surface: str,
    *,
    left_context: str = "",
    right_context: str = "",
) -> LocatedSpan:
    """Resolve a model-supplied surface/context to one end-exclusive span."""
    if not surface:
        raise ValueError("surface must not be empty")
    positions: list[int] = []
    cursor = 0
    while True:
        index = text.find(surface, cursor)
        if index < 0:
            break
        positions.append(index)
        cursor = index + 1
    matches = []
    for start in positions:
        end = start + len(surface)
        if left_context and text[max(0, start - len(left_context)) : start] != left_context:
            continue
        if right_context and text[end : end + len(right_context)] != right_context:
            continue
        matches.append((start, end))
    if len(matches) != 1:
        raise ValueError(
            f"surface/context must identify exactly one occurrence; found {len(matches)}"
        )
    start, end = matches[0]
    return LocatedSpan(start, end, surface, left_context, right_context)


def alignment_operations(source: str, target: str) -> list[dict[str, Any]]:
    return [
        {
            "operation": tag,
            "source_start": source_start,
            "source_end": source_end,
            "target_start": target_start,
            "target_end": target_end,
            "source_text": source[source_start:source_end],
            "target_text": target[target_start:target_end],
        }
        for tag, source_start, source_end, target_start, target_end in SequenceMatcher(
            None, source, target, autojunk=False
        ).get_opcodes()
    ]


def register_confidence_calibration(
    database_url: str, spec: ConfidenceCalibrationSpec
) -> tuple[str, bool]:
    """Register one immutable calibration artifact; return ``(id, reused)``."""
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        inserted = connection.execute(
            """
            INSERT INTO evidence.confidence_calibration (
                calibration_id, task_kind, model_name, model_revision, method,
                dataset_id, dataset_sha256, artifact_uri, artifact_sha256, metrics
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
            )
            ON CONFLICT (calibration_id) DO NOTHING
            RETURNING calibration_id
            """,
            (
                spec.calibration_id,
                spec.task_kind,
                spec.model_name,
                spec.model_revision,
                spec.method,
                spec.dataset_id,
                spec.dataset_sha256,
                spec.artifact_uri,
                spec.artifact_sha256,
                json.dumps(spec.metrics, ensure_ascii=False),
            ),
        ).fetchone()
        stored = connection.execute(
            """
            SELECT task_kind, model_name, model_revision, method, dataset_id,
                   dataset_sha256, artifact_uri, artifact_sha256, metrics
            FROM evidence.confidence_calibration WHERE calibration_id = %s
            """,
            (spec.calibration_id,),
        ).fetchone()
        expected = {
            "task_kind": spec.task_kind,
            "model_name": spec.model_name,
            "model_revision": spec.model_revision,
            "method": spec.method,
            "dataset_id": spec.dataset_id,
            "dataset_sha256": spec.dataset_sha256,
            "artifact_uri": spec.artifact_uri,
            "artifact_sha256": spec.artifact_sha256,
            "metrics": spec.metrics,
        }
        if stored != expected:
            raise ValueError("stored confidence calibration differs from immutable artifact")
    return str(spec.calibration_id), inserted is None


def persist_visual_model_outputs(
    database_url: str,
    *,
    run_id: UUID,
    outputs: Iterable[VisualModelOutputSpec],
) -> VisualOutputPersistResult:
    """Persist exact spotting/layout/recognition outputs and archive paths."""
    records = tuple(outputs)
    if len({item.visual_output_id for item in records}) != len(records):
        raise ValueError("visual output IDs must be unique within one artifact")
    all_path_ids = [
        path.evidence_path_id for item in records for path in item.evidence_paths
    ]
    if len(set(all_path_ids)) != len(all_path_ids):
        raise ValueError("evidence-path IDs must be unique within one artifact")

    psycopg, dict_row = _clients()
    inserted_outputs = 0
    inserted_paths = 0
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        run = connection.execute(
            """
            SELECT status FROM evidence.processing_run WHERE run_id = %s
            """,
            (run_id,),
        ).fetchone()
        if run is None or run["status"] != "completed":
            raise ValueError("visual outputs require a completed registered run")
        for item in records:
            if item.calibration_id is not None:
                calibration = connection.execute(
                    """
                    SELECT calibration_id FROM evidence.confidence_calibration
                    WHERE calibration_id = %s
                    """,
                    (item.calibration_id,),
                ).fetchone()
                if calibration is None:
                    raise ValueError("visual output references unknown calibration")
            cursor = connection.execute(
                """
                INSERT INTO evidence.visual_model_output (
                    visual_output_id, run_id, output_kind, artifact_uri,
                    artifact_sha256, raw_output, raw_output_sha256,
                    structured_output, confidence, confidence_status, calibration_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s
                )
                ON CONFLICT (visual_output_id) DO NOTHING
                RETURNING visual_output_id
                """,
                (
                    item.visual_output_id,
                    run_id,
                    item.output_kind,
                    item.artifact_uri,
                    item.artifact_sha256,
                    item.raw_output,
                    item.raw_output_sha256,
                    json.dumps(item.structured_output, ensure_ascii=False),
                    item.confidence,
                    item.confidence_status,
                    item.calibration_id,
                ),
            ).fetchone()
            inserted_outputs += cursor is not None
            stored = connection.execute(
                """
                SELECT run_id, output_kind, artifact_uri, artifact_sha256,
                       raw_output, raw_output_sha256, structured_output,
                       confidence, confidence_status, calibration_id
                FROM evidence.visual_model_output WHERE visual_output_id = %s
                """,
                (item.visual_output_id,),
            ).fetchone()
            expected = {
                "run_id": run_id,
                "output_kind": item.output_kind,
                "artifact_uri": item.artifact_uri,
                "artifact_sha256": item.artifact_sha256,
                "raw_output": item.raw_output,
                "raw_output_sha256": item.raw_output_sha256,
                "structured_output": item.structured_output,
                "confidence": item.confidence,
                "confidence_status": item.confidence_status,
                "calibration_id": item.calibration_id,
            }
            if stored != expected:
                raise ValueError("stored visual output differs from immutable artifact")
            for path in item.evidence_paths:
                inserted = connection.execute(
                    """
                    INSERT INTO evidence.visual_model_evidence_path (
                        evidence_path_id, visual_output_id, path_role,
                        source_object_id, page_id, derivative_id,
                        layout_region_id, region_id, text_version_id,
                        evidence_span_id, source_uri, image_uri, image_sha256,
                        crop_uri, crop_sha256, crop_bounds
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (evidence_path_id) DO NOTHING
                    RETURNING evidence_path_id
                    """,
                    (
                        path.evidence_path_id,
                        item.visual_output_id,
                        path.path_role,
                        path.source_object_id,
                        path.page_id,
                        path.derivative_id,
                        path.layout_region_id,
                        path.region_id,
                        path.text_version_id,
                        path.evidence_span_id,
                        path.source_uri,
                        path.image_uri,
                        path.image_sha256,
                        path.crop_uri,
                        path.crop_sha256,
                        json.dumps(path.crop_bounds, ensure_ascii=False)
                        if path.crop_bounds is not None
                        else None,
                    ),
                ).fetchone()
                inserted_paths += inserted is not None
                stored_path = connection.execute(
                    """
                    SELECT visual_output_id, path_role, source_object_id,
                           page_id, derivative_id, layout_region_id, region_id,
                           text_version_id, evidence_span_id, source_uri,
                           image_uri, image_sha256, crop_uri, crop_sha256, crop_bounds
                    FROM evidence.visual_model_evidence_path
                    WHERE evidence_path_id = %s
                    """,
                    (path.evidence_path_id,),
                ).fetchone()
                expected_path = {
                    "visual_output_id": item.visual_output_id,
                    "path_role": path.path_role,
                    "source_object_id": path.source_object_id,
                    "page_id": path.page_id,
                    "derivative_id": path.derivative_id,
                    "layout_region_id": path.layout_region_id,
                    "region_id": path.region_id,
                    "text_version_id": path.text_version_id,
                    "evidence_span_id": path.evidence_span_id,
                    "source_uri": path.source_uri,
                    "image_uri": path.image_uri,
                    "image_sha256": path.image_sha256,
                    "crop_uri": path.crop_uri,
                    "crop_sha256": path.crop_sha256,
                    "crop_bounds": path.crop_bounds,
                }
                if stored_path != expected_path:
                    raise ValueError(
                        "stored visual evidence path differs from immutable artifact"
                    )
    return VisualOutputPersistResult(
        str(run_id),
        len(records),
        len(all_path_ids),
        inserted_outputs == 0 and inserted_paths == 0,
    )


def persist_article_local_identity_clusters(
    database_url: str,
    *,
    local_coreference_run_id: UUID,
    processing_run_id: UUID,
    coherent_unit_revision_id: UUID,
    input_sha256: str,
    configuration_sha256: str,
    clusters: Iterable[LocalIdentityClusterSpec],
) -> LocalIdentityPersistResult:
    """Store mention-occurrence clusters scoped to one active article revision."""
    _validate_sha256(input_sha256, "input_sha256")
    _validate_sha256(configuration_sha256, "configuration_sha256")
    records = tuple(clusters)
    if len({item.local_cluster_id for item in records}) != len(records):
        raise ValueError("local identity cluster IDs must be unique")
    mention_ids = [mention_id for item in records for mention_id in item.mention_ids]
    if len(set(mention_ids)) != len(mention_ids):
        raise ValueError("a mention occurrence may belong to only one local cluster")

    psycopg, dict_row = _clients()
    reused = False
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        run = connection.execute(
            """
            SELECT status FROM evidence.processing_run WHERE run_id = %s
            """,
            (processing_run_id,),
        ).fetchone()
        if run is None or run["status"] != "completed":
            raise ValueError("local identity requires a completed registered run")
        revision = connection.execute(
            """
            SELECT revision_id FROM evidence.coherent_unit_revision
            WHERE revision_id = %s AND superseded_at IS NULL
            """,
            (coherent_unit_revision_id,),
        ).fetchone()
        if revision is None:
            raise ValueError("local identity requires one active coherent-unit revision")
        if mention_ids:
            mention_rows = connection.execute(
                """
                SELECT mention_id, coherent_unit_revision_id, evidence_span_id
                FROM evidence.entity_mention
                WHERE mention_id = ANY(%s::uuid[])
                """,
                (mention_ids,),
            ).fetchall()
            mentions = {row["mention_id"]: row for row in mention_rows}
            if mentions.keys() != set(mention_ids):
                raise ValueError("local identity contains an unknown mention occurrence")
            if any(
                row["coherent_unit_revision_id"] != coherent_unit_revision_id
                or row["evidence_span_id"] is None
                for row in mentions.values()
            ):
                raise ValueError(
                    "every local identity member must retain exact evidence in the same revision"
                )
        inserted_run = connection.execute(
            """
            INSERT INTO evidence.local_coreference_run (
                local_coreference_run_id, processing_run_id,
                coherent_unit_revision_id, input_sha256, configuration_sha256
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (
                coherent_unit_revision_id, input_sha256, configuration_sha256
            ) DO NOTHING
            RETURNING local_coreference_run_id
            """,
            (
                local_coreference_run_id,
                processing_run_id,
                coherent_unit_revision_id,
                input_sha256,
                configuration_sha256,
            ),
        ).fetchone()
        stored_run = connection.execute(
            """
            SELECT local_coreference_run_id, processing_run_id
            FROM evidence.local_coreference_run
            WHERE coherent_unit_revision_id = %s
              AND input_sha256 = %s AND configuration_sha256 = %s
            """,
            (coherent_unit_revision_id, input_sha256, configuration_sha256),
        ).fetchone()
        if stored_run != {
            "local_coreference_run_id": local_coreference_run_id,
            "processing_run_id": processing_run_id,
        }:
            raise ValueError("stored local identity run differs from immutable input")
        reused = inserted_run is None
        for cluster in records:
            connection.execute(
                """
                INSERT INTO evidence.local_coreference_cluster (
                    local_cluster_id, local_coreference_run_id,
                    coherent_unit_revision_id, review_status
                ) VALUES (%s, %s, %s, 'candidate')
                ON CONFLICT (local_cluster_id) DO NOTHING
                """,
                (
                    cluster.local_cluster_id,
                    local_coreference_run_id,
                    coherent_unit_revision_id,
                ),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO evidence.local_coreference_member (
                        local_cluster_id, local_coreference_run_id,
                        mention_id, coherent_unit_revision_id
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (local_cluster_id, mention_id) DO NOTHING
                    """,
                    [
                        (
                            cluster.local_cluster_id,
                            local_coreference_run_id,
                            mention_id,
                            coherent_unit_revision_id,
                        )
                        for mention_id in cluster.mention_ids
                    ],
                )
        stored_members = connection.execute(
            """
            SELECT cluster.local_cluster_id, member.mention_id
            FROM evidence.local_coreference_cluster cluster
            JOIN evidence.local_coreference_member member
              USING (local_cluster_id, local_coreference_run_id)
            WHERE cluster.local_coreference_run_id = %s
              AND cluster.coherent_unit_revision_id = %s
            ORDER BY cluster.local_cluster_id, member.mention_id
            """,
            (local_coreference_run_id, coherent_unit_revision_id),
        ).fetchall()
        observed = [
            (row["local_cluster_id"], row["mention_id"]) for row in stored_members
        ]
        expected = sorted(
            (item.local_cluster_id, mention_id)
            for item in records
            for mention_id in item.mention_ids
        )
        if observed != expected:
            raise ValueError("stored local identity clusters differ from immutable output")
    return LocalIdentityPersistResult(
        str(local_coreference_run_id), len(records), len(mention_ids), reused
    )


def review_local_identity_cluster(
    database_url: str,
    local_cluster_id: UUID,
    *,
    decision: Literal["accept", "reject"],
    reviewer: str,
    note: str | None = None,
    review_id: UUID | None = None,
) -> LocalIdentityReviewResult:
    """Append one terminal local-cluster review without changing memberships."""
    if not reviewer.strip():
        raise ValueError("reviewer must not be blank")
    desired = "reviewed" if decision == "accept" else "rejected"
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        cluster = connection.execute(
            """
            SELECT local_cluster_id, local_coreference_run_id,
                   coherent_unit_revision_id, review_status, created_at
            FROM evidence.local_coreference_cluster
            WHERE local_cluster_id = %s FOR UPDATE
            """,
            (local_cluster_id,),
        ).fetchone()
        if cluster is None:
            raise ValueError("unknown local identity cluster")
        membership_count = connection.execute(
            """
            SELECT count(*) AS total
            FROM evidence.local_coreference_member
            WHERE local_cluster_id = %s
            """,
            (local_cluster_id,),
        ).fetchone()["total"]
        if membership_count < 1:
            raise ValueError("local identity cluster has no mention occurrences")

        if cluster["review_status"] != "candidate":
            if cluster["review_status"] != desired:
                raise ValueError("local identity cluster already has a different review")
            existing = connection.execute(
                """
                SELECT review_id, reviewer, note
                FROM evidence.review_decision
                WHERE target_kind = 'local_coreference_cluster'
                  AND target_id = %s AND decision = %s
                ORDER BY reviewed_at, review_id
                LIMIT 1
                """,
                (local_cluster_id, decision),
            ).fetchone()
            if existing is None:
                raise ValueError("reviewed local cluster lacks its append-only decision")
            if review_id is not None and existing["review_id"] != review_id:
                raise ValueError("review_id differs from the completed local review")
            if existing["reviewer"] != reviewer or existing["note"] != note:
                raise ValueError("retry differs from the completed local review")
            return LocalIdentityReviewResult(
                str(local_cluster_id),
                str(existing["review_id"]),
                desired,
                membership_count,
                True,
            )

        review_id = review_id or uuid4()
        previous = {
            "review_status": cluster["review_status"],
            "local_coreference_run_id": str(cluster["local_coreference_run_id"]),
            "coherent_unit_revision_id": str(cluster["coherent_unit_revision_id"]),
            "memberships": membership_count,
        }
        new_value = {
            "review_status": desired,
            "memberships_unchanged": membership_count,
        }
        inserted = connection.execute(
            """
            INSERT INTO evidence.review_decision (
                review_id, target_kind, target_id, decision, reviewer, note,
                previous_value, new_value
            ) VALUES (
                %s, 'local_coreference_cluster', %s, %s, %s, %s,
                %s::jsonb, %s::jsonb
            )
            ON CONFLICT (review_id) DO NOTHING
            RETURNING review_id
            """,
            (
                review_id,
                local_cluster_id,
                decision,
                reviewer,
                note,
                json.dumps(previous, ensure_ascii=False),
                json.dumps(new_value, ensure_ascii=False),
            ),
        ).fetchone()
        if inserted is None:
            existing = connection.execute(
                """
                SELECT target_kind, target_id, decision, reviewer, note,
                       previous_value, new_value
                FROM evidence.review_decision WHERE review_id = %s
                """,
                (review_id,),
            ).fetchone()
            if existing != {
                "target_kind": "local_coreference_cluster",
                "target_id": local_cluster_id,
                "decision": decision,
                "reviewer": reviewer,
                "note": note,
                "previous_value": previous,
                "new_value": new_value,
            }:
                raise ValueError("review_id already identifies a different decision")
        updated = connection.execute(
            """
            UPDATE evidence.local_coreference_cluster
            SET review_status = %s
            WHERE local_cluster_id = %s AND review_status = 'candidate'
            RETURNING local_cluster_id
            """,
            (desired, local_cluster_id),
        ).fetchone()
        if updated is None:
            raise ValueError("local identity cluster review raced with another decision")
        after_count = connection.execute(
            """
            SELECT count(*) AS total
            FROM evidence.local_coreference_member
            WHERE local_cluster_id = %s
            """,
            (local_cluster_id,),
        ).fetchone()["total"]
        if after_count != membership_count:
            raise ValueError("local identity review must not change mention memberships")
    return LocalIdentityReviewResult(
        str(local_cluster_id), str(review_id), desired, membership_count, False
    )


def create_candidate_text_version(
    database_url: str,
    *,
    region_id: UUID,
    variant: TextVariant,
    text_content: str,
    parent_text_version_id: UUID | None = None,
    producing_run_id: UUID | None = None,
    language: str = "zh-Hant",
    configuration: dict[str, Any] | None = None,
) -> TextVersionResult:
    """Append a hashed transcription and deterministic parent alignment."""
    if variant == "raw_ocr":
        raise ValueError("raw OCR versions are created only by OCR artifact ingestion")
    if not text_content:
        raise ValueError("text_content must not be empty")
    psycopg, dict_row = _clients()
    text_sha256 = _sha256_text(text_content)
    alignment_id: UUID | None = None
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        region = connection.execute(
            "SELECT region_id FROM evidence.ocr_region WHERE region_id = %s",
            (region_id,),
        ).fetchone()
        if region is None:
            raise ValueError("unknown OCR region")
        parent_text: str | None = None
        if parent_text_version_id is not None:
            parent = connection.execute(
                """
                SELECT region_id, text_content
                FROM evidence.text_version WHERE text_version_id = %s
                """,
                (parent_text_version_id,),
            ).fetchone()
            if parent is None or parent["region_id"] != region_id:
                raise ValueError("parent text version must belong to the same OCR region")
            parent_text = parent["text_content"]
        row = connection.execute(
            """
            INSERT INTO evidence.text_version (
                region_id, parent_text_version_id, producing_run_id, variant,
                text_content, text_sha256, language, review_status, configuration
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'candidate', %s::jsonb)
            ON CONFLICT (region_id, variant, text_sha256) DO UPDATE SET
                text_content = EXCLUDED.text_content
            RETURNING text_version_id
            """,
            (
                region_id,
                parent_text_version_id,
                producing_run_id,
                variant,
                text_content,
                text_sha256,
                language,
                json.dumps(configuration or {}, ensure_ascii=False),
            ),
        ).fetchone()
        text_version_id = row["text_version_id"]
        if parent_text is not None:
            operations = alignment_operations(parent_text, text_content)
            alignment_sha256 = hashlib.sha256(_canonical_bytes(operations)).hexdigest()
            alignment_id = connection.execute(
                """
                INSERT INTO evidence.text_version_alignment (
                    source_text_version_id, target_text_version_id,
                    operations, alignment_sha256
                ) VALUES (%s, %s, %s::jsonb, %s)
                ON CONFLICT (
                    source_text_version_id, target_text_version_id, alignment_sha256
                ) DO UPDATE SET operations = EXCLUDED.operations
                RETURNING alignment_id
                """,
                (
                    parent_text_version_id,
                    text_version_id,
                    json.dumps(operations, ensure_ascii=False),
                    alignment_sha256,
                ),
            ).fetchone()["alignment_id"]
    return TextVersionResult(str(text_version_id), text_sha256, str(alignment_id) if alignment_id else None)


def review_and_select_text_version(
    database_url: str,
    text_version_id: UUID,
    *,
    decision: Literal["accept", "reject", "needs_review"],
    reviewer: str,
    note: str | None = None,
    review_id: UUID | None = None,
) -> TextReviewResult:
    """Review a transcription and, on acceptance, make it the active version."""
    if not reviewer.strip():
        raise ValueError("reviewer must not be blank")
    psycopg, dict_row = _clients()
    review_id = review_id or uuid4()
    desired = {"accept": "reviewed", "reject": "rejected", "needs_review": "candidate"}[
        decision
    ]
    from . import coherent_jobs

    selection_id: UUID | None = None
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        if decision == "accept":
            coherent_jobs.lock_coherent_mutation(connection)
        version = connection.execute(
            """
            SELECT text_version_id, region_id, variant, text_sha256, review_status
            FROM evidence.text_version WHERE text_version_id = %s FOR UPDATE
            """,
            (text_version_id,),
        ).fetchone()
        if version is None:
            raise ValueError("unknown text version")
        if version["review_status"] != "candidate":
            raise ValueError("only candidate text versions can enter this review action")
        connection.execute(
            """
            INSERT INTO evidence.review_decision (
                review_id, target_kind, target_id, decision, reviewer, note,
                previous_value, new_value
            ) VALUES (%s, 'text_version', %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            """,
            (
                review_id,
                text_version_id,
                decision,
                reviewer,
                note,
                json.dumps(dict(version), default=str),
                json.dumps({"review_status": desired}, ensure_ascii=False),
            ),
        )
        if decision != "needs_review":
            connection.execute(
                "UPDATE evidence.text_version SET review_status = %s WHERE text_version_id = %s",
                (desired, text_version_id),
            )
        if decision == "accept":
            connection.execute(
                """
                UPDATE evidence.region_text_selection
                SET superseded_at = now()
                WHERE region_id = %s AND superseded_at IS NULL
                """,
                (version["region_id"],),
            )
            selection_id = connection.execute(
                """
                INSERT INTO evidence.region_text_selection (
                    region_id, text_version_id, review_id, selection_basis,
                    selected_by, note
                ) VALUES (%s, %s, %s, 'historian_approved', %s, %s)
                RETURNING selection_id
                """,
                (
                    version["region_id"],
                    text_version_id,
                    review_id,
                    reviewer,
                    note,
                ),
            ).fetchone()["selection_id"]
            coherent_jobs.enqueue_coherent_jobs(
                connection, created_by=reviewer, max_revisions=100_000
            )
    return TextReviewResult(str(review_id), str(text_version_id), desired, str(selection_id) if selection_id else None)


def create_exact_evidence_span(
    database_url: str,
    *,
    text_version_id: UUID,
    surface: str,
    left_context: str = "",
    right_context: str = "",
    span_role: str = "evidence",
    polygon: dict[str, Any] | None = None,
) -> EvidenceSpanResult:
    """Let code—not a model—locate and persist authoritative offsets."""
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        version = connection.execute(
            """
            SELECT text_content FROM evidence.text_version WHERE text_version_id = %s
            """,
            (text_version_id,),
        ).fetchone()
        if version is None:
            raise ValueError("unknown text version")
        located = locate_unique_surface(
            version["text_content"],
            surface,
            left_context=left_context,
            right_context=right_context,
        )
        span_id = connection.execute(
            """
            INSERT INTO evidence.evidence_span (
                text_version_id, text_start, text_end, surface_text,
                surface_sha256, polygon, span_role
            ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (text_version_id, text_start, text_end, surface_sha256)
            DO UPDATE SET surface_text = EXCLUDED.surface_text
            RETURNING evidence_span_id
            """,
            (
                text_version_id,
                located.text_start,
                located.text_end,
                surface,
                _sha256_text(surface),
                json.dumps(polygon) if polygon is not None else None,
                span_role,
            ),
        ).fetchone()["evidence_span_id"]
    return EvidenceSpanResult(
        str(span_id), str(text_version_id), located.text_start, located.text_end, surface
    )


def propose_event(
    database_url: str,
    *,
    run_id: UUID,
    event_type: str,
    trigger_evidence_span_id: UUID,
    participants: Iterable[EventParticipantSpec],
    evidence: Iterable[EventEvidenceSpec],
    coherent_unit_revision_id: UUID,
    date_start: Any | None = None,
    date_end: Any | None = None,
    date_precision: str | None = None,
    date_uncertainty: str | None = None,
    location_entity_id: UUID | None = None,
    location_literal: str | None = None,
    aspect: str | None = None,
    confidence: float | None = None,
    attributes: dict[str, Any] | None = None,
    event_id: UUID | None = None,
) -> EventResult:
    """Persist a candidate event whose arguments are supplied immutable IDs."""
    participant_list = list(participants)
    evidence_list = list(evidence)
    if not event_type.strip() or not evidence_list:
        raise ValueError("event_type and at least one evidence span are required")
    if trigger_evidence_span_id not in {item.evidence_span_id for item in evidence_list}:
        raise ValueError("trigger span must also be explicit event evidence")
    event_id = event_id or uuid4()
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        run = connection.execute(
            "SELECT status FROM evidence.processing_run WHERE run_id = %s",
            (run_id,),
        ).fetchone()
        if run is None or run["status"] != "completed":
            raise ValueError("event run must be a completed registered semantic run")
        unit = connection.execute(
            """
            SELECT revision_id FROM evidence.coherent_unit_revision
            WHERE revision_id = %s AND superseded_at IS NULL
            """,
            (coherent_unit_revision_id,),
        ).fetchone()
        if unit is None:
            raise ValueError("events require an active reviewed coherent-unit revision")
        event_attributes = {
            **(attributes or {}),
            "coherent_unit_revision_id": str(coherent_unit_revision_id),
        }
        existing = connection.execute(
            """
            SELECT run_id, event_type, trigger_evidence_span_id, event_status
            FROM evidence.event WHERE event_id = %s
            """,
            (event_id,),
        ).fetchone()
        if existing is not None:
            if (
                existing["run_id"] != run_id
                or existing["event_type"] != event_type
                or existing["trigger_evidence_span_id"] != trigger_evidence_span_id
            ):
                raise ValueError("event UUID already has different immutable provenance")
            participant_count = connection.execute(
                "SELECT count(*) AS count FROM evidence.event_participant WHERE event_id = %s",
                (event_id,),
            ).fetchone()["count"]
            evidence_count = connection.execute(
                "SELECT count(*) AS count FROM evidence.event_evidence WHERE event_id = %s",
                (event_id,),
            ).fetchone()["count"]
            return EventResult(
                str(event_id), participant_count, evidence_count, existing["event_status"]
            )
        connection.execute(
            """
            INSERT INTO evidence.event (
                event_id, run_id, event_type, trigger_evidence_span_id,
                date_start, date_end, date_precision, date_uncertainty,
                location_entity_id, location_literal, aspect, event_status,
                confidence, attributes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                'candidate', %s, %s::jsonb
            )
            """,
            (
                event_id,
                run_id,
                event_type,
                trigger_evidence_span_id,
                date_start,
                date_end,
                date_precision,
                date_uncertainty,
                location_entity_id,
                location_literal,
                aspect,
                confidence,
                json.dumps(event_attributes, ensure_ascii=False),
            ),
        )
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO evidence.event_participant (
                    event_id, entity_id, mention_id, participant_role,
                    evidence_span_id, review_status
                ) VALUES (%s, %s, %s, %s, %s, 'candidate')
                """,
                [
                    (
                        event_id,
                        item.entity_id,
                        item.mention_id,
                        item.participant_role,
                        item.evidence_span_id,
                    )
                    for item in participant_list
                ],
            )
            cursor.executemany(
                """
                INSERT INTO evidence.event_evidence (
                    event_id, evidence_span_id, support_role, review_status
                ) VALUES (%s, %s, %s, 'candidate')
                """,
                [
                    (event_id, item.evidence_span_id, item.support_role)
                    for item in evidence_list
                ],
            )
    return EventResult(str(event_id), len(participant_list), len(evidence_list), "candidate")


def review_event(
    database_url: str,
    event_id: UUID,
    *,
    decision: Literal["accept", "reject", "dispute", "needs_review"],
    reviewer: str,
    note: str | None = None,
    review_id: UUID | None = None,
) -> EventResult:
    """Review an event only after exact evidence and resolved participants pass."""
    desired = {
        "accept": "reviewed",
        "reject": "rejected",
        "dispute": "disputed",
        "needs_review": "candidate",
    }[decision]
    review_id = review_id or uuid4()
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        event = connection.execute(
            "SELECT * FROM evidence.event WHERE event_id = %s FOR UPDATE",
            (event_id,),
        ).fetchone()
        if event is None:
            raise ValueError("unknown event")
        if event["event_status"] != "candidate":
            raise ValueError("only candidate events can enter this review action")
        evidence_count = connection.execute(
            """
            SELECT count(*) AS total,
                   count(*) FILTER (WHERE support_role = 'direct_support') AS direct
            FROM evidence.event_evidence WHERE event_id = %s
            """,
            (event_id,),
        ).fetchone()
        participant_count = connection.execute(
            "SELECT count(*) AS total FROM evidence.event_participant WHERE event_id = %s",
            (event_id,),
        ).fetchone()["total"]
        unresolved = connection.execute(
            """
            SELECT count(*) AS count
            FROM evidence.event_participant participant
            LEFT JOIN evidence.mention_resolution resolution
              ON resolution.mention_id = participant.mention_id
             AND resolution.review_status = 'reviewed'
             AND resolution.superseded_at IS NULL
             AND NOT resolution.is_nil
            LEFT JOIN evidence.entity entity
              ON entity.entity_id = COALESCE(
                  participant.entity_id, resolution.proposed_entity_id
              )
            WHERE participant.event_id = %s
              AND (entity.entity_id IS NULL OR entity.entity_status <> 'reviewed')
            """,
            (event_id,),
        ).fetchone()["count"]
        if decision == "accept" and (
            evidence_count["total"] == 0
            or evidence_count["direct"] == 0
            or unresolved
        ):
            raise ValueError(
                "reviewed events require direct evidence and reviewed participant resolutions"
            )
        connection.execute(
            """
            INSERT INTO evidence.review_decision (
                review_id, target_kind, target_id, decision, reviewer, note,
                previous_value, new_value
            ) VALUES (%s, 'event', %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            """,
            (
                review_id,
                event_id,
                decision,
                reviewer,
                note,
                json.dumps(dict(event), default=str, ensure_ascii=False),
                json.dumps({"event_status": desired}, ensure_ascii=False),
            ),
        )
        if decision != "needs_review":
            connection.execute(
                "UPDATE evidence.event SET event_status = %s WHERE event_id = %s",
                (desired, event_id),
            )
            child_status = "reviewed" if decision == "accept" else "rejected"
            connection.execute(
                "UPDATE evidence.event_participant SET review_status = %s WHERE event_id = %s",
                (child_status, event_id),
            )
            connection.execute(
                "UPDATE evidence.event_evidence SET review_status = %s WHERE event_id = %s",
                (child_status, event_id),
            )
    return EventResult(str(event_id), participant_count, evidence_count["total"], desired)


def accept_entity_redirect(
    database_url: str,
    *,
    superseded_entity_id: UUID,
    canonical_entity_id: UUID,
    reviewer: str,
    reason: str,
    review_id: UUID | None = None,
    entity_redirect_id: UUID | None = None,
) -> EntityRedirectResult:
    """Apply one historian-reviewed, reversible merge without rewriting occurrences."""
    if superseded_entity_id == canonical_entity_id:
        raise ValueError("an entity cannot redirect to itself")
    if not reviewer.strip() or not reason.strip():
        raise ValueError("reviewer and reason must not be blank")
    review_id = review_id or uuid4()
    entity_redirect_id = entity_redirect_id or uuid4()
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        entities = connection.execute(
            """
            SELECT entity_id, entity_type, canonical_name, entity_status
            FROM evidence.entity
            WHERE entity_id = ANY(%s::uuid[])
            ORDER BY entity_id FOR UPDATE
            """,
            ([superseded_entity_id, canonical_entity_id],),
        ).fetchall()
        by_id = {row["entity_id"]: row for row in entities}
        if by_id.keys() != {superseded_entity_id, canonical_entity_id}:
            raise ValueError("both redirect entities must exist")
        source = by_id[superseded_entity_id]
        target = by_id[canonical_entity_id]
        if source["entity_status"] != "reviewed" or target["entity_status"] != "reviewed":
            raise ValueError("both redirect entities must be reviewed")
        if source["entity_type"] != target["entity_type"]:
            raise ValueError("entity redirects require the same entity type")
        target_redirect = connection.execute(
            """
            SELECT entity_redirect_id FROM evidence.entity_redirect
            WHERE superseded_entity_id = %s AND reversed_at IS NULL
            """,
            (canonical_entity_id,),
        ).fetchone()
        if target_redirect is not None:
            raise ValueError("canonical target must already be the terminal entity")
        previous = {
            "superseded": dict(source),
            "canonical": dict(target),
        }
        connection.execute(
            """
            INSERT INTO evidence.review_decision (
                review_id, target_kind, target_id, decision, reviewer, note,
                previous_value, new_value
            ) VALUES (
                %s, 'entity_redirect', %s, 'accept', %s, %s,
                %s::jsonb, %s::jsonb
            )
            """,
            (
                review_id,
                entity_redirect_id,
                reviewer,
                reason,
                json.dumps(previous, default=str, ensure_ascii=False),
                json.dumps(
                    {
                        "superseded_entity_id": str(superseded_entity_id),
                        "canonical_entity_id": str(canonical_entity_id),
                    }
                ),
            ),
        )
        connection.execute(
            """
            INSERT INTO evidence.entity_redirect (
                entity_redirect_id, superseded_entity_id,
                canonical_entity_id, review_id, reason
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            (
                entity_redirect_id,
                superseded_entity_id,
                canonical_entity_id,
                review_id,
                reason,
            ),
        )
        connection.execute(
            """
            UPDATE evidence.entity
            SET entity_status = 'merged', updated_at = now()
            WHERE entity_id = %s
            """,
            (superseded_entity_id,),
        )
    return EntityRedirectResult(
        str(entity_redirect_id),
        str(superseded_entity_id),
        str(canonical_entity_id),
        str(review_id),
        True,
    )


def reverse_entity_redirect(
    database_url: str,
    entity_redirect_id: UUID,
    *,
    reviewer: str,
    reason: str,
    review_id: UUID | None = None,
) -> EntityRedirectResult:
    """Reverse a merge while retaining the original redirect and both reviews."""
    if not reviewer.strip() or not reason.strip():
        raise ValueError("reviewer and reason must not be blank")
    review_id = review_id or uuid4()
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        redirect = connection.execute(
            """
            SELECT * FROM evidence.entity_redirect
            WHERE entity_redirect_id = %s FOR UPDATE
            """,
            (entity_redirect_id,),
        ).fetchone()
        if redirect is None or redirect["reversed_at"] is not None:
            raise ValueError("redirect is absent or already reversed")
        connection.execute(
            """
            INSERT INTO evidence.review_decision (
                review_id, target_kind, target_id, decision, reviewer, note,
                previous_value, new_value
            ) VALUES (
                %s, 'entity_redirect', %s, 'supersede', %s, %s,
                %s::jsonb, %s::jsonb
            )
            """,
            (
                review_id,
                entity_redirect_id,
                reviewer,
                reason,
                json.dumps(dict(redirect), default=str, ensure_ascii=False),
                json.dumps({"active": False}),
            ),
        )
        connection.execute(
            """
            UPDATE evidence.entity_redirect
            SET reversed_at = now(), reversal_review_id = %s
            WHERE entity_redirect_id = %s
            """,
            (review_id, entity_redirect_id),
        )
        connection.execute(
            """
            UPDATE evidence.entity
            SET entity_status = 'reviewed', updated_at = now()
            WHERE entity_id = %s
            """,
            (redirect["superseded_entity_id"],),
        )
    return EntityRedirectResult(
        str(entity_redirect_id),
        str(redirect["superseded_entity_id"]),
        str(redirect["canonical_entity_id"]),
        str(review_id),
        False,
    )


ARTICLE_LOCAL_IDENTITY_EVIDENCE_SQL = """
    SELECT cluster.local_cluster_id, run.local_coreference_run_id,
           run.coherent_unit_revision_id,
           mention.mention_id, mention.mention_text,
           span.evidence_span_id, span.text_start, span.text_end,
           span.surface_text, version.text_version_id, version.variant,
           region.region_id, region.polygon,
           layout.layout_region_id, layout.polygon AS layout_polygon,
           page.page_id, page.page_number,
           derivative.derivative_id, derivative.image_uri,
           derivative.image_sha256,
           volume.volume_number, volume.publication_year,
           source.source_object_id, source.source_uri, source.sha256 AS source_sha256
    FROM evidence.local_coreference_run run
    JOIN evidence.local_coreference_cluster cluster
      USING (local_coreference_run_id, coherent_unit_revision_id)
    JOIN evidence.local_coreference_member member
      USING (
          local_cluster_id, local_coreference_run_id,
          coherent_unit_revision_id
      )
    JOIN evidence.entity_mention mention USING (mention_id)
    JOIN evidence.evidence_span span USING (evidence_span_id)
    JOIN evidence.text_version version USING (text_version_id)
    JOIN evidence.ocr_region region USING (region_id)
    LEFT JOIN evidence.layout_region layout USING (layout_region_id)
    JOIN evidence.ocr_run_input input
      ON input.run_id = region.run_id AND input.page_id = region.page_id
    JOIN archive.page_derivative derivative
      ON derivative.derivative_id = input.derivative_id
     AND derivative.page_id = input.page_id
    JOIN archive.page page USING (page_id)
    JOIN archive.volume volume USING (volume_id)
    JOIN archive.source_object source USING (source_object_id)
    WHERE run.local_coreference_run_id = %s
    ORDER BY cluster.local_cluster_id, mention.mention_id
"""


ENTITY_REVERSE_EVIDENCE_SQL = """
    SELECT entity.entity_id, entity.canonical_name,
           mention.mention_id, mention.mention_text,
           span.evidence_span_id, span.surface_text,
           span.text_start, span.text_end, version.text_version_id,
           version.variant, version.text_content,
           region.region_id, region.polygon,
           page.page_id, page.page_number, page.source_image_uri,
           volume.volume_number, volume.publication_year,
           source.source_object_id, source.source_uri
    FROM evidence.mention_resolution resolution
    LEFT JOIN evidence.entity_redirect redirect
      ON redirect.superseded_entity_id = resolution.proposed_entity_id
     AND redirect.reversed_at IS NULL
    JOIN evidence.entity entity
      ON entity.entity_id = COALESCE(
          redirect.canonical_entity_id, resolution.proposed_entity_id
      )
    JOIN evidence.entity_mention mention USING (mention_id)
    JOIN evidence.evidence_span span USING (evidence_span_id)
    JOIN evidence.text_version version USING (text_version_id)
    JOIN evidence.ocr_region region ON region.region_id = version.region_id
    JOIN archive.page page USING (page_id)
    JOIN archive.volume volume USING (volume_id)
    JOIN archive.source_object source USING (source_object_id)
    WHERE entity.entity_id = %s
      AND resolution.review_status = 'reviewed'
      AND resolution.superseded_at IS NULL
      AND mention.mention_status = 'reviewed'
    ORDER BY volume.volume_number, page.page_number,
             region.reading_order, span.text_start, mention.mention_id
"""


EVENT_REVERSE_EVIDENCE_SQL = """
    SELECT event.event_id, event.event_type,
           event_evidence.event_evidence_id, event_evidence.support_role,
           span.evidence_span_id, span.surface_text,
           span.text_start, span.text_end, version.text_version_id,
           version.variant, version.text_content,
           region.region_id, region.polygon,
           page.page_id, page.page_number, page.source_image_uri,
           volume.volume_number, volume.publication_year,
           source.source_object_id, source.source_uri
    FROM evidence.event
    JOIN evidence.event_evidence event_evidence USING (event_id)
    JOIN evidence.evidence_span span USING (evidence_span_id)
    JOIN evidence.text_version version USING (text_version_id)
    JOIN evidence.ocr_region region ON region.region_id = version.region_id
    JOIN archive.page page USING (page_id)
    JOIN archive.volume volume USING (volume_id)
    JOIN archive.source_object source USING (source_object_id)
    WHERE event.event_id = %s
      AND event.event_status = 'reviewed'
      AND event_evidence.review_status = 'reviewed'
    ORDER BY volume.volume_number, page.page_number,
             region.reading_order, span.text_start
"""


def entity_reverse_evidence(database_url: str, entity_id: UUID) -> list[dict[str, Any]]:
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        return [dict(row) for row in connection.execute(ENTITY_REVERSE_EVIDENCE_SQL, (entity_id,))]


def event_reverse_evidence(database_url: str, event_id: UUID) -> list[dict[str, Any]]:
    psycopg, dict_row = _clients()
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        return [dict(row) for row in connection.execute(EVENT_REVERSE_EVIDENCE_SQL, (event_id,))]
