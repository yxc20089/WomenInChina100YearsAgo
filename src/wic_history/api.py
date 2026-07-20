"""Local researcher API for evidence-citing retrieval and model context export."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, assert_never
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .embedding_pipeline import BGEEmbedder
from .claim_context import load_reviewed_claim_items
from .coherent_api_search import (
    IncompleteCoherentEmbeddingIdentityError,
    run_coherent_api_search,
)
from .evidence import (
    RetrievalResponse,
    ScenarioContextBundle,
    ScenarioEvidenceItem,
)
from .generation import (
    ChatTurn,
    GenerationResponse,
    GenerationTask,
    OpenAICompatibleGenerator,
    TextGenerator,
    generate,
    has_direct_evidence,
)
from .exploration import ExplorationReport, build_exploration_report
from .insights import InsightReport, build_insight_report
from .ingestion_jobs import batch_failures, batch_status
from .search import DEFAULT_ALIAS, dense_search, hybrid_search, lexical_search
from .segmentation_review import (
    SegmentationActivationRequest,
    SegmentationActivationResultView,
    SegmentationDetailResponse,
    SegmentationImportRequest,
    SegmentationProposalResultView,
    SegmentationQueueResponse,
    SegmentationReviewRequest,
    SegmentationReviewResultView,
    activate_reviewed_segmentation,
    import_segmentation_edit,
    list_segmentation_queue,
    record_segmentation_review,
    segmentation_detail,
)
from .review_workflow import (
    ClaimQueueResponse,
    ClaimReviewRequest,
    ClaimReviewResult,
    EntityResolutionRequest,
    MentionQueueResponse,
    MentionReviewRequest,
    ReviewConflictError,
    ReviewNotFoundError,
    ReviewResult,
    list_claim_queue,
    list_mention_queue,
    resolve_entity,
    review_claim,
    review_mention,
)


PAGE_IMAGE_ROOTS = (
    Path("artifacts"),
)


def resolve_local_page_image(image_uri: str, workspace_root: Path) -> Path:
    candidate = Path(image_uri)
    path = (candidate if candidate.is_absolute() else workspace_root / candidate).resolve()
    allowed_roots = [(workspace_root / root).resolve() for root in PAGE_IMAGE_ROOTS]
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    if (
        not any(path.is_relative_to(root) for root in allowed_roots)
        or path.suffix.lower() not in allowed_suffixes
        or not path.is_file()
    ):
        raise ValueError("local page derivative is outside the controlled image roots")
    return path


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["lexical", "dense", "hybrid"] = "hybrid"
    corpus: Literal["region", "reviewed_coherent_unit"] = "region"
    limit: int = Field(default=10, ge=1, le=50)
    year_start: int | None = Field(default=None, ge=1872, le=1949)
    year_end: int | None = Field(default=None, ge=1872, le=1949)


class GenerationRequest(SearchRequest):
    task: GenerationTask = GenerationTask.RESEARCH_BRIEF


class ChatRequest(SearchRequest):
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)


class PageDerivativeView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    derivative_id: UUID
    image_sha256: str
    width: int
    height: int
    dpi: int | None = None
    media_type: str
    evidence_tier: str
    render_manifest_uri: str | None = None
    preferred: bool


class PageDerivativeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    volume_number: int
    page_number: int
    items: list[PageDerivativeView]


class IngestionBatchStatusView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_id: UUID
    name: str
    status: Literal["active", "completed", "failed", "cancelled"]
    total_jobs: int
    ready_jobs: int
    blocked_jobs: int
    dead_letter_jobs: int
    by_status: dict[str, int]
    by_stage: dict[str, dict[str, int]]


class IngestionFailureView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: UUID
    stage: str
    volume_number: int | None
    page_number: int | None
    attempt_count: int
    max_attempts: int
    error_details: dict[str, Any] | None
    completed_at: str | None


def scenario_context(
    response: RetrievalResponse,
    reviewed_items: list[ScenarioEvidenceItem] | None = None,
) -> ScenarioContextBundle:
    """Prepare a safe handoff; only reviewed claims become evidence statements."""
    claim_ids = {claim_id for hit in response.hits for claim_id in hit.claim_ids}
    evidence_items = reviewed_items or []
    warnings = list(response.warnings)
    if not claim_ids:
        warnings.append(
            "No reviewed claims are present in these results. Retrieved OCR may be used for research, "
            "but a model must not present it as a verified historical claim."
        )
    elif not evidence_items:
        warnings.append(
            "Retrieved claim identifiers could not be resolved into reviewed, cited claims; "
            "they are excluded from model evidence."
        )
    return ScenarioContextBundle(
        research_query=response.query,
        evidence_items=evidence_items,
        retrieved_context=response.hits,
        warnings=warnings,
    )


def create_app(
    opensearch_url: str | None = None,
    database_url: str | None = None,
    index: str = DEFAULT_ALIAS,
    embedder_factory: Callable[[], BGEEmbedder] = BGEEmbedder,
    generator_factory: Callable[[], TextGenerator | None] = OpenAICompatibleGenerator.from_environment,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
) -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the API extra: uv sync --extra api") from exc

    search_url = opensearch_url or os.environ.get("OPENSEARCH_URL", "http://127.0.0.1:9200")
    db_url = database_url or os.environ.get("DATABASE_URL")
    graph_uri = neo4j_uri or os.environ.get("NEO4J_URI")
    graph_user = neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    graph_password = neo4j_password or os.environ.get("NEO4J_PASSWORD")
    static_dir = Path(__file__).with_name("static")
    app = FastAPI(title="Women in China 100 Years Ago Research API", version="0.1.0")
    app.state.embedder = None
    app.state.coherent_embedder = None
    app.state.generator = None
    app.state.generator_loaded = False

    @app.middleware("http")
    async def cap_segmentation_mutations(request: Request, call_next: Any) -> Any:
        if (
            request.method == "POST"
            and request.url.path.startswith("/api/review/segmentation")
        ):
            maximum = 5 * 1024 * 1024
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > maximum:
                return JSONResponse(
                    {"detail": "segmentation request exceeds 5 MiB"}, status_code=413
                )
            if not content_length and len(await request.body()) > maximum:
                return JSONResponse(
                    {"detail": "segmentation request exceeds 5 MiB"}, status_code=413
                )
        return await call_next(request)

    def run_search(request: SearchRequest) -> RetrievalResponse:
        if request.year_start and request.year_end and request.year_end < request.year_start:
            raise HTTPException(status_code=422, detail="year_end cannot precede year_start")
        try:
            match request.corpus, request.mode:
                case "region", "lexical":
                    return lexical_search(
                        search_url,
                        request.query,
                        index,
                        request.limit,
                        request.year_start,
                        request.year_end,
                    )
                case "region", "dense":
                    if app.state.embedder is None:
                        app.state.embedder = embedder_factory()
                    return dense_search(
                        search_url,
                        request.query,
                        app.state.embedder,
                        index,
                        request.limit,
                        request.year_start,
                        request.year_end,
                    )
                case "region", "hybrid":
                    if app.state.embedder is None:
                        app.state.embedder = embedder_factory()
                    return hybrid_search(
                        search_url,
                        request.query,
                        app.state.embedder,
                        index,
                        request.limit,
                        year_start=request.year_start,
                        year_end=request.year_end,
                    )
                case "reviewed_coherent_unit", _:
                    result = run_coherent_api_search(
                        search_url, request, app.state.coherent_embedder
                    )
                    app.state.coherent_embedder = result.embedder
                    return result.response
                case unreachable:
                    assert_never(unreachable)
        except IncompleteCoherentEmbeddingIdentityError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Retrieval unavailable: {exc}") from exc

    def build_context(response: RetrievalResponse) -> ScenarioContextBundle:
        claim_ids = {claim_id for hit in response.hits for claim_id in hit.claim_ids}
        if not claim_ids:
            return scenario_context(response)
        if not db_url:
            return scenario_context(response)
        try:
            return scenario_context(response, load_reviewed_claim_items(db_url, claim_ids))
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Reviewed claim context unavailable: {exc}"
            ) from exc

    def require_region(request: SearchRequest) -> None:
        if request.corpus != "region":
            raise HTTPException(
                status_code=422,
                detail="Context and generation endpoints currently require corpus=region",
            )

    def require_database() -> str:
        if not db_url:
            raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")
        return db_url

    def run_generation(
        bundle: ScenarioContextBundle,
        task: GenerationTask,
        history: Sequence[ChatTurn] = (),
    ) -> GenerationResponse:
        if not bundle.retrieved_context or (
            task == GenerationTask.RECONSTRUCTED_SCENE
            and not has_direct_evidence(bundle)
        ):
            return generate(bundle, task, None, history)
        if not app.state.generator_loaded:
            try:
                app.state.generator = generator_factory()
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail=f"Generation configuration invalid: {exc}"
                ) from exc
            app.state.generator_loaded = True
        try:
            return generate(bundle, task, app.state.generator, history)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Generation unavailable: {exc}") from exc

    def review_error(exc: Exception) -> None:
        if isinstance(exc, ReviewNotFoundError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, ReviewConflictError):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=503, detail=f"Review workflow unavailable: {exc}") from exc

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            from opensearchpy import OpenSearch

            search_ok = bool(OpenSearch(hosts=[search_url]).ping())
        except Exception:
            search_ok = False
        generation_error = None
        try:
            generation_configured = (
                OpenAICompatibleGenerator.from_environment() is not None
            )
        except Exception as exc:
            generation_configured = False
            generation_error = str(exc)
        return {
            "status": "ok" if search_ok else "degraded",
            "opensearch": search_ok,
            "database_configured": bool(db_url),
            "neo4j_configured": bool(graph_uri and graph_password),
            "generation_configured": generation_configured,
            "generation_configuration_error": generation_error,
            "index": index,
        }

    @app.get(
        "/api/ingestion/batches/{batch_id}",
        response_model=IngestionBatchStatusView,
    )
    def ingestion_batch(batch_id: UUID) -> IngestionBatchStatusView:
        try:
            status = batch_status(require_database(), batch_id)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Ingestion status unavailable: {exc}"
            ) from exc
        return IngestionBatchStatusView.model_validate(asdict(status))

    @app.get(
        "/api/ingestion/batches/{batch_id}/failures",
        response_model=list[IngestionFailureView],
    )
    def ingestion_failures(batch_id: UUID) -> list[IngestionFailureView]:
        try:
            failures = batch_failures(require_database(), batch_id)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Ingestion failures unavailable: {exc}"
            ) from exc
        return [IngestionFailureView.model_validate(asdict(item)) for item in failures]

    @app.post("/api/search", response_model=RetrievalResponse)
    def search(request: SearchRequest) -> RetrievalResponse:
        return run_search(request)

    @app.post("/api/context", response_model=ScenarioContextBundle)
    def context(request: SearchRequest) -> ScenarioContextBundle:
        require_region(request)
        return build_context(run_search(request))

    @app.post("/api/generate", response_model=GenerationResponse)
    def generate_output(request: GenerationRequest) -> GenerationResponse:
        require_region(request)
        bundle = build_context(run_search(request))
        return run_generation(bundle, request.task)

    @app.post("/api/chat", response_model=GenerationResponse)
    def chat(request: ChatRequest) -> GenerationResponse:
        require_region(request)
        bundle = build_context(run_search(request))
        return run_generation(bundle, GenerationTask.CHAT_ANSWER, request.history)

    @app.get("/api/review/mentions", response_model=MentionQueueResponse)
    def mention_queue(
        status: Literal["candidate", "reviewed", "rejected"] = "candidate",
        limit: int = 25,
        offset: int = 0,
        model_name: str | None = None,
        dataset_id: str | None = None,
        ner_run_id: UUID | None = None,
        source_ocr_run_id: UUID | None = None,
    ) -> MentionQueueResponse:
        if not 1 <= limit <= 100 or offset < 0:
            raise HTTPException(status_code=422, detail="limit must be 1–100 and offset nonnegative")
        try:
            return list_mention_queue(
                require_database(),
                status,
                limit=limit,
                offset=offset,
                model_name=model_name,
                dataset_id=dataset_id,
                ner_run_id=ner_run_id,
                source_ocr_run_id=source_ocr_run_id,
            )
        except HTTPException:
            raise
        except Exception as exc:
            review_error(exc)

    @app.post("/api/review/mentions/{mention_id}", response_model=ReviewResult)
    def decide_mention(mention_id: UUID, request: MentionReviewRequest) -> ReviewResult:
        try:
            return review_mention(require_database(), mention_id, request)
        except HTTPException:
            raise
        except Exception as exc:
            review_error(exc)

    @app.post(
        "/api/review/mentions/{mention_id}/entity-resolution",
        response_model=ReviewResult,
    )
    def decide_entity(
        mention_id: UUID, request: EntityResolutionRequest
    ) -> ReviewResult:
        try:
            return resolve_entity(require_database(), mention_id, request)
        except HTTPException:
            raise
        except Exception as exc:
            review_error(exc)

    @app.get("/api/review/claims", response_model=ClaimQueueResponse)
    def claim_queue(
        status: Literal["candidate", "reviewed", "disputed", "rejected", "superseded"] = "candidate",
        limit: int = 25,
        offset: int = 0,
        model_name: str | None = None,
    ) -> ClaimQueueResponse:
        if not 1 <= limit <= 100 or offset < 0:
            raise HTTPException(status_code=422, detail="limit must be 1–100 and offset nonnegative")
        try:
            return list_claim_queue(
                require_database(),
                status,
                limit=limit,
                offset=offset,
                model_name=model_name,
            )
        except HTTPException:
            raise
        except Exception as exc:
            review_error(exc)

    @app.post("/api/review/claims/{claim_id}", response_model=ClaimReviewResult)
    def decide_claim(claim_id: UUID, request: ClaimReviewRequest) -> ClaimReviewResult:
        try:
            return review_claim(require_database(), claim_id, request)
        except HTTPException:
            raise
        except Exception as exc:
            review_error(exc)

    @app.get("/api/review/segmentations", response_model=SegmentationQueueResponse)
    def segmentation_queue(limit: int = 25, offset: int = 0) -> SegmentationQueueResponse:
        try:
            return list_segmentation_queue(
                require_database(), limit=limit, offset=offset
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Segmentation queue unavailable: {exc}"
            ) from exc

    @app.get(
        "/api/review/segmentations/{run_id}",
        response_model=SegmentationDetailResponse,
    )
    def segmentation_proposal(run_id: UUID) -> SegmentationDetailResponse:
        try:
            return segmentation_detail(require_database(), run_id)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Segmentation detail unavailable: {exc}"
            ) from exc

    @app.post(
        "/api/review/segmentation-imports",
        response_model=SegmentationProposalResultView,
    )
    def import_segmentation(
        request: SegmentationImportRequest,
    ) -> SegmentationProposalResultView:
        try:
            return SegmentationProposalResultView.model_validate(
                asdict(import_segmentation_edit(require_database(), request))
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Segmentation import unavailable: {exc}"
            ) from exc

    @app.post(
        "/api/review/segmentations/{run_id}/reviews",
        response_model=SegmentationReviewResultView,
    )
    def decide_segmentation(
        run_id: UUID, request: SegmentationReviewRequest
    ) -> SegmentationReviewResultView:
        try:
            if request.decision == "accept":
                detail = segmentation_detail(require_database(), run_id)
                if not detail.reviewable:
                    raise ValueError(
                        "segmentation is not reviewable: "
                        + "; ".join(detail.review_blockers)
                    )
            return SegmentationReviewResultView.model_validate(
                asdict(record_segmentation_review(require_database(), run_id, request))
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Segmentation review unavailable: {exc}"
            ) from exc

    @app.post(
        "/api/review/segmentation-reviews/{review_id}/activate",
        response_model=SegmentationActivationResultView,
    )
    def activate_segmentation_review(
        review_id: UUID, request: SegmentationActivationRequest
    ) -> SegmentationActivationResultView:
        try:
            return SegmentationActivationResultView.model_validate(
                asdict(
                    activate_reviewed_segmentation(
                        require_database(), review_id, request
                    )
                )
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Segmentation activation unavailable: {exc}"
            ) from exc

    @app.get("/api/insights", response_model=InsightReport)
    def insights() -> InsightReport:
        try:
            return build_insight_report(
                require_database(),
                neo4j_uri=graph_uri,
                neo4j_user=graph_user,
                neo4j_password=graph_password,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Insight analysis unavailable: {exc}") from exc

    @app.get("/api/exploration", response_model=ExplorationReport)
    def exploration(examples_per_theme: int = 3) -> ExplorationReport:
        if not 1 <= examples_per_theme <= 10:
            raise HTTPException(
                status_code=422, detail="examples_per_theme must be between 1 and 10"
            )
        try:
            return build_exploration_report(
                require_database(), examples_per_theme=examples_per_theme
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Exploratory analysis unavailable: {exc}"
            ) from exc

    @app.get("/api/page-image/{volume_number}/{page_number}")
    def page_image(
        volume_number: int,
        page_number: int,
        derivative_id: UUID | None = None,
    ) -> Any:
        require_database()
        try:
            import psycopg

            with psycopg.connect(db_url) as connection:
                row = connection.execute(
                    """
                    SELECT d.derivative_id, d.image_uri, d.image_sha256,
                           d.evidence_tier
                    FROM archive.page p
                    JOIN archive.volume v USING (volume_id)
                    JOIN archive.page_derivative d
                      ON d.derivative_id = COALESCE(%s, p.preferred_derivative_id)
                     AND d.page_id = p.page_id
                    WHERE v.volume_number = %s AND p.page_number = %s
                    """,
                    (derivative_id, volume_number, page_number),
                ).fetchone()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Page lookup unavailable: {exc}") from exc
        if row is None:
            raise HTTPException(status_code=404, detail="Page image is unavailable")
        try:
            path = resolve_local_page_image(row[1], Path.cwd())
        except ValueError:
            raise HTTPException(status_code=404, detail="Local page derivative is unavailable")
        return FileResponse(
            path,
            headers={
                "X-WIC-Derivative-ID": str(row[0]),
                "X-WIC-Image-SHA256": row[2],
                "X-WIC-Evidence-Tier": row[3],
            },
        )

    @app.get(
        "/api/pages/{volume_number}/{page_number}/derivatives",
        response_model=PageDerivativeResponse,
    )
    def page_derivatives(
        volume_number: int, page_number: int
    ) -> PageDerivativeResponse:
        try:
            import psycopg
            from psycopg.rows import dict_row

            with psycopg.connect(require_database(), row_factory=dict_row) as connection:
                rows = connection.execute(
                    """
                    SELECT d.derivative_id, d.image_sha256, d.width, d.height,
                           d.dpi, d.media_type, d.evidence_tier,
                           d.render_manifest_uri,
                           d.derivative_id = p.preferred_derivative_id AS preferred
                    FROM archive.page p
                    JOIN archive.volume v USING (volume_id)
                    JOIN archive.page_derivative d USING (page_id)
                    WHERE v.volume_number = %s AND p.page_number = %s
                    ORDER BY d.preference_rank DESC, d.width DESC, d.height DESC,
                             d.created_at, d.derivative_id
                    """,
                    (volume_number, page_number),
                ).fetchall()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Page derivative lookup unavailable: {exc}"
            ) from exc
        if not rows:
            raise HTTPException(status_code=404, detail="Page derivatives are unavailable")
        return PageDerivativeResponse(
            volume_number=volume_number,
            page_number=page_number,
            items=[PageDerivativeView.model_validate(row) for row in rows],
        )

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def home() -> Any:
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--opensearch-url", default=os.environ.get("OPENSEARCH_URL"))
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--index", default=DEFAULT_ALIAS)
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the API extra: uv sync --extra api") from exc
    uvicorn.run(
        create_app(
            args.opensearch_url,
            args.database_url,
            args.index,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
        ),
        host=args.host,
        port=args.port,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
