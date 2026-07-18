"""Local researcher API for evidence-citing retrieval and model context export."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .embedding_pipeline import BGEEmbedder
from .claim_context import load_reviewed_claim_items
from .evidence import (
    RetrievalResponse,
    ScenarioContextBundle,
    ScenarioEvidenceItem,
)
from .generation import (
    GenerationResponse,
    GenerationTask,
    OpenAICompatibleGenerator,
    TextGenerator,
    generate,
)
from .search import DEFAULT_ALIAS, dense_search, hybrid_search, lexical_search


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["lexical", "dense", "hybrid"] = "hybrid"
    limit: int = Field(default=10, ge=1, le=50)
    year_start: int | None = Field(default=None, ge=1872, le=1949)
    year_end: int | None = Field(default=None, ge=1872, le=1949)


class GenerationRequest(SearchRequest):
    task: GenerationTask = GenerationTask.RESEARCH_BRIEF


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
) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the API extra: uv sync --extra api") from exc

    search_url = opensearch_url or os.environ.get("OPENSEARCH_URL", "http://127.0.0.1:9200")
    db_url = database_url or os.environ.get("DATABASE_URL")
    static_dir = Path(__file__).with_name("static")
    app = FastAPI(title="Women in China 100 Years Ago Research API", version="0.1.0")
    app.state.embedder = None
    app.state.generator = None
    app.state.generator_loaded = False

    def run_search(request: SearchRequest) -> RetrievalResponse:
        if request.year_start and request.year_end and request.year_end < request.year_start:
            raise HTTPException(status_code=422, detail="year_end cannot precede year_start")
        try:
            if request.mode == "lexical":
                return lexical_search(
                    search_url,
                    request.query,
                    index,
                    request.limit,
                    request.year_start,
                    request.year_end,
                )
            if app.state.embedder is None:
                app.state.embedder = embedder_factory()
            if request.mode == "dense":
                return dense_search(
                    search_url,
                    request.query,
                    app.state.embedder,
                    index,
                    request.limit,
                    request.year_start,
                    request.year_end,
                )
            return hybrid_search(
                search_url,
                request.query,
                app.state.embedder,
                index,
                request.limit,
                year_start=request.year_start,
                year_end=request.year_end,
            )
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

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            from opensearchpy import OpenSearch

            search_ok = bool(OpenSearch(hosts=[search_url]).ping())
        except Exception:
            search_ok = False
        return {
            "status": "ok" if search_ok else "degraded",
            "opensearch": search_ok,
            "database_configured": bool(db_url),
            "generation_configured": bool(os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_MODEL")),
            "index": index,
        }

    @app.post("/api/search", response_model=RetrievalResponse)
    def search(request: SearchRequest) -> RetrievalResponse:
        return run_search(request)

    @app.post("/api/context", response_model=ScenarioContextBundle)
    def context(request: SearchRequest) -> ScenarioContextBundle:
        return build_context(run_search(request))

    @app.post("/api/generate", response_model=GenerationResponse)
    def generate_output(request: GenerationRequest) -> GenerationResponse:
        bundle = build_context(run_search(request))
        if request.task == GenerationTask.RECONSTRUCTED_SCENE and not bundle.evidence_items:
            return generate(bundle, request.task, None)
        if not app.state.generator_loaded:
            try:
                app.state.generator = generator_factory()
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Generation configuration invalid: {exc}") from exc
            app.state.generator_loaded = True
        try:
            return generate(bundle, request.task, app.state.generator)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Generation unavailable: {exc}") from exc

    @app.get("/api/page-image/{volume_number}/{page_number}")
    def page_image(volume_number: int, page_number: int) -> Any:
        if not db_url:
            raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")
        try:
            import psycopg

            with psycopg.connect(db_url) as connection:
                row = connection.execute(
                    """
                    SELECT p.source_image_uri
                    FROM archive.page p JOIN archive.volume v USING (volume_id)
                    WHERE v.volume_number = %s AND p.page_number = %s
                    """,
                    (volume_number, page_number),
                ).fetchone()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Page lookup unavailable: {exc}") from exc
        if row is None or not row[0]:
            raise HTTPException(status_code=404, detail="Page image is unavailable")
        path = Path(row[0]).resolve()
        allowed_root = (Path.cwd() / "artifacts" / "benchmark-pages" / "images").resolve()
        if not path.is_relative_to(allowed_root) or not path.is_file():
            raise HTTPException(status_code=404, detail="Local page derivative is unavailable")
        return FileResponse(path)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the API extra: uv sync --extra api") from exc
    uvicorn.run(
        create_app(args.opensearch_url, args.database_url, args.index),
        host=args.host,
        port=args.port,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
