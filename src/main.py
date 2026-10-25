import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import settings
from src.graph import build_search_graph
from src.retrieval import Hit, Mode, RetrievalResult, Retriever

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 280


class SearchQuery(BaseModel):
    """Request model for the search endpoint."""

    query: str


class Source(BaseModel):
    """A passage given to the LLM; `n` is the number the answer cites as [n]."""

    n: int
    title: str
    source: str
    chunk_index: int | None
    text: str
    score: float | None


class SearchResponse(BaseModel):
    """Response model for the search endpoint."""

    answer: str
    routing: str
    sources: list[Source]


class RetrieveRequest(BaseModel):
    """Request model for the retrieval-only endpoint."""

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(10, ge=1, le=50)
    rrf_k: int | None = Field(None, ge=0, le=1000, description="Defaults to RRF_K")


class RetrievedItem(BaseModel):
    """One ranked chunk."""

    rank: int
    score: float
    id: str
    title: str
    source: str
    chunk_index: int | None
    snippet: str
    dense_rank: int | None = None
    sparse_rank: int | None = None


class FusionWeights(BaseModel):
    """RRF weights used for the fused list, with the reasons that produced them."""

    dense: float
    sparse: float
    reasons: list[str]


class RetrieveResponse(BaseModel):
    """Per-retriever rankings plus the weighted RRF fusion of both."""

    query: str
    top_k: int
    rrf_k: int
    weights: FusionWeights
    sparse_fallback: bool
    sparse: list[RetrievedItem]
    dense: list[RetrievedItem]
    fused: list[RetrievedItem]
    timing_ms: dict[str, float]


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Collapse whitespace and cut text to at most `limit` characters on a word boundary."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def to_item(hit: Hit) -> RetrievedItem:
    p = hit.payload
    return RetrievedItem(
        rank=hit.rank,
        score=hit.score,
        id=hit.id,
        title=p.get("title", ""),
        source=p.get("source", ""),
        chunk_index=p.get("chunk_index"),
        snippet=snippet(p.get("text", "")),
        dense_rank=hit.dense_rank,
        sparse_rank=hit.sparse_rank,
    )


def to_response(query: str, res: RetrievalResult) -> RetrieveResponse:
    return RetrieveResponse(
        query=query,
        top_k=res.top_k,
        rrf_k=res.rrf_k,
        weights=FusionWeights(
            dense=res.weights.dense, sparse=res.weights.sparse, reasons=list(res.weights.reasons)
        ),
        sparse_fallback=res.sparse_fallback,
        sparse=[to_item(h) for h in res.sparse],
        dense=[to_item(h) for h in res.dense],
        fused=[to_item(h) for h in res.fused],
        timing_ms={k: round(v, 2) for k, v in res.timings_ms.items()},
    )


@dataclass
class Resources:
    """Everything the endpoints need, built once at startup."""

    retriever: Retriever
    graph: Any | None = None  # None when no LLM is configured; /search is then disabled


def load_resources() -> Resources:
    """Load Qdrant, embedding models and (if MODEL_PATH is set) the LLM."""
    from src.database import init_qdrant
    from src.embedder import LocalHybridEmbedder
    from src.inference import LocalLLM
    from src.weighting import QueryWeighting

    embedder = LocalHybridEmbedder()
    client = init_qdrant(dense_dim=embedder.dense_dim)
    weighting = QueryWeighting(settings.DYNAMIC_STRENGTH) if settings.DYNAMIC_WEIGHTS else None
    retriever = Retriever(client, embedder, weighting=weighting)
    if not settings.MODEL_PATH:
        logger.warning("MODEL_PATH not set: /search is disabled, /retrieve still works")
        return Resources(retriever=retriever)
    return Resources(retriever=retriever, graph=build_search_graph(retriever, LocalLLM()))


def create_app(resource_factory: Callable[[], Resources] = load_resources) -> FastAPI:
    """Build the FastAPI app; models load in the lifespan, never at import time."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.resources = resource_factory()
        yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.mount("/static", StaticFiles(directory="src/static"), name="static")

    @app.get("/")
    async def serve_ui() -> FileResponse:
        """Serve the main search UI."""
        return FileResponse("src/static/index.html")

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchQuery, request: Request) -> SearchResponse:
        """Run the full LangGraph search pipeline and return the answer."""
        graph = request.app.state.resources.graph
        if graph is None:
            raise HTTPException(status_code=503, detail="No LLM configured (set MODEL_PATH)")
        # The graph is synchronous (embedding, Qdrant, llama.cpp); keep it off the event loop.
        result = await run_in_threadpool(
            graph.invoke,
            {
                "query": req.query,
                "routing": "",
                "documents": [],
                "response": "",
            },
        )
        return SearchResponse(
            answer=result["response"],
            routing=result["routing"],
            sources=[Source(n=i + 1, **d) for i, d in enumerate(result["documents"])],
        )

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(req: RetrieveRequest, request: Request) -> RetrieveResponse:
        """Hybrid retrieval only (no router, no LLM): sparse, dense and fused rankings."""
        retriever: Retriever = request.app.state.resources.retriever
        mode: Mode = "hybrid"
        res = await run_in_threadpool(
            retriever.retrieve, req.query, mode, req.top_k, rrf_k=req.rrf_k
        )
        return to_response(req.query, res)

    @app.get("/health")
    async def health() -> dict:
        """Return the service health status."""
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
