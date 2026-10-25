import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import settings
from src.graph import build_search_graph
from src.retrieval import Retriever

logger = logging.getLogger(__name__)


class SearchQuery(BaseModel):
    """Request model for the search endpoint."""

    query: str


class SearchResponse(BaseModel):
    """Response model for the search endpoint."""

    answer: str
    routing: str
    sources: list[str]


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
            sources=[d["text"] for d in result["documents"]],
        )

    @app.get("/health")
    async def health() -> dict:
        """Return the service health status."""
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
