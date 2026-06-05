from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.database import init_qdrant
from src.graph import search_graph


class SearchQuery(BaseModel):
    """Request model for the search endpoint."""

    query: str


class SearchResponse(BaseModel):
    """Response model for the search endpoint."""

    answer: str
    routing: str
    sources: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Qdrant client on application startup."""
    import src.graph as graph_module

    graph_module.qdrant_client = init_qdrant()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchQuery) -> SearchResponse:
    """Run the full LangGraph search pipeline and return the answer."""
    result = search_graph.invoke(
        {
            "query": req.query,
            "routing": "",
            "documents": [],
            "response": "",
        }
    )
    return SearchResponse(
        answer=result["response"],
        routing=result["routing"],
        sources=result["documents"],
    )
