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


app = FastAPI()
