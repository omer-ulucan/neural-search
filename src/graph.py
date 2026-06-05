import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient, models

from src.config import settings
from src.embedder import LocalHybridEmbedder
from src.inference import LocalLLM

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for the LangGraph search workflow."""

    query: str
    routing: str
    documents: list[str]
    response: str
