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


embedder = LocalHybridEmbedder()
llm = LocalLLM()
qdrant_client: QdrantClient = None


def router_node(state: AgentState) -> AgentState:
    """Classify query as local or web using the LLM."""
    prompt = (
        f"<|im_start|>system\n"
        f"You are a routing assistant. Respond with ONLY the single word 'local' or 'web'.\n"
        f"local = queries about proprietary/internal code, architecture, or docs in a private knowledge base.\n"
        f"web = queries about open-source frameworks, current events, error messages, general technical knowledge.\n"
        f"<|im_start|>user\n{state['query']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    try:
        decision = llm.generate(prompt, temperature=0.1, max_tokens=10).strip().lower()
    except Exception:
        logger.warning("Router LLM call failed; defaulting to local")
        decision = "local"
    if decision not in {"local", "web"}:
        logger.warning("Unexpected router decision: %s; defaulting to local", decision)
        decision = "local"
    return {**state, "routing": decision}
