import logging
from typing import TypedDict

import requests
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


def local_search_node(state: AgentState) -> AgentState:
    """Retrieve top documents using hybrid dense + sparse RRF over Qdrant."""
    dense_vec = embedder.get_dense(state["query"])
    sparse_vec = embedder.get_sparse(state["query"])
    results = qdrant_client.query_points(
        collection_name=settings.COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec, using="", limit=10),
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(sparse_vec.keys()),
                    values=list(sparse_vec.values()),
                ),
                using="lexical-sparse",
                limit=10,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5,
    )
    texts = [r.payload["text"] for r in results.points]
    return {**state, "documents": texts}


def web_search_node(state: AgentState) -> AgentState:
    """Fallback to SearXNG web search with graceful error handling."""
    try:
        resp = requests.get(
            f"{settings.SEARXNG_HOST}/search",
            params={
                "q": state["query"],
                "format": "json",
                "engines": "google,bing",
                "language": "en",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_results = data.get("results", [])[:3]
        snippets = [f"{r['title']}: {r['content']}" for r in raw_results]
    except Exception:
        logger.warning("Web search failed; returning empty documents")
        return {**state, "documents": []}
    return {**state, "documents": snippets}


def generator_node(state: AgentState) -> AgentState:
    """Generate an answer using retrieved documents and the local LLM."""
    prompt = llm.build_rag_prompt(state["query"], state["documents"])
    answer = llm.generate(prompt)
    return {**state, "response": answer}
