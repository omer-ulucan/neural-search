import logging
import re
from collections.abc import Callable
from typing import TypedDict

import requests
from langgraph.graph import END, StateGraph

from src.config import settings
from src.inference import build_rag_prompt
from src.retrieval import Retriever

logger = logging.getLogger(__name__)

ROUTES = ("local", "web")


class AgentState(TypedDict):
    """State schema for the LangGraph search workflow."""

    query: str
    routing: str
    documents: list[dict]
    response: str


def router_prompt(query: str) -> str:
    """Build the Gemma chat prompt that asks the LLM for a single routing word."""
    return (
        f"<start_of_turn>user\n"
        f"You are a routing assistant. Respond with ONLY the single word 'local' or 'web'.\n"
        "local = queries about proprietary/internal code, architecture, or docs in a "
        "private knowledge base.\n"
        "web = queries about open-source frameworks, current events, error messages, "
        "general technical knowledge.\n\n"
        f"Query: {query}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def parse_route(raw: str) -> str:
    """Extract 'local' or 'web' from raw router output; anything else means 'local'."""
    text = raw.strip().lower()
    # Gemma 4 may prefix a thinking channel, e.g. "<|channel>thought\n<channel|>web".
    if "<channel|>" in text:
        text = text.split("<channel|>")[-1]
    words = re.findall(r"[a-z]+", text)
    decision = words[0] if words else ""
    if decision not in ROUTES:
        logger.warning("Unexpected router decision %r; defaulting to local", raw[:80])
        return "local"
    return decision


def build_search_graph(
    retriever: Retriever,
    llm,
    *,
    top_k: int | None = None,
    searxng_host: str | None = None,
    http_get: Callable = requests.get,
):
    """Compile the router -> (local | web) search -> generator workflow."""
    top_k = top_k or settings.TOP_K
    searxng_host = searxng_host or settings.SEARXNG_HOST

    def router_node(state: AgentState) -> AgentState:
        """Classify query as local or web using the LLM."""
        try:
            raw = llm.generate(router_prompt(state["query"]), temperature=0.1, max_tokens=10)
        except Exception:
            logger.warning("Router LLM call failed; defaulting to local")
            return {**state, "routing": "local"}
        return {**state, "routing": parse_route(raw)}

    def local_search_node(state: AgentState) -> AgentState:
        """Retrieve top chunks with hybrid dense + sparse weighted RRF."""
        result = retriever.retrieve(state["query"], "hybrid", top_k)
        docs = [
            {
                "title": h.payload.get("title", ""),
                "source": h.payload.get("source", ""),
                "chunk_index": h.payload.get("chunk_index"),
                "text": h.payload.get("text", ""),
                "score": h.score,
            }
            for h in result.fused
        ]
        return {**state, "documents": docs}

    def web_search_node(state: AgentState) -> AgentState:
        """Fallback to SearXNG web search with graceful error handling."""
        try:
            resp = http_get(
                f"{searxng_host}/search",
                params={
                    "q": state["query"],
                    "format": "json",
                    "engines": "google,bing",
                    "language": "en",
                },
                timeout=10,
            )
            resp.raise_for_status()
            raw_results = resp.json().get("results", [])[:3]
            docs = [
                {
                    "title": r.get("title", ""),
                    "source": r.get("url", ""),
                    "chunk_index": None,
                    "text": f"{r.get('title', '')}: {r.get('content', '')}",
                    "score": None,
                }
                for r in raw_results
            ]
        except Exception:
            logger.warning("Web search failed; returning empty documents")
            return {**state, "documents": []}
        return {**state, "documents": docs}

    def generator_node(state: AgentState) -> AgentState:
        """Generate an answer using retrieved documents and the local LLM."""
        prompt = build_rag_prompt(state["query"], [d["text"] for d in state["documents"]])
        return {**state, "response": llm.generate(prompt)}

    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("local_search", local_search_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generator", generator_node)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["routing"],
        {"local": "local_search", "web": "web_search"},
    )
    workflow.add_edge("local_search", "generator")
    workflow.add_edge("web_search", "generator")
    workflow.add_edge("generator", END)
    return workflow.compile()
