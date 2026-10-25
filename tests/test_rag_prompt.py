from fastapi.testclient import TestClient

from src.graph import build_search_graph
from src.inference import INSUFFICIENT_CONTEXT, build_rag_prompt
from src.main import Resources, create_app
from tests.conftest import FakeLLM

DOCS = [
    {"title": "Install", "source": "install.md", "chunk_index": 0, "text": "Use pip."},
    {"title": "", "source": "https://x.example", "chunk_index": None, "text": "Web text."},
]


def test_prompt_grounds_the_answer_and_asks_for_citations():
    prompt = build_rag_prompt("How to install?", DOCS)
    assert "ONLY the numbered context" in prompt
    assert "[1]" in prompt and "square brackets" in prompt
    assert INSUFFICIENT_CONTEXT in prompt
    assert "[1] Install (install.md, chunk 0)\nUse pip." in prompt
    assert "[2] Untitled (https://x.example)\nWeb text." in prompt
    assert prompt.startswith("<|turn>user\n")
    assert prompt.endswith(
        "Question: How to install?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )


def test_no_documents_answers_insufficient_without_calling_the_llm(seeded_retriever):
    def http_get(*a, **k):
        raise ConnectionError

    llm = FakeLLM(["web"])
    graph = build_search_graph(seeded_retriever, llm, http_get=http_get)
    out = graph.invoke({"query": "q", "routing": "", "documents": [], "response": ""})
    assert out["response"] == INSUFFICIENT_CONTEXT
    assert len(llm.prompts) == 1  # only the router ran


def test_search_returns_numbered_source_metadata(seeded_retriever):
    graph = build_search_graph(seeded_retriever, FakeLLM(["local", "Use pip [1]."]), top_k=2)
    app = create_app(lambda: Resources(retriever=seeded_retriever, graph=graph))
    with TestClient(app) as c:
        body = c.post("/search", json={"query": "pip install"}).json()
    first = body["sources"][0]
    assert [s["n"] for s in body["sources"]] == [1, 2]
    assert first["title"] == "Install" and first["source"] == "install.md"
    assert first["chunk_index"] == 0 and "pip install" in first["text"]
    assert isinstance(first["score"], float)
