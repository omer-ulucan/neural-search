from src.graph import build_search_graph
from tests.conftest import FakeLLM

EMPTY = {"query": "", "routing": "", "documents": [], "response": ""}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def run(graph, query):
    return graph.invoke({**EMPTY, "query": query})


def test_local_route_retrieves_chunks_and_generates(seeded_retriever):
    llm = FakeLLM(["local", "Use pip [1]."])
    out = run(build_search_graph(seeded_retriever, llm, top_k=2), "pip install")
    assert out["routing"] == "local" and out["response"] == "Use pip [1]."
    assert len(out["documents"]) == 2
    assert out["documents"][0]["source"] == "install.md"
    assert "pip install" in llm.prompts[-1]


def test_web_route_uses_searxng_results(seeded_retriever):
    def http_get(url, params, timeout):
        assert url.endswith("/search") and params["format"] == "json"
        return FakeResponse({"results": [{"title": "T", "content": "C", "url": "https://x"}]})

    llm = FakeLLM(["web", "answer"])
    out = run(build_search_graph(seeded_retriever, llm, http_get=http_get), "latest news")
    assert out["routing"] == "web"
    assert out["documents"] == [
        {"title": "T", "source": "https://x", "chunk_index": None, "text": "T: C", "score": None}
    ]


def test_web_search_failure_yields_no_documents(seeded_retriever):
    def http_get(*a, **k):
        raise ConnectionError("down")

    out = run(build_search_graph(seeded_retriever, FakeLLM(["web", "x"]), http_get=http_get), "q")
    assert out["documents"] == []


def test_router_exception_defaults_to_local(seeded_retriever):
    class BrokenRouterLLM(FakeLLM):
        def generate(self, prompt, temperature=0.2, max_tokens=1024):
            if max_tokens == 10:
                raise RuntimeError("boom")
            return "fine"

    out = run(build_search_graph(seeded_retriever, BrokenRouterLLM()), "pip")
    assert out["routing"] == "local" and out["response"] == "fine"
