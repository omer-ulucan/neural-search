import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.graph import build_search_graph
from src.main import Resources, create_app
from tests.conftest import FakeLLM

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client_factory(seeded_retriever):
    def make(llm=None):
        graph = build_search_graph(seeded_retriever, llm) if llm else None
        app = create_app(lambda: Resources(retriever=seeded_retriever, graph=graph))
        return TestClient(app)

    return make


def test_importing_the_app_loads_no_models():
    code = (
        "import sys, src.main, src.graph; "
        "heavy = {'sentence_transformers', 'llama_cpp', 'src.embedder'} & set(sys.modules); "
        "print(sorted(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_health(client_factory):
    with client_factory() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_search_runs_the_graph(client_factory):
    with client_factory(FakeLLM(["local", "Use pip [1]."])) as c:
        body = c.post("/search", json={"query": "pip install"}).json()
    assert body["answer"] == "Use pip [1]." and body["routing"] == "local"
    assert any("pip install" in s["text"] for s in body["sources"])


def test_search_without_llm_returns_503(client_factory):
    with client_factory() as c:
        assert c.post("/search", json={"query": "x"}).status_code == 503
