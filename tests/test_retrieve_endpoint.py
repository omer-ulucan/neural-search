import pytest
from fastapi.testclient import TestClient

from src.main import Resources, create_app, snippet


@pytest.fixture
def client(seeded_retriever):
    app = create_app(lambda: Resources(retriever=seeded_retriever))
    with TestClient(app) as c:
        yield c


def test_retrieve_returns_sparse_dense_and_fused_side_by_side(client):
    body = client.post("/retrieve", json={"query": "rank fusion dense", "top_k": 2}).json()
    assert body["top_k"] == 2 and body["rrf_k"] == 60
    assert body["weights"] == {"dense": 1.0, "sparse": 1.0, "reasons": []}
    for key in ("sparse", "dense", "fused"):
        assert 1 <= len(body[key]) <= 2
        item = body[key][0]
        assert item["rank"] == 1 and isinstance(item["score"], float)
        assert set(item) >= {"title", "snippet", "source", "chunk_index", "id"}
    top = body["fused"][0]
    assert top["title"] == "Fusion" and top["dense_rank"] and top["sparse_rank"]
    assert set(body["timing_ms"]) == {"dense_encode", "sparse_encode", "qdrant", "fusion", "total"}
    assert body["sparse_fallback"] is False


def test_rrf_k_is_configurable_per_request(client):
    body = client.post("/retrieve", json={"query": "pip install", "rrf_k": 10}).json()
    assert body["rrf_k"] == 10
    top = body["fused"][0]
    assert top["score"] == pytest.approx(
        1 / (10 + top["dense_rank"]) + 1 / (10 + top["sparse_rank"])
    )


def test_retrieve_works_without_an_llm(client):
    assert client.post("/search", json={"query": "x"}).status_code == 503
    assert client.post("/retrieve", json={"query": "x"}).status_code == 200


@pytest.mark.parametrize(
    "payload", [{"query": ""}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 51}, {}]
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/retrieve", json=payload).status_code == 422


def test_snippet_collapses_whitespace_and_cuts_on_word_boundary():
    assert snippet("a \n\n b") == "a b"
    out = snippet("word " * 100, limit=22)
    assert out == "word word word word..."
