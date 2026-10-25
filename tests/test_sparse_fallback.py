import logging

import pytest

from src.database import init_qdrant
from src.ingestion import Document, ingest_documents
from src.retrieval import Retriever
from tests.conftest import FAKE_DIM, FakeEmbedder

COLLECTION = "test_fallback"


class EmptySparseEmbedder(FakeEmbedder):
    def get_sparse(self, text: str) -> dict[int, float]:
        return {}


@pytest.fixture
def retriever(memory_qdrant, fake_embedder):
    client = init_qdrant(memory_qdrant, COLLECTION, FAKE_DIM)
    docs = [Document(f"d{i}.md", f"D{i}", f"topic{i} shared words") for i in range(4)]
    ingest_documents(
        client, COLLECTION, docs, fake_embedder, chunk_size=8, chunk_overlap=1, batch_size=8
    )
    return Retriever(client, EmptySparseEmbedder(), COLLECTION)


def test_hybrid_query_with_empty_sparse_vector_falls_back_to_dense(retriever, caplog):
    with caplog.at_level(logging.WARNING):
        res = retriever.retrieve("topic2", "hybrid", top_k=3)
    assert res.sparse_fallback is True
    assert res.sparse == []
    assert [h.id for h in res.fused] == [h.id for h in res.dense]
    assert len(res.fused) == 3
    assert "falling back to dense-only" in caplog.text


def test_sparse_only_query_with_empty_vector_returns_no_results(retriever):
    res = retriever.retrieve("topic2", "sparse", top_k=3)
    assert res.sparse_fallback is True and res.hits == []


def test_normal_query_does_not_flag_fallback(memory_qdrant, fake_embedder):
    client = init_qdrant(memory_qdrant, COLLECTION, FAKE_DIM)
    docs = [Document("a.md", "A", "alpha beta")]
    ingest_documents(client, COLLECTION, docs, fake_embedder, chunk_size=8, chunk_overlap=1)
    res = Retriever(client, fake_embedder, COLLECTION).retrieve("alpha", "hybrid", top_k=1)
    assert res.sparse_fallback is False and res.sparse
