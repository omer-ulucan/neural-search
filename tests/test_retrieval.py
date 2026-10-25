import pytest

from src.database import init_qdrant
from src.ingestion import Document, ingest_documents
from src.retrieval import Retriever, Weights, rrf_scores
from tests.conftest import FAKE_DIM

COLLECTION = "test_retrieval"
DOCS = [
    Document("install.md", "Install", "install the package with pip install neural"),
    Document("errors.md", "Errors", "error E1234 means the index is corrupt"),
    Document("fusion.md", "Fusion", "reciprocal rank fusion merges dense and sparse rankings"),
    Document("long.md", "Long", " ".join(["fusion"] + [f"filler{i}" for i in range(30)])),
]


@pytest.fixture
def retriever(memory_qdrant, fake_embedder):
    client = init_qdrant(memory_qdrant, COLLECTION, FAKE_DIM)
    ingest_documents(
        client, COLLECTION, DOCS, fake_embedder, chunk_size=8, chunk_overlap=2, batch_size=4
    )
    return Retriever(client, fake_embedder, COLLECTION, rrf_k=60, depth=20)


def test_hybrid_returns_each_retriever_and_the_fusion_of_both(retriever):
    res = retriever.retrieve("rank fusion of sparse rankings", "hybrid", top_k=3)
    assert len(res.dense) == len(res.fused) == 3
    assert 1 <= len(res.sparse) <= 3  # sparse only matches chunks sharing a term
    assert res.rrf_k == 60 and res.top_k == 3
    assert set(res.timings_ms) >= {"dense_encode", "sparse_encode", "qdrant", "fusion", "total"}
    assert res.hits is res.fused
    assert res.fused[0].payload["source"] == "fusion.md"


def test_fused_scores_equal_rrf_over_the_full_candidate_lists(retriever):
    full = retriever.retrieve("dense sparse fusion", "hybrid", top_k=20)
    expected = rrf_scores([[h.id for h in full.dense], [h.id for h in full.sparse]], (1, 1), 60)
    assert [(h.id, pytest.approx(h.score)) for h in full.fused] == expected[:20]


def test_top_k_truncates_without_changing_rrf_scores(retriever):
    small = retriever.retrieve("dense sparse fusion", "hybrid", top_k=2)
    big = retriever.retrieve("dense sparse fusion", "hybrid", top_k=6)
    assert [(h.id, h.score) for h in small.fused] == [(h.id, h.score) for h in big.fused[:2]]


def test_single_retriever_modes_only_run_their_encoder(retriever, fake_embedder):
    dense = retriever.retrieve("pip install", "dense", top_k=2)
    assert dense.sparse == [] and dense.fused == [] and dense.hits is dense.dense
    assert dense.timings_ms["sparse_encode"] == 0.0
    sparse = retriever.retrieve("E1234", "sparse", top_k=2)
    assert sparse.dense == [] and sparse.hits[0].payload["source"] == "errors.md"
    assert sparse.timings_ms["dense_encode"] == 0.0


def test_explicit_weights_override_the_default(retriever):
    res = retriever.retrieve("E1234 corrupt", "hybrid", top_k=3, weights=Weights(0.0, 1.0))
    assert [h.id for h in res.fused] == [h.id for h in res.sparse]
    assert res.weights == Weights(0.0, 1.0)


def test_group_by_source_returns_one_hit_per_document(retriever):
    res = retriever.retrieve("fusion filler3", "hybrid", top_k=10, group_by="source")
    ids = [h.id for h in res.fused]
    assert len(ids) == len(set(ids)) and "long.md" in ids


def test_unknown_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("x", "bm25")
