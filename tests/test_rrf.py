import pytest

from src.retrieval import Hit, Weights, collapse, fuse, rrf_scores


def scores(rankings, weights=(1.0, 1.0), rrf_k=60):
    return dict(rrf_scores(rankings, weights, rrf_k))


def test_rank_one_in_both_lists():
    s = scores([["a", "b"], ["a", "c"]])
    assert s["a"] == pytest.approx(2 / 61)
    assert round(s["a"], 4) == 0.0328


def test_ranks_one_and_two():
    s = scores([["a", "b"], ["b", "a"]])
    assert s["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert round(s["a"], 4) == 0.0325


def test_weights_scale_each_retrievers_contribution():
    s = scores([["a", "b"], ["b", "a"]], weights=(2.0, 0.5))
    assert s["a"] == pytest.approx(2.0 / 61 + 0.5 / 62)
    assert s["b"] == pytest.approx(2.0 / 62 + 0.5 / 61)


def test_zero_weight_removes_a_retriever():
    s = scores([["a"], ["b"]], weights=(1.0, 0.0))
    assert s == {"a": pytest.approx(1 / 61)}


def test_document_missing_from_a_list_contributes_zero_for_it():
    s = scores([["a", "b"], ["b"]])
    assert s["a"] == pytest.approx(1 / 61)
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_k_changes_scores():
    assert scores([["a"], ["a"]], rrf_k=10)["a"] == pytest.approx(2 / 11)


def test_repeats_within_one_list_count_once():
    assert scores([["a", "a"], []])["a"] == pytest.approx(1 / 61)


def test_ties_are_broken_by_best_rank_then_id_regardless_of_input_order():
    # x: ranks (1, 3) and y: ranks (3, 1) tie exactly; 1/61 + 1/63 beats z at 2/62
    a = rrf_scores([["x", "z", "y"], ["y", "z", "x"]], (1.0, 1.0))
    b = rrf_scores([["y", "z", "x"], ["x", "z", "y"]], (1.0, 1.0))
    assert [d for d, _ in a] == [d for d, _ in b]
    assert [d for d, _ in a] == ["x", "y", "z"]
    # equal score and equal best rank: falls back to id order
    c = rrf_scores([["q", "p"], ["p", "q"]], (1.0, 1.0))
    assert [d for d, _ in c] == ["p", "q"]


@pytest.mark.parametrize("bad", [dict(weights=(1.0,)), dict(rrf_k=-1)])
def test_invalid_arguments_raise(bad):
    with pytest.raises(ValueError):
        rrf_scores([["a"], ["b"]], **{"weights": (1.0, 1.0), **bad})


def test_negative_weights_rejected():
    with pytest.raises(ValueError):
        Weights(dense=-1.0)


def test_fuse_keeps_per_retriever_ranks_and_payloads():
    dense = [Hit("a", 0.9, 1, {"t": "A"}), Hit("b", 0.8, 2, {"t": "B"})]
    sparse = [Hit("c", 12.0, 1, {"t": "C"}), Hit("a", 7.0, 2, {"t": "A"})]
    fused = fuse(dense, sparse, Weights(), rrf_k=60)
    assert [h.id for h in fused] == ["a", "c", "b"]
    assert [h.rank for h in fused] == [1, 2, 3]
    a = fused[0]
    assert (a.dense_rank, a.sparse_rank, a.payload) == (1, 2, {"t": "A"})
    assert fused[1].dense_rank is None and fused[1].sparse_rank == 1


def test_collapse_keeps_best_chunk_per_document():
    hits = [
        Hit("c1", 0.9, 1, {"source": "doc1"}),
        Hit("c2", 0.8, 2, {"source": "doc2"}),
        Hit("c3", 0.7, 3, {"source": "doc1"}),
    ]
    out = collapse(hits, "source")
    assert [(h.id, h.rank, h.score) for h in out] == [("doc1", 1, 0.9), ("doc2", 2, 0.8)]
