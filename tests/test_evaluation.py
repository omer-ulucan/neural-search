import importlib.util
import json
from pathlib import Path

import pytest

from src.database import init_qdrant
from src.evaluation import (
    as_run,
    drop_identical,
    evaluate,
    expected_chunk_count,
    index_summary,
    load_beir,
    paired_bootstrap,
    per_query,
    split_queries,
    to_documents,
    verify_index,
)
from src.ingestion import ingest_documents
from tests.conftest import FAKE_DIM

ROOT = Path(__file__).resolve().parent.parent


def load_eval_script():
    spec = importlib.util.spec_from_file_location("eval_script", ROOT / "scripts" / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def beir_dir(tmp_path):
    corpus = [
        {"_id": "d1", "title": "Robots", "text": "robots.txt blocks crawlers"},
        {"_id": "d2", "title": "", "text": "use a sitemap for google"},
        {"_id": "d3", "title": "Speed", "text": "enable gzip compression"},
    ]
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in corpus) + "\n")
    queries = [{"_id": "q1", "text": "block crawlers"}, {"_id": "q2", "text": "faster site"}]
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td3\t1\nq2\td2\t0\n"
    )
    return tmp_path


def test_load_beir_reads_corpus_queries_and_positive_qrels(beir_dir):
    data = load_beir(beir_dir)
    assert data.corpus["d1"] == ("Robots", "robots.txt blocks crawlers")
    assert data.queries == {"q1": "block crawlers", "q2": "faster site"}
    assert data.qrels == {"q1": {"d1": 1}, "q2": {"d3": 1}}


def test_to_documents_prepends_title():
    docs = to_documents({"a": ("Title", "body"), "b": ("", "only body")})
    assert [(d.source, d.title, d.text) for d in docs] == [
        ("a", "Title", "Title\n\nbody"),
        ("b", "", "only body"),
    ]


def test_split_is_deterministic_disjoint_and_complete():
    qids = [str(i) for i in range(1000)]
    dev, test = split_queries(qids, 0.5, "seed")
    assert (dev, test) == split_queries(list(reversed(qids)), 0.5, "seed")
    assert set(dev).isdisjoint(test) and set(dev) | set(test) == set(qids)
    assert 400 < len(dev) < 600
    assert split_queries(qids, 0.5, "other")[0] != dev


def test_split_membership_does_not_depend_on_other_queries():
    dev_all, _ = split_queries([str(i) for i in range(100)], 0.5, "s")
    dev_sub, _ = split_queries([str(i) for i in range(0, 100, 2)], 0.5, "s")
    assert set(dev_sub) == {q for q in dev_all if int(q) % 2 == 0}


def test_index_verification_against_the_corpus(memory_qdrant, fake_embedder, beir_dir):
    docs = to_documents(load_beir(beir_dir).corpus)
    client = init_qdrant(memory_qdrant, "eval", FAKE_DIM)
    ingest_documents(client, "eval", docs, fake_embedder, chunk_size=3, chunk_overlap=1)
    exp_docs, exp_chunks = expected_chunk_count(docs, fake_embedder.token_offsets, 3, 1)
    summary = index_summary(client, "eval")
    assert summary["distinct_sources"] == exp_docs == 3
    assert summary["points"] == exp_chunks
    assert summary["dense_size"] == FAKE_DIM and summary["sparse_vectors"] == ["lexical-sparse"]
    checks = verify_index(summary, exp_docs, exp_chunks, FAKE_DIM)
    assert checks["chunks"] == {"found": exp_chunks, "expected": exp_chunks}
    with pytest.raises(RuntimeError, match="chunks"):
        verify_index(summary, exp_docs, exp_chunks + 1, FAKE_DIM)


def test_evaluate_perfect_and_partial_runs():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1, "c": 1}}
    perfect = evaluate({"q1": ["a"], "q2": ["b", "c"]}, qrels, ["q1", "q2"])
    assert perfect == {"nDCG@10": pytest.approx(1.0), "R@100": pytest.approx(1.0)}
    partial = evaluate({"q1": ["a"], "q2": ["b", "x"]}, qrels, ["q1", "q2"])
    idcg = 1 + 1 / 1.584962500721156
    assert partial["nDCG@10"] == pytest.approx((1.0 + 1 / idcg) / 2)
    assert partial["R@100"] == pytest.approx(0.75)


def test_queries_without_results_count_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    assert evaluate({"q1": ["a"], "q2": []}, qrels, ["q1", "q2"])["nDCG@10"] == pytest.approx(0.5)


def test_run_scores_preserve_our_order():
    run = as_run({"q": ["z", "a", "m"]}, depth=100)
    assert [d.doc_id for d in sorted(run, key=lambda d: -d.score)] == ["z", "a", "m"]
    qrels = {"q": {"z": 1}}
    assert evaluate({"q": ["z", "a"]}, qrels, ["q"])["nDCG@10"] == pytest.approx(1.0)
    assert evaluate({"q": ["a", "z"]}, qrels, ["q"])["nDCG@10"] < 1.0


def test_per_query_and_paired_bootstrap():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    a = per_query({"q1": ["x", "a"], "q2": ["b"]}, qrels, ["q1", "q2"])
    b = per_query({"q1": ["a"], "q2": ["b"]}, qrels, ["q1", "q2"])
    assert b == {"q1": pytest.approx(1.0), "q2": pytest.approx(1.0)}
    same = paired_bootstrap(a, a, n=200)
    assert same["mean_delta"] == 0 and same["ci95_low"] == same["ci95_high"] == 0
    better = paired_bootstrap(a, b, n=200)
    assert better["mean_delta"] > 0 and better["ci95_low"] >= 0 and better["queries"] == 2


def test_drop_identical_removes_the_query_itself():
    runs = {"q1": ["q1", "a"], "q2": ["b"]}
    assert drop_identical(runs) == 1 and runs == {"q1": ["a"], "q2": ["b"]}


def test_fused_uses_weighted_rrf_and_truncates():
    ev = load_eval_script()
    runs = {"dense": {"q": ["a", "b", "c"]}, "sparse": {"q": ["c", "b"]}}
    out = ev.fused(runs, lambda q: ev.Weights(1.0, 1.0), rrf_k=60, depth=2)
    assert out == {"q": ["c", "b"]}
    dense_only = ev.fused(runs, lambda q: ev.Weights(1.0, 0.0), rrf_k=60, depth=3)
    assert dense_only == {"q": ["a", "b", "c"]}


def test_config_grid_and_dev_selection_rules():
    ev = load_eval_script()
    names = ev.configs(depth=100)
    assert {"bm25", "splade", "dense", "hybrid_static_d1.0_s1.0_k60"} <= set(names)
    assert len([n for n in names if n.startswith("hybrid_dynamic")]) == 3 * len(ev.K_SWEEP)
    dev = {"a": {"nDCG@10": 0.5}, "b": {"nDCG@10": 0.5}, "c": {"nDCG@10": 0.4}}
    assert ev.best(["a", "b", "c"], dev) == "a"  # ties go to the earlier, simpler config


def test_bm25_ranks_lexical_match_first(beir_dir):
    ev = load_eval_script()
    data = load_beir(beir_dir)
    runs = ev.bm25_rankings(data.corpus, data.queries, ["q1"], depth=10)
    assert runs["q1"][0] == "d1"
