import pytest

from src.retrieval import Weights
from src.weighting import QueryWeighting, lexical_signals


@pytest.mark.parametrize(
    ("query", "signal"),
    [
        ('error "permission denied" on deploy', "quoted"),
        ("run `make install` twice", "quoted"),
        ("cannot read /etc/nginx/nginx.conf", "path"),
        ("redirect rules in .htaccess", "filename"),
        ("settings.py is ignored", "filename"),
        ("os.path.join() returns wrong slashes", "code"),
        ("std::vector resize", "code"),
        ("max_connections too low", "identifier"),
        ("getElementById returns null", "identifier"),
        ("DEBUG_MODE not respected", "constant"),
        ("pip install --upgrade fails", "flag"),
        ("server returns 503 under load", "error_code"),
        ("ORA-01017 invalid username", "error_code"),
        ("broken after upgrading to 2.4.1", "version"),
    ],
)
def test_lexical_signals_detected(query, signal):
    assert signal in lexical_signals(query)


@pytest.mark.parametrize(
    "query",
    [
        "how do I make my site load faster",
        "best way to organise a large documentation site",
        "seo",
    ],
)
def test_prose_has_no_lexical_signals(query):
    assert lexical_signals(query) == []


def test_lexical_queries_lean_sparse_with_reasons():
    w = QueryWeighting(0.5)("nginx returns 502 for /api/v1")
    assert (w.dense, w.sparse) == (0.5, 1.5)
    assert "lexical:error_code" in w.reasons and "lexical:path" in w.reasons


def test_natural_language_questions_lean_dense():
    w = QueryWeighting(0.5)("why does google not index my new pages")
    assert w == Weights(1.5, 0.5, ("natural_language",))


def test_short_neutral_queries_keep_equal_weights():
    assert QueryWeighting(0.5)("seo tips") == Weights(1.0, 1.0, ("neutral",))


def test_zero_strength_is_static_equal_weighting():
    w = QueryWeighting(0.0)
    for q in ["nginx 502", "how do I speed up my site", "seo"]:
        assert (w(q).dense, w(q).sparse) == (1.0, 1.0)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_strength_out_of_range_rejected(bad):
    with pytest.raises(ValueError):
        QueryWeighting(bad)


def test_retriever_uses_weighting_function(seeded_retriever):
    seeded_retriever.weighting = QueryWeighting(1.0)
    res = seeded_retriever.retrieve("error E1234", "hybrid", top_k=3)
    assert (res.weights.dense, res.weights.sparse) == (0.0, 2.0)
    assert [h.id for h in res.fused] == [h.id for h in res.sparse]
