import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load():
    path = ROOT / "scripts" / "make_readme_tables.py"
    spec = importlib.util.spec_from_file_location("make_readme_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def m(ndcg, recall):
    return {"nDCG@10": ndcg, "R@100": recall}


EVAL = {
    "generated_at": "2026-01-01T00:00:00+00:00",
    "git": {"commit": "abcdef1234567890", "dirty": False},
    "models": {"dense": "bge", "sparse": "splade"},
    "subforums": {
        "webmasters": {
            "split": {"dev": 2, "test": 3},
            "dataset": {"corpus_documents": 10},
            "index": {"points": 12},
        }
    },
    "selection": {
        "selected": {
            "hybrid_static_equal_default_k": "s60",
            "hybrid_static_equal_best_k": "s10",
            "hybrid_static_tuned": "s10",
            "hybrid_dynamic_tuned": "d10",
        }
    },
    "configs": {
        "bm25": {"retriever": "bm25"},
        "splade": {"retriever": "splade"},
        "dense": {"retriever": "dense"},
        "s60": {
            "retriever": "hybrid",
            "weights": "static",
            "dense": 1.0,
            "sparse": 1.0,
            "rrf_k": 60,
        },
        "s10": {
            "retriever": "hybrid",
            "weights": "static",
            "dense": 1.0,
            "sparse": 1.0,
            "rrf_k": 10,
        },
        "d10": {"retriever": "hybrid", "weights": "dynamic", "strength": 0.5, "rrf_k": 10},
    },
    "results": {
        "test": {
            "bm25": m(0.1, 0.2),
            "splade": m(0.2, 0.3),
            "dense": m(0.3, 0.4),
            "s60": m(0.4, 0.5),
            "s10": m(0.45, 0.55),
            "d10": m(0.44, 0.56),
        },
        "k_sweep": {
            "dev": {"10": m(0.5, 0), "60": m(0.4, 0)},
            "test": {"10": m(0.45, 0), "60": m(0.4, 0)},
        },
    },
    "dynamic_weights_decision": {
        "keep": False,
        "dev_delta": -0.01,
        "test_paired_bootstrap": {
            "webmasters": {"mean_delta": -0.01, "ci95_low": -0.03, "ci95_high": 0.01}
        },
    },
}


def test_eval_table_lists_every_config_with_test_metrics():
    out = load().eval_tables(EVAL)
    assert "| BM25 baseline (bm25s) | 0.100 | 0.200 |" in out
    assert (
        "| Hybrid RRF, equal weights, k=60 (common default) | 0.400 | 0.500 | k=60, w=(1.0, 1.0) |"
        in out
    )
    assert "| Hybrid RRF, dynamic weights (dev-tuned) | 0.440 | 0.560 | k=10, strength=0.5 |" in out
    assert "| dev | 0.500 | 0.400 |" in out and "| test | 0.450 | 0.400 |" in out
    assert "**not kept (off by default)**" in out
    assert "`abcdef1234`" in out and "held-out **test** split" in out


def test_kept_dynamic_weights_show_a_tiny_dev_delta_instead_of_zero():
    ev = {**EVAL, "dynamic_weights_decision": {**EVAL["dynamic_weights_decision"]}}
    ev["dynamic_weights_decision"].update(keep=True, dev_delta=0.000027)
    out = load().eval_tables(ev)
    assert "dev nDCG@10 delta +0.00003" in out and "**kept (on by default)**" in out


def stages(p50):
    return {"p50": p50, "p95": p50 + 1, "p99": p50 + 2, "n": 500}


BENCH = {
    "generated_at": "2026-01-01T00:00:00+00:00",
    "git": {"commit": "abcdef1234567890", "dirty": False},
    "hardware": {
        "cpu": "cpu-x",
        "physical_cores": 8,
        "logical_cores": 16,
        "torch_threads": 8,
        "gpu": "gpu-y",
        "ram_gb": 32.0,
    },
    "qdrant_server": "1.20.0",
    "corpus": {"index": {"distinct_sources": 10, "points": 12}},
    "retrieval_in_process": {
        "queries": 500,
        "warmup_excluded": 20,
        "top_k": 10,
        "rrf_k": 10,
        "dynamic_weights": True,
        "includes": ["dense encode"],
        "excludes": ["LLM"],
        "stages_ms": {
            s: stages(i)
            for i, s in enumerate(["dense_encode", "sparse_encode", "qdrant", "fusion", "total"])
        },
    },
    "http_retrieve": {"skipped": "--skip-http"},
    "e2e_search": {"skipped": "--skip-e2e"},
    "background_load": {
        "before": {"cpu_busy_pct": 3.2, "gpu": {"utilization_pct": 1.0, "memory_used_mb": 400.0}},
        "after": {"cpu_busy_pct": 41.0, "gpu": None},
    },
}


def test_bench_table_reports_settings_and_background_load():
    out = load().bench_tables(BENCH)
    assert "rrf_k=10, dynamic weights on" in out
    assert "before CPU 3%, GPU 1% (400 MB); after CPU 41%." in out
    assert "| **Retrieval total** | 4.0 | 5.0 | 6.0 | 500 |" in out
    assert "End to end `/search` was not measured: --skip-e2e." in out
    older = {k: v for k, v in BENCH.items() if k != "background_load"}
    assert "Load from other processes" not in load().bench_tables(older)


def test_bench_table_renders_search_levels_as_rows():
    lat = {"p50": 1600.0, "p95": 1900.0, "p99": 2040.0, "n": 20}
    level = {"requests": 20, "error_rate": 0.05, "routing": {"web": 19, "local": 1}}
    e2e = {
        "model": "m.gguf",
        "llm_settings": {"n_ctx": 4096, "n_gpu_layers": -1},
        "note": "llama.cpp calls are serialized",
        "levels": {"1": {**level, "latency_ms": lat}, "5": {**level, "latency_ms": lat}},
    }
    out = load().bench_tables({**BENCH, "e2e_search": e2e})
    assert "| 1 | 20 | local 1, web 19 | 5.0% | 1.6 | 1.9 | 2.0 |" in out
    assert "| 5 | 20 | local 1, web 19 |" in out
    assert "n_ctx 4096, all layers on GPU;" in out and "llama.cpp calls are serialized." in out
    cpu = {**e2e, "llm_settings": {"n_ctx": 4096, "n_gpu_layers": 0}}
    assert "n_ctx 4096, 0 layers on GPU;" in load().bench_tables({**BENCH, "e2e_search": cpu})


def test_update_replaces_only_between_markers():
    mod = load()
    readme = (
        "intro\n<!-- eval:start -->\nold\n<!-- eval:end -->\nmid\n"
        "<!-- bench:start -->\n<!-- bench:end -->\nend"
    )
    out = mod.update(readme)
    assert out.startswith("intro\n<!-- eval:start -->\n") and out.endswith(
        "<!-- bench:end -->\nend"
    )
    assert "old" not in out and "\nmid\n" in out
    with pytest.raises(SystemExit):
        mod.update("no markers here")


def test_committed_readme_matches_results_files():
    """README numbers must come from results/*.json, never typed by hand."""
    mod = load()
    readme = (ROOT / "README.md").read_text()
    assert mod.update(readme) == readme, "run python scripts/make_readme_tables.py"


EM_DASH = chr(0x2014)  # built at runtime so this file does not contain one


def test_no_em_dashes_in_tracked_text_files():
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    suffixes = {".md", ".py", ".html", ".yml", ".toml", ".txt", ".example"}
    offenders = [
        f
        for f in files
        if Path(f).suffix in suffixes
        and (ROOT / f).exists()
        and EM_DASH in (ROOT / f).read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []
