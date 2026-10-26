"""Render the README results tables from results/eval.json and results/bench.json.

Numbers in the README are only ever written by this script. Run it after eval or bench:

    python scripts/make_readme_tables.py          # rewrite README.md in place
    python scripts/make_readme_tables.py --check  # exit 1 if README is out of date
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ("eval", "bench")

LABELS = {
    "bm25": "BM25 baseline (bm25s)",
    "splade": "SPLADE only",
    "dense": "Dense only (BGE-large)",
    "hybrid_static_equal_default_k": "Hybrid RRF, equal weights, k=60 (common default)",
    "hybrid_static_equal_best_k": "Hybrid RRF, equal weights, dev-selected k",
    "hybrid_static_tuned": "Hybrid RRF, dev-tuned static weights",
    "hybrid_dynamic_tuned": "Hybrid RRF, dynamic weights (dev-tuned)",
}


def fmt(x: float) -> str:
    return f"{x:.3f}"


def config_note(name: str, desc: dict) -> str:
    if desc.get("retriever") != "hybrid":
        return ""
    if desc["weights"] == "static":
        return f"k={desc['rrf_k']}, w=({desc['dense']}, {desc['sparse']})"
    return f"k={desc['rrf_k']}, strength={desc['strength']}"


def eval_tables(ev: dict) -> str:
    subs = ev["subforums"]
    test = ev["results"]["test"]
    selected = ev["selection"]["selected"]
    rows = [("bm25", "bm25"), ("splade", "splade"), ("dense", "dense")]
    rows += [(label, name) for label, name in selected.items()]
    split = ", ".join(
        f"{s}: {d['split']['dev']} dev / {d['split']['test']} test queries, "
        f"{d['dataset']['corpus_documents']} docs, {d['index']['points']} chunks"
        for s, d in subs.items()
    )
    lines = [
        f"BEIR CQADupStack ({', '.join(subs)}), held-out **test** split. {split}. "
        f"Every hybrid setting was chosen on the dev split only.",
        "",
        "| Configuration | nDCG@10 | Recall@100 | Setting |",
        "|---|---:|---:|---|",
    ]
    for key, name in rows:
        m = test[name]
        lines.append(
            f"| {LABELS[key]} | {fmt(m['nDCG@10'])} | {fmt(m['R@100'])} "
            f"| {config_note(name, ev['configs'][name])} |"
        )
    sweep = ev["results"]["k_sweep"]
    ks = list(sweep["dev"])
    lines += [
        "",
        f"RRF k sweep, equal weights, nDCG@10 (descriptive; k is selected on dev, "
        f"here k={ev['configs'][selected['hybrid_static_equal_best_k']]['rrf_k']}):",
        "",
        "| Split | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|" + "---:|" * len(ks),
        "| dev | " + " | ".join(fmt(sweep["dev"][k]["nDCG@10"]) for k in ks) + " |",
        "| test | " + " | ".join(fmt(sweep["test"][k]["nDCG@10"]) for k in ks) + " |",
    ]
    d = ev["dynamic_weights_decision"]
    verdict = "kept (on by default)" if d["keep"] else "not kept (off by default)"
    boot = "; ".join(
        f"{s}: test delta {b['mean_delta']:+.3f}, "
        f"95% CI [{b['ci95_low']:+.3f}, {b['ci95_high']:+.3f}]"
        for s, b in d["test_paired_bootstrap"].items()
    )
    git = ev["git"]
    lines += [
        "",
        f"Dynamic weights vs equal weights: dev nDCG@10 delta {d['dev_delta']:+.5f}, "
        f"so dynamic weighting is **{verdict}** ({boot}).",
        "",
        f"Models: `{ev['models']['dense']}` and `{ev['models']['sparse']}`. "
        f"Generated {ev['generated_at']} at commit `{git['commit'][:10]}`"
        f"{' (dirty tree)' if git['dirty'] else ''} from `results/eval.json`.",
    ]
    return "\n".join(lines)


def pct_row(label: str, s: dict) -> str:
    return f"| {label} | {s['p50']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} | {s['n']} |"


def load_note(b: dict) -> str:
    """Other processes' load, which inflates every latency below when it is not near zero."""
    bg = b.get("background_load")
    if not bg:
        return ""

    def one(x: dict) -> str:
        gpu = x["gpu"]
        busy = (
            f", GPU {gpu['utilization_pct']:.0f}% ({gpu['memory_used_mb']:.0f} MB)" if gpu else ""
        )
        return f"CPU {x['cpu_busy_pct']:.0f}%{busy}"

    return (
        f" Load from other processes, sampled while the benchmark held no models: "
        f"before {one(bg['before'])}; after {one(bg['after'])}."
    )


def bench_tables(b: dict) -> str:
    hw = b["hardware"]
    rp = b["retrieval_in_process"]
    stage_labels = {
        "dense_encode": "Dense encode (BGE-large)",
        "sparse_encode": "Sparse encode (SPLADE)",
        "qdrant": "Qdrant query (both retrievers, one batch)",
        "fusion": "RRF fusion",
        "total": "**Retrieval total**",
    }
    lines = [
        f"Hardware: {hw['cpu']}, {hw['physical_cores']} cores / {hw['logical_cores']} threads, "
        f"torch threads {hw['torch_threads']}, GPU: {hw['gpu']}, {hw['ram_gb']} GB RAM. "
        f"Qdrant {b['qdrant_server']} in Docker on the same machine. "
        f"Corpus verified before timing: {b['corpus']['index']['distinct_sources']} documents, "
        f"{b['corpus']['index']['points']} chunks.{load_note(b)}",
        "",
        f"Retrieval only, in process ({rp['queries']} queries, {rp['warmup_excluded']} warmup "
        f"excluded, top_k={rp['top_k']}, rrf_k={rp['rrf_k']}, dynamic weights "
        f"{'on' if rp['dynamic_weights'] else 'off'}, includes {', '.join(rp['includes'])}; "
        f"excludes {', '.join(rp['excludes'])}):",
        "",
        "| Stage | p50 ms | p95 ms | p99 ms | n |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [pct_row(stage_labels[s], rp["stages_ms"][s]) for s in stage_labels]
    http = b["http_retrieve"]
    if "levels" in http:
        lines += [
            "",
            "`POST /retrieve` over loopback HTTP, closed-loop workers "
            "(adds HTTP/JSON and queueing to the above):",
            "",
            "| Concurrency | Requests | Throughput req/s | Error rate | p50 ms | p95 ms "
            "| p99 ms | Server RSS peak MB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c, lv in http["levels"].items():
            lat = lv["latency_ms"]
            lines.append(
                f"| {c} | {lv['requests']} | {lv['throughput_rps']:.2f} | {lv['error_rate']:.1%} "
                f"| {lat['p50']:.0f} | {lat['p95']:.0f} | {lat['p99']:.0f} "
                f"| {lv['server_rss']['peak_mb']:.0f} |"
            )
    e2e = b["e2e_search"]
    lines.append("")
    if "levels" in e2e:
        llm = e2e["llm_settings"]
        layers = "all" if llm["n_gpu_layers"] == -1 else llm["n_gpu_layers"]
        note = f" {e2e['note']}." if e2e.get("note") else ""
        lines += [
            f"End to end `POST /search` ({e2e['model']}, n_ctx {llm['n_ctx']}, {layers} layers "
            f"on GPU; router, retrieval or web search, generation and HTTP/JSON).{note}",
            "",
            "| Concurrency | Queries | Routing | Error rate | p50 s | p95 s | p99 s |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
        for c, lv in e2e["levels"].items():
            lat = lv["latency_ms"]
            routes = ", ".join(f"{r} {n}" for r, n in sorted(lv["routing"].items()))
            lines.append(
                f"| {c} | {lv['requests']} | {routes} | {lv['error_rate']:.1%} "
                f"| {lat['p50'] / 1000:.1f} | {lat['p95'] / 1000:.1f} | {lat['p99'] / 1000:.1f} |"
            )
    else:
        lines.append(f"End to end `/search` was not measured: {e2e.get('skipped')}.")
    git = b["git"]
    lines += [
        "",
        f"Generated {b['generated_at']} at commit `{git['commit'][:10]}`"
        f"{' (dirty tree)' if git['dirty'] else ''} from `results/bench.json`.",
    ]
    return "\n".join(lines)


def render(section: str) -> str:
    path = ROOT / "results" / f"{section}.json"
    if not path.exists():
        script = "eval" if section == "eval" else "bench"
        return f"Not measured yet. Run `python scripts/{script}.py` and then this script."
    data = json.loads(path.read_text())
    return eval_tables(data) if section == "eval" else bench_tables(data)


def update(readme: str) -> str:
    for section in SECTIONS:
        start, end = f"<!-- {section}:start -->", f"<!-- {section}:end -->"
        if start not in readme or end not in readme:
            raise SystemExit(f"README is missing the {start} / {end} markers")
        head, rest = readme.split(start, 1)
        _, tail = rest.split(end, 1)
        readme = f"{head}{start}\n{render(section)}\n{end}{tail}"
    return readme


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="Fail if README.md is stale")
    p.add_argument("--readme", default=str(ROOT / "README.md"))
    args = p.parse_args()
    path = Path(args.readme)
    current = path.read_text()
    new = update(current)
    if args.check:
        if new != current:
            print("README.md results tables are out of date; run scripts/make_readme_tables.py")
            sys.exit(1)
        return
    path.write_text(new)


if __name__ == "__main__":
    main()
