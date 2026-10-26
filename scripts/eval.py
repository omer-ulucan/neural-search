"""Evaluate retrieval on BEIR CQADupStack and write results/eval.json.

Configurations: BM25 baseline, SPLADE only, BGE dense only, and weighted-RRF hybrids.
Every tunable choice (rrf_k, static weights, dynamic strength) is selected on the dev
split only; the test split is touched once, to report the selected configurations.

    python scripts/eval.py --subforums webmasters
"""

import argparse
import datetime as dt
import importlib.metadata
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings  # noqa: E402
from src.evaluation import (  # noqa: E402
    download,
    drop_identical,
    evaluate,
    expected_chunk_count,
    git_state,
    index_summary,
    load_beir,
    paired_bootstrap,
    per_query,
    split_queries,
    to_documents,
    verify_index,
)
from src.retrieval import Weights, rrf_scores  # noqa: E402
from src.weighting import QueryWeighting  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("eval")

K_SWEEP = (10, 30, 60, 100)
STATIC_WEIGHTS = ((1.0, 1.0), (1.25, 0.75), (1.5, 0.5), (0.75, 1.25), (0.5, 1.5))
DYNAMIC_STRENGTHS = (0.25, 0.5, 0.75)
DEFAULT_K = 60
SELECT_BY = "nDCG@10"
BM25_PARAMS = {"k1": 1.2, "b": 0.75, "method": "lucene"}  # Elasticsearch defaults, as in BEIR


def package_versions(*names: str) -> dict[str, str]:
    return {n: importlib.metadata.version(n) for n in names}


def encode_queries(embedder, texts: list[str], cache: Path) -> tuple[list, list]:
    """Dense and sparse query vectors, cached on disk keyed by models and texts."""
    key = {"dense": embedder.dense_model_name, "sparse": embedder.sparse_model_name}
    if cache.exists():
        blob = json.loads(cache.read_text())
        if blob["key"] == key and blob["texts"] == texts:
            sparse = [{int(i): v for i, v in s.items()} for s in blob["sparse"]]
            return blob["dense"], sparse
    dense = embedder.encode_dense(texts)
    sparse = embedder.encode_sparse(texts)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "texts": texts, "dense": dense, "sparse": sparse}))
    return dense, sparse


def bm25_rankings(corpus: dict, queries: dict[str, str], qids: list[str], depth: int) -> dict:
    """Doc-level BM25 (bm25s, English stopwords and Snowball stemming) over title + text."""
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    ids = list(corpus)
    texts = [f"{t} {b}" for t, b in corpus.values()]
    model = bm25s.BM25(**BM25_PARAMS)
    model.index(
        bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False),
        show_progress=False,
    )
    q_tokens = bm25s.tokenize(
        [queries[q] for q in qids], stopwords="en", stemmer=stemmer, show_progress=False
    )
    k = min(depth + 1, len(ids))  # +1 leaves room to drop the query's own id
    docs, scores = model.retrieve(q_tokens, k=k, show_progress=False)
    return {
        qid: [ids[int(i)] for i, s in zip(row, srow, strict=True) if s > 0]
        for qid, row, srow in zip(qids, docs, scores, strict=True)
    }


def fused(runs: dict, weigh, rrf_k: int, depth: int) -> dict[str, list[str]]:
    """Weighted RRF of the dense and sparse runs, per query, via the production rrf_scores."""
    out = {}
    for qid in runs["dense"]:
        w = weigh(qid)
        ranked = rrf_scores([runs["dense"][qid], runs["sparse"][qid]], (w.dense, w.sparse), rrf_k)
        out[qid] = [d for d, _ in ranked[:depth]]
    return out


def prepare_subforum(args, sub: str, embedder, client) -> dict:
    """Download, index, verify, and collect per-retriever runs for one subforum."""
    from src.database import init_qdrant
    from src.ingestion import ingest_documents
    from src.retrieval import Retriever

    if args.dataset_dir:
        path, revision = Path(args.dataset_dir) / sub, "local"
    else:
        path, revision = download(sub, Path(args.data_dir), args.revision)
    data = load_beir(path)
    docs = to_documents(data.corpus)
    collection = f"{args.collection_prefix}{sub}"
    init_qdrant(client, collection, embedder.dense_dim)

    t = time.perf_counter()
    # Length-sorted order keeps embedding batches homogeneous, which cuts padding.
    stats = ingest_documents(
        client,
        collection,
        sorted(docs, key=lambda d: (len(d.text), d.source)),
        embedder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    ingest_s = time.perf_counter() - t
    exp_docs, exp_chunks = expected_chunk_count(
        docs, embedder.token_offsets, args.chunk_size, args.chunk_overlap
    )
    summary = index_summary(client, collection)
    checks = verify_index(summary, exp_docs, exp_chunks, embedder.dense_dim)
    logger.info("%s index verified: %s", sub, checks)

    index = {
        **summary,
        "verification": checks,
        "corpus_documents": len(docs),
        "empty_documents": len(docs) - exp_docs,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "last_ingest": {
            "seconds": round(ingest_s, 1),
            "embedded": stats.embedded,
            "unchanged": stats.unchanged,
            "chunks_upserted": stats.chunks_upserted,
            "stale_chunks_deleted": stats.stale_chunks_deleted,
        },
    }
    if args.index_only:
        return {"index": index}

    qids = sorted(q for q in data.qrels if q in data.queries)
    dev, test = split_queries(qids, args.dev_fraction, args.split_seed)
    texts = [data.queries[q] for q in qids]
    cache = Path(args.data_dir) / "cache" / f"{sub}-{revision[:12]}-queries.json"
    dense_vecs, sparse_vecs = encode_queries(embedder, texts, cache)

    retriever = Retriever(client, embedder, collection)
    runs = {"dense": {}, "sparse": {}}
    fallbacks = 0
    for qid, dv, sv in zip(qids, dense_vecs, sparse_vecs, strict=True):
        res = retriever.search(
            dv,
            sv,
            mode="hybrid",
            top_k=args.depth,
            depth=args.chunk_depth,
            group_by="source",
            with_payload=["source"],
        )
        fallbacks += res.sparse_fallback
        runs["dense"][qid] = [h.id for h in res.dense]
        runs["sparse"][qid] = [h.id for h in res.sparse]
    runs["bm25"] = bm25_rankings(data.corpus, data.queries, qids, args.depth)
    removed = {name: drop_identical(r) for name, r in runs.items()}
    for r in runs.values():
        for qid in r:
            r[qid] = r[qid][: args.depth]

    return {
        "dataset": {
            "name": "BEIR CQADupStack",
            "subforum": sub,
            "source": args.dataset_dir or f"https://huggingface.co/datasets/mteb/cqadupstack-{sub}",
            "revision": revision,
            "corpus_documents": len(data.corpus),
            "queries_with_qrels": len(qids),
            "relevance_judgements": sum(len(v) for v in data.qrels.values()),
        },
        "split": {"dev": len(dev), "test": len(test)},
        "index": index,
        "sparse_fallback_queries": fallbacks,
        "identical_ids_removed": removed,
        "query_types": {
            kind: sum(QueryWeighting(0.5)(t).reasons[0].split(":")[0] == kind for t in texts)
            for kind in ("lexical", "natural_language", "neutral")
        },
        "_runs": runs,
        "_qrels": data.qrels,
        "_queries": data.queries,
        "_dev": dev,
        "_test": test,
    }


def configs(depth: int) -> dict:
    """Every evaluated configuration: name -> (description, function producing rankings)."""
    out = {
        "bm25": ({"retriever": "bm25"}, lambda s: s["_runs"]["bm25"]),
        "splade": ({"retriever": "splade"}, lambda s: s["_runs"]["sparse"]),
        "dense": ({"retriever": "dense"}, lambda s: s["_runs"]["dense"]),
    }
    for k in K_SWEEP:
        for wd, ws in STATIC_WEIGHTS:
            out[f"hybrid_static_d{wd}_s{ws}_k{k}"] = (
                {"retriever": "hybrid", "weights": "static", "dense": wd, "sparse": ws, "rrf_k": k},
                lambda s, wd=wd, ws=ws, k=k: fused(s["_runs"], lambda q: Weights(wd, ws), k, depth),
            )
        for strength in DYNAMIC_STRENGTHS:
            out[f"hybrid_dynamic_s{strength}_k{k}"] = (
                {"retriever": "hybrid", "weights": "dynamic", "strength": strength, "rrf_k": k},
                lambda s, st=strength, k=k: fused(
                    s["_runs"], lambda q: QueryWeighting(st)(s["_queries"][q]), k, depth
                ),
            )
    return out


def mean_over(subs: dict, split: str, rankings: dict) -> dict[str, float]:
    per = [evaluate(rankings[sub], subs[sub]["_qrels"], subs[sub][f"_{split}"]) for sub in subs]
    return {m: sum(p[m] for p in per) / len(per) for m in per[0]}


def best(names: list[str], dev_scores: dict) -> str:
    """Highest dev nDCG@10; ties go to the earlier (simpler) name in the list."""
    return max(names, key=lambda n: (dev_scores[n][SELECT_BY], -names.index(n)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--subforums", nargs="+", default=["webmasters"])
    p.add_argument("--revision", default=None, help="Dataset revision (default: pinned)")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument(
        "--dataset-dir",
        default=None,
        help="Use local BEIR-format dirs <dataset-dir>/<subforum> instead of downloading",
    )
    p.add_argument("--collection-prefix", default="eval_cqadupstack_")
    p.add_argument("--output", default=str(ROOT / "results" / "eval.json"))
    p.add_argument("--dev-fraction", type=float, default=0.5)
    p.add_argument("--split-seed", default="neural-search-eval-v1")
    p.add_argument("--depth", type=int, default=100, help="Documents per retriever")
    p.add_argument("--chunk-depth", type=int, default=200, help="Chunks fetched per retriever")
    p.add_argument("--chunk-size", type=int, default=settings.CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=settings.CHUNK_OVERLAP)
    p.add_argument("--batch-size", type=int, default=128, help="Chunks per embed/upsert batch")
    p.add_argument("--index-only", action="store_true", help="Only build and verify indexes")
    args = p.parse_args()

    from src.database import qdrant_client
    from src.embedder import LocalHybridEmbedder

    started = dt.datetime.now(dt.UTC)
    git = git_state(ROOT)
    if git["dirty"]:
        logger.warning("Working tree is dirty; results will be marked dirty")
    embedder = LocalHybridEmbedder()
    client = qdrant_client()

    subs = {}
    for sub in args.subforums:
        subs[sub] = prepare_subforum(args, sub, embedder, client)
    if args.index_only:
        print(json.dumps({s: d["index"] for s, d in subs.items()}, indent=2))
        return

    all_configs = configs(args.depth)
    rankings = {name: {s: fn(d) for s, d in subs.items()} for name, (_, fn) in all_configs.items()}
    dev = {name: mean_over(subs, "dev", r) for name, r in rankings.items()}

    def static(wd, ws, k):
        return f"hybrid_static_d{wd}_s{ws}_k{k}"

    equal = [static(1.0, 1.0, k) for k in K_SWEEP]
    selected = {
        "hybrid_static_equal_default_k": static(1.0, 1.0, DEFAULT_K),
        "hybrid_static_equal_best_k": best(equal, dev),
        "hybrid_static_tuned": best(
            [static(wd, ws, k) for k in K_SWEEP for wd, ws in STATIC_WEIGHTS], dev
        ),
        "hybrid_dynamic_tuned": best(
            [f"hybrid_dynamic_s{st}_k{k}" for k in K_SWEEP for st in DYNAMIC_STRENGTHS], dev
        ),
    }
    report = ["bm25", "splade", "dense", *selected.values()]
    test = {name: mean_over(subs, "test", rankings[name]) for name in dict.fromkeys(report)}

    # Keep/remove decision for dynamic weights is made on dev; test is reported, not used.
    baseline = selected["hybrid_static_equal_best_k"]
    dynamic = selected["hybrid_dynamic_tuned"]
    significance = {}
    for sub, d in subs.items():
        a = per_query(rankings[baseline][sub], d["_qrels"], d["_test"])
        b = per_query(rankings[dynamic][sub], d["_qrels"], d["_test"])
        significance[sub] = paired_bootstrap(a, b)
    decision = {
        "rule": f"keep dynamic weights only if the dev-selected dynamic config beats the "
        f"dev-selected static equal-weight config on dev {SELECT_BY}",
        "baseline": baseline,
        "dynamic": dynamic,
        "dev_delta": dev[dynamic][SELECT_BY] - dev[baseline][SELECT_BY],
        "test_delta": test[dynamic][SELECT_BY] - test[baseline][SELECT_BY],
        "test_paired_bootstrap": significance,
    }
    decision["keep"] = decision["dev_delta"] > 0

    out = {
        "schema_version": 1,
        "generated_at": started.isoformat(timespec="seconds"),
        "git": git,
        "command": ["python", *sys.argv],
        "config": {
            k: v for k, v in vars(args).items() if k not in ("output", "data_dir", "index_only")
        },
        "models": {
            "dense": embedder.dense_model_name,
            "dense_dim": embedder.dense_dim,
            "sparse": embedder.sparse_model_name,
        },
        "bm25": {
            "library": "bm25s",
            **BM25_PARAMS,
            "stemmer": "snowball english",
            "stopwords": "en",
        },
        "metrics": {"measures": ["nDCG@10", "R@100"], "library": "ir_measures (pytrec_eval)"},
        "selection": {"criterion": f"mean dev {SELECT_BY}", "selected": selected},
        "dynamic_weights_decision": decision,
        "environment": {
            "python": platform.python_version(),
            "packages": package_versions(
                "torch",
                "transformers",
                "sentence-transformers",
                "qdrant-client",
                "bm25s",
                "ir-measures",
                "pytrec-eval-terrier",
            ),
            "qdrant_server": client.info().version,
        },
        "subforums": {
            s: {k: v for k, v in d.items() if not k.startswith("_")} for s, d in subs.items()
        },
        "configs": {name: desc for name, (desc, _) in all_configs.items()},
        "results": {
            "test": test,
            "dev": dev,
            "k_sweep": {
                "weights": "static equal (1.0, 1.0)",
                "note": "descriptive only; rrf_k is selected on dev, never on test",
                "dev": {k: dev[static(1.0, 1.0, k)] for k in K_SWEEP},
                "test": {
                    k: mean_over(subs, "test", rankings[static(1.0, 1.0, k)]) for k in K_SWEEP
                },
            },
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    logger.info("Wrote %s", args.output)
    for name in dict.fromkeys(report):
        logger.info("test %-40s %s", name, {m: round(v, 4) for m, v in test[name].items()})


if __name__ == "__main__":
    main()
