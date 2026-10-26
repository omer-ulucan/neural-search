"""Latency benchmark. Writes results/bench.json.

Three separately reported measurements, never mixed:

1. retrieval_in_process: Retriever.retrieve(mode="hybrid") called directly, split into
   dense encode, sparse encode, Qdrant query and fusion. No HTTP, no LLM.
2. http_retrieve: POST /retrieve against a local uvicorn server under closed-loop
   workloads at several concurrency levels. Adds HTTP, JSON and queueing on top of 1.
3. e2e_search: POST /search, i.e. LLM router + retrieval (or web search) + LLM answer.

Before measuring, the expected corpus is verified to be fully ingested.

    python scripts/bench.py --model-path models/gemma-4-12B-it-Q4_0.gguf
"""

import argparse
import asyncio
import datetime as dt
import json
import logging
import multiprocessing
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark import (  # noqa: E402
    MemorySampler,
    RequestResult,
    hardware,
    run_workload,
    summarize,
)
from src.config import settings  # noqa: E402
from src.evaluation import (  # noqa: E402
    download,
    expected_chunk_count,
    git_state,
    index_summary,
    load_beir,
    to_documents,
    verify_index,
)

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("bench")
STAGES = ("dense_encode", "sparse_encode", "qdrant", "fusion", "total")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """A uvicorn subprocess serving src.main:app with the given environment."""

    def __init__(self, env: dict[str, str], startup_timeout_s: float) -> None:
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.env = {**os.environ, **env}
        self.startup_timeout_s = startup_timeout_s

    def __enter__(self) -> "Server":
        import httpx

        cmd = [sys.executable, "-m", "uvicorn", "src.main:app", "--host", "127.0.0.1"]
        cmd += ["--port", str(self.port), "--log-level", "warning"]
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=self.env)
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode}")
            try:
                if httpx.get(f"{self.url}/health", timeout=1).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(1)
        self.proc.kill()
        raise RuntimeError("server did not become healthy in time")

    def __exit__(self, *exc) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def qdrant_container_memory() -> str | None:
    """Best-effort memory usage of the docker-compose Qdrant container."""
    try:
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        if "qdrant" in line:
            return line.split(" ", 1)[1]
    return None


def verify_corpus(args, embedder, client) -> dict:
    """Fail unless the collection holds exactly the chunks the corpus should produce."""
    if args.dataset_dir:
        path, revision = Path(args.dataset_dir) / args.subforum, "local"
    else:
        path, revision = download(args.subforum, Path(args.data_dir), args.revision)
    data = load_beir(path)
    docs = to_documents(data.corpus)
    exp_docs, exp_chunks = expected_chunk_count(
        docs, embedder.token_offsets, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    )
    summary = index_summary(client, args.collection)
    checks = verify_index(summary, exp_docs, exp_chunks, embedder.dense_dim)
    logger.info("Corpus verified: %s", checks)
    return {
        "dataset": f"BEIR CQADupStack {args.subforum}",
        "revision": revision,
        "corpus_documents": len(docs),
        "index": summary,
        "verification": checks,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "models": {
            "dense": embedder.dense_model_name,
            "dense_dim": embedder.dense_dim,
            "sparse": embedder.sparse_model_name,
        },
        "_queries": [data.queries[q] for q in sorted(data.queries)],
    }


def bench_in_process(args, retriever, queries: list[str]) -> dict:
    for q in queries[: args.warmup]:
        retriever.retrieve(q, "hybrid", args.top_k)
    measured = queries[args.warmup :][: args.queries] if args.queries else queries[args.warmup :]
    if len(measured) < args.min_queries:
        raise RuntimeError(f"only {len(measured)} queries; need at least {args.min_queries}")
    stages = {s: [] for s in STAGES}
    fallbacks = 0
    with MemorySampler() as mem:
        for q in measured:
            res = retriever.retrieve(q, "hybrid", args.top_k)
            fallbacks += res.sparse_fallback
            for s in STAGES:
                stages[s].append(res.timings_ms[s])
    return {
        "includes": ["dense encode", "sparse encode", "Qdrant query", "RRF fusion"],
        "excludes": ["HTTP/JSON", "network", "LLM router", "LLM generation"],
        "queries": len(measured),
        "warmup_excluded": args.warmup,
        "mode": "hybrid",
        "top_k": args.top_k,
        "candidate_depth": retriever.depth,
        "rrf_k": retriever.rrf_k,
        "dynamic_weights": retriever.weighting is not None,
        "sparse_fallback_queries": fallbacks,
        "stages_ms": {s: summarize(v) for s, v in stages.items()},
        "process_rss": mem.summary(),
    }


def bench_http_retrieve(args, queries: list[str]) -> dict:
    import httpx

    env = {"MODEL_PATH": "", "COLLECTION_NAME": args.collection}
    levels = {}
    with Server(env, startup_timeout_s=600) as server:

        async def scenario(concurrency: int, batch: list[str]):
            limits = httpx.Limits(max_connections=concurrency)
            async with httpx.AsyncClient(base_url=server.url, timeout=300, limits=limits) as c:

                async def send(q: str) -> RequestResult:
                    r = await c.post("/retrieve", json={"query": q, "top_k": args.top_k})
                    if r.status_code != 200:
                        return RequestResult(0.0, False, error=f"HTTP {r.status_code}")
                    return RequestResult(0.0, True, server_ms=r.json()["timing_ms"]["total"])

                return await run_workload(send, batch, concurrency)

        asyncio.run(scenario(1, queries[: args.warmup]))
        for i, concurrency in enumerate(args.concurrency):
            n = max(args.http_requests, concurrency * 4)
            start = (i * n) % len(queries)
            batch = [queries[(start + j) % len(queries)] for j in range(n)]
            with MemorySampler(server.proc.pid) as mem:
                result = asyncio.run(scenario(concurrency, batch))
            levels[str(concurrency)] = {
                **result.report(),
                "server_rss": mem.summary(),
                "qdrant_container_memory": qdrant_container_memory(),
            }
            logger.info(
                "http_retrieve c=%d %s", concurrency, levels[str(concurrency)]["latency_ms"]
            )
    return {
        "endpoint": "POST /retrieve",
        "includes": [
            "dense encode",
            "sparse encode",
            "Qdrant query",
            "RRF fusion",
            "HTTP/JSON over loopback",
            "server queueing under concurrency",
        ],
        "excludes": ["LLM router", "LLM generation", "internet network latency"],
        "workload": "closed loop: N workers, each sends its next request when one returns",
        "requests_per_level": args.http_requests,
        "warmup_excluded": args.warmup,
        "top_k": args.top_k,
        "levels": levels,
    }


def bench_e2e_search(args, queries: list[str]) -> dict:
    import httpx

    if not args.model_path or not Path(args.model_path).exists():
        return {"skipped": f"LLM model not found: {args.model_path!r}"}
    try:
        searxng_up = httpx.get(settings.SEARXNG_HOST, timeout=5).status_code < 500
    except httpx.HTTPError:
        searxng_up = False
    env = {
        "MODEL_PATH": str(Path(args.model_path).resolve()),
        "COLLECTION_NAME": args.collection,
        "N_CTX": str(args.e2e_n_ctx),
    }
    levels = {}
    with Server(env, startup_timeout_s=1800) as server:

        async def scenario(concurrency: int, batch: list[str], routes: list[str]):
            async with httpx.AsyncClient(base_url=server.url, timeout=3600) as c:

                async def send(q: str) -> RequestResult:
                    r = await c.post("/search", json={"query": q})
                    if r.status_code != 200:
                        return RequestResult(0.0, False, error=f"HTTP {r.status_code}")
                    routes.append(r.json()["routing"])
                    return RequestResult(0.0, True)

                return await run_workload(send, batch, concurrency)

        asyncio.run(scenario(1, queries[: args.e2e_warmup], []))
        offset = args.e2e_warmup
        for concurrency in args.e2e_concurrency:
            batch = queries[offset : offset + args.e2e_queries]
            offset += args.e2e_queries
            routes: list[str] = []
            with MemorySampler(server.proc.pid) as mem:
                result = asyncio.run(scenario(concurrency, batch, routes))
            levels[str(concurrency)] = {
                **result.report(),
                "routing": {r: routes.count(r) for r in sorted(set(routes))},
                "server_rss": mem.summary(),
            }
            logger.info("e2e_search c=%d %s", concurrency, levels[str(concurrency)]["latency_ms"])
    return {
        "endpoint": "POST /search",
        "includes": [
            "LLM router call",
            "hybrid retrieval (local route) or SearXNG web search (web route)",
            "LLM answer generation",
            "HTTP/JSON over loopback",
        ],
        "excludes": [],
        "model": Path(args.model_path).name,
        "llm_settings": {
            "n_ctx": args.e2e_n_ctx,
            "n_threads": settings.N_THREADS,
            "n_gpu_layers": settings.N_GPU_LAYERS,
            "max_tokens": 1024,
        },
        "searxng_reachable": searxng_up,
        "note": "llama.cpp calls are serialized by a lock, so concurrent /search requests queue",
        "warmup_excluded": args.e2e_warmup,
        "levels": levels,
    }


def model_phase(args) -> dict:
    """Load models, verify the corpus and run the in-process benchmark (child process)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.database import qdrant_client
    from src.embedder import LocalHybridEmbedder
    from src.retrieval import Retriever
    from src.weighting import QueryWeighting

    embedder = LocalHybridEmbedder()
    client = qdrant_client()
    corpus = verify_corpus(args, embedder, client)
    queries = corpus.pop("_queries")
    weighting = QueryWeighting(settings.DYNAMIC_STRENGTH) if settings.DYNAMIC_WEIGHTS else None
    retriever = Retriever(client, embedder, args.collection, weighting=weighting)
    return {
        "hardware": hardware(),
        "qdrant_server": client.info().version,
        "corpus": corpus,
        "retrieval_in_process": bench_in_process(args, retriever, queries),
        "queries": queries,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--subforum", default="webmasters")
    p.add_argument("--revision", default=None)
    p.add_argument("--collection", default=None, help="Default: eval_cqadupstack_<subforum>")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--dataset-dir", default=None, help="Local BEIR dirs instead of downloading")
    p.add_argument("--output", default=str(ROOT / "results" / "bench.json"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--queries", type=int, default=0, help="Measured queries (0 = all)")
    p.add_argument("--min-queries", type=int, default=500)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 20, 50])
    p.add_argument("--http-requests", type=int, default=300, help="Requests per level")
    p.add_argument("--model-path", default=settings.MODEL_PATH)
    p.add_argument("--e2e-queries", type=int, default=20)
    p.add_argument("--e2e-warmup", type=int, default=1)
    p.add_argument("--e2e-concurrency", type=int, nargs="+", default=[1])
    p.add_argument(
        "--e2e-n-ctx",
        type=int,
        default=4096,
        help="LLM context for the e2e run; 5 chunks of 320 tokens plus the answer fit in 4096",
    )
    p.add_argument("--skip-http", action="store_true")
    p.add_argument("--skip-e2e", action="store_true")
    args = p.parse_args()
    args.collection = args.collection or f"eval_cqadupstack_{args.subforum}"

    started = dt.datetime.now(dt.UTC)
    git = git_state(ROOT)
    if git["dirty"]:
        logger.warning("Working tree is dirty; results will be marked dirty")
    # The models live only in a child process that exits before the servers start, so
    # their memory is really returned; a 14 GB machine cannot hold three copies plus an LLM.
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        phase = pool.apply(model_phase, (args,))
    queries = phase.pop("queries")
    out = {
        "schema_version": 1,
        "generated_at": started.isoformat(timespec="seconds"),
        "git": git,
        "command": ["python", *sys.argv],
        "config": {k: v for k, v in vars(args).items() if k not in ("output", "data_dir")},
        **phase,
    }
    out["http_retrieve"] = (
        {"skipped": "--skip-http"} if args.skip_http else bench_http_retrieve(args, queries)
    )
    out["e2e_search"] = (
        {"skipped": "--skip-e2e"} if args.skip_e2e else bench_e2e_search(args, queries)
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    logger.info("Wrote %s", args.output)
    logger.info("in-process total: %s", out["retrieval_in_process"]["stages_ms"]["total"])


if __name__ == "__main__":
    main()
