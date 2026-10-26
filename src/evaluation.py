"""Retrieval evaluation on BEIR CQADupStack: data, dev/test split, runs and metrics.

The pieces here are deliberately free of model loading so they can be unit tested;
scripts/eval.py wires them to the real embedder, Qdrant and BM25.
"""

import hashlib
import json
import random
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import ir_measures
from ir_measures import Qrel, R, ScoredDoc, nDCG
from qdrant_client import QdrantClient

from src.chunking import chunk_text
from src.ingestion import Document

DATASET_REPO = "mteb/cqadupstack-{subforum}"
# Pinned Hugging Face dataset revisions (BEIR CQADupStack as published by MTEB).
DATASET_REVISIONS = {
    "webmasters": "c691d4af16c19a79df7ee93bfbd12cc186a5bc92",
    "unix": "3197b09145642430d9db451fcdc1e1987e54d74c",
}
MEASURES = (nDCG @ 10, R @ 100)


@dataclass
class BeirData:
    """A BEIR dataset: corpus id -> (title, text), query id -> text, and test qrels."""

    corpus: dict[str, tuple[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def download(subforum: str, data_dir: Path, revision: str | None = None) -> tuple[Path, str]:
    """Download one CQADupStack subforum; returns (local dir, resolved revision sha)."""
    from huggingface_hub import HfApi, snapshot_download

    repo = DATASET_REPO.format(subforum=subforum)
    revision = revision or DATASET_REVISIONS.get(subforum) or "main"
    sha = HfApi().dataset_info(repo, revision=revision).sha
    path = snapshot_download(
        repo,
        repo_type="dataset",
        revision=sha,
        local_dir=data_dir / "cqadupstack" / subforum,
        allow_patterns=["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"],
    )
    return Path(path), sha


def load_beir(path: Path) -> BeirData:
    """Load corpus.jsonl, queries.jsonl and qrels/test.tsv in BEIR format."""
    corpus = {}
    with open(path / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            corpus[str(d["_id"])] = (d.get("title") or "", d.get("text") or "")
    queries = {}
    with open(path / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            queries[str(q["_id"])] = q["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels" / "test.tsv", encoding="utf-8") as fh:
        next(fh)  # header: query-id, corpus-id, score
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return BeirData(corpus, queries, qrels)


def to_documents(corpus: dict[str, tuple[str, str]]) -> list[Document]:
    """One Document per corpus entry; the title is prepended to the indexed text."""
    return [
        Document(source=doc_id, title=title, text=f"{title}\n\n{text}" if title else text)
        for doc_id, (title, text) in corpus.items()
    ]


def split_queries(
    qids: Iterable[str], dev_fraction: float, seed: str
) -> tuple[list[str], list[str]]:
    """Deterministically split query ids into (dev, test) by hashing seed:qid.

    Membership of a query never depends on which other queries exist, so the split
    is stable across dataset orderings and subsets.
    """
    dev, test = [], []
    for qid in sorted(qids):
        h = int(hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (dev if h < dev_fraction else test).append(qid)
    return dev, test


def expected_chunk_count(
    docs: Sequence[Document], token_offsets: Callable, chunk_size: int, chunk_overlap: int
) -> tuple[int, int]:
    """(non-empty documents, chunks) that ingestion should produce for docs."""
    n_docs = n_chunks = 0
    for d in docs:
        n = len(chunk_text(d.text, token_offsets, chunk_size, chunk_overlap))
        n_docs += n > 0
        n_chunks += n
    return n_docs, n_chunks


def index_summary(client: QdrantClient, collection: str) -> dict:
    """Distinct sources, point count and vector config of a collection."""
    info = client.get_collection(collection)
    sources: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection, limit=2048, offset=offset, with_payload=["source"], with_vectors=False
        )
        sources.update(p.payload["source"] for p in points)
        if offset is None:
            break
    vectors = info.config.params.vectors
    dense = vectors if not isinstance(vectors, dict) else vectors[""]
    return {
        "collection": collection,
        "points": client.count(collection, exact=True).count,
        "distinct_sources": len(sources),
        "dense_size": dense.size,
        "dense_distance": str(dense.distance.value),
        "sparse_vectors": sorted((info.config.params.sparse_vectors or {}).keys()),
        "status": str(info.status.value),
    }


def verify_index(summary: dict, expected_docs: int, expected_chunks: int, dense_dim: int) -> dict:
    """Compare a collection against what the corpus should have produced; raise if off."""
    checks = {
        "documents": (summary["distinct_sources"], expected_docs),
        "chunks": (summary["points"], expected_chunks),
        "dense_dim": (summary["dense_size"], dense_dim),
    }
    failed = {k: {"found": f, "expected": e} for k, (f, e) in checks.items() if f != e}
    if failed:
        raise RuntimeError(f"Index does not match the corpus: {failed}")
    return {k: {"found": f, "expected": e} for k, (f, e) in checks.items()}


def as_run(rankings: dict[str, Sequence[str]], depth: int) -> list[ScoredDoc]:
    """Turn ranked doc id lists into an ir_measures run.

    Scores are derived from rank so the evaluator sees exactly our (deterministically
    tie-broken) order rather than re-breaking ties among equal RRF scores.
    """
    return [
        ScoredDoc(qid, did, float(depth - i))
        for qid, docs in rankings.items()
        for i, did in enumerate(docs[:depth])
    ]


def as_qrels(qrels: dict[str, dict[str, int]], qids: Iterable[str]) -> list[Qrel]:
    return [Qrel(qid, did, rel) for qid in qids for did, rel in qrels.get(qid, {}).items()]


def evaluate(
    rankings: dict[str, Sequence[str]], qrels: dict[str, dict[str, int]], qids: Sequence[str]
) -> dict[str, float]:
    """Mean nDCG@10 and Recall@100 over qids; queries with no results score zero."""
    run = as_run({q: rankings.get(q, []) for q in qids}, depth=100)
    # ir_measures averages over every judged query, so queries missing from the run count 0.
    agg = ir_measures.calc_aggregate(MEASURES, as_qrels(qrels, qids), run)
    return {str(m): float(agg.get(m, 0.0)) for m in MEASURES}


def per_query(
    rankings: dict[str, Sequence[str]], qrels: dict[str, dict[str, int]], qids: Sequence[str]
) -> dict[str, float]:
    """nDCG@10 per query (0.0 for queries without results)."""
    run = as_run({q: rankings.get(q, []) for q in qids}, depth=100)
    out = dict.fromkeys(qids, 0.0)
    for m in ir_measures.iter_calc([nDCG @ 10], as_qrels(qrels, qids), run):
        out[m.query_id] = float(m.value)
    return out


def paired_bootstrap(
    a: dict[str, float], b: dict[str, float], n: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Mean of (b - a) over shared queries with a 95% percentile bootstrap interval."""
    qids = sorted(set(a) & set(b))
    deltas = [b[q] - a[q] for q in qids]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(n))
    return {
        "mean_delta": sum(deltas) / len(deltas),
        "ci95_low": means[int(0.025 * n)],
        "ci95_high": means[int(0.975 * n) - 1],
        "queries": len(deltas),
    }


def drop_identical(rankings: dict[str, list[str]]) -> int:
    """Remove a query's own id from its ranking (BEIR's ignore_identical_ids)."""
    removed = 0
    for qid, docs in rankings.items():
        if qid in docs:
            docs.remove(qid)
            removed += 1
    return removed


def git_state(root: Path) -> dict:
    """Commit hash and whether the working tree had uncommitted changes."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
