"""Hybrid retrieval core: dense (BGE) and sparse (SPLADE) search fused with weighted RRF.

Fusion runs client-side. Qdrant 1.20 does expose a parameterized RRF query, but it
ranks from 0 (a doc ranked first everywhere scores 2/k, not 2/(k+1)) and its
weighting does not follow the standard sum of w_i / (k + rank_i): with weights
[2, 1] a doc ranked higher by the second list wins. Both per-retriever lists come
back from one batched Qdrant request, so fusing here costs no extra round trip and
gives the demo and the eval the raw per-retriever rankings for free.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from qdrant_client import QdrantClient, models

from src.config import settings
from src.database import DENSE_VECTOR, SPARSE_VECTOR, to_sparse_vector

logger = logging.getLogger(__name__)

Mode = Literal["dense", "sparse", "hybrid"]
MODES: tuple[Mode, ...] = ("dense", "sparse", "hybrid")


@dataclass(frozen=True)
class Hit:
    """One ranked result. Ranks are 1-based."""

    id: str
    score: float
    rank: int
    payload: dict = field(default_factory=dict)
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass(frozen=True)
class Weights:
    """Per-retriever RRF weights, with the reasons that produced them."""

    dense: float = 1.0
    sparse: float = 1.0
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dense < 0 or self.sparse < 0:
            raise ValueError("RRF weights must be non-negative")


@dataclass
class RetrievalResult:
    """Per-retriever rankings plus the fused ranking for one query."""

    mode: Mode
    top_k: int
    rrf_k: int
    weights: Weights
    dense: list[Hit]
    sparse: list[Hit]
    fused: list[Hit]
    timings_ms: dict[str, float]
    sparse_fallback: bool = False

    @property
    def hits(self) -> list[Hit]:
        """The ranking that answers this query's mode."""
        return {"dense": self.dense, "sparse": self.sparse, "hybrid": self.fused}[self.mode]


def rrf_scores(
    rankings: Sequence[Sequence[str]], weights: Sequence[float], rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion.

    score(d) = sum_i w_i / (rrf_k + rank_i(d)), with 1-based ranks. A document missing
    from a ranking contributes nothing for it, a zero-weight ranking is ignored, and
    repeats within one ranking count once. Ties are broken by best single-list rank,
    then by id, so output is stable.
    """
    if len(rankings) != len(weights):
        raise ValueError("Need exactly one weight per ranking")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        if weight == 0:
            continue
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))


def fuse(dense: Sequence[Hit], sparse: Sequence[Hit], weights: Weights, rrf_k: int) -> list[Hit]:
    """Fuse two hit lists with weighted RRF, keeping each hit's per-retriever rank."""
    dense_rank = {h.id: h.rank for h in dense}
    sparse_rank = {h.id: h.rank for h in sparse}
    payloads = {h.id: h.payload for h in (*sparse, *dense)}
    fused = rrf_scores(
        [[h.id for h in dense], [h.id for h in sparse]], [weights.dense, weights.sparse], rrf_k
    )
    return [
        Hit(doc_id, score, rank, payloads[doc_id], dense_rank.get(doc_id), sparse_rank.get(doc_id))
        for rank, (doc_id, score) in enumerate(fused, start=1)
    ]


def collapse(hits: Sequence[Hit], key: str) -> list[Hit]:
    """Keep the best-ranked hit per payload[key] (e.g. one chunk per document), re-ranked."""
    out: list[Hit] = []
    seen: set = set()
    for h in hits:
        group = h.payload.get(key, h.id)
        if group in seen:
            continue
        seen.add(group)
        out.append(replace(h, id=str(group), rank=len(out) + 1))
    return out


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


class Retriever:
    """Runs dense, sparse or hybrid retrieval against one Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient,
        embedder,
        collection: str | None = None,
        *,
        rrf_k: int | None = None,
        depth: int | None = None,
        weights: Weights | None = None,
        weighting: Callable[[str], Weights] | None = None,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection or settings.COLLECTION_NAME
        self.rrf_k = settings.RRF_K if rrf_k is None else rrf_k
        self.depth = depth or settings.CANDIDATE_DEPTH
        self.weights = weights or Weights(settings.DENSE_WEIGHT, settings.SPARSE_WEIGHT)
        self.weighting = weighting

    def weights_for(self, query: str) -> Weights:
        """Static weights, or query-dependent ones when a weighting function is set."""
        return self.weighting(query) if self.weighting else self.weights

    def encode(
        self, query: str, mode: Mode
    ) -> tuple[list[float] | None, dict[int, float] | None, dict[str, float]]:
        """Encode a query with the encoders its mode needs, timing each one."""
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        timings = {"dense_encode": 0.0, "sparse_encode": 0.0}
        dense_vec = sparse_vec = None
        if mode in ("dense", "hybrid"):
            t = time.perf_counter()
            dense_vec = self.embedder.encode_dense([query])[0]
            timings["dense_encode"] = _ms(t)
        if mode in ("sparse", "hybrid"):
            t = time.perf_counter()
            sparse_vec = self.embedder.encode_sparse([query])[0]
            timings["sparse_encode"] = _ms(t)
        return dense_vec, sparse_vec, timings

    def search(
        self,
        dense_vec: list[float] | None,
        sparse_vec: dict[int, float] | None,
        *,
        mode: Mode = "hybrid",
        top_k: int | None = None,
        rrf_k: int | None = None,
        weights: Weights | None = None,
        depth: int | None = None,
        group_by: str | None = None,
        with_payload: bool | list[str] = True,
    ) -> RetrievalResult:
        """Search with already-encoded query vectors and fuse the rankings."""
        top_k = top_k or settings.TOP_K
        rrf_k = self.rrf_k if rrf_k is None else rrf_k
        weights = weights or self.weights
        depth = max(top_k, depth or self.depth)

        requests, names = [], []
        if mode in ("dense", "hybrid"):
            requests.append(
                models.QueryRequest(
                    query=dense_vec, using=DENSE_VECTOR, limit=depth, with_payload=with_payload
                )
            )
            names.append("dense")
        if mode in ("sparse", "hybrid"):
            requests.append(
                models.QueryRequest(
                    query=to_sparse_vector(sparse_vec),
                    using=SPARSE_VECTOR,
                    limit=depth,
                    with_payload=with_payload,
                )
            )
            names.append("sparse")

        t = time.perf_counter()
        responses = self.client.query_batch_points(self.collection, requests=requests)
        qdrant_ms = _ms(t)
        ranked = {"dense": [], "sparse": []}
        for name, resp in zip(names, responses, strict=True):
            hits = [
                Hit(str(p.id), p.score, rank, p.payload or {})
                for rank, p in enumerate(resp.points, start=1)
            ]
            ranked[name] = collapse(hits, group_by) if group_by else hits

        t = time.perf_counter()
        fused = fuse(ranked["dense"], ranked["sparse"], weights, rrf_k) if mode == "hybrid" else []
        fusion_ms = _ms(t)
        return RetrievalResult(
            mode=mode,
            top_k=top_k,
            rrf_k=rrf_k,
            weights=weights,
            dense=ranked["dense"][:top_k],
            sparse=ranked["sparse"][:top_k],
            fused=fused[:top_k],
            timings_ms={"qdrant": qdrant_ms, "fusion": fusion_ms},
        )

    def retrieve(
        self,
        query: str,
        mode: Mode = "hybrid",
        top_k: int | None = None,
        *,
        rrf_k: int | None = None,
        weights: Weights | None = None,
        group_by: str | None = None,
    ) -> RetrievalResult:
        """Encode the query, search, and fuse. The main entry point."""
        t = time.perf_counter()
        dense_vec, sparse_vec, timings = self.encode(query, mode)
        if mode == "hybrid" and weights is None:
            weights = self.weights_for(query)
        result = self.search(
            dense_vec,
            sparse_vec,
            mode=mode,
            top_k=top_k,
            rrf_k=rrf_k,
            weights=weights,
            group_by=group_by,
        )
        result.timings_ms = {**timings, **result.timings_ms, "total": _ms(t)}
        return result
