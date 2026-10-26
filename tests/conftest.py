"""Shared fixtures. Nothing here loads a real model, LLM or Qdrant server."""

import hashlib
import os
import re

import pytest

os.environ.setdefault("MODEL_PATH", "/nonexistent/model.gguf")

from src.config import Settings, settings  # noqa: E402

# Defaults plus explicit environment variables only, so a developer's .env cannot change
# what the tests check.
_isolated = Settings(_env_file=None)
for _name in Settings.model_fields:
    setattr(settings, _name, getattr(_isolated, _name))

FAKE_DIM = 16
_WORD = re.compile(r"\w+")


def _bucket(token: str, size: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % size


class FakeEmbedder:
    """Deterministic bag-of-words embedder with the same interface as LocalHybridEmbedder."""

    dense_dim = FAKE_DIM
    dense_model_name = "fake-dense"
    sparse_model_name = "fake-sparse"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _WORD.finditer(text)]

    def encode_dense(self, texts: list[str]) -> list[list[float]]:
        return [self.get_dense(t) for t in texts]

    def encode_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        return [self.get_sparse(t) for t in texts]

    def get_dense(self, text: str) -> list[float]:
        self.calls.append(text)
        vec = [0.0] * FAKE_DIM
        for tok in _WORD.findall(text.lower()):
            vec[_bucket(tok, FAKE_DIM)] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def get_sparse(self, text: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for tok in _WORD.findall(text.lower()):
            idx = _bucket(tok, 30522)
            out[idx] = out.get(idx, 0.0) + 1.0
        return out


class FakeLLM:
    """LLM stand-in that returns scripted outputs and records prompts."""

    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.prompts: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        self.prompts.append(prompt)
        return self.outputs.pop(0) if self.outputs else "ok"


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def seeded_retriever(memory_qdrant, fake_embedder):
    """A Retriever over a few small documents in in-memory Qdrant."""
    from src.database import init_qdrant
    from src.ingestion import Document, ingest_documents
    from src.retrieval import Retriever

    client = init_qdrant(memory_qdrant, "seeded", FAKE_DIM)
    docs = [
        Document("install.md", "Install", "install the package with pip install neural"),
        Document("errors.md", "Errors", "error E1234 means the index is corrupt"),
        Document("fusion.md", "Fusion", "reciprocal rank fusion merges dense and sparse lists"),
    ]
    ingest_documents(client, "seeded", docs, fake_embedder, chunk_size=16, chunk_overlap=2)
    return Retriever(client, fake_embedder, "seeded")


@pytest.fixture
def memory_qdrant():
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    yield client
    client.close()
