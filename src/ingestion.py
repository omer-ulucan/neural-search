"""Idempotent chunked ingestion of documents into Qdrant.

Every chunk gets a stable point id derived from (source, chunk_index), and every
point stores a fingerprint of the document text plus the chunking and model
config. Re-running ingestion therefore:

- skips documents whose chunks are already stored with the current fingerprint,
- re-embeds documents that changed, then deletes chunks they no longer produce,
- deletes every chunk of documents that disappeared (when `prune` is on).
"""

import hashlib
import json
import logging
import re
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.chunking import chunk_text
from src.database import DENSE_VECTOR, SPARSE_VECTOR, to_sparse_vector

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".txt", ".md")
POINT_NAMESPACE = uuid.UUID("6f0f5d2c-3c1e-4a8e-9a57-2b8f6c1d9e41")
_MD_HEADING = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Document:
    """A source document. `source` is its stable identity, e.g. a relative path."""

    source: str
    title: str
    text: str


@dataclass
class IngestStats:
    """Counts describing what one ingestion run changed."""

    unchanged: int = 0
    embedded: int = 0
    pruned: int = 0
    chunks_upserted: int = 0
    stale_chunks_deleted: int = 0
    empty: list[str] = field(default_factory=list)


def point_id(source: str, chunk_index: int) -> str:
    """Return the stable Qdrant point id of a chunk."""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{source}#{chunk_index}"))


def fingerprint(doc: Document, config: dict) -> str:
    """Hash of document content plus everything that changes its vectors."""
    blob = json.dumps({"title": doc.title, "text": doc.text, **config}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def read_text_file(path: Path) -> str:
    """Decode a text file, honouring UTF-16 and UTF-8 byte order marks."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")  # unreachable: latin-1 accepts any bytes


def extract_title(path: Path, text: str) -> str:
    """Use the first Markdown H1 for .md files, otherwise a cleaned-up file stem."""
    if path.suffix == ".md":
        match = _MD_HEADING.search(text)
        if match:
            return match.group(1)
    return re.sub(r"[-_]+", " ", path.stem).strip()


def read_documents(docs_dir: str | Path) -> list[Document]:
    """Read every .txt and .md file under docs_dir, recursively, in sorted order."""
    root = Path(docs_dir)
    docs = []
    for path in sorted(p for p in root.rglob("*") if p.suffix in SUPPORTED_SUFFIXES):
        if not path.is_file():
            continue
        text = read_text_file(path)
        docs.append(Document(path.relative_to(root).as_posix(), extract_title(path, text), text))
    return docs


def _existing_points(client: QdrantClient, collection: str) -> dict[str, dict[str, str]]:
    """Map source -> {point_id: fingerprint} for every point in the collection."""
    out: dict[str, dict[str, str]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection,
            limit=1024,
            offset=offset,
            with_payload=["source", "fingerprint"],
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            out.setdefault(payload.get("source", ""), {})[str(p.id)] = payload.get("fingerprint")
        if offset is None:
            return out


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ingest_documents(
    client: QdrantClient,
    collection: str,
    docs: Iterable[Document],
    embedder,
    *,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int = 64,
    prune: bool = True,
) -> IngestStats:
    """Chunk, embed and upsert documents, removing stale and orphaned chunks."""
    docs = list(docs)
    sources = [d.source for d in docs]
    if len(set(sources)) != len(sources):
        raise ValueError("Document sources must be unique")

    config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "dense_model": getattr(embedder, "dense_model_name", type(embedder).__name__),
        "sparse_model": getattr(embedder, "sparse_model_name", type(embedder).__name__),
    }
    existing = _existing_points(client, collection)
    stats = IngestStats()
    stale_ids: list[str] = []
    pending: list[models.PointStruct] = []

    def flush() -> None:
        if not pending:
            return
        texts = [p.payload["text"] for p in pending]
        dense = embedder.encode_dense(texts)
        sparse = embedder.encode_sparse(texts)
        for point, d, s in zip(pending, dense, sparse, strict=True):
            point.vector = {DENSE_VECTOR: d, SPARSE_VECTOR: to_sparse_vector(s)}
        client.upsert(collection, points=pending, wait=True)
        stats.chunks_upserted += len(pending)
        pending.clear()

    for doc in docs:
        fp = fingerprint(doc, config)
        chunks = chunk_text(doc.text, embedder.token_offsets, chunk_size, chunk_overlap)
        if not chunks:
            stats.empty.append(doc.source)
        wanted = {point_id(doc.source, c.index): c for c in chunks}
        stored = existing.get(doc.source, {})
        stale_ids.extend(pid for pid in stored if pid not in wanted)
        if all(stored.get(pid) == fp for pid in wanted):
            stats.unchanged += 1
            continue
        stats.embedded += 1
        for pid, c in wanted.items():
            payload = {
                "source": doc.source,
                "title": doc.title,
                "chunk_index": c.index,
                "chunk_count": len(chunks),
                "start_char": c.start_char,
                "end_char": c.end_char,
                "text": c.text,
                "fingerprint": fp,
            }
            pending.append(models.PointStruct(id=pid, vector={}, payload=payload))
            if len(pending) >= batch_size:
                flush()
        logger.info("Queued %s (%d chunks)", doc.source, len(chunks))
    flush()

    if prune:
        keep = set(sources)
        for source, stored in existing.items():
            if source not in keep:
                stale_ids.extend(stored)
                stats.pruned += 1
    # Deletes run only after every upsert has landed, so a document is never missing.
    for batch in _batched(stale_ids, 1024):
        client.delete(collection, points_selector=models.PointIdsList(points=batch), wait=True)
    stats.stale_chunks_deleted = len(stale_ids)
    return stats
