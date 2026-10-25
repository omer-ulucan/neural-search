import pytest
from qdrant_client import models

from src.database import init_qdrant
from src.ingestion import Document, ingest_documents, point_id, read_documents
from tests.conftest import FAKE_DIM

COLLECTION = "test_docs"
CFG = {"chunk_size": 4, "chunk_overlap": 1, "batch_size": 3}


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


@pytest.fixture
def client(memory_qdrant):
    return init_qdrant(memory_qdrant, COLLECTION, FAKE_DIM)


def stored(client) -> dict[str, dict]:
    points, _ = client.scroll(COLLECTION, limit=10_000, with_payload=True, with_vectors=True)
    return {str(p.id): {**p.payload, "_vector": p.vector} for p in points}


def assert_no_orphans(client, docs, embedder, chunk_size=4, chunk_overlap=1):
    from src.chunking import chunk_text

    expected = {
        point_id(d.source, c.index)
        for d in docs
        for c in chunk_text(d.text, embedder.token_offsets, chunk_size, chunk_overlap)
    }
    assert set(stored(client)) == expected


def test_first_ingest_stores_every_chunk_with_metadata(client, fake_embedder):
    docs = [Document("a.md", "A", words(10)), Document("b.txt", "B", words(3))]
    stats = ingest_documents(client, COLLECTION, docs, fake_embedder, **CFG)
    assert stats.embedded == 2 and stats.chunks_upserted == 4
    assert_no_orphans(client, docs, fake_embedder)
    payload = stored(client)[point_id("a.md", 1)]
    assert payload["source"] == "a.md" and payload["title"] == "A"
    assert payload["chunk_index"] == 1 and payload["chunk_count"] == 3
    assert payload["text"] == "w3 w4 w5 w6"


def test_point_ids_are_stable_and_distinct_per_document():
    assert point_id("a.md", 0) == point_id("a.md", 0)
    assert len({point_id("a.md", 0), point_id("b.md", 0), point_id("a.md", 1)}) == 3


def test_reingesting_unchanged_docs_embeds_nothing(client, fake_embedder):
    docs = [Document("a.md", "A", words(10)), Document("b.txt", "B", words(3))]
    ingest_documents(client, COLLECTION, docs, fake_embedder, **CFG)
    before = stored(client)
    fake_embedder.calls.clear()
    stats = ingest_documents(client, COLLECTION, docs, fake_embedder, **CFG)
    assert stats.unchanged == 2 and stats.embedded == 0 and stats.stale_chunks_deleted == 0
    assert fake_embedder.calls == []
    assert stored(client) == before


def test_shrinking_document_leaves_no_orphaned_chunks(client, fake_embedder):
    ingest_documents(client, COLLECTION, [Document("a.md", "A", words(10))], fake_embedder, **CFG)
    docs = [Document("a.md", "A", words(4))]
    stats = ingest_documents(client, COLLECTION, docs, fake_embedder, **CFG)
    assert stats.stale_chunks_deleted == 2
    assert_no_orphans(client, docs, fake_embedder)
    assert {p["chunk_count"] for p in stored(client).values()} == {1}


def test_changed_document_is_reembedded_and_other_docs_untouched(client, fake_embedder):
    b = Document("b.txt", "B", words(5, "b"))
    ingest_documents(client, COLLECTION, [Document("a.md", "A", words(7)), b], fake_embedder, **CFG)
    b_before = {k: v for k, v in stored(client).items() if v["source"] == "b.txt"}
    docs = [Document("a.md", "A", words(7, "x")), b]
    stats = ingest_documents(client, COLLECTION, docs, fake_embedder, **CFG)
    assert stats.embedded == 1 and stats.unchanged == 1
    after = stored(client)
    assert after[point_id("a.md", 0)]["text"] == "x0 x1 x2 x3"
    assert {k: v for k, v in after.items() if v["source"] == "b.txt"} == b_before
    assert_no_orphans(client, docs, fake_embedder)


def test_deleted_document_is_pruned(client, fake_embedder):
    docs = [Document("a.md", "A", words(6)), Document("gone.md", "G", words(9))]
    ingest_documents(client, COLLECTION, docs, fake_embedder, **CFG)
    stats = ingest_documents(client, COLLECTION, docs[:1], fake_embedder, **CFG)
    assert stats.pruned == 1
    assert_no_orphans(client, docs[:1], fake_embedder)


def test_no_prune_keeps_documents_missing_from_this_run(client, fake_embedder):
    docs = [Document("a.md", "A", words(6)), Document("kept.md", "K", words(9))]
    ingest_documents(client, COLLECTION, docs, fake_embedder, **CFG)
    ingest_documents(client, COLLECTION, docs[:1], fake_embedder, prune=False, **CFG)
    assert_no_orphans(client, docs, fake_embedder)


def test_document_that_becomes_empty_loses_all_chunks(client, fake_embedder):
    ingest_documents(client, COLLECTION, [Document("a.md", "A", words(6))], fake_embedder, **CFG)
    stats = ingest_documents(
        client, COLLECTION, [Document("a.md", "A", "  ")], fake_embedder, **CFG
    )
    assert stats.empty == ["a.md"]
    assert stored(client) == {}


def test_leftover_stale_chunks_are_cleaned_without_reembedding(client, fake_embedder):
    """A run interrupted between upsert and delete must be repaired by the next run."""
    ingest_documents(client, COLLECTION, [Document("a.md", "A", words(10))], fake_embedder, **CFG)
    docs = [Document("a.md", "A", words(4))]
    # Simulate the crash: new chunk 0 written, stale chunks 1 and 2 never deleted.
    ingest_documents(client, COLLECTION, docs, fake_embedder, **CFG)
    stale = models.PointStruct(
        id=point_id("a.md", 1),
        vector={"": [1.0] + [0.0] * (FAKE_DIM - 1)},
        payload={"source": "a.md", "fingerprint": "old"},
    )
    client.upsert(COLLECTION, points=[stale])
    fake_embedder.calls.clear()
    stats = ingest_documents(client, COLLECTION, docs, fake_embedder, **CFG)
    assert fake_embedder.calls == [] and stats.stale_chunks_deleted == 1
    assert_no_orphans(client, docs, fake_embedder)


def test_changing_chunk_config_reembeds_everything(client, fake_embedder):
    docs = [Document("a.md", "A", words(10))]
    ingest_documents(client, COLLECTION, docs, fake_embedder, **CFG)
    cfg = {**CFG, "chunk_size": 6}
    stats = ingest_documents(client, COLLECTION, docs, fake_embedder, **cfg)
    assert stats.embedded == 1
    assert_no_orphans(client, docs, fake_embedder, chunk_size=6)


def test_duplicate_sources_are_rejected(client, fake_embedder):
    docs = [Document("a.md", "A", "x"), Document("a.md", "A2", "y")]
    with pytest.raises(ValueError):
        ingest_documents(client, COLLECTION, docs, fake_embedder, **CFG)


def test_read_documents_handles_md_txt_nesting_and_utf16(tmp_path):
    (tmp_path / "guide.md").write_text("intro\n# Install Guide\nbody", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "error_codes.txt").write_bytes("E42 means bad".encode("utf-16"))
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    docs = read_documents(tmp_path)
    assert [(d.source, d.title) for d in docs] == [
        ("guide.md", "Install Guide"),
        ("sub/error_codes.txt", "error codes"),
    ]
    assert docs[1].text == "E42 means bad"
