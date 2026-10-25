from tests.conftest import FAKE_DIM


def test_settings_load_without_real_model():
    from src.config import settings

    assert settings.COLLECTION_NAME


def test_fake_embedder_is_deterministic(fake_embedder):
    a = fake_embedder.get_dense("qdrant hybrid search")
    b = fake_embedder.get_dense("qdrant hybrid search")
    assert a == b and len(a) == FAKE_DIM
    assert fake_embedder.get_sparse("qdrant qdrant")


def test_memory_qdrant_is_usable(memory_qdrant):
    assert memory_qdrant.get_collections().collections == []
