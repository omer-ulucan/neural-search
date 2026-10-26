from src import database
from src.config import Settings, settings


def test_client_uses_configured_host_and_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(database, "QdrantClient", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(settings, "QDRANT_HOST", "http://qdrant:6333")
    monkeypatch.setattr(settings, "QDRANT_TIMEOUT", 90)
    database.qdrant_client()
    assert calls == [(("http://qdrant:6333",), {"timeout": 90})]


def test_default_timeout_exceeds_client_default():
    assert Settings.model_fields["QDRANT_TIMEOUT"].default > 5
