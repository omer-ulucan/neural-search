import pytest
from fastapi.testclient import TestClient

from src.config import Settings, settings
from src.main import Resources, create_app


def preflight(client, origin):
    return client.options(
        "/retrieve",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )


@pytest.fixture
def make_client(seeded_retriever):
    def make():
        return TestClient(create_app(lambda: Resources(retriever=seeded_retriever)))

    return make


@pytest.mark.parametrize("origin", ["https://ulucan.dev", "http://localhost:5173"])
def test_default_origins_are_allowed(make_client, origin):
    with make_client() as c:
        r = preflight(c, origin)
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == origin
        post = c.post("/retrieve", json={"query": "pip"}, headers={"Origin": origin})
        assert post.headers["access-control-allow-origin"] == origin


def test_unknown_origin_is_not_allowed(make_client):
    with make_client() as c:
        r = preflight(c, "https://evil.example")
        assert "access-control-allow-origin" not in r.headers


def test_origins_are_configurable(make_client, monkeypatch):
    monkeypatch.setattr(settings, "CORS_ORIGINS", " https://demo.example ,, ")
    with make_client() as c:
        assert preflight(c, "https://demo.example").status_code == 200
        assert "access-control-allow-origin" not in preflight(c, "https://ulucan.dev").headers


def test_cors_origins_env_parsing(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example,http://localhost:1")
    assert Settings().cors_origins == ["https://a.example", "http://localhost:1"]
