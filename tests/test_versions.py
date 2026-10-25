import importlib.metadata
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_qdrant_server_image_matches_client_minor_version():
    compose = (ROOT / "docker-compose.yml").read_text()
    server = re.search(r"qdrant/qdrant:v(\d+)\.(\d+)", compose).groups()
    client = importlib.metadata.version("qdrant-client").split(".")[:2]
    assert list(server) == client
