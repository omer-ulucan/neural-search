from pathlib import Path

from src.config import Settings

ROOT = Path(__file__).resolve().parent.parent


def test_env_example_lists_every_setting():
    lines = (ROOT / ".env.example").read_text().splitlines()
    keys = {ln.split("=", 1)[0] for ln in lines if "=" in ln and not ln.startswith("#")}
    assert set(Settings.model_fields) <= keys
