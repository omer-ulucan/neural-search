import os
from pathlib import Path

from src.config import Settings, settings

ROOT = Path(__file__).resolve().parent.parent


def test_env_example_lists_every_setting():
    lines = (ROOT / ".env.example").read_text().splitlines()
    keys = {ln.split("=", 1)[0] for ln in lines if "=" in ln and not ln.startswith("#")}
    assert set(Settings.model_fields) <= keys


def test_tests_see_defaults_not_the_local_env_file():
    for name, field in Settings.model_fields.items():
        if name not in os.environ:
            assert getattr(settings, name) == field.default, name
