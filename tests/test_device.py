import pytest
import torch

from src.embedder import resolve_device


@pytest.mark.parametrize(("available", "expected"), [(True, "cuda"), (False, "cpu")])
def test_auto_picks_cuda_only_when_available(monkeypatch, available, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert resolve_device("auto") == expected
    assert resolve_device(" AUTO ") == expected


def test_explicit_cpu_is_kept(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu") == "cpu"


def test_explicit_cuda_without_gpu_fails_loudly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        resolve_device("cuda:0")


def test_default_comes_from_settings(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(settings, "DEVICE", "auto")
    assert resolve_device() == "cpu"
