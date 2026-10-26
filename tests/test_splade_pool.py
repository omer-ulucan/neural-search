import torch

from src.embedder import splade_pool


def reference(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (torch.log1p(torch.relu(logits)) * mask.unsqueeze(-1)).max(dim=1).values


def test_matches_activate_then_pool_exactly():
    torch.manual_seed(0)
    logits = torch.randn(4, 9, 50) * 3
    mask = torch.ones(4, 9, dtype=torch.long)
    mask[1, 5:] = 0
    mask[3, 2:] = 0
    expected = reference(logits, mask)
    assert torch.equal(splade_pool(logits.clone(), mask), expected)


def test_padding_never_contributes():
    logits = torch.tensor([[[-1.0, 2.0], [9.0, 9.0]]])
    mask = torch.tensor([[1, 0]])
    out = splade_pool(logits, mask)
    assert out[0, 0] == 0.0
    assert torch.isclose(out[0, 1], torch.log1p(torch.tensor(2.0)))
