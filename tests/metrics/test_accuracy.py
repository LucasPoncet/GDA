import torch

from src.metrics.accuracy import accuracy


def test_all_correct():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    y = torch.tensor([0, 1])
    assert accuracy(logits, y) == 1.0


def test_all_wrong():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    y = torch.tensor([2, 2])
    assert accuracy(logits, y) == 0.0


def test_half_correct():
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
    y = torch.tensor([0, 1, 1, 0])
    assert accuracy(logits, y) == 0.5