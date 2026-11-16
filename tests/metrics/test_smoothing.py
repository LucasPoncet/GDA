import torch

from src.metrics.smoothing import (
    dirichlet_energy,
    effective_rank,
    mad,
    smoothing_per_layer,
)

EDGES = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])


def test_constant_features():
    h = torch.ones(4, 3)
    assert dirichlet_energy(h, EDGES) == 0
    assert abs(mad(h, EDGES)) < 1e-6
    assert abs(effective_rank(h) - 1) < 1e-6


def test_energy_by_hand():
    h = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    assert abs(dirichlet_energy(h, EDGES, normalized=False) - 6 / 4) < 1e-6


def test_energy_scale_invariant():
    h = torch.randn(4, 3)
    assert abs(dirichlet_energy(h, EDGES) - dirichlet_energy(10 * h, EDGES)) < 1e-5


def test_random_features():
    torch.manual_seed(0)
    h = torch.randn(4, 3)
    assert dirichlet_energy(h, EDGES) > 0
    assert mad(h) > 0
    assert effective_rank(h) > 1


def test_per_layer():
    embeddings = [torch.randn(4, 3), torch.ones(4, 3)]
    results = smoothing_per_layer(embeddings, EDGES)
    assert len(results['energy']) == 2
    assert results['energy'][1] == 0