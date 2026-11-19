import torch
from torch import nn

from src.data.synthetic import to_pyg
from src.graphs.ring import generate_ring_transfer_graph
from src.metrics.sensitivity import (
    gcn_matrix,
    jacobian_sensitivity,
    mpnn_matrix,
    theoretical_sensitivity,
)


class LinearGCN(nn.Module):
    """h^(l+1) = S h^(l): no weights, no nonlinearity, so the Jacobian equals (S^l)[v, u] times identity."""

    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers

    def forward(self, data):
        S = gcn_matrix(data.edge_index, data.num_nodes)
        h = data.x
        embeddings = [h]
        for _ in range(self.num_layers):
            h = S @ h
            embeddings.append(h)
        return embeddings


def ring_data():
    """10-node ring, readout node 0, information node 5, distance 5."""
    return to_pyg(*generate_ring_transfer_graph(10, class_index=0))


def test_normalized_adjacency_on_ring():
    data = ring_data()
    S = gcn_matrix(data.edge_index, data.num_nodes)
    assert torch.isclose(S[0, 0], torch.tensor(1 / 3))
    assert torch.isclose(S[0, 1], torch.tensor(1 / 3))
    assert S[0, 2] == 0


def test_theoretical_sensitivity_on_ring():
    data = ring_data()
    S = gcn_matrix(data.edge_index, data.num_nodes)
    value_5 = theoretical_sensitivity(S, 0, 5, num_layers=5)
    value_4 = theoretical_sensitivity(S, 0, 5, num_layers=4)
    assert abs(value_5 - 2 * (1 / 3) ** 5) < 1e-6
    assert value_4 == 0


def test_jacobian_matches_theory():
    data = ring_data()
    sens = jacobian_sensitivity(LinearGCN(num_layers=5), data)
    assert len(sens) == 6
    assert all(s == 0 for s in sens[:5])
    expected = theoretical_sensitivity(gcn_matrix(data.edge_index, data.num_nodes), 0, 5, num_layers=5)
    assert abs(sens[5] - expected) < 1e-6

def test_mpnn_matrix_on_ring():
    data = ring_data()
    S = mpnn_matrix(data.edge_index, data.num_nodes)
    assert S[0, 0] == 0
    assert torch.isclose(S[0, 1], torch.tensor(0.5))
    assert S[0, 2] == 0