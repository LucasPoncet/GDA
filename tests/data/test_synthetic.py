import torch
from torch_geometric.loader import DataLoader

from src.data.synthetic import make_dataset
from src.graphs.ring import generate_ring_transfer_graph


def test_length_and_balance():
    data = make_dataset(generate_ring_transfer_graph, n_samples=100, classes=5, seed=0, nodes=10)
    assert len(data) == 100
    labels = torch.cat([g.y for g in data])
    assert torch.bincount(labels, minlength=5).tolist() == [20, 20, 20, 20, 20]


def test_same_seed_same_order():
    a = make_dataset(generate_ring_transfer_graph, n_samples=50, classes=5, seed=0, nodes=10)
    b = make_dataset(generate_ring_transfer_graph, n_samples=50, classes=5, seed=0, nodes=10)
    c = make_dataset(generate_ring_transfer_graph, n_samples=50, classes=5, seed=1, nodes=10)
    labels_a = torch.cat([g.y for g in a])
    labels_b = torch.cat([g.y for g in b])
    labels_c = torch.cat([g.y for g in c])
    assert torch.equal(labels_a, labels_b)
    assert not torch.equal(labels_a, labels_c)


def test_not_sorted_by_class():
    data = make_dataset(generate_ring_transfer_graph, n_samples=50, classes=5, seed=0, nodes=10)
    labels = torch.cat([g.y for g in data])
    assert not torch.equal(labels, torch.sort(labels).values)


def test_batching_keeps_masks_and_distance():
    data = make_dataset(generate_ring_transfer_graph, n_samples=10, classes=5, seed=0, nodes=10)
    batch = next(iter(DataLoader(data, batch_size=2)))
    assert batch.readout_mask.sum() == 2
    assert batch.info_mask.sum() == 2
    assert batch.distance.shape == (2,)
    assert batch.y.shape == (2,)