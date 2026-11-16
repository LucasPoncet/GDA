import pytest

from src.data.splits import splits
from src.data.synthetic import make_dataset
from src.graphs.ring import generate_ring_transfer_graph


def test_sizes_add_up():
    graphs = make_dataset(generate_ring_transfer_graph, n_samples=100, classes=5, seed=0, nodes=10)
    train, val, test = splits(graphs, [0.8, 0.1, 0.1], seed=0, batch_size=8)
    assert len(train.dataset) == 80
    assert len(val.dataset) == 10
    assert len(test.dataset) == 10


def test_no_overlap():
    graphs = make_dataset(generate_ring_transfer_graph, n_samples=100, classes=5, seed=0, nodes=10)
    train, val, test = splits(graphs, [0.8, 0.1, 0.1], seed=0, batch_size=8)
    ids_train = {id(g) for g in train.dataset}
    ids_val = {id(g) for g in val.dataset}
    ids_test = {id(g) for g in test.dataset}
    assert not ids_train & ids_val
    assert not ids_train & ids_test
    assert not ids_val & ids_test


def test_bad_ratio_raises():
    graphs = make_dataset(generate_ring_transfer_graph, n_samples=100, classes=5, seed=0, nodes=10)
    with pytest.raises(ValueError):
        splits(graphs, [0.8, 0.1, 0.2])