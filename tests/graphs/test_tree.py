import networkx as nx
import numpy as np

from src.graphs.tree import generate_tree_transfer_graph


def to_nx(edges):
    """Undirected networkx graph from a (2, E) edge array."""
    g = nx.Graph()
    g.add_edges_from(edges.T.tolist())
    return g


def test_shapes():
    edges, x, y, readout, info, dist = generate_tree_transfer_graph(2, class_index=2, classes=5, arity=2)
    assert edges.shape[0] == 2
    assert x.shape == (7, 5)
    assert y.shape == (1,)


def test_is_tree():
    edges, x, y, readout, info, dist = generate_tree_transfer_graph(3, class_index=0, arity=3)
    g = to_nx(edges)
    assert g.number_of_nodes() == 40
    assert nx.is_tree(g)


def test_distance():
    edges, x, y, readout, info, dist = generate_tree_transfer_graph(3, class_index=0, arity=2)
    g = to_nx(edges)
    assert nx.shortest_path_length(g, readout, info) == dist == 3


def test_info_is_a_leaf():
    edges, x, y, readout, info, dist = generate_tree_transfer_graph(3, class_index=0, arity=2)
    g = to_nx(edges)
    assert g.degree(info) == 1


def test_edges_symmetric_no_duplicates():
    edges, x, y, readout, info, dist = generate_tree_transfer_graph(3, class_index=0, arity=2)
    pairs = [tuple(e) for e in edges.T.tolist()]
    assert len(pairs) == len(set(pairs))
    for i, j in pairs:
        assert (j, i) in pairs


def test_features():
    edges, x, y, readout, info, dist = generate_tree_transfer_graph(2, class_index=3, classes=5, arity=2)
    assert info == 6
    assert np.array_equal(x[info], [0, 0, 0, 1, 0])
    assert np.all(x[readout] == 0)
    assert np.all(x[1] == 1)
    assert y[0] == 3