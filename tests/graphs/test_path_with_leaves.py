import networkx as nx
import numpy as np
import torch

from src.graphs.path_with_leaves import generate_path_with_leaves_transfer_graph
from src.graphs.tree import generate_tree_transfer_graph
from src.metrics.sensitivity import mpnn_matrix, theoretical_sensitivity

def to_nx(edges):
    """Undirected networkx graph from a (2, E) edge array."""
    g = nx.Graph()
    g.add_edges_from(edges.T.tolist())
    return g


def test_shapes():
    edges, x, y, readout, info, dist = generate_path_with_leaves_transfer_graph(2, class_index=2, classes=5)
    assert edges.shape == (2, 8)
    assert x.shape == (5, 5)
    assert y.shape == (1,)


def test_edges_symmetric_no_duplicates():
    edges, x, y, readout, info, dist = (generate_path_with_leaves_transfer_graph(3, class_index=0))
    pairs = [tuple(e) for e in edges.T.tolist()]
    assert len(pairs) == len(set(pairs))
    for i, j in pairs:
        assert (j, i) in pairs


def test_distance():
    edges, x, y, readout, info, dist = (generate_path_with_leaves_transfer_graph(3, class_index=0))
    g = to_nx(edges)
    assert nx.shortest_path_length(g, 0, 3) == dist == 3


def test_features():
    edges, x, y, readout, info, dist = (generate_path_with_leaves_transfer_graph(2, class_index=3, classes=5))
    assert np.all(x[readout] == 0)
    assert np.array_equal(x[info], [0, 0, 0, 1, 0])
    assert np.all(x[1] == 1)
    assert np.all(x[3:] == 1)
    

def test_degrees_match_tree():
    edges, x, y, readout, info, dist = (generate_path_with_leaves_transfer_graph(3, class_index=0))
    tree_edges, tree_x, tree_y, tree_readout, tree_info, tree_dist = (generate_tree_transfer_graph(3, class_index=0, arity=2))
    g = to_nx(edges)
    tree = to_nx(tree_edges)
    path = nx.shortest_path(g, readout, info)
    tree_path = nx.shortest_path(tree, tree_readout, tree_info)
    assert [g.degree(i) for i in path] == [tree.degree(i) for i in tree_path]
    

def test_theoretical_sensitivity_matches_tree():
    edges, x, y, readout, info, dist = (generate_path_with_leaves_transfer_graph(3, class_index=0))
    tree_edges, tree_x, tree_y, tree_readout, tree_info, tree_dist = (generate_tree_transfer_graph(3, class_index=0, arity=2))
    matrix = mpnn_matrix(torch.tensor(edges), x.shape[0])
    tree_matrix = mpnn_matrix(torch.tensor(tree_edges), tree_x.shape[0])
    sensitivity = theoretical_sensitivity(matrix, readout, info, num_layers=3)
    tree_sensitivity = theoretical_sensitivity(tree_matrix, tree_readout, tree_info, num_layers=3)
    assert abs(sensitivity - tree_sensitivity) < 1e-8