import networkx as nx
import numpy as np

from src.graphs.clique_path import generate_clique_path_transfer_graph


def test_shapes():
    edges, x, y, readout, info, dist = generate_clique_path_transfer_graph(10, class_index=2, classes=5)
    assert edges.shape[0] == 2
    assert x.shape == (10, 5)
    assert y.shape == (1,)


def test_distance_even():
    edges, x, y, readout, info, dist = generate_clique_path_transfer_graph(10, class_index=0)
    g = nx.Graph()
    g.add_edges_from(edges.T.tolist())
    assert nx.shortest_path_length(g, readout, info) == dist == 6


def test_distance_odd():
    edges, x, y, readout, info, dist = generate_clique_path_transfer_graph(11, class_index=0)
    g = nx.Graph()
    g.add_edges_from(edges.T.tolist())
    assert nx.shortest_path_length(g, readout, info) == dist == 7


def test_edges_symmetric_no_duplicates():
    edges, x, y, readout, info, dist = generate_clique_path_transfer_graph(10, class_index=0)
    pairs = [tuple(e) for e in edges.T.tolist()]
    assert len(pairs) == len(set(pairs))
    for i, j in pairs:
        assert (j, i) in pairs


def test_clique_is_complete():
    edges, x, y, readout, info, dist = generate_clique_path_transfer_graph(10, class_index=0)
    g = nx.Graph()
    g.add_edges_from(edges.T.tolist())
    clique = g.subgraph(range(5))
    assert clique.number_of_edges() == 5 * 4 // 2


def test_features():
    edges, x, y, readout, info, dist = generate_clique_path_transfer_graph(10, class_index=3, classes=5)
    assert info == 9
    assert np.array_equal(x[info], [0, 0, 0, 1, 0])
    assert np.all(x[readout] == 0)
    assert np.all(x[1] == 1)
    assert y[0] == 3