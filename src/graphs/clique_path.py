import numpy as np


def generate_clique_path_transfer_graph(nodes: int, class_index: int, classes: int = 5):
    """
    Generate one clique-path transfer graph: a clique on nodes 0..n//2-1, a path on n//2..n-1, bridged by (n//2-1, n//2).

    Args:
    - nodes (int): Number of nodes in the graph.
    - class_index (int): Class carried by the information node.
    - classes (int): Number of classes.

    Returns:
    - np.ndarray: Edge array of size (2, E), both directions.
    - np.ndarray: X (n, C) node features.
    - np.ndarray: Y, the class index, shape (1,).
    - int: readout node index (0).
    - int: information node index (n - 1, end of the path).
    - int: distance between readout and information nodes (1 + n - n//2).
    """
    if nodes < 4:
        raise ValueError("At least four nodes are required for a clique path.")

    info_node = nodes - 1
    distance = 1 + nodes - nodes // 2

    x = np.ones((nodes, classes))
    x[0, :] = 0.0
    x[info_node, :] = 0.0
    x[info_node, class_index] = 1.0

    edge_index = []
    for i in range(nodes // 2):
        for j in range(i + 1, nodes // 2):
            edge_index.append([i, j])
            edge_index.append([j, i])
    for i in range(nodes // 2, nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index.append([nodes // 2 - 1, nodes // 2])
    edge_index.append([nodes // 2, nodes // 2 - 1])

    y = np.array([class_index], dtype=np.int64)

    return np.array(edge_index, dtype=np.int64).T, x, y, 0, info_node, distance