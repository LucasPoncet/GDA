import numpy as np


def generate_tree_transfer_graph(depth: int, class_index: int, classes: int = 5, arity: int = 2):
    """
    Generate one tree transfer graph: a complete tree in heap order, root 0, parent of node c is (c - 1) // arity.

    Args:
    - depth (int): Depth of the tree (root at level 0, leaves at level depth).
    - class_index (int): Class carried by the information node.
    - classes (int): Number of classes.
    - arity (int): Number of children per internal node.

    Returns:
    - np.ndarray: Edge array of size (2, E), both directions.
    - np.ndarray: X (n, C) node features.
    - np.ndarray: Y, the class index, shape (1,).
    - int: readout node index (0, the root).
    - int: information node index (n - 1, a deepest leaf).
    - int: distance between readout and information nodes (depth).
    """
    if depth < 1:
        raise ValueError("Depth must be at least 1.")
    if arity < 2:
        raise ValueError("Arity must be at least 2.")

    nodes = (arity ** (depth + 1) - 1) // (arity - 1)
    info_node = nodes - 1

    x = np.ones((nodes, classes))
    x[0, :] = 0.0
    x[info_node, :] = 0.0
    x[info_node, class_index] = 1.0

    edge_index = []
    for child in range(1, nodes):
        parent = (child - 1) // arity
        edge_index.append([parent, child])
        edge_index.append([child, parent])

    y = np.array([class_index], dtype=np.int64)

    return np.array(edge_index, dtype=np.int64).T, x, y, 0, info_node, depth