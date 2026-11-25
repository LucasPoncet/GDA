import numpy as np


def generate_path_with_leaves_transfer_graph(depth: int, class_index: int, classes: int = 5):
    """
    Generate a path from node 0 to node depth, with one extra leaf per node except the last.
    The path has the same degrees as a root-to-leaf path in a binary tree.
    
    Args:
    - depth (int): Distance between readout and information nodes (length of the path).
    - class_index (int): Class carried by the information node.
    - classes (int): Number of classes.

    Returns:
    - np.ndarray: Edge array of size (2, 4 * depth), both directions.
    - np.ndarray: X (2 * depth + 1, C) node features.
    - np.ndarray: Y, the class index, shape (1,).
    - int: readout node index (0, start of the path).
    - int: information node index (depth, end of the path).
    - int: distance between readout and information nodes (depth).
    """
    if depth < 1:
        raise ValueError("Depth must be at least 1.")

    nodes = 2*depth + 1
    info_node = depth

    x = np.ones((nodes, classes))
    x[0, :] = 0.0
    x[info_node, :] = 0.0
    x[info_node, class_index] = 1.0

    edge_index = []
    for i in range(depth):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])
        
        leaf = depth + 1 + i
        edge_index.append([i, leaf])
        edge_index.append([leaf, i])

    y = np.array([class_index], dtype=np.int64)

    return np.array(edge_index, dtype=np.int64).T, x, y, 0, info_node, depth