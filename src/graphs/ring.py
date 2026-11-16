import numpy as np


def generate_ring_transfer_graph(nodes:int, classe_index:int, add_crosses:bool=False, classes:int=5):
    """
    Generate one ring transfer graph.

    Args:
    - nodes (int): Number of nodes in each graph.
    - classe_index (int): index of the class of the source node in the graph
    - add_crosses (bool): Whether to add cross edges in the ring.
    - classes (int): Number of different classes or labels.

    Returns:
    - np.ndarray: Edge array of size (2,E) in both directions.
    - np.ndarray: X (n,C) a node feature array.
    - np.ndarray: Y the class index array.
    - int: readout node index (0).
    - int: information node index (n//2 here).
    - int: distance between readout and information nodes (only valid if add_crosses=False).
    """
    if nodes < 3:
        raise ValueError("At least three nodes are required for a ring.")

    source_node = nodes//2

    x = np.ones((nodes,classes))

    x[0, :] = 0.0
    x[source_node, :] = 0.0
    x[source_node, classe_index] = 1.0


    edge_index = []
    for i in range(nodes-1):
        edge_index.append([i,i+1])
        edge_index.append([i+1,i])

        if add_crosses and i<source_node:
            edge_index.append([i, nodes-1-i])
            edge_index.append([nodes-i-1,i])
            if nodes + 1 - i < nodes:
                            edge_index.append([i, nodes + 1 - i])
                            edge_index.append([nodes + 1 - i, i])
            

    edge_index.append([0, nodes - 1])
    edge_index.append([nodes - 1, 0])
    
    y = np.array([classe_index], dtype=np.int64)

    return np.array(edge_index, dtype=np.int64),x,y,0,source_node,source_node