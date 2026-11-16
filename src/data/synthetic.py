from typing import Callable
import warnings

import numpy as np
import torch
from torch_geometric.data import Data

def to_pyg(
    edges: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    readout_node: int,
    info_node: int,
    distance: int,
) -> Data:
    """Returns Data(x, edge_index, y, readout_mask, info_mask, distance)."""
    x = torch.tensor(x, dtype=torch.float32)
    edges = torch.tensor(edges, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    n = x.shape[0]
    readout_mask = torch.zeros(n, dtype=torch.bool)
    readout_mask[readout_node] = True
    info_mask = torch.zeros(n, dtype=torch.bool)
    info_mask[info_node] = True
    data = Data(
        x=x,
        edge_index=edges,
        y=y,
        readout_mask=readout_mask,
        info_mask=info_mask,
        distance=torch.tensor([distance], dtype=torch.long),
    )
    return data

def make_dataset(
    generator: Callable,
    n_samples: int,
    classes: int = 5,
    seed: int = 0,
    **graph_kwargs,
) -> list[Data]:
    """Returns n_samples graphs with balanced classes (remainder padded), shuffled with seed."""
    rng = np.random.default_rng(seed)

    labels = np.repeat(np.arange(classes), n_samples // classes)
    remainder= n_samples % classes
    if remainder:
        warnings.warn(f"n_samples={n_samples} not divisible by classes={classes}: {remainder} classes get one extra sample.")
        extra = rng.choice(classes, size=remainder, replace=False)
        labels = np.concatenate([labels, extra])

    labels = rng.permutation(labels)

    dataset = []
    for label in labels:
        edges, x, y, readout_node, info_node, distance = generator(class_index=label, classes=classes, **graph_kwargs)
        graph = to_pyg( edges, x, y, readout_node, info_node, distance)
        dataset.append(graph)
    return dataset