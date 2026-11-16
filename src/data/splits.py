import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def splits(graphs: list[Data], split_ratio: list[float], seed: int = 0, batch_size: int = 64):
    """Generate the train, validation and test DataLoaders from a shuffled list of graphs."""
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError(f"split_ratio must sum to 1, got {split_ratio}")
    n = len(graphs)
    n_train, n_val = int(n * split_ratio[0]), int(n * split_ratio[1])
    g_train = graphs[:n_train]
    g_val = graphs[n_train:n_train + n_val]
    g_test = graphs[n_train + n_val:]
    train_dataloader = DataLoader(
        g_train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_dataloader = DataLoader(g_val, batch_size=batch_size)
    test_dataloader = DataLoader(g_test, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader