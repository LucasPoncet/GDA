import sys
from pathlib import Path

file = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(file))

import torch
from torch import Tensor


def route(receiver: Tensor, best_slot: Tensor, opened: Tensor, k: int) -> Tensor:
    """
    Route every index of the receiver (index refers to the node it should be sent to), to the slot of this node the more appropriated.
    The rule is:
        - opened slot if opened
        - best slot if not opened

    Args:
        receiver: Tensor [M],dtype=int, the indices of the receiving nodes for each message
        best_slot: Tensor [M],dtype=int, the index of the best matching slot for each message
        opened: Tensor [M],dtype=int, the index of the slot opened by each message, or -1 if it did not open a slot
        k: int, number of slots per node

    Returns:
        target: Tensor [M],dtype=int, the flat index of the target slot for each

    """
    return (
        receiver * k + torch.where(opened >= 0, opened, best_slot)
    )  # best slot is used if opened is -1 eg similarity is high enough to route to an existing node
