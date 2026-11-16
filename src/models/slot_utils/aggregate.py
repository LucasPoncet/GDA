import sys
from pathlib import Path

file = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(file))

import torch
from torch import Tensor
from torch_geometric.utils import scatter

from models.slot_utils.slotstate import SlotState


def aggregate(state: SlotState, sender: Tensor, target: Tensor) -> Tensor:
    """
    Each slot becomes the mean of its old content and of the copies routed to it: a mean aggregation on the slot graph,
    with a self-loop on every occupied slot. An echo leaves the slot unchanged, a new slot gets the mean of its copies.

    Args:
        state: SlotState, the current state of the slots
        sender: Tensor [M],dtype=int, flat indices of the sending slots
        target: Tensor [M],dtype=int, flat indices of the target slots

    Returns:
        Tensor [N, k, d], the updated slot contents after aggregation
    """
    flat = state.flat()
    # Self-loops: the old content of an occupied slot counts as one copy (free slots are empty and do not count)
    kept = state.occ.reshape(-1).nonzero().squeeze(1)
    index = torch.cat([target, kept])
    values = torch.cat([flat[sender], flat[kept]])
    return scatter(values, index, dim=0, dim_size=flat.shape[0], reduce="mean").view_as(
        state.S
    )
