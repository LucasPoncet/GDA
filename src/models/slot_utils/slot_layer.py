import sys
from pathlib import Path

file = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(file))

import torch
from torch import Tensor
from torch.nn import functional as F

from models.slot_utils.aggregate import aggregate
from models.slot_utils.allocate import allocate
from models.slot_utils.emit import emit
from models.slot_utils.match import match
from models.slot_utils.route import route
from models.slot_utils.slotstate import SlotState


def slot_layer(
    state: SlotState, src: Tensor, dst: Tensor, tau: float
) -> tuple[SlotState, dict]:
    """
    One round of message passing.
    - Compares similarity between each source / target slots
    - Allocates as much as possible features to new slots if similarity is not high enough
    - Aggregates the allocated nodes to each slot

    Args:
        state: SlotState
        src: Tensor [E], E number of edges.
        dst: Tensor [E], E number of edges.
        tau: float, threshold for similarity matching

    Returns:
        state: SlotState, current state of the graph
        stats: dict, opened and forced statistics
    """
    sender, receiver = emit(
        state, src, dst
    )  # flat indices of the sending slots and indices of the receiving nodes
    content = state.flat()[sender]
    best_sim, best_slot = match(
        state, content, receiver
    )  # find the best matching slot for each message
    novel = (
        best_sim <= tau
    )  # messages that are novel (not similar enough to any occupied slot)
    opened, occ = allocate(
        content, novel, best_sim, receiver, state.occ, tau
    )  # allocate free slots to novel messages, -1 if no new available
    target = route(
        receiver, best_slot, opened, state.k
    )  # target slot of each message (flat index): the slot it opened, otherwise the closest occupied slot
    S = aggregate(state, sender, target)
    S = torch.cat(
        [state.S[:, :1], S[:, 1:]], 1
    )  # We lock the slot 0 of each node, to keep the initial feature of each node untouched.
    S = F.layer_norm(S, S.shape[-1:]) * occ[..., None]  # Free slots stay empty
    stats = {
        "new": (opened >= 0).float().mean(),
        "forced": (novel & (opened < 0)).float().mean(),
    }
    return SlotState(S, occ), stats
