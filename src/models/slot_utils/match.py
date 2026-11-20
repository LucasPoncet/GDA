import sys
from pathlib import Path

file = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(file))

import torch
import torch.nn.functional as F
from torch import Tensor

from models.slot_utils.slotstate import SlotState


def match(state: SlotState, content: Tensor, receiver: Tensor) -> tuple[Tensor, Tensor]:
    """
    Return the cosine similarity between each message and the occupied slots of its receiver.

    Args:
        state: SlotState, the current state of the slots
        content: Tensor [M, d], the content of the messages
        receiver: Tensor [M], the flat indices of the receiving nodes for each message

    Returns:
        best_sim: Tensor [M], the best cosine similarity for each message with the occupied slots
        best_slot: Tensor [M], the flat index of the best matching slot for each message

    """
    q = F.normalize(content, dim=-1)  # [M, d]
    slots = F.normalize(state.S[receiver], dim=-1)  # [M, k, d]
    sim = torch.einsum("md,mkd->mk", q, slots)
    sim = sim.masked_fill(
        state.occ[receiver].logical_not(), float("-inf")
    )  # - inf for every unoccupied slot
    return sim.max(-1)
