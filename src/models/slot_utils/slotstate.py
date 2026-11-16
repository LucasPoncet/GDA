from typing import NamedTuple

import torch
from torch import Tensor


class SlotState(NamedTuple):
    """
    State of nodes with multiple slots.
    k slots, d-dimensional contents, N nodes.

    Args:
        S: Tensor [N, k, d], dtype = float, containing the contents of the slots
        occ: Tensor [N, k], dtype = bool,  indicating which slots are occupied (True)

    Example:
        >>> S = torch.Tensor([[[1.0, 2.0], [0.0, 0.0]], [[3.0, 4.0], [5.0, 6.0]]])
            occ = torch.BoolTensor([[True, False], [True, True]])
            state = SlotState(S, occ)
            print(state.k)  # Output: 2
            print(state.flat())  # Output: tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [5.0, 6.0]])

    """

    S: Tensor
    occ: Tensor

    @property
    def k(self) -> int:
        """
        Number of slots per node
        """
        return self.S.shape[1]

    def flat(self) -> Tensor:
        """
        Slot contents as N*k virtual nodes
        """
        return self.S.reshape(-1, self.S.shape[-1])


def init_state(h: Tensor, k: int) -> SlotState:
    """
    Initialize the slot state with a given node representation and number of slots
    At initialization, the first slot of each node is filled with its representation.

    Args:
        h: Tensor [N, d], node representations
        k: int, number of slots per node
    Returns:
        SlotState: Initialized slot state with the first slot occupied and the rest empty.
    """
    S = torch.zeros(h.shape[0], k, h.shape[1], device=h.device)
    S[:, 0] = h
    occ = torch.zeros(h.shape[0], k, dtype=torch.bool, device=h.device)
    occ[:, 0] = True
    return SlotState(S, occ)
