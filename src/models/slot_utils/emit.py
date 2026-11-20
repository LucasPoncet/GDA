import sys
from pathlib import Path

file = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(file))

import torch
from torch import Tensor

from models.slot_utils.slotstate import SlotState


def emit(state: SlotState, src: Tensor, dst: Tensor) -> tuple[Tensor, Tensor]:
    """
    Emit the sender and receiver indices of the messages, for the flatten representation (see SlotState.flat()).

        Args:
        state: SlotState, the current state of the slots
        src: Tensor [E] dtype=int, source nodes of the edges, E is the number of edges
        dst: Tensor [E] dtype=int, destination nodes of the edges

    Returns:
        sender: Tensor [E*k] dtype=int, flat indices of the sending slots
        receiver: Tensor [E*k] dtype=int, indices of the receiving nodes corresponding to each message

    Example:
        >>> S = torch.Tensor([[[1.0, 2.0], [0.0, 0.0]], [[3.0, 4.0], [5.0, 6.0]]])
            occ = torch.BoolTensor([[True, False], [True, True]])
            state = SlotState(S, occ)
            src = torch.tensor([0, 1])  # we send both 0 to 1 and 1 to 0
            dst = torch.tensor([1, 0])
            sender, receiver = emit(state, src, dst)
            print(sender)  # Output: tensor([0, 2, 3])
            print(receiver)  # Output: tensor([1, 0, 0]), message 0 goes to node 1, 2nd and 3rd go to node 0

    """
    k = state.k
    sender = (src[:, None] * k + torch.arange(k, device=src.device)).reshape(
        -1
    )  # [E*k]
    receiver = dst.repeat_interleave(k)  # [E*k]
    sent = state.occ.reshape(-1)[sender]
    return sender[sent], receiver[sent]
