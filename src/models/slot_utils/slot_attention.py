import sys
from pathlib import Path

file = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(file))

import torch
from torch import Tensor, nn

from models.slot_utils.slotstate import SlotState


class SlotAttention(nn.Module):
    """
    Attention over the occupied slots of the node, the query comes from slot 0 (identity of the node)
    """

    def __init__(self, d: int):
        super().__init__()
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)

    def forward(self, state: SlotState) -> Tensor:
        S = state.S
        q = self.query(S[:, 0])  # [N, d]
        att = torch.einsum("nd,nkd->nk", q, self.key(S)) / S.shape[-1] ** 0.5
        att = torch.softmax(att.masked_fill(~state.occ, float("-inf")), -1)
        return torch.einsum("nk,nkd->nd", att, self.value(S))
