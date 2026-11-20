import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

file = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(file))


from models.slot_utils.slot_layer import slot_layer
from models.slot_utils.slotstate import SlotState, init_state


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
        q = self.query(S[:, 0])
        att = torch.einsum("nd,nkd->nk", q, self.key(S)) / S.shape[-1] ** 0.5
        att = torch.softmax(
            att.masked_fill(~state.occ, float("-inf")), -1
        )  # state.occ contains True if occupied, so ~False=True masks the unoccupied slots
        return torch.einsum("nk,nkd->nd", att, self.value(S))


class SlotMPNN(nn.Module):
    """
    Slot Message Passing Neural Net.

    Args:
        in_channels: int, number of input channel (nodes attributes)
        hidden_channels: int, number of output channels for the encoder
        out_channels: int, number of output channels for the decoder
        num_layers: int, number of layers for message passing
        num_slots: int, number of slots per nodes
        tau: float, threshold above which messages are considered as similar
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_slots: int,
        tau=0.9,
        include_initial_feature=True,
    ):
        super().__init__()
        d = hidden_channels
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.tau = tau
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, d), nn.ReLU(), nn.Linear(d, d)
        )
        self.attention = SlotAttention(d)

        # include or not the initial feature for the reading
        self.include_initial_feature = include_initial_feature
        if include_initial_feature:
            self.decoder = nn.Sequential(
                nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, out_channels)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(d, d), nn.ReLU(), nn.Linear(d, out_channels)
            )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        h = self.encoder(x)
        state = init_state(F.layer_norm(h, h.shape[-1:]), self.num_slots)
        stats = []
        for _ in range(self.num_layers):
            state, layer_stats = slot_layer(state, src, dst, self.tau)
            stats.append(layer_stats)
        if self.include_initial_feature:
            return self.decoder(torch.cat([state.S[:, 0], self.attention(state)], -1))
        else:
            return self.decoder(self.attention(state))
