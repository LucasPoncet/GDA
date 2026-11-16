import sys
from pathlib import Path

file = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(file))

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter


@torch.no_grad()
def allocate(
    content: Tensor,
    novel: Tensor,
    best_sim: Tensor,
    receiver: Tensor,
    occ: Tensor,
    tau: float,
) -> tuple[Tensor, Tensor]:
    """
    Gives a free slot to the novel messages (best_sim <= tau). Every node does the same thing, all nodes in parallel.

    As long as some messages wait and a slot is free, one round:
        1. the founder is the waiting message farthest from the current slots (lowest best_sim): the most novel one
        2. the founder opens the next free slot, and every waiting message close to the founder (cosine > tau) joins it
        3. the founder is now a current slot: the remaining messages are compared to it too

    Messages still waiting when the node is full get no slot (they will be forced on their closest slot).

    Args:
        content: Tensor [M, d],dtype=float the content of the messages
        novel: Tensor [M], dtype=bool, indicates whether i-th message is novel or not (has to be assigned to a new slot)
        best_sim: Tensor [M],dtype=float, best cosine similarity of each message with the occupied slots of its receiver
        receiver: Tensor [M],dtype=int the indices of the receiving nodes for each message
        occ: Tensor [N, k],dtype=bool, boolean mask indicating which slots are occupied
        tau: float, cosine similarity threshold for joining an opened slot

    Returns:
        opened: Tensor [M],dtype=int, the index of the slot opened by each message, or -1 if it did not open a slot
        occ: Tensor [N, k],dtype=bool, updated occupancy

    Example:
        >>> content = torch.tensor([[0.3, 0.2], [0.9, 0.1], [0.0, 1.0], [0.5, 0.5]])
            best_sim = torch.tensor([0.1, 0.95, 0.2, 0.8])
            receiver = torch.tensor([0, 0, 1, 1]) # first 2 messages go to node 0, last 2 to node 1
            occ = torch.BoolTensor([[True, False], [True, False]]) # node 0 and node 1 have one occupied slot each
            tau = 0.9
            opened, new_occ = allocate(content, best_sim, receiver, occ, tau)
            # The first message is novel (0.1 <= 0.9) and opens slot 1 of node 0, the second message is not novel (0.95 > 0.9) and does not open a slot.
            # The third message is novel so it opens slot 1 of node 1, and the last message stays unallocated because its not similar enough to the message that opened slot 1 of node 1.
            print("Nodes going through opened slots:", opened)  # Output: tensor([1, -1, 1, -1])
            print("New occupancy:\n", new_occ)  # Output: tensor([[ True, True], [ True,  True]])
    """
    N, k = occ.shape
    best_sim = best_sim.clone()  # [M]  Updated when a slot opens (step 3)
    used = occ.sum(
        1
    )  # [N] number of occupied slots of each node = index of its next free slot
    opened = torch.full_like(receiver, -1)  # [M]
    for _ in range(k):
        # Novel messages without a slot yet, on a node that still has a free slot
        waiting = novel & (opened < 0) & (used[receiver] < k)  # [M]
        if not waiting.any():
            break
        msg = waiting.nonzero().squeeze(1)
        node = receiver[msg]

        lowest = scatter(
            best_sim[msg], node, dim_size=N, reduce="min"
        )  # [N] lowest best_sim of each node
        candidate = (
            best_sim[msg] == lowest[node]
        )  # [M] True for the waiting messages that are candidates to be the founder of their node
        # If there are multiple candidates, the first one is chosen (scatter only returns the first index with reduce="min")
        founder = scatter(
            msg[candidate], node[candidate], dim_size=N, reduce="min"
        )  # [N]

        # The founder and the waiting messages close to it share the next free slot of their node
        sim_to_founder = F.cosine_similarity(
            content[msg], content[founder[node]], dim=-1
        )  # [M]
        join = (msg == founder[node]) | (
            sim_to_founder > tau
        )  # [M] True for the founder and the waiting messages close to it
        opened[msg[join]] = used[node[join]]
        has_founder = torch.zeros(N, dtype=torch.bool, device=occ.device)
        has_founder[node] = True
        used = used + has_founder

        # The founder is now a current slot
        best_sim[msg] = torch.maximum(best_sim[msg], sim_to_founder)
    occ = torch.arange(k, device=occ.device) < used[:, None]
    return opened, occ
