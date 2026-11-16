import sys
from pathlib import Path

file = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(file))

import torch

from models.slot_utils.aggregate import aggregate
from models.slot_utils.allocate import allocate
from models.slot_utils.emit import emit
from models.slot_utils.match import match
from models.slot_utils.route import route
from models.slot_utils.slotstate import SlotState


def test_emit():
    print("\nTesting the emit function...\n")
    # 2 nodes, 2 slots each, 2 edges
    S = torch.tensor([[[1.0, 2.0], [0.0, 0.0]], [[3.0, 4.0], [5.0, 6.0]]])  # [2, 2, 2]
    # Second slot of node 0 is free, both slots of node 1 are occupied
    occ = torch.tensor([[True, False], [True, True]])  # [2, 2]
    state = SlotState(S, occ)
    src = torch.tensor([0, 1])  # edges from node 0 to node 1
    dst = torch.tensor([1, 0])  # edges from node 1 to node 0

    sender, receiver = emit(state, src, dst)
    assert sender.tolist() == [0, 2, 3], f"Unexpected sender indices: {sender.tolist()}"
    assert receiver.tolist() == [
        1,
        0,
        0,
    ], f"Unexpected receiver indices: {receiver.tolist()}"


def test_match():
    print("\nTesting the match function...\n")
    S = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]])  # [2, 2, 2]
    occ = torch.tensor([[True, True], [True, False]])  # [2, 2]
    state = SlotState(S, occ)
    content = torch.tensor([[1.0, 0.0], [0.5, 0.6], [0.0, 1.0]])  # [3, 2]
    receiver = torch.tensor([0, 1, 0])  # sends to node 0, node 1, node 0
    # 0 is sent to node 0, closest to slot 0
    # 1 is sent to node 1, closest to slot 0
    # 2 is sent to node 0, closest to slot 1

    best_sim, best_slot = match(state, content, receiver)
    assert best_slot.tolist() == [
        0,
        0,
        1,
    ], f"Unexpected best slots: {best_slot.tolist()}"
    print("Best similarities:", best_sim)
    print("Best slots:", best_slot)


def test_allocate():
    print("\nTesting the allocate function...\n")
    S = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]])  # [2, 2, 2]
    occ = torch.tensor([[True, False], [True, False]])  # [2, 2]
    state = SlotState(S, occ)
    content = torch.tensor([[1.0, 0.0], [0.5, 0.6], [0.0, 1.0]])  # [3, 2]
    receiver = torch.tensor([0, 1, 0])  # sends to node 0, node 1, node 0
    novel = torch.tensor([True, True, True])  # All messages are novel
    tau = 0.9
    best_sim, _ = match(
        state, content, receiver
    )  # [3] best similarity with the occupied slots

    opened, new_occ = allocate(content, novel, best_sim, receiver, occ, tau)
    assert opened.tolist() == [-1, 1, 1], f"Unexpected opened slots: {opened.tolist()}"
    print("Opened slots:", opened)
    print("New occupancy:", new_occ)


def test_route():
    print("\nTesting the route function...\n")
    receiver = torch.tensor([0, 1, 0])  # sends to node 0, node 1, node 0
    best_slot = torch.tensor([0, 0, 1])  # closest occupied slots
    opened = torch.tensor([-1, -1, 1])  # opened slots (-1 if none)
    k = 2

    target = route(receiver, best_slot, opened, k)
    print("Target slots:", target)
    # we route 0 to slot 0 of node 0, 1 to slot 1 of node 1, and 2 to slot 1 of node 0 (index is flat index = node * k + slot)
    assert target.tolist() == [0, 2, 1], f"Unexpected target slots: {target.tolist()}"


def test_aggregate():
    print("\nTesting the aggregate function...\n")
    S = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]])  # [2, 2, 2]
    occ = torch.tensor([[True, False], [True, False]])  # [2, 2]
    state = SlotState(S, occ)
    sender = torch.tensor([0, 2])  # messages from slot 0 of node 0 and slot 0 of node 2
    target = torch.tensor(
        [2, 1]
    )  # 1st goes to the slot 0 of node 2, 2nd to the slot 1 of node 1

    new_S = aggregate(state, sender, target)
    print("New slot contents:", new_S)
    # the 1st slot of the node 0 was empty so we don't apply a mean: we just take the value as it is
    expected_S = torch.tensor([[[1.0, 0.0], [1, 1]], [[1.0, 0.5], [0.0, 0.0]]])
    assert torch.allclose(new_S, expected_S), f"Unexpected new slot contents: {new_S}"
