import matplotlib.pyplot as plt
import torch

from src.graphs.clique_path import generate_clique_path_transfer_graph
from src.graphs.ring import generate_ring_transfer_graph
from src.graphs.tree import generate_tree_transfer_graph
from src.metrics.sensitivity import mpnn_matrix, theoretical_sensitivity
from src.plotting.style import COLORS

distances = range(3, 11)

graphs = {
    "Ring": lambda r: generate_ring_transfer_graph(2 * r, class_index=0),
    "Clique path": lambda r: generate_clique_path_transfer_graph(2 * (r - 1), class_index=0),
    "Tree": lambda r: generate_tree_transfer_graph(r, class_index=0),
}

fig, ax = plt.subplots(figsize=(4, 3))

for (name, generate), color, marker in zip(graphs.items(), COLORS, ["o", "s", "^"]):
    values = []
    for r in distances:
        edges, x, y, readout, info, d = generate(r)
        assert d == r
        S = mpnn_matrix(torch.tensor(edges), x.shape[0])
        values.append(theoretical_sensitivity(S, readout, info, num_layers=r))
    ax.semilogy(list(distances), values, marker=marker, linestyle="", color=color, label=name)

rs = list(distances)
ax.set_xticks(rs)
ax.grid(True)
ax.set_xlabel("distance r (= number of layers)")
ax.set_ylabel(r"$(\mathsf{A}^r)_{vu}$")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

fig.savefig("figures/theoretical_sensitivity.png", dpi=200, bbox_inches="tight")