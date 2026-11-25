import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.synthetic import to_pyg
from src.graphs.path_with_leaves import generate_path_with_leaves_transfer_graph
from src.graphs.ring import generate_ring_transfer_graph
from src.graphs.tree import generate_tree_transfer_graph
from src.metrics.sensitivity import jacobian_sensitivity, mpnn_matrix, theoretical_sensitivity
from src.models.mpnn import MPNN
from src.plotting.style import COLORS

distances = range(3, 11)

graphs = {
    "Ring": lambda r: generate_ring_transfer_graph(2 * r, class_index=0),
    "Path with leaves": lambda r: generate_path_with_leaves_transfer_graph(r, class_index=0),
    "Tree": lambda r: generate_tree_transfer_graph(r, class_index=0),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 3))

for (name, generate), color, marker in zip(graphs.items(), COLORS, ["o", "s", "^"]):
    theory = []
    measured = {True: [], False: []}
    for r in distances:
        edges, x, y, readout, info, d = generate(r)
        assert d == r

        # Theoretical sensitivity using only the graph topology (normalized operator)
        S = mpnn_matrix(torch.tensor(edges), x.shape[0])
        theory.append(theoretical_sensitivity(S, readout, info, num_layers=r))

        # Measured sensitivity through the untrained MPNN (with and without degree normalization)
        data = to_pyg(edges, x, y, readout, info, d)
        for normalize in [True, False]:
            sensitivities = []
            for seed in range(20):
                torch.manual_seed(seed)
                model = MPNN(input_dim=5, hidden_dim=64, num_layers=r, normalize=normalize)
                sensitivities.append(jacobian_sensitivity(model, data)[-1])
            measured[normalize].append(np.median(sensitivities))

    axes[0].semilogy(list(distances), theory, marker=marker, linestyle="", color=color, label=name)
    axes[1].semilogy(list(distances), measured[True], marker=marker, linestyle="", color=color, label=name)
    axes[2].semilogy(list(distances), measured[False], marker=marker, linestyle="", color=color, label=name)

axes[0].set_title("Theoretical sensitivity")
axes[0].set_ylabel(r"$(\mathsf{\hat{A}}^r)_{vu}$")
axes[1].set_title("Measured sensitivity (normalize=True)")
axes[1].set_ylabel("Jacobian sensitivity")
axes[2].set_title("Measured sensitivity (normalize=False)")
axes[2].set_ylabel("Jacobian sensitivity")

for ax in axes:
    ax.set_xticks(list(distances))
    ax.grid(True)
    ax.set_xlabel("distance r (= number of layers)")
axes[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))

fig.tight_layout()
fig.savefig("figures/degree_vs_topology.png", dpi=200, bbox_inches="tight")