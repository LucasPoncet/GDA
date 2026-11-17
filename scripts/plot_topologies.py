import matplotlib.pyplot as plt
import networkx as nx

from src.graphs.clique_path import generate_clique_path_transfer_graph
from src.graphs.ring import generate_ring_transfer_graph
from src.graphs.tree import generate_tree_transfer_graph
from src.plotting.graphs import draw_graph

fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

edges, _, _, readout, info, d = generate_ring_transfer_graph(8, class_index=0)
draw_graph(edges, readout, info, axes[0], pos=nx.circular_layout(range(8)))
axes[0].set_title(f"Ring (r = {d})")

edges, _, _, readout, info, d = generate_clique_path_transfer_graph(6, class_index=0)
draw_graph(edges, readout, info, axes[1])
axes[1].set_title(f"Clique path (r = {d})")

edges, _, _, readout, info, d = generate_tree_transfer_graph(4, class_index=0)
pos = {}
v = 0
for level in range(d + 1):
    for k in range(2**level):
        pos[v] = ((k + 0.5) / 2**level, -level)
        v += 1
draw_graph(edges, readout, info, axes[2], pos=pos)
axes[2].set_title(f"Tree (r = {d})")

fig.savefig("figures/topologies.png", dpi=200, bbox_inches="tight")