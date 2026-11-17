import networkx as nx

from src.plotting.style import FILLER, INFO, READOUT


def draw_graph(edges, readout, info, ax, pos=None):
    g = nx.Graph(edges.T.tolist())
    colors = [READOUT if v == readout else INFO if v == info else FILLER for v in g.nodes]
    nx.draw_networkx(g, pos=pos, ax=ax, node_color=colors, node_size=20, edge_color="gray", with_labels=False)