from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class MPNNLayer(MessagePassing):
    """Update each node using its own features and messages from its neighbors

    Args:
        in_dim: Number of input features per node
        out_dim: Number of output features per node
        aggr: How to combine incoming messages, such as "add", "mean" or "max"
        activation: Activation applied after the update (default ReLU)
        normalize: Whether to normalize messages using node degrees
        bias: Whether the self transformation uses a bias
    """

    def __init__(self, in_dim, out_dim, aggr="add", activation=None, normalize=False, bias=True):
        super().__init__(aggr=aggr)
        self.omega = nn.Linear(in_dim, out_dim, bias=bias)
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        if activation is None:
            activation = nn.ReLU()
        self.activation = activation
        self.normalize = normalize

    def forward(self, x, edge_index):
        """Run one message-passing step and return the updated node features"""
        edge_weight = None
        if self.normalize:
            row, col = edge_index
            deg = degree(col, num_nodes=x.size(0), dtype=x.dtype)
            # For an edge j -> i, the weight is w_ji = 1 / sqrt(deg[j] * deg[i])
            deg_inv_sqrt = deg.rsqrt()
            deg_inv_sqrt[deg == 0] = 0   # nodes with no incoming edges
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # self.propagate calls three methods in order:
        # - self.message builds one message per edge
        # - self.aggregate combines messages by destination node
        # - self.update returns the new state of each node
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        """Build one message per edge"""
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out, x):
        """Combine each node's own features with its aggregated messages"""
        return self.activation(self.omega(x) + self.W(aggr_out))


class MPNN(nn.Module):
    """Encoder followed by a stack of message-passing layers. Returns node embeddings."""

    def __init__(self, input_dim, hidden_dim, num_layers, **layer_kwargs):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MPNNLayer(hidden_dim, hidden_dim, **layer_kwargs))

    def forward(self, batch):
        """Return the list [h0, h1, ..., hL] of node embeddings, each of shape (num_nodes, hidden_dim)"""
        h = self.encoder(batch.x)
        embeddings = [h]
        for layer in self.layers:
            h = layer(h, batch.edge_index)
            embeddings.append(h)
        return embeddings