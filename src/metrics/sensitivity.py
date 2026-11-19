from torch import Tensor, diag, eye, nn, rsqrt, zeros
from torch.autograd.functional import jacobian
from torch.linalg import matrix_power, matrix_norm
from torch_geometric.data import Data


def mpnn_matrix(edge_index: Tensor, num_nodes: int) -> Tensor:
    """D^-1/2 A D^-1/2 (no self-loops): the operator of our MPNN with normalize=True, as in Di Giovanni et al., Sec. 4.1."""
    A = zeros((num_nodes, num_nodes), device=edge_index.device)
    A[edge_index[0], edge_index[1]] = 1.0
    d_rsqrt = diag(rsqrt(A.sum(dim=1)))
    return d_rsqrt @ A @ d_rsqrt


def gcn_matrix(edge_index: Tensor, num_nodes: int) -> Tensor:
    """D^-1/2 (A + I) D^-1/2 with D the degree of A + I: the operator of Kipf & Welling's GCN (PyG GCNConv)."""
    A = zeros((num_nodes, num_nodes), device=edge_index.device)
    A[edge_index[0], edge_index[1]] = 1.0
    A_hat = A + eye(num_nodes, device=edge_index.device)
    d_rsqrt = rsqrt(A_hat.sum(dim=1))
    D_rsqrt = diag(d_rsqrt)
    return D_rsqrt @ A_hat @ D_rsqrt


def theoretical_sensitivity(S: Tensor, readout_node: int, info_node: int, num_layers: int) -> float:
    """(S^L)[readout_node, info_node]: the topology term of Theorem 3.2, for a given message-passing matrix S."""
    return matrix_power(S, num_layers)[readout_node, info_node].item()

def jacobian_sensitivity(model: nn.Module, data: Data) -> list[float]:
    """For each layer l, compute the jacobian norm on one graph."""
    model.eval()

    readout = data.readout_mask.nonzero().item()
    information = data.info_mask.nonzero().item()

    def f(x_info):
        """Embeddings of the readout node at every layer, as a function of the information node's features."""
        graph = data.clone()
        x = data.x.clone()
        x[information] = x_info
        graph.x = x
        embeddings = model(graph)
        return tuple(h[readout] for h in embeddings)

    jacobians = jacobian(f, data.x[information])
    return [matrix_norm(J, ord=1).item() for J in jacobians]

    

    