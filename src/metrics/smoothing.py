import torch.nn.functional as F
from torch import Tensor, bool, eye, exp, log
from torch.linalg import matrix_norm, svdvals


def dirichlet_energy(h: Tensor, edge_index: Tensor, normalized:bool = True) -> float:
    """(1/n) * sum over edges (i, j) of |h_i - h_j|^2 (Rusch et al. 2023); if normalized, divided by |h|_F^2."""
    src, dst = edge_index
    energy = (h[src] - h[dst]).pow(2).sum() / h.shape[0]
    if normalized:
        energy = energy / (matrix_norm(h) ** 2 + 1e-12)
    return energy.item()


def mad(h: Tensor, edge_index: Tensor | None = None) -> float:
    """Mean cosine distance 1 - cos(h_i, h_j) over edges if edge_index is given, else over all node pairs i != j (Chen et al. 2020)."""
    if edge_index is not None:
        src, dst = edge_index
        distances = 1 - F.cosine_similarity(h[src], h[dst], dim=1)
    else:
        h_unit = F.normalize(h, dim=1)
        cos = h_unit @ h_unit.T
        off_diagonal = ~eye(h.shape[0], dtype=bool, device=h.device)
        distances = 1 - cos[off_diagonal]
    return distances.mean().item()


def effective_rank(h: Tensor) -> float:
    """exp(-sum_k p_k * log(p_k)), with p_k = s_k / sum(s) and s_k the singular values of h (n, d) (Roy & Vetterli 2007)."""
    sigmas = svdvals(h)
    total = sigmas.sum()
    if total == 0:
        return 0.0
    p = sigmas / total
    p = p[p > 0]
    return exp(-(p * log(p)).sum()).item()

def smoothing_per_layer(embeddings: list[Tensor], edge_index: Tensor) -> dict[str, list[float]]:
    """Applies dirichlet_energy, mad and effective_rank to each layer: {'energy': [...], 'mad': [...], 'rank': [...]}."""
    results = {'energy': [], 'mad': [], 'rank': []}
    for h in embeddings:
        results['energy'].append(dirichlet_energy(h, edge_index))
        results['mad'].append(mad(h, edge_index))
        results['rank'].append(effective_rank(h))
    return results
