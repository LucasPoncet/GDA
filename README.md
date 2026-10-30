# SlotMPNN - Geometric Data Analysis Project

This repository contains the code developed for the Geometric Data Analysis on the analysis of the paper:
*"On Over-Squashing in Message Passing Neural Networks:  The Impact of Width, Depth, and Topology"* by Di Giovanni et al. ICML 2023. 


It contains the code to reproduce some of the results presented in the paper as well as additional experiments to highlight some limitations, mainly about the over-smoothing induced by common over-squashing solutions.
We then add our own MPNN model: **SlotMPNN** which effectively use the width argument not in the hidden layer but in the memory each node have to better preserve message information. The trade-off comes from the increased memory needs. We provide runs experiments against the same dataset presented by Di Giovanni et al. 
We additionnaly add an interesting dataset: *Tree-NeighborsMatch* by  Alon & Yahav 2021. What is interesting in this dataset is the graph itself, a tree, that posseses edges with an even higher negative curvature than the already present dataset.

---

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) for deterministic dependency resolution.
Python is pinned to **3.12**.

```bash
git clone https://github.com/LucasPoncet/GDA.git
cd GDA
uv sync
```

### Install the dependencies for your hardware

PyTorch is installed through an *extra* matching your CUDA version. Run `nvidia-smi` and read the
"CUDA Version" field (the highest version your driver supports), then run **one** of:

| Your setup | Command |
|---|---|
| CUDA 13.x driver | `uv sync --extra cu130` |
| CUDA 12.8+ driver | `uv sync --extra cu128` (torch ≤ 2.11) |
| CUDA 12.6+ driver | `uv sync --extra cu126` |
| No NVIDIA GPU / macOS | `uv sync --extra cpu` |

> `uv sync` without an extra installs **no** PyTorch.

---

## Repository structure

```text
gda-oversquashing/
├── README.md
├── pyproject.toml
│
├── configs/
│   ├── ring_transfer.yaml
│   ├── clique_path.yaml
│   ├── tree_neighbors.yaml
│   └── rewiring_tradeoff.yaml
│
├── src/
│   ├── graphs/
│   │   ├── ring.py
│   │   ├── clique_path.py
│   │   ├── tree.py
│   │   └── tasks.py
│   ├── data/
│   ├── models/
│   │   ├── interface.py
│   │   ├── mpnn.py
│   │   ├── slot_mpnn.py
│   │   ├── pyg_baselines.py
│   │   └── readout.py
│   ├── rewiring/
│   ├── train/
│   ├── metrics/
│   │   ├── accuracy.py
│   │   ├── sensitivity.py     # measured Jacobian + theoretical Ŝ^L
│   │   └── smoothing.py       # Dirichlet energy, MAD, effective rank
│   └── plotting/
│
├── third_party/
│   ├── on_oversquashing/
│   └── fosr/
├── tests/
│   ├── conftest.py
│   ├── test_graphs.py
│   ├── test_mpnn.py
│   └── test_metrics.py
│
├── scripts/
│   ├── run_experiment.py
│   ├── compare.py
│   └── make_figures.py
│
├── runs/
├── figures/
└── data/raw/
```