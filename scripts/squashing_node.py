"""Squashing vs smoothing in a single node with sum aggregation.
What do we show in this script ?
1. A minority neighbour is lost in the sum by the majority as the degree grows.
2. Two neighbourhoods that differ by only a single feature (neighbour) become indistinguishable.
3. If the feature dimension d is smaller than the number of distinct features K,
    the neighbours cannot be recovered from the sum.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.data.features import random_unit_vectors
from src.plotting.style import COLORS

N_SEEDS = 50

# For 1/2. We vary the degrees to see the impact when growing
FEATURE_DIM = 5
DEGREES = np.arange(2, 101)

# For 3.
NUM_FEATURES = 8
NUM_NEIGHBOURS = 20
DIMS = np.arange(
    1, NUM_FEATURES + 3
)  # We add 3 to see the impact when the dim is higher than N_FEATURES


def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def relative_distance(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


cos_majority = np.zeros((N_SEEDS, len(DEGREES)))
cos_minority = np.zeros((N_SEEDS, len(DEGREES)))
rel_distance = np.zeros((N_SEEDS, len(DEGREES)))

for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    majority, minority, other = random_unit_vectors(3, FEATURE_DIM, rng)

    for i, degree in enumerate(DEGREES):
        # node 1 : (degree - 1) neighbours with majority feature, one with minority
        # node 2 : same, but the last neighbour has an other feature
        sum_1 = (degree - 1) * majority + minority
        sum_2 = (degree - 1) * majority + other

        cos_majority[seed, i] = cosine(majority, sum_1)
        cos_minority[seed, i] = cosine(minority, sum_1)
        rel_distance[seed, i] = relative_distance(sum_1, sum_2)

recon_error = np.zeros((N_SEEDS, len(DIMS)))

for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(NUM_NEIGHBOURS, np.ones(NUM_FEATURES) / NUM_FEATURES)

    for i, d in enumerate(DIMS):
        features = random_unit_vectors(NUM_FEATURES, d, rng)
        aggregated = (
            features.T @ counts
        )  # sum over neighbours, this line can be changed if willing to see the impact using an other agg scheme.
        recovered_counts = (
            np.linalg.pinv(features.T) @ aggregated
        )  # best guess with pinv
        recon_error[seed, i] = relative_distance(counts, recovered_counts)


# Plots.
def plot_mean_std(ax, x, values, color, label, **kwargs):
    mean, std = values.mean(axis=0), values.std(axis=0)
    ax.plot(x, mean, color=color, label=label, **kwargs)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(12, 3))

plot_mean_std(ax_a, DEGREES, cos_majority, COLORS[0], "majority class")
plot_mean_std(ax_a, DEGREES, cos_minority, COLORS[1], "minority class")
ax_a.set_xlabel("Degree")
ax_a.set_ylabel("Cosine similarity")
ax_a.legend(loc="center right")

plot_mean_std(ax_b, DEGREES, rel_distance, COLORS[2], "measured")
ax_b.plot(DEGREES, 1 / DEGREES, "k--", label=r"$1/\deg$")
ax_b.set_xscale("log")
ax_b.set_yscale("log")
ax_b.set_xlabel("Degree")
ax_b.set_ylabel("Relative distance")
ax_b.legend()

plot_mean_std(
    ax_c, DIMS, recon_error, COLORS[3], "neighbours recovered from the sum", marker="o"
)
ax_c.axvline(
    NUM_FEATURES, color="k", linestyle="--", label=f"{NUM_FEATURES} distinct features"
)
ax_c.set_xlabel("Feature dimension $d$")
ax_c.set_ylabel("Relative error")
ax_c.legend()

for ax in (ax_a, ax_b, ax_c):
    ax.grid(True)

fig.tight_layout()
fig.savefig("figures/squashing_node.png", dpi=200, bbox_inches="tight")
