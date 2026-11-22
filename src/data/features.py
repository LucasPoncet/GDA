# Control the features converging to a single node, to show that sum aggregation "squashes" them.

import numpy as np


def random_unit_vectors(k: int, dim: int, rng: np.random.Generator):
    """k random unit vectors of dimension dim (not orthogonal when k > dim)."""
    v = rng.standard_normal((k, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)
