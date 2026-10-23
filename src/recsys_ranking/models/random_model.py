"""Random baseline. Named ``random_model`` so it never shadows the stdlib ``random``."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import scipy.sparse as sp


class RandomModel:
    """Uniform random scores, a pure function of (seed, user id)."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.n_items = 0

    def params(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def fit(self, train: sp.csr_matrix) -> Self:
        self.n_items = int(train.shape[1])
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        out = np.empty((len(users), self.n_items), dtype=np.float32)
        for row, u in enumerate(np.asarray(users)):
            # One generator per user, so a user's scores do not depend on batching.
            out[row] = np.random.default_rng([self.seed, int(u)]).random(self.n_items)
        return out
