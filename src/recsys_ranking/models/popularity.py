"""Global popularity baseline."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import scipy.sparse as sp


class Popularity:
    """Scores every item by its number of fit interactions; identical for all users."""

    name = "popularity"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._counts: np.ndarray | None = None

    def params(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def fit(self, train: sp.csr_matrix) -> Self:
        self._counts = np.asarray(train.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        assert self._counts is not None, "call fit first"
        return np.tile(self._counts, (len(users), 1))
