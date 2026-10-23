"""Item-based neighbourhood model with shrunk cosine similarity."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import scipy.sparse as sp


class ItemKNN:
    """Item-item cosine on the binary matrix, keeping the top neighbours per item.

    ``sim_ij = dot_ij / (norm_i * norm_j + shrink)``; the shrink term damps
    similarities that rest on few co-occurrences. The score of item j for a user
    is the sum of similarities between j and the items the user has seen.
    """

    name = "itemknn"

    def __init__(self, n_neighbors: int = 50, shrink: float = 0.0, seed: int = 0) -> None:
        self.n_neighbors = n_neighbors
        self.shrink = shrink
        self.seed = seed
        self._train: sp.csr_matrix | None = None
        self._sim: sp.csr_matrix | None = None

    def params(self) -> dict[str, Any]:
        return {"n_neighbors": self.n_neighbors, "shrink": self.shrink, "seed": self.seed}

    def fit(self, train: sp.csr_matrix, chunk: int = 512) -> Self:
        x = sp.csr_matrix(train, dtype=np.float32)
        x.data[:] = 1.0
        n_items = x.shape[1]
        co = (x.T @ x).tocsc()  # co-occurrence counts, items x items
        norm = np.sqrt(np.asarray(x.sum(axis=0)).ravel()).astype(np.float32)
        k = min(self.n_neighbors, max(n_items - 1, 1))

        rows, cols, vals = [], [], []
        for start in range(0, n_items, chunk):  # dense column blocks bound memory
            stop = min(start + chunk, n_items)
            block = co[:, start:stop].toarray()
            block /= norm[:, None] * norm[None, start:stop] + self.shrink + 1e-12
            block[np.arange(start, stop), np.arange(stop - start)] = 0.0  # no self-similarity
            if k < n_items:
                top = np.argpartition(-block, k - 1, axis=0)[:k]
            else:
                top = np.tile(np.arange(n_items)[:, None], (1, stop - start))
            top_vals = np.take_along_axis(block, top, axis=0)
            keep = top_vals > 0
            rows.append(top[keep])
            cols.append(np.broadcast_to(np.arange(start, stop), top.shape)[keep])
            vals.append(top_vals[keep])

        # Column j holds the neighbours of item j, so score = user_rows @ sim.
        self._sim = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
            dtype=np.float32,
        )
        self._train = x
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        assert self._train is not None and self._sim is not None, "call fit first"
        out = self._train[np.asarray(users)] @ self._sim
        return np.asarray(out.toarray(), dtype=np.float32)
