"""Bayesian personalised ranking matrix factorisation (Rendle et al., 2009)."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import scipy.sparse as sp


def _scatter_add(target: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """``target[idx] += vals`` with repeated indices summed (a faster ``np.add.at``)."""
    rows, inverse = np.unique(idx, return_inverse=True)
    pool = sp.csr_matrix(
        (np.ones(len(idx)), (inverse, np.arange(len(idx)))), shape=(len(rows), len(idx))
    )
    target[rows] += pool @ vals


class BPR:
    """BPR-MF trained by vectorised mini-batch SGD on sampled (user, positive, negative) triples.

    Score is ``p_u . q_i + b_i``. Each epoch draws as many triples as there are
    observations: an observed (u, i) pair uniformly, and j uniformly among items the user
    has not interacted with. Updates within a batch are summed (scatter-add), so an
    item that appears several times in a batch moves proportionally further.
    """

    name = "bpr"

    def __init__(
        self,
        factors: int = 64,
        lr: float = 0.05,
        reg: float = 1e-3,
        epochs: int = 30,
        seed: int = 0,
        batch_size: int = 1024,
    ) -> None:
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None

    def params(self) -> dict[str, Any]:
        return {
            "factors": self.factors,
            "lr": self.lr,
            "reg": self.reg,
            "epochs": self.epochs,
            "seed": self.seed,
            "batch_size": self.batch_size,
        }

    def fit(self, train: sp.csr_matrix) -> Self:
        x = sp.csr_matrix(train)
        x.sort_indices()
        n_users, n_items = x.shape
        rng = np.random.default_rng(self.seed)
        rows = np.repeat(np.arange(n_users, dtype=np.int64), np.diff(x.indptr))
        cols = x.indices.astype(np.int64)
        keys = rows * n_items + cols  # sorted: rows ascend, columns ascend within a row
        nnz = len(keys)

        p = rng.standard_normal((n_users, self.factors)) * 0.1
        q = rng.standard_normal((n_items, self.factors)) * 0.1
        b = np.zeros(n_items)
        lr, reg, bs = self.lr, self.reg, self.batch_size

        for _ in range(self.epochs if nnz else 0):
            pick = rng.integers(0, nnz, size=nnz)
            us, ii = rows[pick], cols[pick]
            jj = rng.integers(0, n_items, size=nnz)
            bad = np.arange(nnz)
            for _retry in range(50):  # resample negatives that are actually observed
                probe = us[bad] * n_items + jj[bad]
                hit = keys[np.minimum(np.searchsorted(keys, probe), nnz - 1)] == probe
                bad = bad[hit]
                if not len(bad):
                    break
                jj[bad] = rng.integers(0, n_items, size=len(bad))
            for s in range(0, nnz, bs):
                u, i, j = us[s : s + bs], ii[s : s + bs], jj[s : s + bs]
                pu, qi, qj = p[u], q[i], q[j]
                diff = qi - qj
                x_uij = np.einsum("bf,bf->b", pu, diff) + b[i] - b[j]
                g = 1.0 / (1.0 + np.exp(np.clip(x_uij, -30.0, 30.0)))  # sigmoid(-x_uij)
                gc = g[:, None]
                gp = gc * pu
                _scatter_add(p, u, lr * (gc * diff - reg * pu))
                ij = np.concatenate([i, j])
                _scatter_add(q, ij, lr * np.concatenate([gp - reg * qi, -gp - reg * qj]))
                b += np.bincount(
                    ij, lr * np.concatenate([g - reg * b[i], -g - reg * b[j]]), n_items
                )

        self.user_factors, self.item_factors, self.item_bias = p, q, b
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        assert self.user_factors is not None, "call fit first"
        assert self.item_factors is not None and self.item_bias is not None
        s = self.user_factors[np.asarray(users)] @ self.item_factors.T + self.item_bias
        return s.astype(np.float32)
