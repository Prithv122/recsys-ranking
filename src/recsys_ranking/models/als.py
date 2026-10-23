"""Implicit-feedback alternating least squares (Hu, Koren, Volinsky, 2008)."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import scipy.sparse as sp

_BUDGET = 500_000  # max elements of one gathered (rows, nnz, factors) block
_MAX_BATCH = 512


class ALS:
    """iALS with confidence ``c = 1 + alpha * r`` on observed entries, preference 1 there.

    With ``C^u`` the diagonal confidence matrix of user u and ``p_u`` the 0/1 preferences,
    the paper's per-user solution (eq. 4) is ``x_u = (Y^T C^u Y + reg I)^-1 Y^T C^u p_u``.
    Writing ``Y^T C^u Y = Y^T Y + Y^T (C^u - I) Y`` (eq. 5) lets ``Y^T Y`` be computed
    once per sweep, so each user costs only their observed items. Items are solved
    symmetrically. ``reg`` is a plain ridge, not scaled by the number of interactions.
    """

    name = "als"

    def __init__(
        self,
        factors: int = 64,
        reg: float = 0.1,
        alpha: float = 10.0,
        iterations: int = 15,
        seed: int = 0,
    ) -> None:
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def params(self) -> dict[str, Any]:
        return {
            "factors": self.factors,
            "reg": self.reg,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "seed": self.seed,
        }

    def _sweep(self, mat: sp.csr_matrix, other: np.ndarray) -> np.ndarray:
        """Solve every row of ``mat`` against the fixed ``other`` factors."""
        f = other.shape[1]
        gram = other.T @ other + self.reg * np.eye(f)
        counts = np.diff(mat.indptr)
        out = np.zeros((mat.shape[0], f))  # rows without interactions stay at zero
        active = np.flatnonzero(counts > 0)
        # Sorted by length, so the padding inside a batch of equal-ish rows stays small.
        active = active[np.argsort(counts[active], kind="stable")]
        pos = 0
        while pos < len(active):
            end = min(pos + _MAX_BATCH, len(active))
            while end - pos > 1 and (end - pos) * counts[active[end - 1]] * f > _BUDGET:
                end = pos + (end - pos) // 2
            rows = active[pos:end]
            pos = end

            width = int(counts[rows[-1]])
            offs = np.arange(width)
            valid = offs[None, :] < counts[rows][:, None]
            at = np.minimum(mat.indptr[rows][:, None] + offs[None, :], mat.nnz - 1)
            r = np.where(valid, mat.data[at], 0.0)
            yg = np.take(other, mat.indices[at], axis=0)  # (batch, width, f)
            # Y^T (C-I) Y over the observed items, and Y^T C p (p = 1 on observed items).
            # Z^T Z with the same buffer on both sides lets BLAS use its symmetric product.
            b = np.einsum("bl,blf->bf", np.where(valid, 1.0 + self.alpha * r, 0.0), yg)
            yg *= np.sqrt(self.alpha * r)[..., None]  # in place: yg is not needed after b
            a = np.matmul(yg.transpose(0, 2, 1), yg) + gram
            out[rows] = np.linalg.solve(a, b[..., None])[..., 0]
        return out

    def fit(self, train: sp.csr_matrix) -> Self:
        by_user = sp.csr_matrix(train, dtype=np.float64)
        by_item = sp.csr_matrix(by_user.T)
        rng = np.random.default_rng(self.seed)
        items = rng.standard_normal((by_user.shape[1], self.factors)) * 0.01
        users = rng.standard_normal((by_user.shape[0], self.factors)) * 0.01
        for _ in range(self.iterations):
            users = self._sweep(by_user, items)
            items = self._sweep(by_item, users)
        self.user_factors, self.item_factors = users, items
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        assert self.user_factors is not None and self.item_factors is not None, "call fit first"
        return (self.user_factors[np.asarray(users)] @ self.item_factors.T).astype(np.float32)
