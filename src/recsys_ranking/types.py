"""Shared data shapes. Every module codes against these, so they change only with GUIDELINES.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Self

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Interactions:
    """Positive interactions in raw-id space, one row per (user, item) event.

    Arrays are aligned and the same length. ``rating`` is kept so the positive
    threshold can be applied (and audited) after loading.
    """

    user: np.ndarray  # raw user ids, int64
    item: np.ndarray  # raw item ids, int64
    ts: np.ndarray  # unix seconds, int64
    rating: np.ndarray  # float32; synthetic data uses 5.0 for every row

    def __len__(self) -> int:
        return int(self.user.shape[0])


@dataclass(frozen=True)
class Fold:
    """One fit/evaluate pair in a dense index space built from the fit rows only.

    ``fit`` and ``truth`` are binary CSR matrices of shape (n_users, n_items).
    Users and items with no fit interaction are not in the index space; truth rows that
    referred to them were dropped and counted in ``dropped``. Truth never contains a
    (user, item) pair that is also in ``fit``.
    """

    name: str  # "val", "test" or "loo"
    fit: sp.csr_matrix
    truth: sp.csr_matrix
    user_ids: np.ndarray  # raw id of each dense user index
    item_ids: np.ndarray  # raw id of each dense item index
    dropped: dict[str, int] = field(default_factory=dict)
    # keys: "cold_user_rows", "cold_item_rows", "seen_pair_rows"

    @property
    def n_users(self) -> int:
        return int(self.fit.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.fit.shape[1])

    def eval_users(self) -> np.ndarray:
        """Dense indices of users with at least one truth item, ascending."""
        return np.flatnonzero(np.diff(self.truth.indptr) > 0)


class Recommender(Protocol):
    """A model scores every catalogue item for a batch of users.

    ``score`` must not mask seen items: protocols do that. Higher is better.
    """

    name: str

    def fit(self, train: sp.csr_matrix) -> Self: ...

    def score(self, users: np.ndarray) -> np.ndarray:
        """Return float32 array of shape (len(users), n_items)."""
        ...
