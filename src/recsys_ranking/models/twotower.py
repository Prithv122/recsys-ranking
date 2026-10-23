"""Two-tower retrieval model (torch, optional extra).

A user tower and an item tower each map an id to a unit-length vector; the score is their dot
product. Training is sampled softmax over observed (user, item) pairs: every other item in the
mini-batch is a negative for a pair, which is cheap but samples items in proportion to how often
they are observed, so popular items are over-represented as negatives. The standard logQ
correction (Yi et al., 2019) subtracts ``log p(item)`` from each logit, where ``p`` is the
probability that the item is drawn into the batch, i.e. its share of the observed pairs. Without it
the model learns to push popular items down.

Only ids are available (no side features), so each tower is an embedding table followed by a
small MLP. CPU only; a fixed seed gives the same weights on the same machine and thread count.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


class _Tower(nn.Module):
    """Id embedding followed by a one-hidden-layer MLP with a residual connection."""

    def __init__(self, n: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(n, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(ids)
        return F.normalize(e + self.mlp(e), dim=-1)


class TwoTower:
    """Two-tower model trained with in-batch sampled softmax and logQ correction.

    ``temperature`` divides the cosine logits; 0.1 is a common default for normalised towers
    (lower sharpens the softmax). It only matters for training, since ranking is unchanged by it.
    """

    name = "twotower"

    def __init__(
        self,
        dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 1024,
        temperature: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.temperature = temperature
        self.seed = seed
        self._user_tower: _Tower | None = None
        self._item_vecs: torch.Tensor | None = None

    def params(self) -> dict:
        return {
            "dim": self.dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "seed": self.seed,
        }

    def fit(self, train: sp.csr_matrix) -> Self:
        coo = train.tocoo()
        users = torch.from_numpy(coo.row.astype(np.int64))
        items = torch.from_numpy(coo.col.astype(np.int64))
        n_users, n_items = train.shape
        n_pairs = users.shape[0]
        if n_pairs == 0:
            raise ValueError("cannot fit on an empty interaction matrix")

        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        user_tower = _Tower(n_users, self.dim)
        item_tower = _Tower(n_items, self.dim)
        params = list(user_tower.parameters()) + list(item_tower.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)

        # p(item) = share of observed pairs; clamp so an unseen item cannot give log(0).
        counts = np.asarray(train.sum(axis=0)).ravel().clip(min=1)
        log_q = torch.from_numpy(np.log(counts / counts.sum()).astype(np.float32))

        user_tower.train()
        item_tower.train()
        for _ in range(self.epochs):
            order = torch.randperm(n_pairs, generator=gen)
            for start in range(0, n_pairs, self.batch_size):
                idx = order[start : start + self.batch_size]
                if idx.shape[0] < 2:
                    continue
                bu, bi = users[idx], items[idx]
                logits = user_tower(bu) @ item_tower(bi).T / self.temperature
                logits = logits - log_q[bi][None, :]
                # The same item appearing twice in a batch is not a negative for itself.
                same_item = bi[:, None] == bi[None, :]
                diag = torch.eye(idx.shape[0], dtype=torch.bool)
                logits = logits.masked_fill(same_item & ~diag, float("-inf"))
                loss = F.cross_entropy(logits, torch.arange(idx.shape[0]))
                opt.zero_grad()
                loss.backward()
                opt.step()

        user_tower.eval()
        item_tower.eval()
        with torch.no_grad():
            self._item_vecs = item_tower(torch.arange(n_items))
        self._user_tower = user_tower
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        if self._user_tower is None or self._item_vecs is None:
            raise RuntimeError("fit() must be called before score()")
        with torch.no_grad():
            u = self._user_tower(torch.from_numpy(np.asarray(users, dtype=np.int64)))
            return (u @ self._item_vecs.T).numpy().astype(np.float32, copy=False)
