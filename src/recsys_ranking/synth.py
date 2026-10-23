"""Synthetic implicit-feedback data with planted latent structure.

Used by CI and the smoke run, so nothing here needs the real ratings.
"""

from __future__ import annotations

import numpy as np

from recsys_ranking.types import Interactions

EPOCH_BASE = 946_684_800  # 2000-01-01 00:00:00 UTC
SPAN_SECONDS = 3 * 365 * 86_400  # interactions are spread over three years
# Sampling probability is softmax(affinity / TEMPERATURE). A low value makes behaviour follow the
# latent factors closely, so a model that recovers them clearly beats popularity (at 0.3 the
# oracle's recall@10 is about 0.94 against 0.74 for popularity on the default size).
TEMPERATURE = 0.3
_CHUNK_USERS = 512


def make_synthetic(
    n_users: int = 300,
    n_items: int = 200,
    n_factors: int = 8,
    density: float = 0.05,
    pop_skew: float = 1.0,
    seed: int = 0,
) -> tuple[Interactions, np.ndarray]:
    """Generate interactions and the true affinity matrix that produced them.

    Raw user and item ids are ``0..n-1``, so ``truth_affinity[user, item]`` is indexed by raw id
    (shape ``(n_users, n_items)``, float32). Affinity is ``U V^T / sqrt(f) + b`` with Gaussian
    factors and a Zipf-like popularity bias ``b_i = pop_skew * log(w_i)``, ``w_i`` proportional
    to ``1/rank_i`` with ranks assigned to items at random.

    Each user draws ``max(3, Poisson(density * n_items))`` distinct items with probability
    proportional to ``softmax(affinity / TEMPERATURE)`` (Gumbel top-k). Timestamps are i.i.d.
    uniform over a fixed window, independent of user and item, so behaviour before and after any
    time cutoff has the same distribution. Every rating is 5.0. Rows are ordered by timestamp.
    """
    if n_users < 1 or n_items < 3:
        raise ValueError("need n_users >= 1 and n_items >= 3")
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")

    rng = np.random.default_rng(seed)
    users_f = rng.standard_normal((n_users, n_factors))
    items_f = rng.standard_normal((n_items, n_factors))

    ranks = rng.permutation(n_items) + 1  # item i has popularity rank ranks[i]
    pop_bias = pop_skew * np.log(1.0 / ranks)
    pop_bias -= pop_bias.mean()  # only differences matter in a softmax

    scale = np.float32(1.0 / np.sqrt(n_factors))
    affinity = (users_f.astype(np.float32) @ items_f.astype(np.float32).T) * scale
    affinity += pop_bias.astype(np.float32)

    counts = np.clip(rng.poisson(density * n_items, size=n_users), 3, n_items)

    user_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    for start in range(0, n_users, _CHUNK_USERS):
        stop = min(start + _CHUNK_USERS, n_users)
        logits = affinity[start:stop] / np.float32(TEMPERATURE)
        # Gumbel noise from float32 uniforms: much cheaper than rng.gumbel at this size.
        u = rng.random(logits.shape, dtype=np.float32)
        np.clip(u, 1e-7, 1.0 - 1e-7, out=u)
        keys = logits - np.log(-np.log(u))
        c = counts[start:stop]
        m = int(c.max())
        top = np.argpartition(-keys, m - 1, axis=1)[:, :m]
        top_keys = np.take_along_axis(keys, top, axis=1)
        order = np.argsort(-top_keys, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        keep = np.arange(m)[None, :] < c[:, None]
        rows = np.repeat(np.arange(start, stop), c)
        user_parts.append(rows)
        item_parts.append(top[keep])  # row-major mask keeps each user's items in rank order

    user = np.concatenate(user_parts).astype(np.int64)
    item = np.concatenate(item_parts).astype(np.int64)
    ts = EPOCH_BASE + rng.integers(0, SPAN_SECONDS, size=user.shape[0], dtype=np.int64)

    order = np.argsort(ts, kind="stable")
    inter = Interactions(
        user=user[order],
        item=item[order],
        ts=ts[order],
        rating=np.full(user.shape[0], 5.0, dtype=np.float32),
    )
    return inter, affinity
