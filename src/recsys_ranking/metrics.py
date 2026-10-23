"""Ranking and catalogue metrics with binary relevance.

Per-user functions take ``recs`` (int64, shape ``(n, >=k)``, best first) and ``truth``
(a list of ``n`` int arrays of relevant item indices) and return a float64 array of length ``n``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _hits(recs: np.ndarray, truth: Sequence[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the (n, k) boolean hit matrix and the (n,) truth sizes."""
    recs = np.asarray(recs)
    if recs.ndim != 2:
        raise ValueError(f"recs must be 2-D, got shape {recs.shape}")
    if k < 1:
        raise ValueError("k must be at least 1")
    if recs.shape[1] < k:
        raise ValueError(f"recs has {recs.shape[1]} columns but k={k}")
    if len(truth) != recs.shape[0]:
        raise ValueError(f"{recs.shape[0]} rows of recs but {len(truth)} truth arrays")
    top = recs[:, :k]
    hits = np.zeros(top.shape, dtype=bool)
    sizes = np.empty(top.shape[0], dtype=np.int64)
    for i, t in enumerate(truth):
        t = np.asarray(t)
        if t.size == 0:
            raise ValueError(f"user {i} has empty truth")
        sizes[i] = t.size
        hits[i] = np.isin(top[i], t)
    return hits, sizes


def _discount(k: int) -> np.ndarray:
    """Gain 1/log2(rank+1) for ranks 1..k."""
    return 1.0 / np.log2(np.arange(2, k + 2))


def recall_at_k(recs: np.ndarray, truth: Sequence[np.ndarray], k: int) -> np.ndarray:
    """Hits in the top k divided by the size of the truth set."""
    hits, sizes = _hits(recs, truth, k)
    return hits.sum(axis=1) / sizes


def ndcg_at_k(recs: np.ndarray, truth: Sequence[np.ndarray], k: int) -> np.ndarray:
    """DCG of the top k over the ideal DCG for ``min(k, |truth|)`` relevant items."""
    hits, sizes = _hits(recs, truth, k)
    disc = _discount(k)
    dcg = hits @ disc
    idcg = np.cumsum(disc)[np.minimum(k, sizes) - 1]
    return dcg / idcg


def map_at_k(recs: np.ndarray, truth: Sequence[np.ndarray], k: int) -> np.ndarray:
    """Sum of precision@i at each hit, divided by ``min(k, |truth|)``."""
    hits, sizes = _hits(recs, truth, k)
    ranks = np.arange(1, k + 1)
    precision = np.cumsum(hits, axis=1) / ranks
    return (precision * hits).sum(axis=1) / np.minimum(k, sizes)


def mrr_at_k(recs: np.ndarray, truth: Sequence[np.ndarray], k: int) -> np.ndarray:
    """Reciprocal rank of the first hit within the top k, else 0."""
    hits, _ = _hits(recs, truth, k)
    first = hits.argmax(axis=1)
    return np.where(hits.any(axis=1), 1.0 / (first + 1), 0.0)


def coverage(recs: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears at least once in ``recs``."""
    return float(np.unique(recs).size / n_items)


def mean_popularity(recs: np.ndarray, item_pop: np.ndarray) -> float:
    """Mean of ``item_pop[item]`` over every recommended slot."""
    return float(np.asarray(item_pop)[np.asarray(recs)].mean())


def gini_exposure(recs: np.ndarray, n_items: int) -> float:
    """Gini of per-item appearance counts, zero-count items included.

    0 means every item is shown equally often; the value tends to 1 as exposure
    concentrates on one item (the maximum for ``n_items`` items is ``(n-1)/n``).
    """
    counts = np.bincount(np.asarray(recs).ravel(), minlength=n_items).astype(np.float64)
    if counts.size != n_items:
        raise ValueError("recs contains an item index >= n_items")
    total = counts.sum()
    if total == 0:
        return 0.0
    counts.sort()
    n = counts.size
    weights = np.arange(1, n + 1)
    return float(2.0 * (weights @ counts) / (n * total) - (n + 1) / n)
