"""The two evaluation protocols: full-catalogue top-k and sampled negatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys_ranking.types import Fold, Recommender

SCHEMES = ("uniform", "popularity")


def _seen_mask_coords(matrix, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row position, column) coordinates of the stored entries of ``matrix[rows]``."""
    sub = matrix[rows]
    return np.repeat(np.arange(len(rows)), np.diff(sub.indptr)), sub.indices


def full_topk(
    model: Recommender, fold: Fold, users: np.ndarray, k: int, batch: int = 1024
) -> np.ndarray:
    """Top-k unseen items per user over the whole catalogue, best first.

    Items in the user's fit row are excluded. Ties go to the lower item index
    (stable sort on the negated score). Returns int64 of shape ``(len(users), k)``.
    """
    users = np.asarray(users, dtype=np.int64)
    if k < 1 or k > fold.n_items:
        raise ValueError(f"k={k} is not in 1..{fold.n_items}")
    fit = fold.fit.tocsr()
    out = np.empty((len(users), k), dtype=np.int64)
    for start in range(0, len(users), batch):
        chunk = users[start : start + batch]
        scores = np.array(model.score(chunk), dtype=np.float64)
        _require_finite(model, scores)
        rows, cols = _seen_mask_coords(fit, chunk)
        scores[rows, cols] = -np.inf
        unseen = fold.n_items - np.bincount(rows, minlength=len(chunk))
        if (unseen < k).any():
            bad = int(chunk[np.flatnonzero(unseen < k)[0]])
            raise ValueError(f"user {bad} has fewer than k={k} unseen items")
        order = np.argsort(-scores, axis=1, kind="stable")
        out[start : start + batch] = order[:, :k]
    return out


def _require_finite(model: Recommender, scores: np.ndarray) -> None:
    # A NaN positive would rank first under the sampled protocol and a NaN item would sort
    # below masked seen items under the full one, so a diverged model must fail loudly.
    if not np.isfinite(scores).all():
        raise ValueError(f"{model.name} produced non-finite scores")


@dataclass(frozen=True)
class SampledSet:
    """Candidate sets for the sampled protocol: one row per (eval user, truth item) pair.

    Pairs are ordered by eval user, then by ascending truth item, so the users are
    contiguous and ``np.unique(user_idx)`` equals ``fold.eval_users()``.
    """

    user_idx: np.ndarray  # (n_pairs,) dense user index of each pair
    pos_item: np.ndarray  # (n_pairs,) the held-out item
    neg_items: np.ndarray  # (n_pairs, n_neg) sampled negatives
    scheme: str
    seed: int


def sample_negatives(
    fold: Fold,
    n_neg: int = 100,
    scheme: str = "uniform",
    seed: int = 20261023,
    batch: int = 2048,
) -> SampledSet:
    """Draw ``n_neg`` negatives per (user, truth item) pair without replacement.

    Eligible items are those in neither the user's fit row nor the user's truth row.
    ``scheme="popularity"`` draws proportionally to fit item counts. Each candidate gets a
    random key and the ``n_neg`` smallest are kept: key = u for uniform, and
    key = -log(u) / p for popularity (Efraimidis-Spirakis weighted sampling without
    replacement). Excluded items get key = inf. The uniform stream is consumed pair by
    pair, so the result does not depend on ``batch``.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    n_items = fold.n_items
    fit = fold.fit.tocsr()
    truth = fold.truth.tocsr().copy()
    truth.sort_indices()

    user_idx = np.repeat(np.arange(truth.shape[0]), np.diff(truth.indptr)).astype(np.int64)
    pos_item = truth.indices.astype(np.int64)
    n_pairs = len(pos_item)

    neg_inv_p = None
    if scheme == "popularity":
        counts = np.asarray(fit.sum(axis=0)).ravel().astype(np.float64)
        with np.errstate(divide="ignore"):
            neg_inv_p = -counts.sum() / counts  # -1/p; items never seen in fit get -inf

    rng = np.random.default_rng(seed)
    neg_items = np.empty((n_pairs, n_neg), dtype=np.int64)
    for start in range(0, n_pairs, batch):
        users = user_idx[start : start + batch]
        key = rng.random((len(users), n_items))
        if neg_inv_p is not None:
            with np.errstate(divide="ignore"):
                np.log(key, out=key)  # u == 0 gives -inf * -inf = +inf: never chosen
            key *= neg_inv_p
        for matrix in (fit, truth):
            rows, cols = _seen_mask_coords(matrix, users)
            key[rows, cols] = np.inf
        top = np.argpartition(key, n_neg - 1, axis=1)[:, :n_neg]
        short = ~np.isfinite(np.take_along_axis(key, top, axis=1)).all(axis=1)
        if short.any():
            bad = int(users[np.flatnonzero(short)[0]])
            raise ValueError(f"user {bad} has fewer than n_neg={n_neg} eligible items")
        top.sort(axis=1)
        neg_items[start : start + len(users)] = top
    return SampledSet(user_idx, pos_item, neg_items, scheme, seed)


def sampled_ranks(model: Recommender, sampled: SampledSet, batch: int = 1024) -> np.ndarray:
    """1-based rank of each pair's positive among its candidates; ties count against it.

    ``rank = 1 + #(negatives whose score >= the positive's score)``. Returns int64 of
    length ``n_pairs``.
    """
    n_pairs = len(sampled.pos_item)
    ranks = np.empty(n_pairs, dtype=np.int64)
    if n_pairs == 0:
        return ranks
    users, inverse = np.unique(sampled.user_idx, return_inverse=True)
    order = np.argsort(inverse, kind="stable")  # pairs grouped by user
    sorted_inv = inverse[order]
    for start in range(0, len(users), batch):
        stop = min(start + batch, len(users))
        lo, hi = np.searchsorted(sorted_inv, [start, stop])
        pair_ids = order[lo:hi]
        scores = np.asarray(model.score(users[start:stop]))
        _require_finite(model, scores)
        local = sorted_inv[lo:hi] - start
        pos = scores[local, sampled.pos_item[pair_ids]]
        neg = scores[local[:, None], sampled.neg_items[pair_ids]]
        ranks[pair_ids] = 1 + (neg >= pos[:, None]).sum(axis=1)
    return ranks


def sampled_user_metrics(sampled: SampledSet, ranks: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Per-user HR@k and NDCG@k, each averaged over the user's pairs.

    One relevant item per pair, so ``hr = rank <= k`` and ``ndcg = 1/log2(rank+1)`` inside
    the top k. Users come back in ``fold.eval_users()`` order.
    """
    ranks = np.asarray(ranks)
    if ranks.shape != sampled.pos_item.shape:
        raise ValueError("ranks must have one entry per sampled pair")
    hit = ranks <= k
    gain = np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)
    _, inverse, per_user = np.unique(sampled.user_idx, return_inverse=True, return_counts=True)
    return {
        "hr": np.bincount(inverse, weights=hit.astype(np.float64)) / per_user,
        "ndcg": np.bincount(inverse, weights=gain) / per_user,
    }
