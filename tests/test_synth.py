"""Synthetic generator: shape, determinism, stationarity and a planted signal."""

from __future__ import annotations

import time

import numpy as np
import pytest

from recsys_ranking.split import global_time_folds
from recsys_ranking.synth import EPOCH_BASE, SPAN_SECONDS, make_synthetic
from recsys_ranking.types import Fold


def _recall_at_k(fold: Fold, scores: np.ndarray, k: int = 10) -> float:
    """Mean per-user recall@k with seen-in-fit items masked; scores are dense (users, items)."""
    users = fold.eval_users()
    masked = scores[users].astype(np.float64)
    masked[fold.fit[users].toarray() > 0] = -np.inf
    top = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    truth = fold.truth[users].toarray() > 0
    hits = np.take_along_axis(truth, top, axis=1).sum(axis=1)
    return float(np.mean(hits / truth.sum(axis=1)))


def test_shapes_dtypes_and_ranges() -> None:
    inter, affinity = make_synthetic(n_users=120, n_items=80, n_factors=4, density=0.1, seed=3)
    assert affinity.shape == (120, 80)
    assert affinity.dtype == np.float32
    assert inter.user.dtype == np.int64
    assert inter.item.dtype == np.int64
    assert inter.ts.dtype == np.int64
    assert inter.rating.dtype == np.float32
    assert np.all(inter.rating == 5.0)
    assert inter.user.min() >= 0 and inter.user.max() < 120
    assert inter.item.min() >= 0 and inter.item.max() < 80
    assert inter.ts.min() >= EPOCH_BASE
    assert inter.ts.max() < EPOCH_BASE + SPAN_SECONDS
    assert np.all(np.diff(inter.ts) >= 0)


def test_no_repeated_pairs_and_minimum_three_per_user() -> None:
    inter, _ = make_synthetic(n_users=150, n_items=60, density=0.02, seed=1)
    pairs = inter.user * 60 + inter.item
    assert np.unique(pairs).size == pairs.size
    per_user = np.bincount(inter.user, minlength=150)
    assert per_user.min() >= 3  # density * n_items = 1.2, so the floor is what applies


def test_density_is_about_right() -> None:
    inter, _ = make_synthetic(n_users=400, n_items=200, density=0.05, seed=2)
    mean_per_user = len(inter) / 400
    assert 9.0 < mean_per_user < 11.0  # target density * n_items = 10


def test_same_seed_same_output_and_other_seed_differs() -> None:
    a, aff_a = make_synthetic(seed=7)
    b, aff_b = make_synthetic(seed=7)
    c, _ = make_synthetic(seed=8)
    for x, y in ((a.user, b.user), (a.item, b.item), (a.ts, b.ts), (a.rating, b.rating)):
        assert np.array_equal(x, y)
    assert np.array_equal(aff_a, aff_b)
    assert len(a) != len(c) or not np.array_equal(a.item, c.item)


def test_popularity_is_skewed_and_pop_skew_zero_flattens_it() -> None:
    def top_decile_share(pop_skew: float) -> float:
        inter, _ = make_synthetic(n_users=400, n_items=200, pop_skew=pop_skew, seed=0)
        counts = np.sort(np.bincount(inter.item, minlength=200))[::-1]
        return float(counts[:20].sum() / counts.sum())

    skewed = top_decile_share(1.0)
    flat = top_decile_share(0.0)
    assert skewed > flat + 0.15
    assert skewed > 0.4


def test_stationary_in_time() -> None:
    """Item mix and activity in the first and last third of time look alike."""
    inter, _ = make_synthetic(n_users=600, n_items=100, density=0.1, seed=0)
    lo = inter.ts < EPOCH_BASE + SPAN_SECONDS // 3
    hi = inter.ts >= EPOCH_BASE + 2 * SPAN_SECONDS // 3
    share_lo = np.bincount(inter.item[lo], minlength=100) / lo.sum()
    share_hi = np.bincount(inter.item[hi], minlength=100) / hi.sum()
    assert 0.9 < lo.sum() / hi.sum() < 1.1
    # Total variation distance between two samples of ~2,000 events over 100 items is about
    # 0.08 by chance alone; a drifting process would sit far above this.
    assert 0.5 * np.abs(share_lo - share_hi).sum() < 0.15


def test_oracle_beats_popularity_on_held_out_interactions() -> None:
    inter, affinity = make_synthetic(seed=0)
    _, test = global_time_folds(inter)
    truth_aff = affinity[np.ix_(test.user_ids, test.item_ids)]
    popularity = np.asarray(test.fit.sum(axis=0)).ravel()
    pop_scores = np.tile(popularity, (test.n_users, 1))

    oracle = _recall_at_k(test, truth_aff)
    pop = _recall_at_k(test, pop_scores)
    assert test.eval_users().size > 100
    assert oracle > 0.85
    assert oracle > pop + 0.15  # measured: about 0.94 against 0.74


def test_big_config_is_fast_enough() -> None:
    start = time.perf_counter()
    inter, affinity = make_synthetic(n_users=6000, n_items=3700, density=0.03, seed=0)
    elapsed = time.perf_counter() - start
    assert affinity.shape == (6000, 3700)
    assert 0.025 * 6000 * 3700 < len(inter) < 0.035 * 6000 * 3700
    assert elapsed < 30  # about 4 s on a laptop; the bound only catches a loop regression


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_users": 0},
        {"n_items": 2},
        {"n_factors": 0},
        {"density": 0.0},
        {"density": 1.5},
    ],
)
def test_invalid_arguments(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        make_synthetic(**kwargs)
