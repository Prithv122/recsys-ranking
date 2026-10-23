"""Protocol tests on tiny hand-built folds and a fake model with a fixed score matrix."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from recsys_ranking.protocols import (
    SampledSet,
    full_topk,
    sample_negatives,
    sampled_ranks,
    sampled_user_metrics,
)
from recsys_ranking.types import Fold


class FixedModel:
    """Returns rows of a fixed score matrix; seen items are deliberately not masked."""

    name = "fixed"

    def __init__(self, scores: np.ndarray):
        self.scores = np.asarray(scores, dtype=np.float32)

    def fit(self, train):
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.scores[users].copy()


def make_fold(fit_dense: np.ndarray, truth_dense: np.ndarray) -> Fold:
    n_users, n_items = fit_dense.shape
    return Fold(
        name="test",
        fit=sp.csr_matrix(fit_dense.astype(np.float32)),
        truth=sp.csr_matrix(truth_dense.astype(np.float32)),
        user_ids=np.arange(n_users),
        item_ids=np.arange(n_items),
    )


def random_fold(
    n_users: int = 60,
    n_items: int = 200,
    fit_density: float = 0.1,
    truth_density: float = 0.03,
    seed: int = 0,
    skew: bool = False,
) -> Fold:
    rng = np.random.default_rng(seed)
    if skew:
        item_p = fit_density * 4 / (1 + np.arange(n_items) / 10)  # a few very popular items
        item_p = np.clip(item_p, 0, 0.9)
    else:
        item_p = np.full(n_items, fit_density)
    fit = rng.random((n_users, n_items)) < item_p
    truth = (rng.random((n_users, n_items)) < truth_density) & ~fit
    return make_fold(fit, truth)


# --- full_topk ---------------------------------------------------------------------------


def test_full_topk_hand_computed_and_masks_seen():
    fit = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]])
    truth = np.zeros_like(fit)
    truth[0, 5] = 1
    truth[1, 0] = 1
    scores = np.array(
        [
            [9.0, 1.0, 5.0, 3.0, 4.0, 2.0],  # item 0 is best but seen -> 2, 4, 3
            [1.0, 2.0, 9.0, 8.0, 3.0, 0.0],  # items 2, 3 are seen -> 4, 1, 0
        ]
    )
    fold = make_fold(fit, truth)
    recs = full_topk(FixedModel(scores), fold, np.array([0, 1]), k=3)
    assert recs.dtype == np.int64
    np.testing.assert_array_equal(recs, [[2, 4, 3], [4, 1, 0]])


def test_full_topk_ties_go_to_lower_item_index():
    fit = np.zeros((1, 6), dtype=int)
    fit[0, 1] = 1
    fold = make_fold(fit, np.zeros_like(fit))
    model = FixedModel(np.zeros((1, 6)))
    # constant scores: unseen items in index order, skipping the seen item 1
    np.testing.assert_array_equal(full_topk(model, fold, np.array([0]), k=4), [[0, 2, 3, 4]])


def test_full_topk_never_returns_seen_items_and_batching_is_invariant():
    fold = random_fold(n_users=40, n_items=80, fit_density=0.2, seed=1)
    rng = np.random.default_rng(2)
    scores = rng.integers(0, 5, size=(40, 80)).astype(np.float32)  # lots of ties
    model = FixedModel(scores)
    users = np.arange(40)
    one = full_topk(model, fold, users, k=10, batch=1)
    for batch in (3, 16, 1024):
        np.testing.assert_array_equal(full_topk(model, fold, users, k=10, batch=batch), one)
    fit = fold.fit.toarray().astype(bool)
    for u in users:
        assert not fit[u, one[u]].any()
        assert len(set(one[u])) == 10


def test_full_topk_respects_user_subset_and_order():
    fold = random_fold(n_users=10, n_items=30, seed=3)
    model = FixedModel(np.random.default_rng(4).random((10, 30)))
    full = full_topk(model, fold, np.arange(10), k=5)
    sub = full_topk(model, fold, np.array([7, 2]), k=5)
    np.testing.assert_array_equal(sub, full[[7, 2]])


def test_full_topk_raises_when_fewer_than_k_unseen():
    fit = np.array([[1, 1, 1, 0]])
    fold = make_fold(fit, np.zeros_like(fit))
    with pytest.raises(ValueError, match="fewer than k"):
        full_topk(FixedModel(np.ones((1, 4))), fold, np.array([0]), k=2)


def test_non_finite_scores_fail_loudly_in_both_protocols():
    fold = random_fold(n_users=6, n_items=60, seed=12)
    sampled = sample_negatives(fold, n_neg=20, seed=1)
    users = np.arange(6)
    clean = np.random.default_rng(0).random((6, 60))
    assert full_topk(FixedModel(clean), fold, users, k=5).shape == (6, 5)  # the control passes
    for bad in (np.nan, np.inf):
        scores = clean.copy()
        scores[3, 41] = bad  # one cell in one row is enough
        model = FixedModel(scores)
        with pytest.raises(ValueError, match="non-finite"):
            full_topk(model, fold, users, k=5)
        with pytest.raises(ValueError, match="non-finite"):
            sampled_ranks(model, sampled)
    all_nan = FixedModel(np.full((6, 60), np.nan))
    with pytest.raises(ValueError, match="non-finite"):
        full_topk(all_nan, fold, users, k=5)
    with pytest.raises(ValueError, match="non-finite"):
        sampled_ranks(all_nan, sampled)


# --- sample_negatives --------------------------------------------------------------------


def _check_valid(fold: Fold, sampled: SampledSet, n_neg: int) -> None:
    fit = fold.fit.toarray().astype(bool)
    truth = fold.truth.toarray().astype(bool)
    assert sampled.neg_items.shape == (len(sampled.pos_item), n_neg)
    for i in range(len(sampled.pos_item)):
        u = sampled.user_idx[i]
        negs = sampled.neg_items[i]
        assert len(set(negs)) == n_neg
        assert not fit[u, negs].any()
        assert not truth[u, negs].any()
        assert truth[u, sampled.pos_item[i]]


@pytest.mark.parametrize("scheme", ["uniform", "popularity"])
def test_sample_negatives_valid_and_ordered(scheme):
    fold = random_fold(seed=5)
    sampled = sample_negatives(fold, n_neg=30, scheme=scheme, seed=1)
    _check_valid(fold, sampled, 30)
    assert sampled.scheme == scheme and sampled.seed == 1
    # pair order: eval users ascending, then ascending truth item
    expected_users = np.repeat(np.arange(fold.n_users), np.diff(fold.truth.indptr))
    expected_items = np.concatenate([np.sort(fold.truth[u].indices) for u in range(fold.n_users)])
    np.testing.assert_array_equal(sampled.user_idx, expected_users)
    np.testing.assert_array_equal(sampled.pos_item, expected_items)
    np.testing.assert_array_equal(np.unique(sampled.user_idx), fold.eval_users())


@pytest.mark.parametrize("scheme", ["uniform", "popularity"])
def test_sample_negatives_deterministic_and_batch_independent(scheme):
    fold = random_fold(seed=6)
    a = sample_negatives(fold, n_neg=20, scheme=scheme, seed=3)
    b = sample_negatives(fold, n_neg=20, scheme=scheme, seed=3)
    np.testing.assert_array_equal(a.neg_items, b.neg_items)
    for batch in (1, 7, 100):
        c = sample_negatives(fold, n_neg=20, scheme=scheme, seed=3, batch=batch)
        np.testing.assert_array_equal(a.neg_items, c.neg_items)
        np.testing.assert_array_equal(a.pos_item, c.pos_item)
    other = sample_negatives(fold, n_neg=20, scheme=scheme, seed=4)
    assert not np.array_equal(a.neg_items, other.neg_items)


def test_sample_negatives_uniform_is_roughly_flat():
    # one user, 400 items, no fit, one truth item: with many pairs-worth of draws every
    # eligible item should be picked about equally often.
    n_items = 400
    fit = np.zeros((300, n_items), dtype=int)
    truth = np.zeros_like(fit)
    truth[:, 0] = 1
    fold = make_fold(fit, truth)
    sampled = sample_negatives(fold, n_neg=40, scheme="uniform", seed=9)
    counts = np.bincount(sampled.neg_items.ravel(), minlength=n_items)
    assert counts[0] == 0  # the truth item is never a negative
    expected = 300 * 40 / (n_items - 1)  # 30.1 per item
    assert abs(counts[1:].mean() - expected) < 1e-9
    assert counts[1:].std() < 3 * np.sqrt(expected)


def test_sample_negatives_popularity_skews_toward_popular_items():
    fold = random_fold(n_users=200, n_items=300, seed=7, skew=True)
    pop = np.asarray(fold.fit.sum(axis=0)).ravel()
    uniform = sample_negatives(fold, n_neg=30, scheme="uniform", seed=2)
    popular = sample_negatives(fold, n_neg=30, scheme="popularity", seed=2)
    _check_valid(fold, popular, 30)
    assert pop[popular.neg_items].mean() > 1.5 * pop[uniform.neg_items].mean()
    # the most popular decile is drawn far more often than the least popular decile
    order = np.argsort(-pop, kind="stable")
    top, bottom = order[:30], order[-30:]
    counts = np.bincount(popular.neg_items.ravel(), minlength=300)
    assert counts[top].sum() > 3 * counts[bottom].sum()


def test_sample_negatives_raises_when_too_few_eligible():
    fit = np.zeros((1, 10), dtype=int)
    fit[0, :4] = 1
    truth = np.zeros_like(fit)
    truth[0, 4] = 1
    fold = make_fold(fit, truth)  # 5 eligible items
    sample_negatives(fold, n_neg=5)  # exactly enough
    with pytest.raises(ValueError, match="eligible"):
        sample_negatives(fold, n_neg=6)


def test_sample_negatives_popularity_raises_when_support_too_small():
    # items 5..9 have no fit interactions, so popularity sampling can only reach items 0..4
    fit = np.zeros((2, 10), dtype=int)
    fit[0, :5] = 1
    fit[1, 0] = 1
    truth = np.zeros_like(fit)
    truth[1, 1] = 1
    fold = make_fold(fit, truth)
    with pytest.raises(ValueError, match="eligible"):
        sample_negatives(fold, n_neg=4, scheme="popularity")


def test_sample_negatives_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="scheme"):
        sample_negatives(random_fold(), scheme="zipf")


# --- sampled_ranks / sampled_user_metrics ------------------------------------------------


def test_sampled_ranks_constant_model_ties_count_against_positive():
    fold = random_fold(n_users=30, n_items=300, seed=8)
    sampled = sample_negatives(fold, n_neg=100, seed=1)
    ranks = sampled_ranks(FixedModel(np.zeros((30, 300))), sampled)
    assert ranks.dtype == np.int64
    assert (ranks == 101).all()


def _manual_set() -> tuple[SampledSet, FixedModel]:
    # item scores per user (4 items):
    #   user 0: [5, 3, 9, 1]    user 1: [0, 2, 7, 7]
    scores = np.array([[5, 3, 9, 1], [0, 2, 7, 7]], dtype=np.float32)
    sampled = SampledSet(
        user_idx=np.array([0, 0, 1, 1]),
        pos_item=np.array([0, 3, 2, 1]),
        neg_items=np.array([[2, 3], [1, 0], [3, 0], [0, 2]]),
        scheme="uniform",
        seed=0,
    )
    return sampled, FixedModel(scores)


def test_sampled_ranks_hand_computed():
    sampled, model = _manual_set()
    # pair 0: pos score 5, negs 9, 1       -> one neg >= 5            -> rank 2
    # pair 1: pos score 1, negs 3, 5       -> both >= 1               -> rank 3
    # pair 2: pos score 7, negs 7, 0       -> the tie counts          -> rank 2
    # pair 3: pos score 2, negs 0, 7       -> only the 7 is >= 2      -> rank 2
    np.testing.assert_array_equal(sampled_ranks(model, sampled), [2, 3, 2, 2])
    np.testing.assert_array_equal(sampled_ranks(model, sampled, batch=1), [2, 3, 2, 2])


def test_sampled_ranks_batch_invariant_on_random_fold():
    fold = random_fold(n_users=50, n_items=250, seed=10)
    sampled = sample_negatives(fold, n_neg=50, seed=4)
    model = FixedModel(np.random.default_rng(1).integers(0, 20, size=(50, 250)))
    base = sampled_ranks(model, sampled, batch=1024)
    for batch in (1, 4, 13):
        np.testing.assert_array_equal(sampled_ranks(model, sampled, batch=batch), base)
    assert base.min() >= 1 and base.max() <= 51


def test_sampled_user_metrics_hand_computed():
    sampled, _ = _manual_set()
    ranks = np.array([2, 3, 2, 2])
    at2 = sampled_user_metrics(sampled, ranks, k=2)
    # user 0: ranks 2, 3 -> HR@2 = (1 + 0)/2; NDCG@2 = (1/log2(3) + 0)/2
    # user 1: ranks 2, 2 -> HR@2 = 1;         NDCG@2 = 1/log2(3)
    np.testing.assert_allclose(at2["hr"], [0.5, 1.0])
    np.testing.assert_allclose(at2["ndcg"], [0.5 / np.log2(3), 1 / np.log2(3)])
    at3 = sampled_user_metrics(sampled, ranks, k=3)
    # user 0 at k=3: HR = 1; NDCG = (1/log2(3) + 1/log2(4)) / 2
    np.testing.assert_allclose(at3["hr"], [1.0, 1.0])
    np.testing.assert_allclose(at3["ndcg"][0], (1 / np.log2(3) + 0.5) / 2)
    assert set(at2) == {"hr", "ndcg"}


def test_sampled_user_metrics_rank_one_and_users_in_eval_order():
    fold = random_fold(n_users=40, n_items=200, seed=11)
    sampled = sample_negatives(fold, n_neg=20, seed=5)
    out = sampled_user_metrics(sampled, np.ones(len(sampled.pos_item), dtype=np.int64), k=10)
    assert out["hr"].shape == (len(fold.eval_users()),)
    np.testing.assert_allclose(out["hr"], 1.0)
    np.testing.assert_allclose(out["ndcg"], 1.0)  # 1/log2(2)


def test_sampled_user_metrics_length_check():
    sampled, _ = _manual_set()
    with pytest.raises(ValueError, match="one entry per"):
        sampled_user_metrics(sampled, np.array([1, 2]), k=10)
