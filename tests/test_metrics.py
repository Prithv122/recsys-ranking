"""Metric tests: every expected value is worked out by hand in the comment above it."""

from __future__ import annotations

import numpy as np
import pytest

from recsys_ranking.metrics import (
    coverage,
    gini_exposure,
    map_at_k,
    mean_popularity,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def _t(*items: int) -> np.ndarray:
    return np.array(items, dtype=np.int64)


def _recs(*rows: list[int]) -> np.ndarray:
    return np.array(rows, dtype=np.int64)


# truth {2, 5}, recs [5, 1, 2], k = 3: hits at rank 1 and rank 3.
TRUTH = [_t(2, 5)]
RECS = _recs([5, 1, 2])


def test_recall_hand_computed():
    # 2 hits of 2 relevant = 1.0 at k=3; only item 5 in the top 1 -> 1/2.
    assert recall_at_k(RECS, TRUTH, 3)[0] == pytest.approx(1.0)
    assert recall_at_k(RECS, TRUTH, 1)[0] == pytest.approx(0.5)


def test_recall_denominator_is_full_truth_size():
    # truth {1,2,3,4}, top 3 = [1, 2, 9] -> 2 hits / 4 relevant = 0.5
    assert recall_at_k(_recs([1, 2, 9]), [_t(1, 2, 3, 4)], 3)[0] == pytest.approx(0.5)


def test_ndcg_hand_computed():
    # DCG  = 1/log2(1+1) + 1/log2(3+1) = 1 + 1/2 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3)      = 1 + 1/log2(3) = 1.6309...
    expected = 1.5 / (1 + 1 / np.log2(3))
    assert ndcg_at_k(RECS, TRUTH, 3)[0] == pytest.approx(expected)
    assert expected == pytest.approx(0.91972, abs=1e-4)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(_recs([2, 5, 9]), TRUTH, 3)[0] == pytest.approx(1.0)


def test_ndcg_idcg_uses_min_k_and_truth_size():
    # truth {1,2,3,4}, recs [1, 2, 9], k = 3: DCG = 1 + 1/log2(3).
    # IDCG over min(3, 4) = 3 ideal slots = 1 + 1/log2(3) + 1/2.
    dcg = 1 + 1 / np.log2(3)
    idcg = dcg + 0.5
    assert ndcg_at_k(_recs([1, 2, 9]), [_t(1, 2, 3, 4)], 3)[0] == pytest.approx(dcg / idcg)


def test_ndcg_single_truth_item_uses_one_slot_idcg():
    # truth {7}, hit at rank 2, k = 5: DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
    assert ndcg_at_k(_recs([1, 7, 2, 3, 4]), [_t(7)], 5)[0] == pytest.approx(1 / np.log2(3))


def test_map_hand_computed():
    # hits at ranks 1 and 3: precision@1 = 1/1, precision@3 = 2/3 -> sum 5/3.
    # denominator min(3, 2) = 2 -> 5/6
    assert map_at_k(RECS, TRUTH, 3)[0] == pytest.approx(5 / 6)


def test_map_denominator_is_min_k_truth_size():
    # truth of 5 items, k = 3, hits at ranks 1 and 3: (1 + 2/3) / min(3, 5) = 5/9
    value = map_at_k(_recs([1, 9, 2]), [_t(1, 2, 3, 4, 5)], 3)[0]
    assert value == pytest.approx(5 / 9)


def test_mrr_first_hit_only():
    # recs [9, 5, 2], truth {2, 5}: first hit at rank 2 -> 1/2; the later hit is ignored.
    assert mrr_at_k(_recs([9, 5, 2]), TRUTH, 3)[0] == pytest.approx(0.5)


def test_mrr_zero_when_first_hit_beyond_k():
    # first hit at rank 3 but k = 2 -> 0
    assert mrr_at_k(_recs([9, 8, 5]), TRUTH, 2)[0] == 0.0
    assert mrr_at_k(_recs([9, 8, 5]), TRUTH, 3)[0] == pytest.approx(1 / 3)


def test_only_first_k_columns_are_used():
    # Hit sits in column 4 (rank 4); with k = 3 every metric ignores it.
    recs = _recs([9, 8, 7, 5, 2])
    for fn in (recall_at_k, ndcg_at_k, map_at_k, mrr_at_k):
        assert fn(recs, TRUTH, 3)[0] == 0.0
        assert fn(recs, TRUTH, 4)[0] > 0.0


def test_batch_returns_float64_per_user():
    recs = _recs([5, 1, 2], [9, 8, 7])
    truth = [_t(2, 5), _t(1)]
    out = recall_at_k(recs, truth, 3)
    assert out.dtype == np.float64
    assert out.shape == (2,)
    # user 0: 2/2; user 1: no hit -> 0
    np.testing.assert_allclose(out, [1.0, 0.0])
    np.testing.assert_allclose(ndcg_at_k(recs, truth, 3)[1], 0.0)


def test_input_validation():
    with pytest.raises(ValueError, match="columns"):
        recall_at_k(_recs([1, 2]), [_t(1)], 3)
    with pytest.raises(ValueError, match="empty truth"):
        ndcg_at_k(_recs([1, 2, 3]), [_t()], 3)
    with pytest.raises(ValueError, match="truth arrays"):
        map_at_k(_recs([1, 2, 3]), [_t(1), _t(2)], 3)


def test_coverage():
    # recs [[0,1],[1,2]]: unique items {0,1,2} = 3 of 4
    assert coverage(_recs([0, 1], [1, 2]), 4) == pytest.approx(0.75)


def test_mean_popularity():
    pop = np.array([10, 20, 30, 40])
    # slots hold items 0,1,1,2 -> (10 + 20 + 20 + 30) / 4 = 20
    assert mean_popularity(_recs([0, 1], [1, 2]), pop) == pytest.approx(20.0)


def test_gini_equal_counts_is_zero():
    # every one of 4 items appears once
    assert gini_exposure(_recs([0, 1], [2, 3]), 4) == pytest.approx(0.0)


def test_gini_counts_zero_exposure_items():
    # counts [1, 3] over 2 items: mean |xi - xj| / (2 * mean) = (2 * 2) / (2 * 2^2 * 2) -> 0.25
    # Same data with a third, never-shown item: counts [0, 1, 3] -> sorted weights 1,2,3:
    # 2 * (0*1 + 1*2 + 3*3) / (3 * 4) - 4/3 = 22/12 - 16/12 = 0.5
    assert gini_exposure(_recs([0, 1], [1, 1]), 2) == pytest.approx(0.25)
    assert gini_exposure(_recs([1, 2], [2, 2]), 3) == pytest.approx(0.5)


def test_gini_all_on_one_item_is_n_minus_one_over_n():
    recs = np.zeros((50, 5), dtype=np.int64)
    # counts [0]*9 + [250] -> 2 * (10 * 250) / (10 * 250) - 11/10 = 0.9 = (n-1)/n
    assert gini_exposure(recs, 10) == pytest.approx(0.9)
    assert gini_exposure(recs, 1000) == pytest.approx(0.999)
