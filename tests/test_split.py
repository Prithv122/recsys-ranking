"""Splits: hand-derived answers on the fixture, invariants on synthetic data.

Fixture facts (positives, rating >= 4): 47 rows, users 1-8, movies 1-11. Sorted by time, the
rows at 0-based positions 36 and 41 are the quantile cutoffs for 10% + 10%:
(47 - 1) * 0.8 = 36.8 -> index 36 and (47 - 1) * 0.9 = 41.4 -> index 41 with method="lower".
Position 36 ties with 37, and 41 ties with 42, so ties at a cutoff must land on the truth side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from recsys_ranking.data import load_ml1m, positives
from recsys_ranking.split import global_time_folds, leave_last_out, loo_future_share, time_cutoffs
from recsys_ranking.synth import make_synthetic
from recsys_ranking.types import Fold, Interactions

FIXTURE = Path(__file__).parent / "data" / "ratings_fixture.dat"
T_VAL = 1_015_804_800  # 2000-01-01 + 40 * 20 days
T_TEST = 1_033_084_800  # 2000-01-01 + 50 * 20 days


@pytest.fixture(scope="module")
def pos() -> Interactions:
    return positives(load_ml1m(FIXTURE))


def _cells(mat: sp.csr_matrix) -> set[tuple[int, int]]:
    rows, cols = mat.nonzero()
    return set(zip(rows.tolist(), cols.tolist(), strict=True))


def _raw_pairs(fold: Fold, mat: sp.csr_matrix) -> set[tuple[int, int]]:
    rows, cols = mat.nonzero()
    return set(zip(fold.user_ids[rows].tolist(), fold.item_ids[cols].tolist(), strict=True))


def _check_format(fold: Fold) -> None:
    assert fold.fit.shape == fold.truth.shape == (fold.user_ids.size, fold.item_ids.size)
    assert np.all(np.diff(fold.user_ids) > 0) and np.all(np.diff(fold.item_ids) > 0)
    for mat in (fold.fit, fold.truth):
        assert isinstance(mat, sp.csr_matrix)
        assert mat.dtype == np.float32
        assert mat.has_sorted_indices
        assert np.all(mat.data == 1.0)
    assert not _cells(fold.fit) & _cells(fold.truth)
    # Every dense user and item has at least one fit row.
    assert np.all(np.diff(fold.fit.indptr) > 0)
    assert np.all(fold.fit.getnnz(axis=0) > 0)
    assert set(fold.dropped) == {"cold_user_rows", "cold_item_rows", "seen_pair_rows"}


# --- fixture, global time ------------------------------------------------------------------


def test_cutoffs_on_fixture(pos: Interactions) -> None:
    assert len(pos) == 47
    assert time_cutoffs(pos) == (T_VAL, T_TEST)


def test_val_fold_on_fixture(pos: Interactions) -> None:
    val, _ = global_time_folds(pos)
    _check_format(val)
    assert val.name == "val"
    # Fit: the 36 rows before T_VAL; users 1-6 and movies 1-9. User 2 rated movie 3 twice.
    assert val.user_ids.tolist() == [1, 2, 3, 4, 5, 6]
    assert val.item_ids.tolist() == list(range(1, 10))
    assert val.fit.nnz == 35
    assert val.fit.sum() == 35.0
    # User 2 (dense 1) has seen movies 1, 2, 3, 5, 6, 8 -> dense 0, 1, 2, 4, 5, 7.
    assert val.fit[1].indices.tolist() == [0, 1, 2, 4, 5, 7]
    # Truth rows at time T_VAL..T_TEST-1 (5 rows): (u1,m2) seen in fit, (u8,m4) cold user,
    # (u2,m10) cold item, (u3,m9) and (u4,m7) valid.
    assert _raw_pairs(val, val.truth) == {(3, 9), (4, 7)}
    assert _cells(val.truth) == {(2, 8), (3, 6)}
    assert val.eval_users().tolist() == [2, 3]
    assert val.dropped == {"cold_user_rows": 1, "cold_item_rows": 1, "seen_pair_rows": 1}


def test_test_fold_on_fixture(pos: Interactions) -> None:
    _, test = global_time_folds(pos)
    _check_format(test)
    assert test.name == "test"
    # Fit: 41 rows before T_TEST, two of them repeats ((u2,m3) and (u1,m2)). User 8 and movie 10
    # now exist because their first rows fall in the val window.
    assert test.user_ids.tolist() == [1, 2, 3, 4, 5, 6, 8]
    assert test.item_ids.tolist() == list(range(1, 11))
    assert test.fit.nnz == 39
    # Truth at T_TEST and later (6 rows): (u1,m10) and (u1,m3) tie on time and are both kept,
    # (u5,m4) valid, (u3,m11) cold item, (u2,m8) seen, (u7,m11) cold user AND cold item.
    assert _raw_pairs(test, test.truth) == {(1, 10), (1, 3), (5, 4)}
    assert _cells(test.truth) == {(0, 9), (0, 2), (4, 3)}
    assert test.eval_users().tolist() == [0, 4]
    # The both-cold row counts under cold_user_rows only.
    assert test.dropped == {"cold_user_rows": 1, "cold_item_rows": 1, "seen_pair_rows": 1}


def test_tie_at_cutoff_goes_to_truth(pos: Interactions) -> None:
    # (u8,m4) sits exactly at T_VAL: it is not in val fit (so user 8 is cold there)...
    val, test = global_time_folds(pos)
    assert 8 not in val.user_ids
    # ...and (u1,m10), exactly at T_TEST, is truth for test, not fit.
    assert (1, 10) not in _raw_pairs(test, test.fit)
    assert (1, 10) in _raw_pairs(test, test.truth)


def test_fractions_change_cutoffs(pos: Interactions) -> None:
    # 20% test: q = 0.8 -> index 36 for the test cutoff; 5% val: q = 0.75 -> floor(34.5) = 34.
    t_val, t_test = time_cutoffs(pos, val_frac=0.05, test_frac=0.2)
    sorted_ts = np.sort(pos.ts)
    assert t_test == sorted_ts[36]
    assert t_val == sorted_ts[34]


@pytest.mark.parametrize(
    ("val_frac", "test_frac"), [(0.0, 0.1), (0.1, 0.0), (0.5, 0.5), (0.7, 0.4), (-0.1, 0.1)]
)
def test_bad_fractions_raise(pos: Interactions, val_frac: float, test_frac: float) -> None:
    with pytest.raises(ValueError):
        global_time_folds(pos, val_frac, test_frac)


# --- fixture, leave-last-out ---------------------------------------------------------------


def test_loo_on_fixture(pos: Interactions) -> None:
    loo = leave_last_out(pos)
    _check_format(loo)
    assert loo.name == "loo"
    # Users with one positive (7 and 8) are fit-only but present. Movie 11 is in fit through user 7.
    assert loo.user_ids.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert loo.item_ids.tolist() == list(range(1, 12))
    # Latest positive per user: u1 -> m10 (ties with m3 at the same second, larger id wins),
    # u2 -> m8 (already in fit: dropped as seen), u3 -> m11, u4 -> m7, u5 -> m4, u6 -> m3.
    assert _raw_pairs(loo, loo.truth) == {(1, 10), (3, 11), (4, 7), (5, 4), (6, 3)}
    assert loo.eval_users().tolist() == [0, 2, 3, 4, 5]
    assert loo.dropped == {"cold_user_rows": 0, "cold_item_rows": 0, "seen_pair_rows": 1}
    # 41 fit rows with two repeats ((u2,m3) and (u1,m2)).
    assert loo.fit.nnz == 39
    # The tied loser (u1,m3) stays in fit.
    assert (1, 3) in _raw_pairs(loo, loo.fit)
    # Fit-only users have no truth.
    assert loo.truth[6].nnz == 0 and loo.truth[7].nnz == 0


def test_loo_future_share_on_fixture(pos: Interactions) -> None:
    # Surviving truth pairs and their time codes (days/20 since 2000-01-01): u6 31, u4 43, u1 50,
    # u5 51, u3 52. Fit time codes are 1-36 without 31, then 40, 40, 41, 42, 50, 54 (41 rows).
    # Fit rows strictly later: 11, 2, 1, 1, 1 -> mean = 16 / (5 * 41).
    assert loo_future_share(pos) == pytest.approx(16 / 205)


# --- tiny hand-made cases --------------------------------------------------------------------


def _inter(rows: list[tuple[int, int, int]]) -> Interactions:
    arr = np.array(rows, dtype=np.int64)
    return Interactions(
        user=arr[:, 0].copy(),
        item=arr[:, 1].copy(),
        ts=arr[:, 2].copy(),
        rating=np.full(len(rows), 5.0, dtype=np.float32),
    )


def test_loo_future_share_hand_case() -> None:
    inter = _inter(
        [
            (1, 1, 10), (1, 2, 30),  # user 1: latest is item 2 at t=30
            (2, 1, 20), (2, 2, 22),  # user 2: latest is item 2 at t=22
            (3, 2, 50),  # single positive: fit-only
            (4, 1, 25),  # single positive: fit-only
            (5, 9, 5), (5, 8, 6),  # latest item 8 is in no other row -> cold item, not counted
        ]
    )  # fmt: skip
    # Fit timestamps: 10, 20, 25, 50, 5 -> five rows.
    # user 1, t=30: later {50} = 1/5.  user 2, t=22: later {25, 50} = 2/5.  Mean = 0.3.
    assert loo_future_share(inter) == pytest.approx(0.3)
    fold = leave_last_out(inter)
    assert fold.dropped["cold_item_rows"] == 1
    assert fold.eval_users().tolist() == [0, 1]


def test_loo_future_share_no_truth_is_nan() -> None:
    inter = _inter([(1, 1, 10), (2, 2, 20)])
    assert np.isnan(loo_future_share(inter))
    assert leave_last_out(inter).truth.nnz == 0


def test_loo_tie_takes_larger_item_id_regardless_of_row_order() -> None:
    rows = [(1, 7, 100), (1, 3, 100), (1, 5, 50), (2, 4, 60), (2, 9, 70)]
    for ordering in (rows, rows[::-1]):
        fold = leave_last_out(_inter(ordering))
        assert _raw_pairs(fold, fold.truth) == set()  # item 7 / 9 are cold: only one row each...
    # ...so give those items a second user to make them non-cold and check the pick.
    rows += [(3, 7, 10), (3, 9, 11), (4, 7, 1)]
    for ordering in (rows, rows[::-1]):
        fold = leave_last_out(_inter(ordering))
        assert (1, 7) in _raw_pairs(fold, fold.truth)
        assert (1, 3) in _raw_pairs(fold, fold.fit)


def test_duplicate_rows_collapse_to_binary_entries() -> None:
    inter = _inter(
        [
            (1, 1, 1), (1, 1, 2), (1, 2, 3), (2, 1, 4),  # fit: (1,1) is repeated
            (1, 2, 90), (1, 2, 91),  # repeats of a fit pair
            (2, 2, 99), (2, 2, 100),  # the same truth pair twice
        ]
    )  # fmt: skip
    # 8 rows: q=0.8 -> index floor(7 * 0.8) = 5 -> t_val = 91;
    # q=0.9 -> floor(6.3) = 6 -> t_test = 99.
    assert time_cutoffs(inter) == (91, 99)
    val, test = global_time_folds(inter)
    # val: fit ts < 91 -> (1,1)x2, (1,2)x2, (2,1) = 3 distinct cells; truth (1,2,91) is a seen pair.
    assert val.fit.nnz == 3 and val.fit.sum() == 3.0
    assert val.truth.nnz == 0
    assert val.dropped == {"cold_user_rows": 0, "cold_item_rows": 0, "seen_pair_rows": 1}
    # test: fit ts < 99 -> same 3 cells; truth (2,2) appears twice and is stored once.
    assert test.fit.nnz == 3
    assert _raw_pairs(test, test.truth) == {(2, 2)}
    assert test.truth.sum() == 1.0
    assert test.dropped == {"cold_user_rows": 0, "cold_item_rows": 0, "seen_pair_rows": 0}


# --- invariants on synthetic data ------------------------------------------------------------


@pytest.fixture(scope="module")
def synth() -> Interactions:
    inter, _ = make_synthetic(n_users=400, n_items=150, density=0.08, seed=5)
    return inter


def test_no_leakage_and_exact_membership(synth: Interactions) -> None:
    val, test = global_time_folds(synth)
    t_val, t_test = time_cutoffs(synth)
    assert t_val <= t_test
    ts_of = {
        (int(u), int(i)): int(t) for u, i, t in zip(synth.user, synth.item, synth.ts, strict=True)
    }
    assert len(ts_of) == len(synth)  # the generator never repeats a pair

    for fold, lo, hi in ((val, t_val, t_test), (test, t_test, None)):
        _check_format(fold)
        fit_pairs = _raw_pairs(fold, fold.fit)
        truth_pairs = _raw_pairs(fold, fold.truth)
        # Every fit row is strictly before the fit cutoff; every truth row is at or after it.
        assert max(ts_of[p] for p in fit_pairs) < lo
        assert min(ts_of[p] for p in truth_pairs) >= lo
        if hi is not None:
            assert max(ts_of[p] for p in truth_pairs) < hi
        assert not fit_pairs & truth_pairs

        # Membership equals an independent set computation.
        exp_fit = {p for p, t in ts_of.items() if t < lo}
        exp_truth_raw = {p for p, t in ts_of.items() if t >= lo and (hi is None or t < hi)}
        fit_users = {u for u, _ in exp_fit}
        fit_items = {i for _, i in exp_fit}
        exp_truth = {
            (u, i) for u, i in exp_truth_raw if u in fit_users and i in fit_items
        } - exp_fit
        assert fit_pairs == exp_fit
        assert truth_pairs == exp_truth
        assert set(fold.user_ids.tolist()) == fit_users
        assert set(fold.item_ids.tolist()) == fit_items

        # Dropped counts add up to the raw truth rows.
        d = fold.dropped
        assert d["cold_user_rows"] + d["cold_item_rows"] + d["seen_pair_rows"] == (
            len(exp_truth_raw) - len(exp_truth)
        )
        assert d["cold_user_rows"] == sum(1 for u, _ in exp_truth_raw if u not in fit_users)
        assert d["seen_pair_rows"] == 0  # no repeated pairs in this generator

        # Eval users are exactly the users with truth.
        truth_users = {u for u, _ in exp_truth}
        assert {int(fold.user_ids[r]) for r in fold.eval_users()} == truth_users


def test_val_fit_is_prefix_of_test_fit(synth: Interactions) -> None:
    val, test = global_time_folds(synth)
    assert _raw_pairs(val, val.fit) <= _raw_pairs(test, test.fit)


def test_loo_on_synthetic(synth: Interactions) -> None:
    loo = leave_last_out(synth)
    _check_format(loo)
    ts_of = {
        (int(u), int(i)): int(t) for u, i, t in zip(synth.user, synth.item, synth.ts, strict=True)
    }
    rows_per_user: dict[int, list[tuple[int, int]]] = {}
    for (u, i), t in ts_of.items():
        rows_per_user.setdefault(u, []).append((t, i))

    truth_pairs = _raw_pairs(loo, loo.truth)
    fit_pairs = _raw_pairs(loo, loo.fit)
    for u, rows in rows_per_user.items():
        last_t, last_i = max(rows)  # latest time, ties -> larger item id
        in_truth = (u, last_i) in truth_pairs
        if len(rows) < 2:
            assert not in_truth and (u, last_i) in fit_pairs
        elif in_truth:
            assert all(ts_of[(u, i)] <= last_t for i in [p[1] for p in rows])
            assert len([p for p in truth_pairs if p[0] == u]) == 1
            assert (u, last_i) not in fit_pairs
        else:  # truth row dropped because the item has no other row
            assert loo.dropped["cold_item_rows"] > 0
    # Everything is either truth, fit, or a counted drop.
    n_users_multi = sum(1 for r in rows_per_user.values() if len(r) >= 2)
    assert len(truth_pairs) + sum(loo.dropped.values()) == n_users_multi
    assert len(fit_pairs) == len(synth) - n_users_multi

    share = loo_future_share(synth)
    assert 0.0 < share < 1.0
