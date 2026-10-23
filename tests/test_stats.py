"""Bootstrap and Kendall tau tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from recsys_ranking.stats import kendall_tau, ordering, ordering_tau_ci, paired_bootstrap


def test_paired_bootstrap_constant_difference():
    a = np.full(50, 0.7)
    b = np.full(50, 0.2)
    point, lo, hi = paired_bootstrap(a, b, n_boot=200)
    assert point == pytest.approx(0.5)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_paired_bootstrap_identical_inputs_is_zero():
    x = np.random.default_rng(0).random(100)
    assert paired_bootstrap(x, x, n_boot=100) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_interval_covers_true_shift_and_excludes_zero():
    rng = np.random.default_rng(1)
    b = rng.normal(size=2000)
    a = b + 0.3 + rng.normal(scale=0.5, size=2000)
    point, lo, hi = paired_bootstrap(a, b)
    assert lo <= point <= hi
    assert lo < 0.3 < hi
    assert lo > 0


def test_paired_bootstrap_ci_width_shrinks_with_n():
    rng = np.random.default_rng(2)
    d_small = rng.normal(size=100)
    d_large = rng.normal(size=10_000)
    _, lo_s, hi_s = paired_bootstrap(d_small, np.zeros(100))
    _, lo_l, hi_l = paired_bootstrap(d_large, np.zeros(10_000))
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_paired_bootstrap_deterministic_per_seed():
    rng = np.random.default_rng(3)
    a, b = rng.random(300), rng.random(300)
    assert paired_bootstrap(a, b, seed=5) == paired_bootstrap(a, b, seed=5)
    assert paired_bootstrap(a, b, seed=5) != paired_bootstrap(a, b, seed=6)


def test_paired_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([]), np.array([]))


def test_kendall_tau_hand_computed():
    # x = 1,2,3 ; y = 1,3,2: pairs (1,2) concordant, (1,3) concordant, (2,3) discordant
    # -> (2 - 1) / 3
    assert kendall_tau(np.array([1, 2, 3]), np.array([1, 3, 2])) == pytest.approx(1 / 3)
    assert kendall_tau(np.array([1, 2, 3]), np.array([3, 2, 1])) == pytest.approx(-1.0)


def test_kendall_tau_hand_computed_with_ties():
    # x = 1,1,2 ; y = 1,2,3. Pairs: (1,2) tied in x; (1,3) concordant; (2,3) concordant.
    # n0 = 3, n1 = 1 (x ties), n2 = 0 -> tau-b = 2 / sqrt((3-1) * 3)
    assert kendall_tau(np.array([1, 1, 2]), np.array([1, 2, 3])) == pytest.approx(
        2 / np.sqrt(2 * 3)
    )


@pytest.mark.parametrize("seed", range(8))
def test_kendall_tau_matches_scipy_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 120))
    x = rng.integers(0, 6, size=n).astype(float)  # many ties
    y = (x + rng.integers(-2, 3, size=n)).astype(float)
    assert kendall_tau(x, y) == pytest.approx(sps.kendalltau(x, y, variant="b").statistic)


def test_kendall_tau_matches_scipy_continuous():
    rng = np.random.default_rng(99)
    x, y = rng.normal(size=200), rng.normal(size=200)
    assert kendall_tau(x, y) == pytest.approx(sps.kendalltau(x, y).statistic)


def test_kendall_tau_constant_vector_is_nan():
    assert np.isnan(kendall_tau(np.ones(4), np.arange(4)))


def test_ordering_best_first_ties_by_name():
    assert ordering({"b": 0.5, "a": 0.5, "c": 0.9, "d": 0.1}) == ["c", "a", "b", "d"]


def _per_user(values: dict[str, float], n: int) -> dict[str, np.ndarray]:
    return {m: np.full(n, v) for m, v in values.items()}


def test_identical_inputs_give_tau_one_and_degenerate_ci():
    rng = np.random.default_rng(0)
    full = {
        m: rng.random(200) + shift for m, shift in zip("abcd", (0.0, 0.2, 0.4, 0.6), strict=True)
    }
    tau, lo, hi = ordering_tau_ci(full, {m: v.copy() for m, v in full.items()}, n_boot=300)
    assert (tau, lo, hi) == (1.0, 1.0, 1.0)


def test_reversed_orderings_give_tau_minus_one():
    full = _per_user({"a": 0.9, "b": 0.5, "c": 0.1}, 40)
    sampled = _per_user({"a": 0.1, "b": 0.5, "c": 0.9}, 40)
    assert ordering_tau_ci(full, sampled, n_boot=200) == (-1.0, -1.0, -1.0)


def test_two_models_swap():
    # full: a > b > c; sampled: b > a > c. Pairs (a,b) discordant, (a,c) and (b,c) concordant
    # -> tau = (2 - 1) / 3
    full = _per_user({"a": 0.5, "b": 0.4, "c": 0.1}, 30)
    sampled = _per_user({"a": 0.4, "b": 0.5, "c": 0.1}, 30)
    tau, lo, hi = ordering_tau_ci(full, sampled, n_boot=200)
    assert tau == pytest.approx(1 / 3)
    assert (lo, hi) == (pytest.approx(1 / 3), pytest.approx(1 / 3))


def test_noisy_swap_has_a_real_interval_and_is_deterministic():
    rng = np.random.default_rng(4)
    n = 150
    base = {"a": 0.50, "b": 0.48, "c": 0.20}
    full = {m: v + rng.normal(scale=0.3, size=n) for m, v in base.items()}
    sampled = {
        "a": full["a"] - 0.02 + rng.normal(scale=0.05, size=n),
        "b": full["b"] + 0.02 + rng.normal(scale=0.05, size=n),
        "c": full["c"] + rng.normal(scale=0.05, size=n),
    }
    first = ordering_tau_ci(full, sampled, n_boot=400, seed=11)
    assert first == ordering_tau_ci(full, sampled, n_boot=400, seed=11)
    _, lo, hi = first
    assert -1.0 <= lo <= hi <= 1.0
    assert lo < hi  # resampling users flips the a/b order in some resamples


def test_bootstrap_resamples_users_jointly():
    # a and b are nearly tied, so independent resampling per protocol would flip their order
    # in some resamples of one protocol but not the other. Identical per-user values under
    # both protocols keep tau at exactly 1 only if the same users are drawn for both.
    rng = np.random.default_rng(8)
    full = {"a": rng.random(60), "b": rng.random(60), "c": rng.random(60) - 5}
    sampled = {m: 2 * v + 1 for m, v in full.items()}
    assert ordering_tau_ci(full, sampled, n_boot=300) == (1.0, 1.0, 1.0)


def test_ordering_tau_ci_validates_inputs():
    with pytest.raises(ValueError, match="model names"):
        ordering_tau_ci({"a": np.ones(3)}, {"b": np.ones(3)})
    with pytest.raises(ValueError, match="same length"):
        ordering_tau_ci({"a": np.ones(3), "b": np.ones(3)}, {"a": np.ones(3), "b": np.ones(4)})
