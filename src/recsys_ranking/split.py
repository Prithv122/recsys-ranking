"""Train/evaluation splits. Input is already positives-only; ratings are not looked at here.

Every fold is reindexed from its fit rows: dense indices follow ascending raw id, users and items
with no fit row leave the index space, and the truth rows that referred to them are dropped
and counted in ``Fold.dropped``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys_ranking.types import Fold, Interactions


def _binary_csr(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    """Binary float32 CSR from (row, col) pairs; duplicate pairs collapse to one entry."""
    if rows.size == 0:
        return sp.csr_matrix(shape, dtype=np.float32)
    keys = np.unique(rows.astype(np.int64) * shape[1] + cols)
    r, c = np.divmod(keys, shape[1])
    mat = sp.csr_matrix((np.ones(keys.size, dtype=np.float32), (r, c)), shape=shape)
    mat.sum_duplicates()
    mat.sort_indices()
    return mat


def _build_fold(
    name: str,
    fit_user: np.ndarray,
    fit_item: np.ndarray,
    truth_user: np.ndarray,
    truth_item: np.ndarray,
) -> tuple[Fold, np.ndarray]:
    """Build a fold from raw-id rows; also return a mask of the truth rows that were kept."""
    user_ids = np.unique(fit_user)
    item_ids = np.unique(fit_item)
    shape = (user_ids.size, item_ids.size)

    fit = _binary_csr(
        np.searchsorted(user_ids, fit_user), np.searchsorted(item_ids, fit_item), shape
    )

    cold_user = ~np.isin(truth_user, user_ids)
    # A row that is both cold-user and cold-item is counted as cold_user only.
    cold_item = ~cold_user & ~np.isin(truth_item, item_ids)
    ok = ~cold_user & ~cold_item

    t_rows = np.searchsorted(user_ids, truth_user[ok])
    t_cols = np.searchsorted(item_ids, truth_item[ok])
    t_keys = t_rows.astype(np.int64) * shape[1] + t_cols
    fit_r, fit_c = fit.nonzero()
    fit_keys = fit_r.astype(np.int64) * shape[1] + fit_c
    seen = np.isin(t_keys, fit_keys)

    keep = np.zeros(truth_user.shape[0], dtype=bool)
    keep[np.flatnonzero(ok)[~seen]] = True
    truth = _binary_csr(t_rows[~seen], t_cols[~seen], shape)

    dropped = {
        "cold_user_rows": int(cold_user.sum()),
        "cold_item_rows": int(cold_item.sum()),
        "seen_pair_rows": int(seen.sum()),
    }
    fold = Fold(
        name=name, fit=fit, truth=truth, user_ids=user_ids, item_ids=item_ids, dropped=dropped
    )
    return fold, keep


def time_cutoffs(
    inter: Interactions, val_frac: float = 0.1, test_frac: float = 0.1
) -> tuple[int, int]:
    """Return ``(t_val, t_test)``: timestamp quantiles ``q(1-val-test)`` and ``q(1-test)``.

    Uses ``np.quantile(..., method="lower")``: the cutoff is always an observed timestamp, the
    element at sorted index ``floor((n - 1) * q)``, so ``ts < t`` and ``ts >= t`` are exact.
    """
    if not (val_frac > 0 and test_frac > 0 and val_frac + test_frac < 1):
        raise ValueError("val_frac and test_frac must be positive and sum to less than 1")
    if len(inter) == 0:
        raise ValueError("cannot split an empty interaction set")
    # Rounding removes float noise such as 1 - 0.1 - 0.1 = 0.7999999999999999.
    q_val = round(1.0 - val_frac - test_frac, 12)
    q_test = round(1.0 - test_frac, 12)
    t_val, t_test = np.quantile(inter.ts, [q_val, q_test], method="lower")
    return int(t_val), int(t_test)


def global_time_folds(
    inter: Interactions, val_frac: float = 0.1, test_frac: float = 0.1
) -> tuple[Fold, Fold]:
    """Global-time split: ``val`` (fit before t_val, truth in [t_val, t_test)) and ``test``."""
    t_val, t_test = time_cutoffs(inter, val_frac, test_frac)

    fit_val = inter.ts < t_val
    truth_val = (inter.ts >= t_val) & (inter.ts < t_test)
    val, _ = _build_fold(
        "val",
        inter.user[fit_val],
        inter.item[fit_val],
        inter.user[truth_val],
        inter.item[truth_val],
    )

    fit_test = inter.ts < t_test
    truth_test = inter.ts >= t_test
    test, _ = _build_fold(
        "test",
        inter.user[fit_test],
        inter.item[fit_test],
        inter.user[truth_test],
        inter.item[truth_test],
    )
    return val, test


def _loo_truth_mask(inter: Interactions) -> np.ndarray:
    """Boolean mask of the leave-last-out truth rows."""
    n = len(inter)
    # Sorted by user, then time, then item id: the last row of a user's block is their latest
    # positive, and a timestamp tie resolves to the larger item id.
    order = np.lexsort((inter.item, inter.ts, inter.user))
    u_sorted = inter.user[order]
    is_last = np.ones(n, dtype=bool)
    is_last[:-1] = u_sorted[1:] != u_sorted[:-1]
    is_first = np.ones(n, dtype=bool)
    is_first[1:] = u_sorted[1:] != u_sorted[:-1]
    truth_sorted = is_last & ~is_first  # users with one positive stay fit-only
    truth = np.zeros(n, dtype=bool)
    truth[order[truth_sorted]] = True
    return truth


def _leave_last_out(inter: Interactions) -> tuple[Fold, np.ndarray, np.ndarray]:
    if len(inter) == 0:
        raise ValueError("cannot split an empty interaction set")
    truth_mask = _loo_truth_mask(inter)
    fit_mask = ~truth_mask
    fold, keep = _build_fold(
        "loo",
        inter.user[fit_mask],
        inter.item[fit_mask],
        inter.user[truth_mask],
        inter.item[truth_mask],
    )
    return fold, inter.ts[truth_mask][keep], inter.ts[fit_mask]


def leave_last_out(inter: Interactions) -> Fold:
    """Each user's latest positive is truth (ties: larger item id); the rest is fit."""
    return _leave_last_out(inter)[0]


def loo_future_share(inter: Interactions) -> float:
    """Leak size of ``leave_last_out``.

    For every truth pair that survives into the fold, the share of fit rows (raw rows, not
    deduplicated) with a strictly later timestamp; the result is the mean over those pairs.
    NaN when the fold has no truth pairs.
    """
    _, truth_ts, fit_ts = _leave_last_out(inter)
    if truth_ts.size == 0 or fit_ts.size == 0:
        return float("nan")
    fit_sorted = np.sort(fit_ts)
    later = fit_sorted.size - np.searchsorted(fit_sorted, truth_ts, side="right")
    return float(np.mean(later / fit_sorted.size))
