"""Bootstrap intervals and rank correlation between model orderings."""

from __future__ import annotations

import numpy as np

_CHUNK_ELEMENTS = 4_000_000  # cap on gathered elements per bootstrap chunk


def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 7
) -> tuple[float, float, float]:
    """Mean of ``a - b`` with a 95% percentile interval, resampling users with replacement."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.ndim != 1 or diff.size == 0:
        raise ValueError("a and b must be non-empty 1-D arrays of equal length")
    n = diff.size
    rng = np.random.default_rng(seed)
    chunk = max(1, _CHUNK_ELEMENTS // n)
    means = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(size, n))
        means[start : start + size] = diff[idx].mean(axis=1)
    lo, hi = _percentile_ci(means)
    return float(diff.mean()), lo, hi


def _tau_b_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tau-b for each row of ``x`` against the same row of ``y`` (shape ``(B, m)``).

    Uses all ordered pairs: numerator sum(sign dx * sign dy), denominator
    sqrt(#untied-x pairs * #untied-y pairs). Costs O(m^2) per row.
    """
    sx = np.sign(x[:, :, None] - x[:, None, :]).astype(np.int8)
    sy = np.sign(y[:, :, None] - y[:, None, :]).astype(np.int8)
    num = (sx * sy).sum(axis=(1, 2), dtype=np.int64)
    nx = (sx != 0).sum(axis=(1, 2))
    ny = (sy != 0).sum(axis=(1, 2))
    denom = np.sqrt(nx.astype(np.float64) * ny)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, num / denom, np.nan)


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b (tie-corrected); NaN if a vector is constant. O(n^2): for short vectors."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1-D arrays of equal length")
    return float(_tau_b_rows(x[None, :], y[None, :])[0])


def ordering(means: dict[str, float]) -> list[str]:
    """Model names best first by mean; ties are broken by name."""
    return sorted(means, key=lambda name: (-means[name], name))


def ordering_tau_ci(
    full: dict[str, np.ndarray],
    sampled: dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Tau-b between the model orderings of two protocols, with a 95% bootstrap interval.

    ``full`` and ``sampled`` map model name to a per-user array; every array has the same
    length and user order. Users are resampled jointly: the same indices are used for every
    model under both protocols before the per-model means are recomputed.
    """
    if set(full) != set(sampled):
        raise ValueError("full and sampled must have the same model names")
    names = sorted(full)
    columns = [np.asarray(d[m], dtype=np.float64) for d in (full, sampled) for m in names]
    if len({c.shape for c in columns}) != 1 or columns[0].ndim != 1:
        raise ValueError("every per-user array must be 1-D with the same length")
    f = np.column_stack(columns[: len(names)])
    s = np.column_stack(columns[len(names) :])
    n, m = f.shape
    if n == 0:
        raise ValueError("per-user arrays are empty")
    point = float(_tau_b_rows(f.mean(axis=0)[None, :], s.mean(axis=0)[None, :])[0])

    rng = np.random.default_rng(seed)
    chunk = max(1, _CHUNK_ELEMENTS // (n * m))
    taus = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(size, n))
        taus[start : start + size] = _tau_b_rows(f[idx].mean(axis=1), s[idx].mean(axis=1))
    lo, hi = _percentile_ci(taus)
    return point, lo, hi
