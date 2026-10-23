"""Two-tower model on a small planted dataset (skipped when torch is not installed)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("torch")

from recsys_ranking.models.twotower import TwoTower

N_GROUPS, USERS_PER_GROUP, BLOCK = 3, 40, 30


def _planted(seed: int = 0) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Each user group prefers one block of items; 5 own-block items per user are held out."""
    rng = np.random.default_rng(seed)
    n_users, n_items = N_GROUPS * USERS_PER_GROUP, N_GROUPS * BLOCK
    fit = np.zeros((n_users, n_items), dtype=np.float32)
    held = np.zeros_like(fit)
    for u in range(n_users):
        g = u // USERS_PER_GROUP
        own = rng.permutation(np.arange(g * BLOCK, (g + 1) * BLOCK))[:20]
        fit[u, own[:15]] = 1
        held[u, own[15:]] = 1
        fit[u, rng.integers(0, n_items, size=3)] = 1  # off-block noise
    held[fit > 0] = 0
    return sp.csr_matrix(fit), sp.csr_matrix(held)


def _model(**kw: object) -> TwoTower:
    return TwoTower(dim=16, lr=5e-3, epochs=25, batch_size=256, **kw)


def _recall_at_10(scores: np.ndarray, fit: sp.csr_matrix, held: sp.csr_matrix) -> float:
    scores = scores.copy()
    scores[fit.toarray() > 0] = -np.inf
    top = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    held_d = held.toarray()
    hits = np.take_along_axis(held_d, top, axis=1).sum(axis=1)
    return float(np.mean(hits / held_d.sum(axis=1)))


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    """Tiny matmuls gain nothing from threads, and a loaded CI box makes them crawl."""
    import torch

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def planted() -> tuple[sp.csr_matrix, sp.csr_matrix]:
    return _planted()


@pytest.fixture(scope="module")
def fitted(planted: tuple[sp.csr_matrix, sp.csr_matrix]) -> TwoTower:
    return _model(seed=0).fit(planted[0])


def test_shape_dtype_params(planted, fitted) -> None:
    fit, _ = planted
    s = fitted.score(np.arange(7))
    assert s.shape == (7, fit.shape[1])
    assert s.dtype == np.float32
    assert np.isfinite(s).all()
    assert fitted.name == "twotower"
    assert fitted.params()["dim"] == 16
    assert fitted.params()["seed"] == 0


def test_deterministic_per_seed(planted, fitted) -> None:
    fit, _ = planted
    again = _model(seed=0).fit(fit)
    np.testing.assert_array_equal(fitted.score(np.arange(20)), again.score(np.arange(20)))
    other = _model(seed=1).fit(fit)
    assert not np.array_equal(fitted.score(np.arange(20)), other.score(np.arange(20)))


def test_batch_independence(fitted) -> None:
    users = np.array([5, 70, 33, 5])
    together = fitted.score(users)
    for row, u in zip(together, users, strict=True):
        np.testing.assert_allclose(row, fitted.score(np.array([u]))[0], atol=1e-6)


def test_beats_popularity_on_heldout_recall(planted, fitted) -> None:
    fit, held = planted
    pop = np.tile(np.asarray(fit.sum(axis=0)).ravel(), (fit.shape[0], 1)).astype(np.float32)
    model_recall = _recall_at_10(fitted.score(np.arange(fit.shape[0])), fit, held)
    pop_recall = _recall_at_10(pop, fit, held)
    assert model_recall > pop_recall + 0.2


def test_score_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        TwoTower().score(np.array([0]))
