"""Model contract tests on a small planted dataset (no dependence on synth.py)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from recsys_ranking import evaluate
from recsys_ranking.models import REGISTRY, UNAVAILABLE
from recsys_ranking.models.als import ALS
from recsys_ranking.models.bpr import BPR
from recsys_ranking.models.itemknn import ItemKNN
from recsys_ranking.models.popularity import Popularity
from recsys_ranking.models.random_model import RandomModel

N_GROUPS, USERS_PER_GROUP, BLOCK, N_POPULAR = 3, 50, 30, 20
N_USERS = N_GROUPS * USERS_PER_GROUP
N_ITEMS = N_GROUPS * BLOCK + N_POPULAR

CONFIGS: dict[str, dict] = {
    "random": {},
    "popularity": {},
    "itemknn": {"n_neighbors": 20, "shrink": 1.0},
    "als": {"factors": 8, "reg": 0.1, "alpha": 10.0, "iterations": 8},
    "bpr": {"factors": 8, "lr": 0.05, "reg": 1e-3, "epochs": 20},
}


def make_planted(seed: int = 3) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return (train, heldout): each group likes its own item block; some items are liked by all."""
    rng = np.random.default_rng(seed)
    prob = np.full((N_USERS, N_ITEMS), 0.01)
    prob[:, N_GROUPS * BLOCK :] = 0.25  # globally popular items
    for g in range(N_GROUPS):
        users = slice(g * USERS_PER_GROUP, (g + 1) * USERS_PER_GROUP)
        prob[users, g * BLOCK : (g + 1) * BLOCK] = 0.5
    full = rng.random((N_USERS, N_ITEMS)) < prob
    hold = full & (rng.random((N_USERS, N_ITEMS)) < 0.3)
    train = full & ~hold
    return sp.csr_matrix(train.astype(np.float32)), sp.csr_matrix(hold.astype(np.float32))


def recall_at_10(model, train: sp.csr_matrix, heldout: sp.csr_matrix) -> float:
    users = np.flatnonzero(np.diff(heldout.indptr) > 0)
    scores = model.score(users).copy()
    scores[train[users].toarray() > 0] = -np.inf  # mask seen items
    top = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    truth = heldout[users].toarray() > 0
    hits = np.take_along_axis(truth, top, axis=1).sum(axis=1)
    return float(np.mean(hits / truth.sum(axis=1)))


@pytest.fixture(scope="module")
def data() -> tuple[sp.csr_matrix, sp.csr_matrix]:
    return make_planted()


@pytest.fixture(scope="module")
def fitted(data) -> dict:
    train, _ = data
    return {name: REGISTRY[name](**cfg, seed=0).fit(train) for name, cfg in CONFIGS.items()}


def test_registry_has_core_models() -> None:
    assert {"random", "popularity", "itemknn", "als", "bpr"} <= set(REGISTRY)
    for name, cls in REGISTRY.items():
        assert cls.name == name


def test_registry_and_grids_agree() -> None:
    """Every runnable model has a registered grid, and every grid point builds its model."""
    assert not set(REGISTRY) & set(UNAVAILABLE)  # a model is available or has a reason, not both
    known = {*REGISTRY, *UNAVAILABLE}
    for grids in (evaluate.GRIDS, evaluate.SMOKE_GRIDS):
        assert set(grids) == known
        for name, points in grids.items():
            assert points, name
            if name in REGISTRY:
                for params in points:
                    assert REGISTRY[name](**params, seed=0).params()["seed"] == 0, (name, params)
    # evaluate's default model order is its grid order restricted to what is registered
    assert evaluate.Config().model_names() == [m for m in evaluate.GRIDS if m in REGISTRY]


@pytest.mark.parametrize("name", list(CONFIGS))
def test_score_shape_dtype_finite(name, fitted) -> None:
    users = np.array([0, 7, 120, 149])
    s = fitted[name].score(users)
    assert s.shape == (4, N_ITEMS)
    assert s.dtype == np.float32
    assert np.isfinite(s).all()


@pytest.mark.parametrize("name", list(CONFIGS))
def test_deterministic_and_batch_independent(name, data, fitted) -> None:
    train, _ = data
    again = REGISTRY[name](**CONFIGS[name], seed=0).fit(train)
    everyone = np.arange(N_USERS)
    full = fitted[name].score(everyone)
    np.testing.assert_array_equal(full, again.score(everyone))
    for u in (0, 61, 149):
        np.testing.assert_allclose(fitted[name].score(np.array([u]))[0], full[u], rtol=1e-5)
    # Order of the batch does not matter either.
    shuffled = np.array([149, 3, 77])
    np.testing.assert_allclose(fitted[name].score(shuffled), full[shuffled], rtol=1e-5)


@pytest.mark.parametrize("name", ["random", "bpr", "als"])
def test_seed_changes_result(name, data, fitted) -> None:
    train, _ = data
    other = REGISTRY[name](**CONFIGS[name], seed=1).fit(train)
    assert not np.array_equal(other.score(np.arange(5)), fitted[name].score(np.arange(5)))


@pytest.mark.parametrize("name", list(CONFIGS))
def test_params_round_trip(name, fitted) -> None:
    model = fitted[name]
    params = model.params()
    assert params["seed"] == 0
    assert REGISTRY[name](**params).params() == params
    for key in CONFIGS[name]:
        assert key in params


def test_popularity_same_row_for_everyone(data, fitted) -> None:
    train, _ = data
    s = fitted["popularity"].score(np.arange(N_USERS))
    assert (s == s[0]).all()
    np.testing.assert_array_equal(s[0], np.asarray(train.sum(axis=0)).ravel())


def test_itemknn_structure(data) -> None:
    train, _ = data
    model = ItemKNN(n_neighbors=5, shrink=0.0).fit(train)
    sim = model._sim.tocsc()
    assert sim.diagonal().sum() == 0
    assert np.diff(sim.indptr).max() <= 5  # at most n_neighbors per item
    assert sim.data.max() <= 1.0 + 1e-6  # cosine on binary data


def test_cold_users_and_items_are_safe() -> None:
    train = sp.csr_matrix(
        (np.ones(4, dtype=np.float32), ([0, 0, 1, 1], [0, 1, 1, 2])), shape=(4, 5)
    )  # user 2, 3 and items 3, 4 have no interactions
    for name, cfg in CONFIGS.items():
        s = REGISTRY[name](**cfg).fit(train).score(np.arange(4))
        assert s.shape == (4, 5)
        assert np.isfinite(s).all(), name


def test_beats_baselines_on_planted_data(data, fitted) -> None:
    train, heldout = data
    recall = {name: recall_at_10(m, train, heldout) for name, m in fitted.items()}
    assert recall["popularity"] > recall["random"] + 0.02
    for name in ("itemknn", "als", "bpr"):
        assert recall[name] > recall["popularity"] + 0.02, recall


def test_classes_exported() -> None:
    assert REGISTRY["random"] is RandomModel
    assert REGISTRY["popularity"] is Popularity
    assert REGISTRY["itemknn"] is ItemKNN
    assert REGISTRY["als"] is ALS
    assert REGISTRY["bpr"] is BPR


def test_als_sweep_matches_dense_formula(data) -> None:
    """One sweep equals the textbook per-user solve (Y^T C Y + reg I) x = Y^T C p."""
    train, _ = data
    rng = np.random.default_rng(0)
    y = rng.standard_normal((N_ITEMS, 6))
    model = ALS(factors=6, reg=0.3, alpha=7.0)
    got = model._sweep(sp.csr_matrix(train, dtype=np.float64), y)
    dense = train.toarray()
    for u in (0, 80, 149):
        c = np.diag(1.0 + model.alpha * dense[u])
        lhs = y.T @ c @ y + model.reg * np.eye(6)
        np.testing.assert_allclose(
            got[u], np.linalg.solve(lhs, y.T @ c @ (dense[u] > 0)), atol=1e-9
        )
