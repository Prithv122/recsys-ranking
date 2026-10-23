"""Runner tests on small synthetic data: tuning, staging, merging, aggregation and the CLI.

Everything here uses the five numpy models (no torch), the smoke grids and a few hundred
bootstrap draws, so the whole file runs in well under a minute.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from recsys_ranking import cli, evaluate
from recsys_ranking.evaluate import Config
from recsys_ranking.synth import make_synthetic
from recsys_ranking.types import Interactions

FIVE = ("random", "popularity", "itemknn", "als", "bpr")
DERIVED = ("summary.csv", "pairs.csv", "ordering.csv")


def quiet(_msg: str) -> None:
    pass


def make_cfg(models: tuple[str, ...] | None = FIVE) -> Config:
    # 50 negatives need well over 50 active items; 200 x 150 at density 0.15 has about 140
    return Config(n_neg=50, n_boot=100, grids=evaluate.SMOKE_GRIDS, models=models)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_per_user(out: Path) -> dict[str, np.ndarray]:
    with np.load(out / "raw" / "per_user.npz") as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def inter() -> Interactions:
    return make_synthetic(n_users=200, n_items=150, density=0.15, seed=0)[0]


@pytest.fixture(scope="module")
def all_run(inter, tmp_path_factory) -> Path:
    """One end-to-end run of the five models; shared by the read-only tests below."""
    out = tmp_path_factory.mktemp("all")
    evaluate.run(inter, out, make_cfg(), log=quiet, stage="all")
    return out


# --- best_params -------------------------------------------------------------------------


def _row(model: str, config: int, score: float) -> dict:
    return {"model": model, "config": config, "params": {"c": config}, "val_ndcg10": score}


def test_best_params_highest_score_wins():
    rows = [_row("m", 0, 0.1), _row("m", 1, 0.3), _row("m", 2, 0.2)]
    assert evaluate.best_params(rows) == {"m": {"c": 1}}


def test_best_params_tie_goes_to_earlier_grid_point():
    rows = [_row("m", 0, 0.25), _row("m", 1, 0.25), _row("m", 2, 0.1), _row("m", 3, 0.25)]
    assert evaluate.best_params(rows) == {"m": {"c": 0}}
    # a later strict improvement still beats an earlier tie
    rows.append(_row("m", 4, 0.2500001))
    assert evaluate.best_params(rows) == {"m": {"c": 4}}


def test_best_params_models_are_independent():
    rows = [_row("a", 0, 0.1), _row("b", 0, 0.9), _row("a", 1, 0.2), _row("b", 1, 0.9)]
    assert evaluate.best_params(rows) == {"a": {"c": 1}, "b": {"c": 0}}


# --- tuning.csv round trip ---------------------------------------------------------------


def test_tuning_csv_round_trip_keeps_selection(inter, tmp_path, monkeypatch):
    """Writer and reader of tuning.csv, with `tune` replaced by fixed scores.

    This does not exercise real scoring: it checks that what `run(stage="tune")` writes and
    `read_tuning` parses selects the same grid point as the in-memory rows. Scores that
    collapse to one value under 6 significant digits would tie and pick config 0 if the
    writer rounded them.
    """
    scores = [0.1234561, 0.1234564, 0.1234562]
    params = [{"n_neighbors": n, "shrink": 0} for n in (10, 20, 30)]
    rows = [
        {"model": "itemknn", "config": i, "params": p, "val_ndcg10": s, "fit_s": 0.01}
        for i, (p, s) in enumerate(zip(params, scores, strict=True))
    ]
    assert len({f"{s:.6g}" for s in scores}) == 1  # rounded output could not tell them apart
    in_memory = evaluate.best_params(rows)
    assert in_memory == {"itemknn": params[1]}

    monkeypatch.setattr(evaluate, "tune", lambda fold, cfg, log: [dict(r) for r in rows])
    cfg = Config(models=("itemknn",), grids={"itemknn": params})
    out = evaluate.run(inter, tmp_path, cfg, log=quiet, stage="tune")

    assert out == tmp_path
    assert not (tmp_path / "raw").exists()  # tune stage never scores
    read_back = evaluate.read_tuning(tmp_path / "tuning.csv")
    assert [r["val_ndcg10"] for r in read_back] == scores  # exact floats, not 6-digit copies
    assert [r["params"] for r in read_back] == params
    assert evaluate.best_params(read_back) == in_memory


# --- staged runs -------------------------------------------------------------------------


def test_tune_then_score_matches_single_pass(inter, all_run, tmp_path):
    evaluate.run(inter, tmp_path, make_cfg(), log=quiet, stage="tune")
    assert (tmp_path / "tuning.csv").exists()
    assert not (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "raw").exists()
    evaluate.run(inter, tmp_path, make_cfg(), log=quiet, stage="score")
    for name in DERIVED:
        assert (tmp_path / name).read_bytes() == (all_run / name).read_bytes(), name


def test_scoring_a_subset_later_merges(inter, tmp_path):
    first = ("random", "popularity", "itemknn")
    evaluate.run(inter, tmp_path, make_cfg(first), log=quiet, stage="all")
    before = read_csv(tmp_path / "summary.csv")
    before_arrays = load_per_user(tmp_path)
    before_timings = {r["model"] for r in read_csv(tmp_path / "timings.csv")}
    assert {r["model"] for r in before} == set(first)
    assert before_timings == set(first)

    evaluate.run(inter, tmp_path, make_cfg(("als",)), log=quiet, stage="tune")
    assert {r["model"] for r in read_csv(tmp_path / "tuning.csv")} == {*first, "als"}
    evaluate.run(inter, tmp_path, make_cfg(("als",)), log=quiet, stage="score")

    after = read_csv(tmp_path / "summary.csv")
    assert {r["model"] for r in after} == {*first, "als"}
    assert [r for r in after if r["model"] in first] == before  # earlier rows are untouched
    after_arrays = load_per_user(tmp_path)
    for key, vals in before_arrays.items():
        np.testing.assert_array_equal(after_arrays[key], vals)
    assert {k.split("|")[2] for k in after_arrays} == {*first, "als"}
    assert {r["model"] for r in read_csv(tmp_path / "timings.csv")} == {*first, "als"}
    assert {r["model_a"] for r in read_csv(tmp_path / "pairs.csv")} <= {*first, "als"}


def test_rescoring_a_scored_model_needs_force(inter, tmp_path):
    cfg = make_cfg(("random", "popularity"))
    evaluate.run(inter, tmp_path, cfg, log=quiet, stage="all")
    kept = {n: (tmp_path / n).read_bytes() for n in (*DERIVED, "raw/per_user.npz")}

    with pytest.raises(ValueError, match="already scored"):
        evaluate.run(inter, tmp_path, cfg, log=quiet, stage="all")
    with pytest.raises(ValueError, match="already scored"):
        evaluate.run(inter, tmp_path, make_cfg(("popularity",)), log=quiet, stage="score")
    for name, data in kept.items():  # the refusal happens before anything is overwritten
        assert (tmp_path / name).read_bytes() == data, name

    evaluate.run(inter, tmp_path, cfg, log=quiet, stage="all", force=True)
    for name in DERIVED:  # same seeds and data, so a forced re-score reproduces the files
        assert (tmp_path / name).read_bytes() == kept[name], name


# --- Config.model_names ------------------------------------------------------------------


def test_model_names_rejects_unknown_or_miscased_names():
    with pytest.raises(ValueError, match="unknown model 'itemKNN'"):
        Config(models=("itemKNN",)).model_names()
    with pytest.raises(ValueError, match="unknown model"):
        Config(models=("random", "nope")).model_names()
    # registered but missing from the grid table is also an error
    with pytest.raises(ValueError, match="unknown model"):
        Config(models=("bpr",), grids={"random": [{}]}).model_names()


def test_model_names_unavailable_model_reports_its_reason(monkeypatch):
    monkeypatch.setitem(evaluate.UNAVAILABLE, "twotower", "ImportError: no module named torch")
    with pytest.raises(ValueError, match=r"unavailable.*no module named torch"):
        Config(models=("twotower",)).model_names()
    with pytest.raises(ValueError, match="unavailable"):
        Config(models=("random", "twotower")).model_names()


def test_model_names_default_is_registry_models_in_grid_order():
    names = Config().model_names()
    assert names == [m for m in evaluate.GRIDS if m in evaluate.REGISTRY]
    assert names[:5] == list(FIVE)
    reordered = Config(grids={"bpr": [{}], "random": [{}], "not_a_model": [{}]})
    assert reordered.model_names() == ["bpr", "random"]
    assert Config(models=("bpr", "random")).model_names() == ["bpr", "random"]  # request order


# --- aggregate ---------------------------------------------------------------------------


def _split_key(key: str) -> tuple[str, str, str, str, int]:
    fold, protocol, model, metric, k = key.split("|")
    return fold, protocol, model, metric, int(k)


def test_pairs_diff_is_mean_of_per_user_differences(all_run):
    per_user = load_per_user(all_run)
    pairs = read_csv(all_run / "pairs.csv")
    assert pairs
    seen = set()
    for row in pairs:
        assert row["model_a"] < row["model_b"]  # alphabetical
        a = per_user["|".join([row["fold"], row["protocol"], row["model_a"], "ndcg", "10"])]
        b = per_user["|".join([row["fold"], row["protocol"], row["model_b"], "ndcg", "10"])]
        assert a.shape == b.shape
        assert float(row["diff"]) == pytest.approx(float((a - b).mean()), rel=1e-5, abs=1e-9)
        assert float(row["ci_lo"]) <= float(row["diff"]) <= float(row["ci_hi"])
        seen.add((row["fold"], row["protocol"], row["model_a"], row["model_b"]))
    # every fold x protocol x unordered model pair appears exactly once
    assert len(seen) == len(pairs) == 2 * 3 * (5 * 4 // 2)


def test_per_user_arrays_share_one_length_per_fold(all_run):
    per_user = load_per_user(all_run)
    folds = {r["fold"]: int(r["n_eval_users"]) for r in read_csv(all_run / "folds.csv")}
    for key, vals in per_user.items():
        fold = _split_key(key)[0]
        assert vals.ndim == 1
        assert len(vals) == folds[fold], key
    lengths = {}
    for key, vals in per_user.items():
        lengths.setdefault(_split_key(key)[0], set()).add(len(vals))
    assert set(lengths) == {"test", "loo"}
    assert all(len(v) == 1 for v in lengths.values())


def test_summary_means_match_per_user_means(all_run):
    per_user = load_per_user(all_run)
    summary = read_csv(all_run / "summary.csv")
    checked = 0
    for row in summary:
        if row["ci_lo"] == "":  # catalogue metrics are not per-user and have no CI
            assert row["metric"] in {"coverage", "mean_popularity", "gini_exposure"}
            continue
        key = "|".join([row["fold"], row["protocol"], row["model"], row["metric"], row["k"]])
        vals = per_user[key]
        assert int(row["n_users"]) == len(vals)
        assert float(row["mean"]) == pytest.approx(float(vals.mean()), rel=1e-5, abs=1e-9)
        assert float(row["ci_lo"]) <= float(row["mean"]) <= float(row["ci_hi"])
        checked += 1
    assert checked == len(per_user)


# --- CLI ---------------------------------------------------------------------------------


def test_cli_synthetic_smoke_writes_report(tmp_path, capsys):
    argv = ["evaluate", "--synthetic", "--smoke", "--out", str(tmp_path)]
    assert cli.main([*argv, "--models", "random,popularity"]) == 0
    report = tmp_path / "REPORT.md"
    assert report.exists() and report.read_text(encoding="utf-8").strip()
    assert f"wrote {report}" in capsys.readouterr().out
    for name in (*DERIVED, "tuning.csv", "folds.csv", "timings.csv"):
        assert (tmp_path / name).exists(), name
    assert {r["model"] for r in read_csv(tmp_path / "summary.csv")} == {"random", "popularity"}


def test_cli_stage_defaults_follow_the_data_source(inter, tmp_path, monkeypatch, capsys):
    """Real data tunes only by default; synthetic runs everything. No file under data/ is read."""
    from recsys_ranking import data

    monkeypatch.setattr(data, "load_ml1m", lambda path: inter)
    monkeypatch.setattr(data, "positives", lambda x, *a, **kw: x)
    calls: list[dict] = []

    def fake_run(inter_, out, cfg=None, log=print, stage="all", force=False):
        calls.append({"out": Path(out), "stage": stage, "force": force, "inter": inter_})
        return Path(out)

    monkeypatch.setattr(evaluate, "run", fake_run)

    real = tmp_path / "real"
    assert cli.main(["evaluate", "--out", str(real)]) == 0
    assert calls[-1]["stage"] == "tune" and calls[-1]["inter"] is inter
    assert f"wrote {real / 'tuning.csv'}" in capsys.readouterr().out
    assert not (real / "REPORT.md").exists()  # no report before the test fold is scored

    from recsys_ranking import report

    monkeypatch.setattr(report, "write_report", lambda out: Path(out) / "REPORT.md")
    assert cli.main(["evaluate", "--out", str(real), "--stage", "score", "--force"]) == 0
    assert calls[-1]["stage"] == "score" and calls[-1]["force"] is True
    assert f"wrote {real / 'REPORT.md'}" in capsys.readouterr().out

    assert cli.main(["evaluate", "--synthetic", "--smoke", "--out", str(tmp_path / "syn")]) == 0
    assert calls[-1]["stage"] == "all" and calls[-1]["force"] is False
    assert calls[-1]["inter"] is not inter  # the synthetic generator, not the patched loader
