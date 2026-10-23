"""Report wording branches, driven by the hand-written fixture in tests/data/results."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from recsys_ranking.report import HARMLESS_LINE, verdict, write_report

FIXTURE = Path(__file__).parent / "data" / "results"


def _pair(v: dict, a: str, b: str) -> dict:
    return next(p for p in v["pairwise"] if {p["a"], p["b"]} == {a, b})


def _strong(v: dict, proto: str, a: str, b: str) -> dict:
    return next(r for r in v["strong_reversals"][proto] if {r["a"], r["b"]} == {a, b})


@pytest.fixture
def results(tmp_path: Path) -> Path:
    dest = tmp_path / "results"
    shutil.copytree(FIXTURE, dest)
    return dest


def test_table_sorted_best_first_and_fold_sizes() -> None:
    v = verdict(FIXTURE)
    assert v["ordering_full"] == ["als", "itemknn", "popularity", "random"]
    assert [r["model"] for r in v["table"]] == v["ordering_full"]
    assert v["table"][0]["mean"] == pytest.approx(0.30)
    assert v["fold_sizes"] == {"val": 480, "test": 500, "loo": 600}


def test_clear_win() -> None:
    p = _pair(verdict(FIXTURE), "als", "popularity")
    assert p["outcome"] == "beats"
    assert (p["winner"], p["loser"]) == ("als", "popularity")


def test_no_clear_difference() -> None:
    p = _pair(verdict(FIXTURE), "als", "itemknn")
    assert p["outcome"] == "no_clear"
    assert p["winner"] is None


def test_swap_that_is_not_a_strong_reversal() -> None:
    v = verdict(FIXTURE)
    assert ("als", "itemknn") in v["tau"]["sampled_uniform"]["swapped_pairs"]
    r = _strong(v, "sampled_uniform", "als", "itemknn")
    assert r["swapped"] is True
    assert r["strong"] is False
    assert not any(x["strong"] for x in v["strong_reversals"]["sampled_uniform"])


def test_strong_reversal_and_flipped_pair_orientation() -> None:
    # pairs.csv stores (popularity, itemknn) for this protocol; the report must flip it.
    v = verdict(FIXTURE)
    r = _strong(v, "sampled_popularity", "itemknn", "popularity")
    assert (r["a"], r["b"]) == ("itemknn", "popularity")
    assert r["full_direction"] == 1
    assert r["sampled_direction"] == -1
    assert r["strong"] is True
    # als vs popularity is swapped as well, but the sampled CI includes 0.
    assert _strong(v, "sampled_popularity", "als", "popularity")["strong"] is False
    assert not v["sampling_harmless"]


def test_report_text_wording(results: Path) -> None:
    text = write_report(results).read_text(encoding="utf-8")
    assert "als beats popularity" in text
    assert "No clear difference between als and itemknn" in text
    assert "Strong reversal: itemknn and popularity." in text
    assert "als and itemknn: order swapped, not a strong reversal." in text
    assert HARMLESS_LINE not in text
    assert "| test | 500 |" in text
    assert "Leave-last-out" in text
    assert text.count("Strong reversal:") == 1


def test_loo_section() -> None:
    loo = verdict(FIXTURE)["loo"]
    assert loo["ordering"] == ["itemknn", "als", "popularity", "random"]
    assert loo["tau_vs_test_full"] == pytest.approx(2 / 3, abs=1e-6)  # one swapped pair of six
    assert loo["tau_vs_sampled"][0]["tau"] == 1.0


def test_tau_one_branch(results: Path) -> None:
    ordering = pd.read_csv(results / "ordering.csv")
    pairs = pd.read_csv(results / "pairs.csv")
    summary = pd.read_csv(results / "summary.csv")
    sel = (ordering["fold"] == "test") & (ordering["protocol_a"] == "full")
    ordering.loc[sel, ["tau", "ci_lo", "ci_hi"]] = 1.0
    ordering.loc[sel, "ordering_b"] = ordering.loc[sel, "ordering_a"]
    ordering.to_csv(results / "ordering.csv", index=False)
    pairs.to_csv(results / "pairs.csv", index=False)
    summary.to_csv(results / "summary.csv", index=False)

    v = verdict(results)
    assert v["sampling_harmless"] is True
    text = write_report(results).read_text(encoding="utf-8")
    assert HARMLESS_LINE in text
    assert "Sampling was harmless on this data (tau = 1 on both sampled protocols)." in text
    assert "Strong reversal:" not in text
    assert "order swapped" not in text


def test_harmless_requires_both_protocols(results: Path) -> None:
    ordering = pd.read_csv(results / "ordering.csv")
    sel = (ordering["fold"] == "test") & (ordering["protocol_b"] == "sampled_popularity")
    ordering[~sel].to_csv(results / "ordering.csv", index=False)
    ordering = pd.read_csv(results / "ordering.csv")
    ordering.loc[ordering["fold"] == "test", "tau"] = 1.0
    ordering.to_csv(results / "ordering.csv", index=False)
    v = verdict(results)
    assert v["sampling_harmless"] is False
    assert "sampled_popularity: not evaluated." in write_report(results).read_text("utf-8")


def test_missing_fold_raises(results: Path) -> None:
    with pytest.raises(ValueError):
        verdict(results, fold="nope")
