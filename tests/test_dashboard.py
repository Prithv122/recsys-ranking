"""Dashboard figures build and validate headless from the fixture results."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("altair")

from recsys_ranking import dashboard
from recsys_ranking.report import load_results

FIXTURE = Path(__file__).parent / "data" / "results"
NAMES = ["ndcg_by_protocol", "rank_slope", "catalogue_metrics", "pair_differences"]


def _layer_marks(spec: dict) -> list[str]:
    return [layer["mark"]["type"] for layer in spec["layer"]]


def test_check_returns_figure_names() -> None:
    assert dashboard.check(FIXTURE) == NAMES


def test_ndcg_chart_has_ci_bars_and_protocol_colour() -> None:
    figs = dashboard.build_figures(*load_results(FIXTURE))
    spec = figs["ndcg_by_protocol"].to_dict()
    assert _layer_marks(spec) == ["rule", "point"]
    rule = spec["layer"][0]["encoding"]
    assert rule["y"]["field"] == "ci_lo"
    assert rule["y2"]["field"] == "ci_hi"
    assert rule["color"]["field"] == "protocol"
    assert spec["layer"][1]["encoding"]["y"]["field"] == "mean"
    # test fold only: 4 models x 3 protocols, NDCG@10 only (no k=5, recall or hr rows)
    rows = next(iter(spec["datasets"].values()))
    assert len(rows) == 12
    assert {r["protocol"] for r in rows} == set(dashboard.PROTOCOLS)


def test_slope_chart_ranks_follow_protocol_means() -> None:
    figs = dashboard.build_figures(*load_results(FIXTURE))
    spec = figs["rank_slope"].to_dict()
    assert _layer_marks(spec) == ["line", "point"]
    assert spec["layer"][0]["encoding"]["x"]["field"] == "protocol"
    assert spec["layer"][0]["encoding"]["y"]["field"] == "rank"
    rows = next(iter(spec["datasets"].values()))
    rank = {(r["protocol"], r["model"]): r["rank"] for r in rows}
    assert rank[("full", "als")] == 1
    assert rank[("sampled_uniform", "itemknn")] == 1
    assert rank[("sampled_popularity", "popularity")] == 1
    assert "tau" in spec["title"]["subtitle"]


def test_catalogue_chart_has_three_panels() -> None:
    figs = dashboard.build_figures(*load_results(FIXTURE))
    spec = figs["catalogue_metrics"].to_dict()
    assert len(spec["hconcat"]) == 3
    assert all(p["mark"]["type"] == "bar" for p in spec["hconcat"])
    assert [len(d) for d in (spec["datasets"][k] for k in spec["datasets"])] == [4, 4, 4]


def test_pair_chart_marks_and_zero_line() -> None:
    figs = dashboard.build_figures(*load_results(FIXTURE))
    spec = figs["pair_differences"].to_dict()
    assert _layer_marks(spec) == ["rule", "point", "rule"]


def test_app_path_points_at_this_module() -> None:
    p = dashboard.app_path()
    assert p.name == "dashboard.py"
    assert p.exists()


def test_streamlit_not_imported_at_module_top() -> None:
    src = dashboard.app_path().read_text(encoding="utf-8").split("def main")[0]
    assert "import streamlit" not in src
