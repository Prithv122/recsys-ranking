"""Dashboard over the aggregate results CSVs (optional ``dashboard`` extra: Streamlit + Altair).

``build_figures`` and ``check`` need only pandas and Altair, so the headless check runs without a
Streamlit server. Streamlit is imported inside ``main`` for the same reason.
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd

from recsys_ranking.report import load_results

PROTOCOLS = ["full", "sampled_uniform", "sampled_popularity"]
CAPTION = (
    "Aggregate metrics on one dataset (MovieLens 1M), one global-time split, CPU-sized models. "
    "Intervals are paired-bootstrap 95% percentile intervals over evaluation users."
)


def _ndcg10(summary: pd.DataFrame, fold: str) -> pd.DataFrame:
    sel = summary[(summary["fold"] == fold) & (summary["k"] == 10)]
    sel = sel[sel["metric"].str.lower().isin(["ndcg", "ndcg@10"])]
    return sel[sel["protocol"].isin(PROTOCOLS)][["protocol", "model", "mean", "ci_lo", "ci_hi"]]


def _pick_fold(summary: pd.DataFrame, fold: str | None) -> str:
    folds = list(summary["fold"].drop_duplicates())
    if fold is not None:
        return fold
    return "test" if "test" in folds else str(folds[0])


def _ndcg_chart(nd: pd.DataFrame, fold: str) -> alt.Chart:
    order = list(
        nd[nd["protocol"] == "full"].sort_values("mean", ascending=False)["model"]
    ) or sorted(nd["model"].unique())
    base = alt.Chart(nd).encode(
        x=alt.X("model:N", sort=order, title="Model (best by full NDCG@10 first)"),
        xOffset=alt.XOffset("protocol:N", sort=PROTOCOLS),
        color=alt.Color("protocol:N", sort=PROTOCOLS, title="Protocol"),
        tooltip=["model", "protocol", "mean", "ci_lo", "ci_hi"],
    )
    bars = base.mark_rule(strokeWidth=2).encode(
        y=alt.Y("ci_lo:Q", title="NDCG@10 (95% CI)"), y2="ci_hi:Q"
    )
    dots = base.mark_point(filled=True, size=60).encode(y="mean:Q")
    return (bars + dots).properties(
        title=f"NDCG@10 by model and protocol ({fold} fold)", width=520, height=320
    )


def _slope_chart(nd: pd.DataFrame, ordering: pd.DataFrame, fold: str) -> alt.Chart:
    ranks = nd.copy()
    ranks["rank"] = ranks.groupby("protocol")["mean"].rank(ascending=False, method="min")
    sub = ""
    o = ordering[(ordering["fold"] == fold) & (ordering["protocol_a"] == "full")]
    taus = [
        f"{r.protocol_b}: tau {r.tau:.2f}"
        for r in o.itertuples(index=False)
        if r.protocol_b in PROTOCOLS
    ]
    if taus:
        sub = "Kendall tau vs full: " + "; ".join(taus)
    base = alt.Chart(ranks).encode(
        x=alt.X(
            "protocol:N",
            sort=PROTOCOLS,
            title="Protocol (uniform / popularity = 100 sampled negatives)",
            # Short labels so all three fit; Vega otherwise hides overlapping ones.
            axis=alt.Axis(
                labelAngle=0,
                labelOverlap=False,
                labelLimit=0,
                labelExpr="replace(datum.label, 'sampled_', '')",
            ),
        ),
        y=alt.Y("rank:Q", scale=alt.Scale(reverse=True, domainMin=1), title="Rank (1 = best)"),
        color=alt.Color("model:N", title="Model"),
    )
    lines = base.mark_line(strokeWidth=2).encode(detail="model:N")
    points = base.mark_point(filled=True, size=70).encode(
        tooltip=["model", "protocol", "rank", "mean"]
    )
    return (lines + points).properties(
        # Streamlit's theme colours titles but not subtitles; mid grey reads on light and dark.
        title=alt.TitleParams(
            f"Model rank by protocol ({fold} fold)", subtitle=sub, subtitleColor="#8a8f98"
        ),
        width=420,
        height=320,
    )


def _catalogue_chart(summary: pd.DataFrame, fold: str) -> alt.HConcatChart:
    cat = summary[(summary["fold"] == fold) & (summary["protocol"] == "full")]
    cat = cat[cat["k"] == 10]
    lowered = cat["metric"].str.lower()
    panels = []
    for key, title in (
        ("cover", "Catalogue coverage@10"),
        ("pop", "Mean popularity of recommended items@10"),
        ("gini", "Gini of exposure@10 (0 = equal)"),
    ):
        sub = cat[lowered.str.contains(key)][["model", "mean"]]
        panels.append(
            alt.Chart(sub)
            .mark_bar()
            .encode(
                x=alt.X("model:N", sort="-y", title=None),
                y=alt.Y("mean:Q", title=None),
                color=alt.Color("model:N", legend=None),
                tooltip=["model", "mean"],
            )
            .properties(title=title, width=190, height=260)
        )
    return alt.hconcat(*panels)


def _pairs_chart(pairs: pd.DataFrame, fold: str) -> alt.Chart:
    sel = pairs[(pairs["fold"] == fold) & (pairs["k"] == 10)]
    sel = sel[sel["metric"].str.lower().isin(["ndcg", "ndcg@10"])].copy()
    sel["pair"] = sel["model_a"] + " - " + sel["model_b"]
    sel["excludes_zero"] = (sel["ci_lo"] > 0) | (sel["ci_hi"] < 0)
    base = alt.Chart(sel).encode(
        y=alt.Y("pair:N", title=None),
        yOffset=alt.YOffset("protocol:N", sort=PROTOCOLS),
        color=alt.Color("protocol:N", sort=PROTOCOLS, title="Protocol"),
        tooltip=["pair", "protocol", "diff", "ci_lo", "ci_hi"],
    )
    bars = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("ci_lo:Q", title="NDCG@10 difference, first minus second (95% CI)"), x2="ci_hi:Q"
    )
    dots = base.mark_point(filled=True, size=50).encode(x="diff:Q")
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="z:Q")
    return (bars + dots + zero).properties(
        title=f"Paired differences between models ({fold} fold)", width=480
    )


def build_figures(
    summary: pd.DataFrame,
    ordering: pd.DataFrame,
    pairs: pd.DataFrame,
    fold: str | None = None,
) -> dict[str, alt.TopLevelMixin]:
    """Build every figure for one fold (``test`` when present)."""
    fold = _pick_fold(summary, fold)
    nd = _ndcg10(summary, fold)
    return {
        "ndcg_by_protocol": _ndcg_chart(nd, fold),
        "rank_slope": _slope_chart(nd, ordering, fold),
        "catalogue_metrics": _catalogue_chart(summary, fold),
        "pair_differences": _pairs_chart(pairs, fold),
    }


def check(results_dir: Path | str) -> list[str]:
    """Headless check: load the CSVs, build and validate every figure, return the names."""
    summary, ordering, pairs = load_results(results_dir)
    figures = build_figures(summary, ordering, pairs)
    for fig in figures.values():
        fig.to_dict()  # validates the Vega-Lite spec against the schema
    return list(figures)


def app_path() -> Path:
    """Path of this file, for ``streamlit run``."""
    return Path(__file__).resolve()


def main() -> None:
    """Streamlit app body. Reads ``RESULTS_DIR`` (default ``results/``)."""
    import streamlit as st

    results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
    st.set_page_config(page_title="Recommender ranking under sampling", layout="wide")
    st.title("Does the model ordering survive sampled evaluation?")
    try:
        summary, ordering, pairs = load_results(results_dir)
    except FileNotFoundError as e:
        st.error(f"Results not found in {results_dir}: {e}")
        return
    st.caption(CAPTION)

    folds = list(summary["fold"].drop_duplicates())
    fold = st.sidebar.selectbox("Fold", folds, index=folds.index(_pick_fold(summary, None)))
    figures = build_figures(summary, ordering, pairs, fold=fold)
    st.altair_chart(figures["ndcg_by_protocol"], use_container_width=False)
    st.altair_chart(figures["rank_slope"], use_container_width=False)
    st.altair_chart(figures["pair_differences"], use_container_width=False)
    st.altair_chart(figures["catalogue_metrics"], use_container_width=False)

    report = results_dir / "REPORT.md"
    if report.exists():
        with st.expander("REPORT.md", expanded=False):
            st.markdown(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
