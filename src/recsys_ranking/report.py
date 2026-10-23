"""Turn ``summary.csv``, ``ordering.csv`` and ``pairs.csv`` into ``REPORT.md``, mechanically.

Every sentence is a template filled from the three files, using the wording rules in
``PREREGISTRATION.md``: "A beats B" only when the paired 95% CI excludes 0, otherwise "no clear
difference between A and B"; "reversal" only for a strong reversal (full-protocol CI excludes 0 one
way, sampled-protocol CI excludes 0 the other way). No interpretation is written here.

The headline metric is NDCG at k = 10 (``metric`` ``ndcg``). Catalogue metrics are matched by
substring on ``metric`` (``cover``, ``pop``, ``gini``) because they carry no CI.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import kendalltau

FULL = "full"
SAMPLED = ("sampled_uniform", "sampled_popularity")
HARMLESS_LINE = "Sampling was harmless on this data (tau = 1 on both sampled protocols)."
_EPS = 1e-9


def load_results(results_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read ``summary.csv``, ``ordering.csv`` and ``pairs.csv`` from ``results_dir``."""
    d = Path(results_dir)
    return (
        pd.read_csv(d / "summary.csv"),
        pd.read_csv(d / "ordering.csv"),
        pd.read_csv(d / "pairs.csv"),
    )


def _is_ndcg10(df: pd.DataFrame) -> pd.Series:
    return df["metric"].str.lower().isin(["ndcg", "ndcg@10"]) & (df["k"] == 10)


def ndcg10_table(summary: pd.DataFrame, fold: str, protocol: str) -> pd.DataFrame:
    """NDCG@10 rows for one fold and protocol, best first (ties by model name)."""
    sel = summary[(summary["fold"] == fold) & (summary["protocol"] == protocol)]
    sel = sel[_is_ndcg10(sel)]
    return sel.sort_values(["mean", "model"], ascending=[False, True]).reset_index(drop=True)


def _direction(lo: float, hi: float) -> int:
    """+1 when the CI is entirely above 0, -1 when entirely below, else 0."""
    if lo > 0:
        return 1
    if hi < 0:
        return -1
    return 0


def _pair_lookup(pairs: pd.DataFrame, fold: str, protocol: str) -> dict[tuple[str, str], dict]:
    """NDCG@10 pair rows for one fold/protocol, keyed by (model_a, model_b) in both orientations."""
    sel = pairs[(pairs["fold"] == fold) & (pairs["protocol"] == protocol)]
    sel = sel[_is_ndcg10(sel)]
    out: dict[tuple[str, str], dict] = {}
    for r in sel.itertuples(index=False):
        d, lo, hi = float(r.diff), float(r.ci_lo), float(r.ci_hi)
        out[(r.model_a, r.model_b)] = {"diff": d, "ci_lo": lo, "ci_hi": hi}
        out[(r.model_b, r.model_a)] = {"diff": -d, "ci_lo": -hi, "ci_hi": -lo}
    return out


def _split(ordering: str) -> list[str]:
    return [m for m in str(ordering).split(">") if m]


def _swapped(order_a: list[str], order_b: list[str]) -> list[tuple[str, str]]:
    """Pairs (x, y) with x above y in ``order_a`` and below y in ``order_b``."""
    pos_b = {m: i for i, m in enumerate(order_b)}
    return [
        (x, y)
        for x, y in combinations(order_a, 2)
        if x in pos_b and y in pos_b and pos_b[x] > pos_b[y]
    ]


def _pair_outcome(a: str, b: str, row: dict | None) -> dict[str, Any]:
    """Wording-rule outcome for ``a`` vs ``b`` (a is the one ranked higher by mean)."""
    if row is None:
        return {"a": a, "b": b, "outcome": "missing", "direction": 0}
    direction = _direction(row["ci_lo"], row["ci_hi"])
    outcome = "no_clear" if direction == 0 else "beats"
    winner = a if direction > 0 else b if direction < 0 else None
    loser = b if direction > 0 else a if direction < 0 else None
    return {
        "a": a,
        "b": b,
        **row,
        "direction": direction,
        "outcome": outcome,
        "winner": winner,
        "loser": loser,
    }


def verdict(results_dir: Path | str, fold: str = "test") -> dict[str, Any]:
    """Structured facts behind the report, for ``fold`` (the headline fold, ``test``)."""
    summary, ordering, pairs = load_results(results_dir)

    fold_sizes: dict[str, int] = {}
    for f in summary["fold"].drop_duplicates():
        full = summary[(summary["fold"] == f) & (summary["protocol"] == FULL)]
        full = full[_is_ndcg10(full)]
        if len(full):
            fold_sizes[str(f)] = int(full["n_users"].max())

    table = ndcg10_table(summary, fold, FULL)
    if table.empty:
        raise ValueError(f"no full-protocol NDCG@10 rows for fold {fold!r} in summary.csv")
    models = list(table["model"])

    full_pairs = _pair_lookup(pairs, fold, FULL)
    pairwise = [_pair_outcome(a, b, full_pairs.get((a, b))) for a, b in combinations(models, 2)]

    taus: dict[str, dict[str, Any]] = {}
    strong: dict[str, list[dict[str, Any]]] = {}
    for proto in SAMPLED:
        o = ordering[
            (ordering["fold"] == fold)
            & (ordering["protocol_a"] == FULL)
            & (ordering["protocol_b"] == proto)
        ]
        o = o[_is_ndcg10(o)]
        if o.empty:
            continue
        r = o.iloc[0]
        order_a, order_b = _split(r["ordering_a"]), _split(r["ordering_b"])
        swapped = _swapped(order_a, order_b)
        taus[proto] = {
            "tau": float(r["tau"]),
            "ci_lo": float(r["ci_lo"]),
            "ci_hi": float(r["ci_hi"]),
            "ordering_full": order_a,
            "ordering_sampled": order_b,
            "swapped_pairs": swapped,
            "is_one": float(r["tau"]) >= 1 - _EPS,
        }
        samp_pairs = _pair_lookup(pairs, fold, proto)
        rows = []
        for a, b in combinations(order_a, 2):  # a is above b in the full ordering
            f_row, s_row = full_pairs.get((a, b)), samp_pairs.get((a, b))
            if f_row is None or s_row is None:
                continue
            fd = _direction(f_row["ci_lo"], f_row["ci_hi"])
            sd = _direction(s_row["ci_lo"], s_row["ci_hi"])
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "full": f_row,
                    "sampled": s_row,
                    "full_direction": fd,
                    "sampled_direction": sd,
                    "swapped": (a, b) in swapped,
                    "strong": fd * sd == -1,
                }
            )
        strong[proto] = rows

    harmless = all(p in taus and taus[p]["is_one"] for p in SAMPLED)

    catalogue: dict[str, dict[str, float]] = {}
    cat = summary[(summary["fold"] == fold) & (summary["protocol"] == FULL) & (summary["k"] == 10)]
    for key, label in (("cover", "coverage"), ("pop", "mean_popularity"), ("gini", "gini")):
        sub = cat[cat["metric"].str.lower().str.contains(key)]
        if len(sub):
            catalogue[label] = {str(r.model): float(r.mean) for r in sub.itertuples(index=False)}

    return {
        "fold": fold,
        "fold_sizes": fold_sizes,
        "table": [
            {
                "model": r.model,
                "mean": float(r.mean),
                "ci_lo": float(r.ci_lo),
                "ci_hi": float(r.ci_hi),
            }
            for r in table.itertuples(index=False)
        ],
        "ordering_full": models,
        "pairwise": pairwise,
        "tau": taus,
        "strong_reversals": strong,
        "sampling_harmless": harmless,
        "catalogue": catalogue,
        "loo": _loo(summary, ordering, models),
    }


def _loo(summary: pd.DataFrame, ordering: pd.DataFrame, test_models: list[str]) -> dict | None:
    table = ndcg10_table(summary, "loo", FULL)
    if table.empty:
        return None
    loo_models = list(table["model"])
    common = [m for m in test_models if m in loo_models]
    tau_vs_test = None
    if len(common) >= 2:
        loo_rank = [loo_models.index(m) for m in common]
        test_rank = [test_models.index(m) for m in common]
        tau_vs_test = float(kendalltau(loo_rank, test_rank).statistic)  # tau-b
    o = ordering[(ordering["fold"] == "loo") & (ordering["protocol_a"] == FULL)]
    o = o[_is_ndcg10(o)]
    return {
        "table": [
            {
                "model": r.model,
                "mean": float(r.mean),
                "ci_lo": float(r.ci_lo),
                "ci_hi": float(r.ci_hi),
            }
            for r in table.itertuples(index=False)
        ],
        "ordering": loo_models,
        "tau_vs_test_full": tau_vs_test,
        "tau_vs_sampled": [
            {
                "protocol": r.protocol_b,
                "tau": float(r.tau),
                "ci_lo": float(r.ci_lo),
                "ci_hi": float(r.ci_hi),
            }
            for r in o.itertuples(index=False)
        ],
    }


def _f(x: float) -> str:
    return f"{x:.4f}"


def _ci(lo: float, hi: float) -> str:
    return f"[{_f(lo)}, {_f(hi)}]"


def _signed(x: float) -> str:
    return f"{x:+.4f}"


def _table(rows: list[dict]) -> list[str]:
    out = ["| Rank | Model | NDCG@10 | 95% CI |", "|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        out.append(f"| {i} | {r['model']} | {_f(r['mean'])} | {_ci(r['ci_lo'], r['ci_hi'])} |")
    return out


def render(v: dict[str, Any]) -> str:
    """Render the verdict dict as markdown."""
    fold = v["fold"]
    L: list[str] = ["# Results report", ""]
    L += [
        f"Headline fold: `{fold}`. Metric: NDCG@10. Generated from `summary.csv`, "
        "`ordering.csv` and `pairs.csv`; one dataset, one split, CPU-sized models.",
        "",
    ]

    if v["fold_sizes"]:
        L += ["## Fold sizes", "", "| Fold | Evaluation users |", "|---|---|"]
        L += [f"| {f} | {n} |" for f, n in v["fold_sizes"].items()]
        L.append("")

    L += [f"## Full-catalogue NDCG@10 ({fold} fold)", "", *_table(v["table"]), ""]

    L += ["## Pairwise comparisons (full protocol)", ""]
    for p in v["pairwise"]:
        if p["outcome"] == "missing":
            L.append(f"- {p['a']} vs {p['b']}: pair not in `pairs.csv`.")
            continue
        stat = f"difference {_signed(p['diff'])}, 95% CI {_ci(p['ci_lo'], p['ci_hi'])}"
        if p["outcome"] == "beats":
            L.append(f"- {p['winner']} beats {p['loser']} ({stat}).")
        else:
            L.append(f"- No clear difference between {p['a']} and {p['b']} ({stat}).")
    L.append("")

    L += ["## Full vs sampled orderings", ""]
    if not v["tau"]:
        L += ["No ordering rows for this fold in `ordering.csv`.", ""]
    else:
        L += ["| Sampled protocol | Kendall tau-b | 95% CI |", "|---|---|---|"]
        for proto, t in v["tau"].items():
            L.append(f"| {proto} | {_f(t['tau'])} | {_ci(t['ci_lo'], t['ci_hi'])} |")
        L.append("")
        for proto, t in v["tau"].items():
            L.append(f"- full: {' > '.join(t['ordering_full'])}")
            L.append(f"- {proto}: {' > '.join(t['ordering_sampled'])}")
        L.append("")

        L += ["## Strong-reversal check", ""]
        L += [
            "A pair is a strong reversal when the full-protocol difference CI excludes 0 in one "
            "direction and the sampled-protocol difference CI excludes 0 in the other. "
            "Differences are first minus second model, the first being ranked higher by full "
            "NDCG@10.",
            "",
        ]
        for proto, rows in v["strong_reversals"].items():
            L += [f"### full vs {proto}", ""]
            L += [
                "| Pair | Full diff (CI) | Sampled diff (CI) | Order swapped | Strong reversal |",
                "|---|---|---|---|---|",
            ]
            for r in rows:
                fr, sr = r["full"], r["sampled"]
                L.append(
                    f"| {r['a']} vs {r['b']} "
                    f"| {_signed(fr['diff'])} {_ci(fr['ci_lo'], fr['ci_hi'])} "
                    f"| {_signed(sr['diff'])} {_ci(sr['ci_lo'], sr['ci_hi'])} "
                    f"| {'yes' if r['swapped'] else 'no'} "
                    f"| {'yes' if r['strong'] else 'no'} |"
                )
            L.append("")

        L += ["## Result", ""]
        if v["sampling_harmless"]:
            L += [HARMLESS_LINE, ""]
        else:
            for proto in SAMPLED:
                t = v["tau"].get(proto)
                if t is None:
                    L.append(f"- {proto}: not evaluated.")
                    continue
                if t["is_one"]:
                    L.append(f"- {proto}: tau = 1, no swapped pairs.")
                    continue
                L.append(
                    f"- {proto}: tau = {_f(t['tau'])}, "
                    f"swapped pairs: {_swap_text(t['swapped_pairs'])}."
                )
                strongs = [r for r in v["strong_reversals"].get(proto, []) if r["strong"]]
                for r in strongs:
                    L.append(f"  - Strong reversal: {r['a']} and {r['b']}.")
                for r in v["strong_reversals"].get(proto, []):
                    if r["swapped"] and not r["strong"]:
                        L.append(
                            f"  - {r['a']} and {r['b']}: order swapped, not a strong reversal."
                        )
            L.append("")

    if v["catalogue"]:
        models = v["ordering_full"]
        names = {
            "coverage": "Coverage@10",
            "mean_popularity": "Mean popularity@10",
            "gini": "Gini of exposure@10",
        }
        L += ["## Catalogue metrics (full protocol, k = 10, descriptive only)", ""]
        L += [
            "| Model | " + " | ".join(names[k] for k in v["catalogue"]) + " |",
            "|---|" + "---|" * len(v["catalogue"]),
        ]
        for m in models:
            cells = [f"{c[m]:.4f}" if m in c else "" for c in v["catalogue"].values()]
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")

    loo = v["loo"]
    if loo:
        L += ["## Leave-last-out ablation (`loo` fold)", ""]
        L += [*_table(loo["table"]), ""]
        if loo["tau_vs_test_full"] is not None:
            L.append(
                f"- Kendall tau-b between the loo full ordering and the {fold} full ordering: "
                f"{_f(loo['tau_vs_test_full'])} (point estimate, no CI)."
            )
        for t in loo["tau_vs_sampled"]:
            L.append(
                f"- loo full vs {t['protocol']}: tau = {_f(t['tau'])}, "
                f"95% CI {_ci(t['ci_lo'], t['ci_hi'])}."
            )
        L.append("")

    return "\n".join(L).rstrip() + "\n"


def _swap_text(swapped: list[tuple[str, str]]) -> str:
    return ", ".join(f"{a}/{b}" for a, b in swapped) if swapped else "none"


def write_report(results_dir: Path | str) -> Path:
    """Write ``REPORT.md`` next to the CSVs and return its path."""
    d = Path(results_dir)
    out = d / "REPORT.md"
    out.write_text(render(verdict(d)), encoding="utf-8")
    return out
