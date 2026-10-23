"""Fold x model x protocol runner.

`run` fits and scores, saving per-user metrics to ``results/raw/`` (git-ignored: they are a
MovieLens derivative). `aggregate` turns those into the committed aggregate CSVs, so the
protocol comparison can be redone without refitting.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from . import metrics, protocols, stats
from .models import REGISTRY, UNAVAILABLE
from .split import global_time_folds, leave_last_out, loo_future_share
from .types import Fold, Interactions

PROTOCOLS = ("full", "sampled_uniform", "sampled_popularity")
FULL_METRICS = ("recall", "ndcg", "map", "mrr")
SAMPLED_METRICS = ("hr", "ndcg")
PRIMARY = ("ndcg", 10)

# Grids exactly as registered in PREREGISTRATION.md; order matters for tie-breaking.
GRIDS: dict[str, list[dict]] = {
    "random": [{}],
    "popularity": [{}],
    "itemknn": [{"n_neighbors": n, "shrink": s} for n in (50, 200) for s in (0, 10)],
    "als": [
        {"factors": f, "reg": r, "alpha": a, "iterations": 15}
        for f in (32, 64, 128)
        for r in (0.01, 0.1)
        for a in (1, 10)
    ],
    "bpr": [
        {"factors": f, "lr": lr, "reg": r, "epochs": 30}
        for f in (32, 64)
        for lr in (0.01, 0.05)
        for r in (1e-4, 1e-3)
    ],
    "twotower": [{"dim": 64, "lr": lr, "epochs": 20, "batch_size": 1024} for lr in (1e-3, 3e-3)],
}

SMOKE_GRIDS: dict[str, list[dict]] = {
    "random": [{}],
    "popularity": [{}],
    "itemknn": [{"n_neighbors": 20, "shrink": 0}],
    "als": [{"factors": 8, "reg": 0.1, "alpha": 10, "iterations": 3}],
    "bpr": [{"factors": 8, "lr": 0.05, "reg": 1e-4, "epochs": 3}],
    "twotower": [{"dim": 16, "lr": 3e-3, "epochs": 2, "batch_size": 256}],
}


@dataclass
class Config:
    ks: tuple[int, ...] = (5, 10, 20)
    n_neg: int = 100
    neg_seed: int = 20261023
    boot_seed: int = 7
    n_boot: int = 2000
    grids: dict[str, list[dict]] = field(default_factory=lambda: GRIDS)
    models: tuple[str, ...] | None = None  # None = every registered model

    def model_names(self) -> list[str]:
        """Models to run. An explicit request for an unknown or unavailable model is an error."""
        if self.models is None:
            return [m for m in self.grids if m in REGISTRY]
        for m in self.models:
            if m in UNAVAILABLE:
                raise ValueError(f"model {m!r} is unavailable here ({UNAVAILABLE[m]})")
            if m not in REGISTRY or m not in self.grids:
                raise ValueError(f"unknown model {m!r}; known: {sorted(REGISTRY)}")
        return list(self.models)


def _key(fold: str, protocol: str, model: str, metric: str, k: int) -> str:
    return f"{fold}|{protocol}|{model}|{metric}|{k}"


def _parse(key: str) -> tuple[str, str, str, str, int]:
    fold, protocol, model, metric, k = key.split("|")
    return fold, protocol, model, metric, int(k)


def _fit(name: str, params: dict, fold: Fold):
    t0 = time.perf_counter()
    model = REGISTRY[name](**params).fit(fold.fit)
    return model, time.perf_counter() - t0


def tune(fold: Fold, cfg: Config, log: Callable[[str], None]) -> list[dict]:
    """Score every grid point on ``fold`` (the val fold) by full NDCG@10."""
    users = fold.eval_users()
    truth = [fold.truth.indices[fold.truth.indptr[u] : fold.truth.indptr[u + 1]] for u in users]
    rows = []
    for name in cfg.model_names():
        for i, params in enumerate(cfg.grids[name]):
            model, secs = _fit(name, params, fold)
            recs = protocols.full_topk(model, fold, users, PRIMARY[1])
            score = float(metrics.ndcg_at_k(recs, truth, PRIMARY[1]).mean())
            rows.append(
                {"model": name, "config": i, "params": params, "val_ndcg10": score, "fit_s": secs}
            )
            log(f"tune {name}[{i}] {params} ndcg@10={score:.4f} ({secs:.1f}s)")
    return rows


def read_tuning(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [
            {**r, "params": json.loads(r["params"]), "val_ndcg10": float(r["val_ndcg10"])}
            for r in csv.DictReader(fh)
        ]


def best_params(tuning: list[dict]) -> dict[str, dict]:
    """Highest val NDCG@10 per model; ties go to the earlier grid point."""
    best: dict[str, dict] = {}
    for row in tuning:  # rows arrive in grid order
        cur = best.get(row["model"])
        if cur is None or row["val_ndcg10"] > cur["val_ndcg10"]:
            best[row["model"]] = row
    return {m: r["params"] for m, r in best.items()}


def score_fold(
    fold: Fold, chosen: dict[str, dict], cfg: Config, log: Callable[[str], None]
) -> tuple[dict[str, np.ndarray], list[dict], list[dict]]:
    """Per-user metrics for every model and protocol on one fold."""
    users = fold.eval_users()
    truth = [fold.truth.indices[fold.truth.indptr[u] : fold.truth.indptr[u + 1]] for u in users]
    item_pop = np.asarray(fold.fit.sum(axis=0)).ravel()
    sampled = {
        "sampled_uniform": protocols.sample_negatives(
            fold, cfg.n_neg, scheme="uniform", seed=cfg.neg_seed
        ),
        "sampled_popularity": protocols.sample_negatives(
            fold, cfg.n_neg, scheme="popularity", seed=cfg.neg_seed
        ),
    }
    per_user: dict[str, np.ndarray] = {}
    catalogue: list[dict] = []
    timings: list[dict] = []
    kmax = max(cfg.ks)
    for name, params in chosen.items():
        model, secs = _fit(name, params, fold)
        timings.append({"fold": fold.name, "model": name, "fit_s": round(secs, 2)})
        recs = protocols.full_topk(model, fold, users, kmax)
        for k in cfg.ks:
            for metric in FULL_METRICS:
                fn = getattr(metrics, f"{metric}_at_k")
                per_user[_key(fold.name, "full", name, metric, k)] = fn(recs, truth, k)
        top10 = recs[:, :10]
        catalogue += [
            {"fold": fold.name, "model": name, "metric": "coverage", "k": 10,
             "mean": metrics.coverage(top10, fold.n_items)},
            {"fold": fold.name, "model": name, "metric": "mean_popularity", "k": 10,
             "mean": metrics.mean_popularity(top10, item_pop)},
            {"fold": fold.name, "model": name, "metric": "gini_exposure", "k": 10,
             "mean": metrics.gini_exposure(top10, fold.n_items)},
        ]  # fmt: skip
        for proto, sset in sampled.items():
            ranks = protocols.sampled_ranks(model, sset)
            for k in cfg.ks:
                um = protocols.sampled_user_metrics(sset, ranks, k)
                for metric in SAMPLED_METRICS:
                    per_user[_key(fold.name, proto, name, metric, k)] = um[metric]
        ndcg = per_user[_key(fold.name, "full", name, *PRIMARY)].mean()
        log(f"{fold.name} {name} full ndcg@10={ndcg:.4f} ({secs:.1f}s fit)")
    return per_user, catalogue, timings


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({c: _fmt(row.get(c, "")) for c in columns})


def _fmt(v: object) -> object:
    return f"{v:.6g}" if isinstance(v, float) else v


DROP_KEYS = ("cold_user_rows", "cold_item_rows", "seen_pair_rows")


def fold_row(fold: Fold, inter: Interactions | None = None) -> dict:
    row = {
        "fold": fold.name,
        "n_users": fold.n_users,
        "n_items": fold.n_items,
        "n_fit": int(fold.fit.nnz),
        "n_truth": int(fold.truth.nnz),
        "n_eval_users": len(fold.eval_users()),
        **{k: fold.dropped.get(k, 0) for k in DROP_KEYS},
    }
    if fold.name == "loo" and inter is not None:
        row["future_share"] = loo_future_share(inter)
    return row


FOLD_COLUMNS = [
    "fold", "n_users", "n_items", "n_fit", "n_truth", "n_eval_users",
    "cold_user_rows", "cold_item_rows", "seen_pair_rows", "future_share",
]  # fmt: skip


def run(
    inter: Interactions,
    out_dir: Path,
    cfg: Config | None = None,
    log: Callable[[str], None] = print,
    stage: str = "all",
    force: bool = False,
) -> Path:
    """Tune on val, score test and loo, save per-user metrics, then aggregate.

    ``stage="tune"`` stops after writing ``tuning.csv``; ``stage="score"`` reuses that file, so
    the test fold is touched only once the harness has been reviewed. Scoring merges into
    earlier per-user results by model and refuses to re-score a model already there (the test
    fold is scored once per model) unless ``force`` is set.
    """
    cfg = cfg or Config()
    out_dir = Path(out_dir)
    val, test = global_time_folds(inter)
    loo = leave_last_out(inter)
    _write_csv(out_dir / "folds.csv", [fold_row(f, inter) for f in (val, test, loo)], FOLD_COLUMNS)

    if stage == "score":
        tuning = read_tuning(out_dir / "tuning.csv")
    else:
        tuning = tune(val, cfg, log)
        path = out_dir / "tuning.csv"
        if path.exists():  # keep rows for models tuned in an earlier call
            names = set(cfg.model_names())
            tuning = [r for r in read_tuning(path) if r["model"] not in names] + tuning
        _write_csv(
            out_dir / "tuning.csv",
            # repr keeps every digit, so selection after a round-trip matches selection in memory
            [
                {
                    **r,
                    "params": json.dumps(r["params"], sort_keys=True),
                    "val_ndcg10": repr(r["val_ndcg10"]),
                }
                for r in tuning
            ],
            ["model", "config", "params", "val_ndcg10", "fit_s"],
        )
        if stage == "tune":
            return out_dir
    chosen = best_params(tuning)
    missing = set(cfg.model_names()) - set(chosen)
    if missing:
        raise ValueError(f"no tuning rows for {sorted(missing)}")
    chosen = {m: chosen[m] for m in cfg.model_names()}

    raw = out_dir / "raw"
    per_user, catalogue, timings = _load_scored(out_dir)
    done = sorted({_parse(k)[2] for k in per_user} & set(chosen))
    if done and not force:
        raise ValueError(f"already scored on the test fold: {done} (pass force to re-score)")
    per_user = {k: v for k, v in per_user.items() if _parse(k)[2] not in chosen}
    catalogue = [r for r in catalogue if r["model"] not in chosen]
    timings = [r for r in timings if r["model"] not in chosen]
    for fold in (test, loo):
        pu, cat, tim = score_fold(fold, chosen, cfg, log)
        per_user |= pu
        catalogue += cat
        timings += tim
    raw.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw / "per_user.npz", **per_user)
    _write_csv(raw / "catalogue.csv", catalogue, ["fold", "model", "metric", "k", "mean"])
    _write_csv(out_dir / "timings.csv", timings, ["fold", "model", "fit_s"])
    aggregate(out_dir, cfg)
    return out_dir


def _load_scored(out_dir: Path) -> tuple[dict[str, np.ndarray], list[dict], list[dict]]:
    """Per-user metrics, catalogue rows and timings from an earlier scoring call, if any."""
    raw = out_dir / "raw"
    if not (raw / "per_user.npz").exists():
        return {}, [], []
    with np.load(raw / "per_user.npz") as z:
        per_user = {k: z[k] for k in z.files}
    with (raw / "catalogue.csv").open(encoding="utf-8") as fh:
        catalogue = [{**r, "k": int(r["k"]), "mean": float(r["mean"])} for r in csv.DictReader(fh)]
    timings = []
    if (out_dir / "timings.csv").exists():
        with (out_dir / "timings.csv").open(encoding="utf-8") as fh:
            timings = [{**r, "fit_s": float(r["fit_s"])} for r in csv.DictReader(fh)]
    return per_user, catalogue, timings


def aggregate(out_dir: Path, cfg: Config | None = None) -> None:
    """Per-user metrics -> summary.csv, pairs.csv, ordering.csv (all aggregate)."""
    cfg = cfg or Config()
    out_dir = Path(out_dir)
    with np.load(out_dir / "raw" / "per_user.npz") as z:
        per_user = {k: z[k] for k in z.files}

    summary = []
    for key, vals in sorted(per_user.items()):
        fold, proto, model, metric, k = _parse(key)
        mean, lo, hi = stats.paired_bootstrap(vals, np.zeros_like(vals), cfg.n_boot, cfg.boot_seed)
        summary.append(
            {"fold": fold, "protocol": proto, "model": model, "metric": metric, "k": k,
             "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_users": len(vals)}
        )  # fmt: skip
    with (out_dir / "raw" / "catalogue.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            summary.append({**row, "protocol": "full", "mean": float(row["mean"])})
    _write_csv(
        out_dir / "summary.csv",
        summary,
        ["fold", "protocol", "model", "metric", "k", "mean", "ci_lo", "ci_hi", "n_users"],
    )

    folds = sorted({_parse(k)[0] for k in per_user})
    pairs, ordering = [], []
    metric, k = PRIMARY
    for fold in folds:
        by_proto = {
            proto: {
                _parse(key)[2]: v
                for key, v in per_user.items()
                if _parse(key)[:2] == (fold, proto) and _parse(key)[3:] == (metric, k)
            }
            for proto in PROTOCOLS
        }
        for proto, series in by_proto.items():
            for a, b in combinations(sorted(series), 2):
                d, lo, hi = stats.paired_bootstrap(series[a], series[b], cfg.n_boot, cfg.boot_seed)
                pairs.append(
                    {"fold": fold, "protocol": proto, "metric": metric, "k": k,
                     "model_a": a, "model_b": b, "diff": d, "ci_lo": lo, "ci_hi": hi}
                )  # fmt: skip
        full = by_proto["full"]
        for proto in PROTOCOLS[1:]:
            tau, lo, hi = stats.ordering_tau_ci(full, by_proto[proto], cfg.n_boot, cfg.boot_seed)
            ordering.append(
                {"fold": fold, "protocol_a": "full", "protocol_b": proto, "metric": metric, "k": k,
                 "tau": tau, "ci_lo": lo, "ci_hi": hi,
                 "ordering_a": _order(full), "ordering_b": _order(by_proto[proto])}
            )  # fmt: skip
    if {"test", "loo"} <= set(folds):
        # Different users on each fold, so no joint bootstrap: point tau only.
        t = _means(per_user, "test")
        lo_ = _means(per_user, "loo")
        common = sorted(set(t) & set(lo_))
        tau = stats.kendall_tau(
            np.array([t[m] for m in common]), np.array([lo_[m] for m in common])
        )
        ordering.append(
            {"fold": "test_vs_loo", "protocol_a": "full", "protocol_b": "full", "metric": metric,
             "k": k, "tau": tau, "ordering_a": _order_means(t), "ordering_b": _order_means(lo_)}
        )  # fmt: skip
    _write_csv(
        out_dir / "pairs.csv",
        pairs,
        ["fold", "protocol", "metric", "k", "model_a", "model_b", "diff", "ci_lo", "ci_hi"],
    )
    _write_csv(
        out_dir / "ordering.csv",
        ordering,
        ["fold", "protocol_a", "protocol_b", "metric", "k", "tau", "ci_lo", "ci_hi",
         "ordering_a", "ordering_b"],
    )  # fmt: skip


def _means(per_user: dict[str, np.ndarray], fold: str) -> dict[str, float]:
    return {
        _parse(key)[2]: float(v.mean())
        for key, v in per_user.items()
        if _parse(key)[:2] == (fold, "full") and _parse(key)[3:] == PRIMARY
    }


def _order(series: dict[str, np.ndarray]) -> str:
    return _order_means({m: float(v.mean()) for m, v in series.items()})


def _order_means(means: dict[str, float]) -> str:
    return ">".join(stats.ordering(means))
