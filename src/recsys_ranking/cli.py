"""recsys-ranking command line: download | evaluate | compare-protocols | dashboard."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_DATA = Path("data")
DEFAULT_RESULTS = Path("results")
SYNTHETIC_RESULTS = Path("results/synthetic")


def _cmd_download(args: argparse.Namespace) -> int:
    from .data import download

    folder = download(args.data_dir)
    print(f"MovieLens 1M ready in {folder}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from . import evaluate, report

    cfg = evaluate.Config()
    if args.models:
        cfg.models = tuple(args.models.split(","))
    if args.synthetic:
        from .synth import make_synthetic

        n_users, n_items = (300, 200) if args.smoke else (3000, 1500)
        inter, _ = make_synthetic(n_users=n_users, n_items=n_items, seed=args.seed)
        out = args.out or SYNTHETIC_RESULTS
    else:
        from .data import load_ml1m, positives

        inter = positives(load_ml1m(args.data_dir / "ml-1m"))
        out = args.out or DEFAULT_RESULTS
    if args.smoke:
        cfg.grids = evaluate.SMOKE_GRIDS
        cfg.n_boot = 200
        cfg.n_neg = 50  # the smoke catalogue has ~120 active items
    # Real data defaults to tuning only: scoring the test fold is a deliberate second step.
    stage = args.stage or ("all" if args.synthetic else "tune")
    evaluate.run(inter, out, cfg, stage=stage, force=args.force)
    if stage == "tune":
        print(f"wrote {out / 'tuning.csv'}")
        return 0
    print(f"wrote {report.write_report(out)}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from . import evaluate, report

    out = args.out or DEFAULT_RESULTS
    evaluate.aggregate(out)
    print(f"wrote {report.write_report(out)}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from . import dashboard

    results = args.out or DEFAULT_RESULTS
    if args.check:
        names = dashboard.check(results)
        print("figures OK: " + ", ".join(names))
        return 0
    env = {**os.environ, "RESULTS_DIR": str(results)}
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard.app_path())]
    return subprocess.call(cmd, env=env)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recsys-ranking", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="fetch and verify MovieLens 1M into data/")
    d.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    d.set_defaults(func=_cmd_download)

    e = sub.add_parser("evaluate", help="tune on val, score test and loo, write results/")
    e.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    e.add_argument("--out", type=Path, default=None)
    e.add_argument("--synthetic", action="store_true", help="use the synthetic generator")
    e.add_argument("--smoke", action="store_true", help="tiny grids and data, seconds to run")
    e.add_argument("--models", default=None, help="comma-separated subset, e.g. popularity,als")
    e.add_argument("--seed", type=int, default=0, help="synthetic data seed")
    e.add_argument(
        "--stage",
        choices=("all", "tune", "score"),
        default=None,
        help="tune: val fold only (default on real data); score: reuse tuning.csv and score "
        "test and loo; all: both (default with --synthetic)",
    )
    e.add_argument("--force", action="store_true", help="allow re-scoring a scored model")
    e.set_defaults(func=_cmd_evaluate)

    c = sub.add_parser("compare-protocols", help="re-aggregate saved per-user metrics")
    c.add_argument("--out", type=Path, default=None)
    c.set_defaults(func=_cmd_compare)

    b = sub.add_parser("dashboard", help="Streamlit view of results/")
    b.add_argument("--out", type=Path, default=None)
    b.add_argument("--check", action="store_true", help="build every figure headless and exit")
    b.set_defaults(func=_cmd_dashboard)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
