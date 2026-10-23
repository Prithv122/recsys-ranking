# recsys-ranking — D8

**Tier:** 3 · **Category:** D — Analytics & data science · **Wave:** 4

The project's own engineering guide. Root rules in `../GUIDELINES.md` apply.

## What this is

A product team must pick which recommender to ship. Papers often score models against 100 sampled
negatives instead of the full catalogue, which is cheap but can reorder models. One question:
under a global-time split, which of random / popularity / item-kNN / ALS / BPR / two-tower ranks
best by full-catalogue NDCG@10, and does that ordering survive the 100-sampled-negatives protocol
(Kendall tau between the two orderings, with a bootstrap CI)?

## Stack

Python 3.12, numpy + scipy + pandas. Models are own code (iALS, BPR-MF, item-kNN in numpy/scipy);
two-tower is an optional `twotower` extra (torch). Dashboard is an optional `dashboard` extra
(Streamlit + Altair) over the aggregate results CSV. MovieLens 1M, downloaded by script, never committed.

## Non-negotiables

1. `PREREGISTRATION.md` is committed before any MovieLens number exists. Grids, seeds, metrics and
   wording rules are not changed after seeing test results; an amendment is dated and says why.
2. MovieLens terms forbid redistribution: no ratings, no per-user files, no derived matrices in git.
   Only aggregate metrics (means, CIs, tau) are committed, under `results/`.
3. Every README number comes from a committed `results/` file written by `recsys-ranking`.
4. CI never needs the real data: tests use `synth.py` and the hand-written fixture in `tests/data/`.
5. Hyperparameters are chosen on the `val` fold only; the `test` fold is scored once per model.

## Contracts (code against these; changing one means editing this file in the same commit)

Shapes live in `src/recsys_ranking/types.py`: `Interactions`, `Fold`, `Recommender`.

- **data.py** — `load_ml1m(path) -> Interactions` parses `ratings.dat` (`UserID::MovieID::Rating::Timestamp`),
  all ratings. `positives(inter, threshold=4.0) -> Interactions` keeps `rating >= threshold`.
  `download(dest_dir) -> Path` fetches `ml-1m.zip`, verifies `ML1M_SHA256` when set, extracts, returns the folder.
- **synth.py** — `make_synthetic(n_users, n_items, n_factors, density, pop_skew, seed) -> (Interactions, truth_affinity)`:
  latent-factor users and items, Zipf-like item popularity, increasing timestamps. Same seed, same output.
- **split.py** — `global_time_folds(inter, val_frac=0.1, test_frac=0.1) -> (val_fold, test_fold)`.
  Cutoffs are timestamp quantiles of the positives: `t_val = q(1-val-test)`, `t_test = q(1-test)`.
  `val`: fit `ts < t_val`, truth `t_val <= ts < t_test`. `test`: fit `ts < t_test`, truth `ts >= t_test`.
  `leave_last_out(inter) -> Fold` (name `"loo"`): each user's latest positive (ties: larger item id) is truth,
  users with fewer than 2 positives are fit-only. Every fold is reindexed from its fit rows (see `Fold`).
  `loo_future_share(inter) -> float`: mean, over loo truth pairs, of the share of loo fit rows with a later timestamp (the leak size), over truth pairs that survive into the fold.
  Split functions take positives only (`positives(load_ml1m(...))`); cutoffs use `np.quantile(method="lower")`,
  so a timestamp tie at a cutoff lands on the truth side. Synthetic raw ids are 0-based and index `truth_affinity`.
- **metrics.py** — per-user arrays, binary relevance. `recs`: int64 `(n, k)` item indices, best first;
  `truth`: list of n int arrays. `recall_at_k` = hits / |truth|; `ndcg_at_k` with IDCG over
  `min(k, |truth|)`, gain `1/log2(rank+1)`; `map_at_k` = sum of precision@i at hits / `min(k, |truth|)`;
  `mrr_at_k` = 1/rank of the first hit within k, else 0. Catalogue-level: `coverage(recs, n_items)`,
  `mean_popularity(recs, item_pop)`, `gini_exposure(recs, n_items)` (0 = equal exposure).
- **protocols.py** — `full_topk(model, fold, users, k, batch=1024) -> recs`: seen-in-fit items set to `-inf`,
  ties broken by lower item index (stable sort on `-score`).
  `sample_negatives(fold, n_neg=100, scheme="uniform"|"popularity", seed) -> SampledSet` (dataclass: `user_idx` dense user of each pair, `pos_item`,
  `neg_items` (n_pairs, n_neg), `scheme`, `seed`): one row per
  (eval user, truth item) pair, `n_neg` items drawn without replacement from items not in the user's fit
  or truth; popularity scheme draws ∝ fit item count. Depends only on the fold and seed, never on a model,
  so every model is ranked against identical negatives. `sampled_ranks(model, sampled) -> int array`:
  1-based rank of the positive among its 101 candidates, **ties count against the positive**
  (rank = 1 + #neg ≥ pos score). `sampled_user_metrics(sampled, ranks, k)` → per-user HR@k and NDCG@k
  (mean over that user's pairs), users in `fold.eval_users()` order.
- **stats.py** — `paired_bootstrap(a, b, n_boot, seed) -> (mean_diff, lo, hi)` over users (95% percentile).
  `kendall_tau(x, y)` (tau-b). `ordering_tau_ci(full: dict[model, per-user], sampled: dict[model, per-user],
  n_boot, seed) -> (tau, lo, hi)`: resample users jointly, recompute each model's mean under both
  protocols, tau between the two orderings.
- **models/** — each implements `Recommender`; constructor takes hyperparameters + `seed`; `fit` on binary CSR.
  `random`, `popularity`, `itemknn` (cosine, top-`n_neighbors`, `shrink`), `als` (implicit ALS,
  Hu–Koren–Volinsky, confidence `1 + alpha * r`, exact per-user solve with the `YᵀY` trick),
  `bpr` (BPR-MF, vectorised mini-batch SGD), `twotower` (torch, optional, sampled-softmax with in-batch
  negatives). `models.REGISTRY: dict[str, type]` maps name → class; `twotower` registers only if torch imports.
- **evaluate.py** — runs a fold × model × protocol grid, writes per-user rows to `results/raw/` (git-ignored)
  and aggregates to `results/summary.csv` with columns
  `fold, protocol, model, metric, k, mean, ci_lo, ci_hi, n_users` (`protocol` ∈ `full`, `sampled_uniform`,
  `sampled_popularity`; catalogue metrics have empty CIs) and `results/ordering.csv`
  (`fold, protocol_a, protocol_b, metric, k, tau, ci_lo, ci_hi, ordering_a, ordering_b`; orderings are
  model names joined by `>` best first) and `results/pairs.csv`
  (`fold, protocol, metric, k, model_a, model_b, diff, ci_lo, ci_hi`; diff = a − b, paired bootstrap).
  `report.py` turns these three files into `results/REPORT.md` mechanically, using the wording rules in
  `PREREGISTRATION.md`.
- **cli.py** — `recsys-ranking download | evaluate | compare-protocols | dashboard [--check]`;
  `evaluate --synthetic --smoke` runs end to end on synthetic data in seconds.

## Acceptance criteria

- [ ] Split has no test-time leakage; metrics match hand-computed cases; sampled protocol deterministic per seed
- [ ] ALS beats popularity on planted synthetic data; no model ever recommends a seen item
- [ ] `PREREGISTRATION.md` committed before the first ML-1M run; results and verdict follow its rules
- [ ] Dashboard builds headless (`dashboard --check`) and shows the protocol comparison
- [ ] Runs from a clean clone using only the documented steps; tests, lint and CI green

## Commands

```
uv sync --all-groups --extra dashboard
uv run recsys-ranking evaluate --synthetic --smoke
uv run recsys-ranking download
uv run recsys-ranking evaluate                 # real data: tunes on val only
uv run recsys-ranking evaluate --stage score   # scores test + loo once; refuses a re-score
uv run pytest ; uv run ruff check . ; uv run ruff format --check .
```

## Project-specific notes

- Local pytest needs `--basetemp=<scratch dir>` on this machine; never put it in `pyproject.toml`.
- ML-1M timestamps cluster in 2000, so the last 10% by time covers fewer, late-active users. That is
  the point of a global split; the fold sizes and dropped-row counts are reported, not hidden.
