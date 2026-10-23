# Pre-registration — recsys-ranking

Committed before any MovieLens 1M number was computed. Changes after that point are amendments at
the bottom, dated, with the reason, and never touch a rule after its result has been seen.

**Status:** accepted as written by the owner on 2026-10-23, before the dataset was downloaded.

## Question

Under a global-time split of MovieLens 1M, which model ranks best by **full-catalogue NDCG@10**, and
does that ordering survive the **100-sampled-negatives** protocol?

## Data

- MovieLens 1M `ratings.dat` (GroupLens; research use, no redistribution). Positive = `rating >= 4`.
  Lower ratings are dropped, not treated as negatives.
- Folds (`split.py`): cutoffs at the 80th and 90th percentile of positive timestamps.
  `val` = fit before t_val, truth in [t_val, t_test) — used only to pick hyperparameters.
  `test` = fit before t_test, truth from t_test on — the headline, scored once per model.
  Users or items with no fit row are dropped from truth; counts are reported.
- Ablation fold `loo`: each user's latest positive is truth, everything else is fit (the popular
  per-user split, which lets the model train on events that happen after other users' test events).

## Models and grids

| Model | Grid (selected on `val` full NDCG@10) |
|---|---|
| random | seed 0 |
| popularity | none |
| itemknn | n_neighbors ∈ {50, 200} × shrink ∈ {0, 10} |
| als | factors ∈ {32, 64, 128} × reg ∈ {0.01, 0.1} × alpha ∈ {1, 10}, 15 iterations |
| bpr | factors ∈ {32, 64} × lr ∈ {0.01, 0.05} × reg ∈ {1e-4, 1e-3}, 30 epochs |
| twotower | dim 64, lr ∈ {1e-3, 3e-3}, 20 epochs, in-batch negatives, batch 1024 |

Ties in selection go to the first configuration in grid order. All seeds are fixed in code
(model seed 0; negative-sampling seed 20261023; bootstrap seed 7). Two-tower is optional: if it
cannot be trained on CPU in the session, it is reported as "not evaluated", not dropped silently.
The selected configuration is refit on the `test` fold's fit rows; the same configuration is used on `loo`.

## Protocols and metrics

- **full**: every catalogue item the user has not seen in fit is ranked.
- **sampled_uniform** and **sampled_popularity**: each (user, truth item) pair is ranked against 100
  negatives drawn without replacement from items the user has not seen in fit or truth, uniformly or
  ∝ fit popularity. Negatives depend only on the fold and seed, so every model sees the same ones.
  Ties count against the positive.
- Per-user metrics at k ∈ {5, 10, 20}: Recall, NDCG, MAP, MRR (full); HR, NDCG (sampled).
  Catalogue metrics (full, k = 10): coverage, mean recommended-item popularity, Gini of exposure — descriptive only.
- Uncertainty: 2,000 bootstrap resamples over evaluation users, 95% percentile intervals.

## Primary

The ordering of models by mean `test` full NDCG@10. Wording rule: "A beats B" only when the paired
bootstrap 95% CI of the NDCG@10 difference excludes 0; otherwise "no clear difference between A and B".

## Secondary

1. Kendall tau-b between the full NDCG@10 ordering and each sampled NDCG@10 ordering on `test`,
   with a bootstrap CI (users resampled jointly across protocols).
2. **Strong reversal** of a pair: the full-protocol difference CI excludes 0 in one direction and the
   sampled-protocol difference CI excludes 0 in the other. Reported per pair.
3. The same ordering on `loo`, and its tau against the `test` full ordering. Leak size on `loo`:
   mean, over truth pairs, of the share of fit rows timestamped after that pair.

## Allowed results

- tau = 1 on both sampled protocols: "sampling was harmless on this data", reported as a null.
- tau < 1: name the swapped pairs; call it a reversal only under the strong-reversal rule above.
- ALS not beating item-kNN within CI is a valid result and is reported as such.
- Every number is one dataset, one split, CPU-sized models; the README says so.

## Amendments

_None._
