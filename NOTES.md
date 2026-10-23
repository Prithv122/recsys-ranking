# Build Notes — recsys-ranking

Working notes: what broke, what was tried, why X over Y.

---

## Log

### 2026-10-23 — contracts first, four parts built in parallel, reviewed before the test fold

- **Order of work.** `GUIDELINES.md` contracts and `PREREGISTRATION.md` were committed before any
  module existed, then data/split, metrics/protocols/stats, the five numpy models, and two-tower/
  report/dashboard were built side by side against them. The owner accepted the pre-registration
  before the dataset was downloaded.
- **Smoke run broke on the sampled protocol.** The 300 × 200 synthetic set only has ~120 items that
  anyone interacted with, so "100 negatives the user hasn't seen" was impossible. Smoke mode uses 50;
  the real run keeps the registered 100.
- **Download failed once with `CERTIFICATE_VERIFY_FAILED`, then worked.** GroupLens chains to an emSign
  root that Windows had not cached yet; after the OS fetched it (a `curl` call did that), Python's
  verification passed. Nothing was disabled. The archive hash is pinned in `data.py`.
- **`.gitignore` swallowed the test fixture.** A bare `data/` pattern also matched `tests/data/`;
  anchored to `/data/`.
- **Staged run.** Tuning on the validation fold started while an independent review read the harness,
  so the test fold was not touched until the review was clean. The review found no blocker and five
  process/robustness fixes, all applied before scoring:
  1. Non-finite scores now raise. A NaN positive would have ranked first under the sampled protocol
     (`neg >= NaN` is false), which would have looked like a "strong reversal".
  2. Scoring merges by model and refuses to re-score a model already scored on the test fold;
     before, adding two-tower later would have wiped the other five from every results file.
  3. An unknown or unavailable model name is an error, not a silent drop (the pre-registration says
     two-tower is reported as "not evaluated" if it can't run, never dropped silently).
  4. `tuning.csv` keeps full precision, so selection after the CSV round-trip equals selection in memory.
  5. On real data `evaluate` defaults to `--stage tune`; scoring the test fold is a separate, explicit command.
- **The first tuning run predates fix 4**, so its `tuning.csv` is rounded to 6 significant digits; it was
  checked for ties created by rounding before scoring (see below).
- **Rounding check (fix 4) passed.** Within every model the 6-digit `val_ndcg10` values are all distinct
  (closest pair: BPR 0.12685 vs 0.126881), so the rounded file selects exactly what the in-memory run
  would have.
- **Validation fold surprises.** Popularity (0.1165) beat item-kNN (0.1062) and ALS (0.1022) on val
  NDCG@10; BPR was best (0.1269). Only 491 evaluation users, so this is noisy. ALS's best point is the
  smallest factor count in the grid (32) and the larger grids got worse: the registered grid may not
  contain ALS's optimum. Disclosed, not re-gridded (the grid is pre-registered).
- **Torch threads.** The first two-tower tuning run used torch's default threads on a loaded machine;
  stopped before it wrote anything and restarted with `OMP_NUM_THREADS=1`.
- **BLAS threads.** With the machine loaded by other jobs, a small dense solve took 24 s with 12 BLAS
  threads and a fraction of a second with one; real runs set `OPENBLAS_NUM_THREADS=1`. Fit times in
  `results/` are from a busy laptop and are not a benchmark.

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| Leave-last-out as the headline split | Lets the model train on events after other users' test events; kept as the ablation and its leak is measured |
| One sampled candidate set per model | Different negatives per model would add noise to exactly the comparison under study; negatives depend only on fold + seed |
| Committing per-user metrics | A MovieLens derivative; only aggregates are committed |
| `np.add.at` for BPR updates | Too slow on 2-D rows; replaced by a sparse scatter-add with the same semantics |

## Open questions

- [ ] Does the protocol result hold on an implicit-only dataset with real purchases (brief §12)?
