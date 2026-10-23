# Resume Bullets — recsys-ranking

Form: **action → technical specifics → measured outcome.** Numbers or it doesn't go on the resume.

---

## Bullets

- Pre-registered and ran a recommender evaluation study on MovieLens 1M (6 models incl. own numpy iALS and
  BPR-MF, a torch two-tower) under a global time split; showed 100-sampled-negative evaluation reorders models (Kendall tau 0.47
  uniform; 0.07 [−0.07, 0.20] popularity-weighted, where 5 of 15 pairs strongly reverse by paired 95% CIs
  and the model ranked 5th of 6 on the full catalogue comes first).
- Built a leakage-safe ranking harness (time-based folds, shared seeded negatives, user-level bootstrap,
  validation-only tuning, one test-fold scoring run committed with raw-file hashes); measured that under leave-last-out, on average 44% of the training set
  is timestamped after the test item.

## Short form (one line)

- Showed popularity-weighted sampled-negative evaluation reorders recommenders on MovieLens 1M under a time split (tau 0.07, 5 strong reversals), pre-registered, own iALS/BPR.
