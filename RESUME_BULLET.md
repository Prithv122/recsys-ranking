# Resume Bullets — recsys-ranking

Form: **action → technical specifics → measured outcome.** Numbers or it doesn't go on the resume.

---

## Bullets

- Pre-registered and ran a recommender evaluation study on MovieLens 1M (6 models incl. own numpy iALS and
  BPR-MF, a torch two-tower) under a global time split; showed the common 100-sampled-negatives protocol
  changes the model decision: Kendall tau 0.07 [−0.07, 0.20] vs full-catalogue NDCG@10 and 5 of 15 pairs
  reversed with paired 95% CIs, promoting the 5th-best model to 1st.
- Built a leakage-safe ranking harness (time-based folds, shared seeded negatives, user-level bootstrap,
  validation-only tuning with a code-enforced single test-fold score); measured that leave-last-out trains on
  data later than the test item for 44% of rows and yields an unrelated model ordering (tau −0.07).

## Short form (one line)

- Showed sampled-negative evaluation reorders recommenders on MovieLens 1M (tau 0.07, 5 strong reversals), pre-registered, own iALS/BPR.

_Not quotable until the repo is public and CI is green (owner's push decision)._
