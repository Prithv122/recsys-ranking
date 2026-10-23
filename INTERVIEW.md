# Interview Prep — recsys-ranking

---

### Q1. Walk me through the architecture in 90 seconds.

**A:** MovieLens 1M ratings become positives (rating ≥ 4). One global time cutoff splits them: everything
before the 90th percentile of timestamps is training data, everything after is test; a second cutoff at the
80th percentile gives a validation fold for tuning. Users or items with no training rows are dropped from
the truth and counted. Six models (random, popularity, item-kNN, implicit ALS, BPR-MF, two-tower) are tuned
on a registered grid on validation only, refit, and scored once on test. Each is scored two ways: against the
whole catalogue, and against 100 sampled negatives per held-out item (uniform and popularity-weighted, drawn
once and shared by every model). Per-user metrics stay local; a bootstrap over users gives CIs, paired
differences and a Kendall tau between the two orderings, and a report writer turns those files into
wording using rules fixed before the data was downloaded. A Streamlit dashboard reads the same CSVs.

### Q2. Why a global time split instead of leave-last-out, the default in most papers?

**A:** Leave-last-out holds out each user's last item, but trains on everyone else's history up to the
present, so the model learns from events that happen after the test event it is judged on. Here that is
not hypothetical: on average 44% of the training rows are later than the item being predicted. A global
cutoff is what deployment looks like: train on the past, predict the future for everyone. I kept
leave-last-out as an ablation, and it produced an unrelated model ordering (tau −0.07 against the global
split). I'm careful about the claim: the two splits also evaluate different users, so that shows the split
choice changes the answer, not that the leak alone causes it.

### Q3. What's the weakest part, and what breaks first at scale?

**A:** Statistically, tuning: the validation fold has only 491 users because ML-1M activity is concentrated
in 2000, and ALS's best grid point is at the edge of the grid (32 factors), so ALS may be under-tuned. It's
one dataset and one split. Engineering-wise, scoring is dense (users × items in memory) and ALS loops over
users in Python; at 100× both need approximate top-k retrieval and batched or GPU solves. Two-tower already
takes ~15 minutes per fit on one CPU thread.

### Q4. How do you know it works? What did you measure, and against what baseline?

**A:** The metric code is checked against hand-computed NDCG/MAP/MRR cases; the sampled protocol is
deterministic per seed and independent of batch size; ALS is checked against a dense textbook solve;
there's a leakage test on the split. Random and popularity are the baselines, and popularity is strong
here: item-kNN does not clearly beat it. An independent review read the harness before the test fold was
touched and found five robustness issues, the important one being that a NaN score would have ranked first
under the sampled protocol and faked a reversal; all fixed and tested before scoring. The result: BPR wins
on the full catalogue (NDCG@10 0.197, beats every other model), but with popularity-weighted negatives the
tau between orderings is 0.07 [−0.07, 0.20] and 5 of 15 pairs strongly reverse; that protocol would have
shipped the two-tower, which the full catalogue ranks 5th of 6.

### Q5. Isn't a reversal just noise? With six models, tau jumps in steps of 0.13.

**A:** That's why tau isn't the only rule. I registered a "strong reversal": a pair counts only when the
paired bootstrap CI excludes 0 in one direction on the full catalogue and in the other direction on the
sampled protocol, with the same users resampled for both. Swaps between models that the full catalogue
can't separate (ALS vs two-tower, item-kNN vs popularity) are reported as "swapped, not a strong reversal".
Five pairs pass the strict rule under popularity-weighted sampling. The negatives are also identical across
models, so the reversal can't come from one model getting easier negatives, and ties were checked: mid-rank
ties move popularity's sampled NDCG@10 by at most 0.0002. The likely mechanism, which I label exploratory
because I didn't register it, is popularity: the models that gain recommend less popular titles, and
popularity-weighted negatives punish models that rank popular titles high.

---

## 30-second pitch

Teams often pick recommenders using a shortcut: rank each held-out item against 100 random items instead
of the whole catalogue. I tested whether that changes the decision. Six models on MovieLens 1M, a time-based
split, everything pre-registered. On the full catalogue BPR wins clearly; with popularity-weighted sampled
negatives the model ordering is essentially uncorrelated (tau 0.07) and five model pairs flip with
confidence intervals that exclude zero, so the shortcut would have shipped the 5th-best model.
