"""Post-hoc diagnostic: does the pessimistic tie rule move popularity's sampled NDCG@10?

Popularity has integer scores, so it ties often (about 10% of its sampled pairs).
Item-kNN also ties when the positive scores exactly 0 (about 2% of its pairs); a separate
check found that mid-rank ties change its sampled NDCG@10 by less than 0.0001.
ALS and random do not tie.
Writes results/tie_check.csv. Run after `recsys-ranking download`.
"""

import csv
from pathlib import Path

import numpy as np

from recsys_ranking import protocols
from recsys_ranking.data import load_ml1m, positives
from recsys_ranking.evaluate import Config
from recsys_ranking.models import REGISTRY
from recsys_ranking.split import global_time_folds


def _ndcg10(rank: np.ndarray) -> np.ndarray:
    return np.where(rank <= 10, 1 / np.log2(rank + 1), 0.0)


def main() -> None:
    cfg = Config()
    _, test = global_time_folds(positives(load_ml1m(Path("data/ml-1m"))))
    model = REGISTRY["popularity"]().fit(test.fit)
    rows = []
    for scheme in ("uniform", "popularity"):
        s = protocols.sample_negatives(test, cfg.n_neg, scheme=scheme, seed=cfg.neg_seed)
        users = np.unique(s.user_idx)
        scores = model.score(users[:1])[0]  # popularity gives every user the same row
        pos, neg = scores[s.pos_item], scores[s.neg_items]
        above = (neg > pos[:, None]).sum(axis=1)
        tied = (neg == pos[:, None]).sum(axis=1)
        per_rule = {}
        for rule, rank in (("pessimistic", 1 + above + tied), ("mid_rank", 1 + above + tied / 2)):
            gain = _ndcg10(rank)
            per_rule[rule] = float(np.mean([gain[s.user_idx == u].mean() for u in users]))
        rows.append(
            {
                "protocol": f"sampled_{scheme}",
                "pairs_with_tie": round(float((tied > 0).mean()), 4),
                "ndcg10_pessimistic": round(per_rule["pessimistic"], 4),
                "ndcg10_mid_rank": round(per_rule["mid_rank"], 4),
            }
        )
    out = Path("results/tie_check.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
