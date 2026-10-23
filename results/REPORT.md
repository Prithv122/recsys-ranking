# Results report

Headline fold: `test`. Metric: NDCG@10. Generated from `summary.csv`, `ordering.csv` and `pairs.csv`; one dataset, one split, CPU-sized models.

## Folds

| Fold | Users | Items | Fit rows | Truth rows | Eval users | Cold-user rows | Cold-item rows | Seen-pair rows |
|---|---|---|---|---|---|---|---|---|
| val | 5378 | 3468 | 460224 | 13632 | 491 | 43883 | 11 | 0 |
| test | 5975 | 3489 | 517750 | 52175 | 1155 | 5257 | 99 | 0 |
| loo | 6038 | 3529 | 569244 | 6033 | 6033 | 0 | 4 | 0 |

Leak size on `loo`: on average 44.2% of fit rows are later than the truth pair they are evaluated against.

## Full-catalogue NDCG@10 (test fold)

| Rank | Model | NDCG@10 | 95% CI |
|---|---|---|---|
| 1 | bpr | 0.1966 | [0.1838, 0.2100] |
| 2 | itemknn | 0.1860 | [0.1730, 0.1986] |
| 3 | popularity | 0.1779 | [0.1651, 0.1916] |
| 4 | als | 0.1675 | [0.1557, 0.1789] |
| 5 | twotower | 0.1631 | [0.1519, 0.1748] |
| 6 | random | 0.0104 | [0.0087, 0.0125] |

## Pairwise comparisons (full protocol)

- bpr beats itemknn (difference +0.0105, 95% CI [0.0026, 0.0185]).
- bpr beats popularity (difference +0.0187, 95% CI [0.0103, 0.0270]).
- bpr beats als (difference +0.0290, 95% CI [0.0189, 0.0387]).
- bpr beats twotower (difference +0.0334, 95% CI [0.0235, 0.0437]).
- bpr beats random (difference +0.1861, 95% CI [0.1737, 0.1994]).
- No clear difference between itemknn and popularity (difference +0.0082, 95% CI [-0.0023, 0.0178]).
- itemknn beats als (difference +0.0185, 95% CI [0.0103, 0.0269]).
- itemknn beats twotower (difference +0.0229, 95% CI [0.0135, 0.0317]).
- itemknn beats random (difference +0.1756, 95% CI [0.1630, 0.1881]).
- No clear difference between popularity and als (difference +0.0104, 95% CI [-0.0014, 0.0222]).
- popularity beats twotower (difference +0.0148, 95% CI [0.0032, 0.0263]).
- popularity beats random (difference +0.1675, 95% CI [0.1545, 0.1805]).
- No clear difference between als and twotower (difference +0.0044, 95% CI [-0.0026, 0.0113]).
- als beats random (difference +0.1571, 95% CI [0.1456, 0.1682]).
- twotower beats random (difference +0.1527, 95% CI [0.1416, 0.1636]).

## Full vs sampled orderings

| Sampled protocol | Kendall tau-b | 95% CI |
|---|---|---|
| sampled_uniform | 0.4667 | [0.2000, 0.6000] |
| sampled_popularity | 0.0667 | [-0.0667, 0.2000] |

- full: bpr > itemknn > popularity > als > twotower > random
- sampled_uniform: bpr > twotower > itemknn > als > popularity > random
- full: bpr > itemknn > popularity > als > twotower > random
- sampled_popularity: twotower > als > bpr > itemknn > popularity > random

## Strong-reversal check

A pair is a strong reversal when the full-protocol difference CI excludes 0 in one direction and the sampled-protocol difference CI excludes 0 in the other. Differences are first minus second model, the first being ranked higher by full NDCG@10.

### full vs sampled_uniform

| Pair | Full diff (CI) | Sampled diff (CI) | Order swapped | Strong reversal |
|---|---|---|---|---|
| bpr vs itemknn | +0.0105 [0.0026, 0.0185] | +0.0156 [0.0099, 0.0212] | no | no |
| bpr vs popularity | +0.0187 [0.0103, 0.0270] | +0.0455 [0.0390, 0.0523] | no | no |
| bpr vs als | +0.0290 [0.0189, 0.0387] | +0.0186 [0.0111, 0.0261] | no | no |
| bpr vs twotower | +0.0334 [0.0235, 0.0437] | +0.0055 [-0.0014, 0.0120] | no | no |
| bpr vs random | +0.1861 [0.1737, 0.1994] | +0.2954 [0.2851, 0.3058] | no | no |
| itemknn vs popularity | +0.0082 [-0.0023, 0.0178] | +0.0300 [0.0226, 0.0373] | no | no |
| itemknn vs als | +0.0185 [0.0103, 0.0269] | +0.0031 [-0.0037, 0.0097] | no | no |
| itemknn vs twotower | +0.0229 [0.0135, 0.0317] | -0.0101 [-0.0170, -0.0034] | yes | yes |
| itemknn vs random | +0.1756 [0.1630, 0.1881] | +0.2798 [0.2696, 0.2907] | no | no |
| popularity vs als | +0.0104 [-0.0014, 0.0222] | -0.0269 [-0.0367, -0.0172] | yes | no |
| popularity vs twotower | +0.0148 [0.0032, 0.0263] | -0.0401 [-0.0493, -0.0316] | yes | yes |
| popularity vs random | +0.1675 [0.1545, 0.1805] | +0.2499 [0.2404, 0.2597] | no | no |
| als vs twotower | +0.0044 [-0.0026, 0.0113] | -0.0132 [-0.0181, -0.0078] | yes | no |
| als vs random | +0.1571 [0.1456, 0.1682] | +0.2768 [0.2662, 0.2869] | no | no |
| twotower vs random | +0.1527 [0.1416, 0.1636] | +0.2899 [0.2801, 0.3000] | no | no |

### full vs sampled_popularity

| Pair | Full diff (CI) | Sampled diff (CI) | Order swapped | Strong reversal |
|---|---|---|---|---|
| bpr vs itemknn | +0.0105 [0.0026, 0.0185] | +0.0002 [-0.0040, 0.0043] | no | no |
| bpr vs popularity | +0.0187 [0.0103, 0.0270] | +0.0287 [0.0245, 0.0329] | no | no |
| bpr vs als | +0.0290 [0.0189, 0.0387] | -0.0148 [-0.0202, -0.0097] | yes | yes |
| bpr vs twotower | +0.0334 [0.0235, 0.0437] | -0.0201 [-0.0257, -0.0148] | yes | yes |
| bpr vs random | +0.1861 [0.1737, 0.1994] | +0.0516 [0.0453, 0.0579] | no | no |
| itemknn vs popularity | +0.0082 [-0.0023, 0.0178] | +0.0285 [0.0233, 0.0339] | no | no |
| itemknn vs als | +0.0185 [0.0103, 0.0269] | -0.0150 [-0.0191, -0.0110] | yes | yes |
| itemknn vs twotower | +0.0229 [0.0135, 0.0317] | -0.0203 [-0.0251, -0.0156] | yes | yes |
| itemknn vs random | +0.1756 [0.1630, 0.1881] | +0.0514 [0.0444, 0.0582] | no | no |
| popularity vs als | +0.0104 [-0.0014, 0.0222] | -0.0434 [-0.0499, -0.0372] | yes | no |
| popularity vs twotower | +0.0148 [0.0032, 0.0263] | -0.0487 [-0.0553, -0.0423] | yes | yes |
| popularity vs random | +0.1675 [0.1545, 0.1805] | +0.0229 [0.0175, 0.0277] | no | no |
| als vs twotower | +0.0044 [-0.0026, 0.0113] | -0.0053 [-0.0093, -0.0016] | yes | no |
| als vs random | +0.1571 [0.1456, 0.1682] | +0.0664 [0.0592, 0.0733] | no | no |
| twotower vs random | +0.1527 [0.1416, 0.1636] | +0.0717 [0.0644, 0.0790] | no | no |

## Result

- sampled_uniform: tau = 0.4667, swapped pairs: itemknn/twotower, popularity/als, popularity/twotower, als/twotower.
  - Strong reversal: itemknn and twotower.
  - Strong reversal: popularity and twotower.
  - popularity and als: order swapped, not a strong reversal.
  - als and twotower: order swapped, not a strong reversal.
- sampled_popularity: tau = 0.0667, swapped pairs: bpr/als, bpr/twotower, itemknn/als, itemknn/twotower, popularity/als, popularity/twotower, als/twotower.
  - Strong reversal: bpr and als.
  - Strong reversal: bpr and twotower.
  - Strong reversal: itemknn and als.
  - Strong reversal: itemknn and twotower.
  - Strong reversal: popularity and twotower.
  - popularity and als: order swapped, not a strong reversal.
  - als and twotower: order swapped, not a strong reversal.

## Catalogue metrics (full protocol, k = 10, descriptive only)

| Model | Coverage@10 | Mean popularity@10 | Gini of exposure@10 |
|---|---|---|---|
| bpr | 0.1568 | 1512.1500 | 0.9667 |
| itemknn | 0.1433 | 1367.2800 | 0.9646 |
| popularity | 0.0275 | 1953.9000 | 0.9931 |
| als | 0.1731 | 1116.4000 | 0.9382 |
| twotower | 0.3141 | 979.4580 | 0.8909 |
| random | 0.9564 | 131.5430 | 0.3095 |

## Leave-last-out ablation (`loo` fold)

| Rank | Model | NDCG@10 | 95% CI |
|---|---|---|---|
| 1 | twotower | 0.0446 | [0.0408, 0.0486] |
| 2 | als | 0.0441 | [0.0401, 0.0482] |
| 3 | itemknn | 0.0351 | [0.0316, 0.0386] |
| 4 | bpr | 0.0254 | [0.0223, 0.0285] |
| 5 | popularity | 0.0193 | [0.0167, 0.0220] |
| 6 | random | 0.0015 | [0.0009, 0.0023] |

- Kendall tau-b between the loo full ordering and the test full ordering: -0.0667 (point estimate, no CI).
- loo full vs sampled_uniform: tau = 1.0000, 95% CI [0.8667, 1.0000].
- loo full vs sampled_popularity: tau = 1.0000, 95% CI [0.7333, 1.0000].
