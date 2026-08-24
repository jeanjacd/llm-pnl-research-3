# Per-league evaluation (nested, holdout-once)

Produced by `python -m wc2026 evaluate --league <id>`.
Each league is modelled, tuned and calibrated **independently**. Hyperparameters
were selected on the DEV region only, the calibrator was fitted on the CAL
window only, and the final holdout was opened exactly once.

## Headline holdout results

| League | n | log-loss | base rate | improvement | 95% CI (cluster bootstrap) | cal. slope | ECE | unrated skipped |
|---|---|---|---|---|---|---|---|---|
| la_liga | 851 | **0.9748** | 1.0629 | 8.3% | [0.9479, 1.0032] | 1.210 | 0.025 | 9 |
| bundesliga | 765 | **0.9960** | 1.0851 | 8.2% | [0.9622, 1.0309] | 0.930 | 0.030 | 10 |
| ligue_1 | 685 | **0.9844** | 1.0618 | 7.3% | [0.9503, 1.0178] | 1.079 | 0.019 | 7 |
| premier_league | 859 | **1.0009** | 1.0769 | 7.1% | [0.9727, 1.0284] | 0.997 | 0.021 | 7 |
| mls | 1740 | **1.0515** | 1.0705 | 1.8% | [1.0339, 1.0695] | 0.792 | 0.018 | 7 |

Selected parameters (DEV only, per league — deliberately different):

| League | half-life (d) | rho | blend_k |
|---|---|---|---|
| premier_league | 365 | -0.03 | 8 |
| la_liga | 365 | -0.10 | 8 |
| bundesliga | 365 | -0.06 | 12 |
| ligue_1 | 180 | -0.06 | 8 |
| mls | 365 | -0.03 | 8 |

## What these numbers do and do not say

**They say:** against a base-rate baseline, on matches never seen during
selection or calibration, the four European leagues beat the baseline by
7–8% of log-loss, and the whole 95% cluster-bootstrap interval sits below the
baseline in each case. Bundesliga and Ligue 1 are close to calibrated;
Premier League is essentially calibrated (slope 0.997).

**They do not say anything about market alpha.** The comparison is
model-vs-base-rate, not model-vs-price. No point-in-time market prices are
stored yet, so no claim of edge over a bookmaker or exchange is supported by
this table.

## Notable findings

- **MLS is the weakest league to model, by a wide margin** (1.8% vs 7–8%), and
  its interval [1.0339, 1.0695] only barely clears its 1.0705 baseline. MLS
  parity is real; the earlier "the model works" conclusion was drawn on the
  hardest league available.
- **MLS is over-confident** (calibration slope 0.792 — probabilities too
  extreme). Any MLS staking must be discounted for this, which is exactly what
  the trust curve in the betting layer is for.
- **La Liga is under-confident** (slope 1.210): its forecasts could be sharpened.
- 7–10 fixtures per league were **skipped because a club had no fitted rating**
  (promoted sides). Previously these were silently scored as league-average;
  they are now refused and counted.
- Every league selected different parameters. Ligue 1 chose a 180-day half-life
  against 365 elsewhere — confirming that a pooled European fit would have been
  wrong for at least one league.

## Reproduce

```bash
python -m wc2026 update   --all
python -m wc2026 evaluate --all
```
