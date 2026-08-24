# wc2026 performance report

**Status: PAPER TRADING ONLY.** Live trading is zero and
unsupported in this phase. Figures below are separated into
model backtest, forward paper results, and counterfactuals --
they are not interchangeable.

## Model backtest (NOT market alpha)

Nested evaluation: hyperparameters chosen on DEV, calibrator on
CAL, headline computed once on an untouched HOLDOUT. The
comparison is model vs BASE RATE, not model vs market price.

| league | n | log-loss | base rate | improvement | 95% CI | cal. slope |
|---|---|---|---|---|---|---|
| bundesliga | 765 | 0.9960 | 1.0851 | 8.2% | [0.9622, 1.0309] | 0.930 |
| la_liga | 851 | 0.9748 | 1.0629 | 8.3% | [0.9479, 1.0032] | 1.210 |
| ligue_1 | 685 | 0.9844 | 1.0618 | 7.3% | [0.9503, 1.0178] | 1.079 |
| mls | 1740 | 1.0515 | 1.0705 | 1.8% | [1.0339, 1.0695] | 0.792 |
| premier_league | 859 | 1.0009 | 1.0769 | 7.1% | [0.9727, 1.0284] | 0.997 |

## Forward paper trading

No paper portfolio yet.


## Coverage and abstention

Supported market families are the six derived from the exact
regulation scoreline grid. Corners, halves, player props,
method-of-victory and season futures are DISCOVERED, RECORDED
and ABSTAINED FROM -- no validated model exists for them, so no
price is produced. See docs/VENUES.md for the audited counts.

