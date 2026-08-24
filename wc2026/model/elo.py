"""
elo.py
======
A transparent World Football Elo rating, computed from the same results dataset the
rest of the model uses -- no scraping, fully reproducible. Used as a shrinkage prior
for the Dixon-Coles attack/defense ratings (model/ratings.py), which keeps sparse
teams (e.g. World Cup minnows with few recent games) from over-fitting their handful
of matches.

Implementation follows the standard eloratings.net conventions:
  * Expected score:  E = 1 / (1 + 10**(-(R_a - R_b + hfa) / 400))
  * Update:          R' = R + K * G * (W - E)
    - W in {1, 0.5, 0} for win/draw/loss
    - G is a goal-difference multiplier (bigger wins move ratings more)
    - K scaled by match importance (config tier weights)
"""
from __future__ import annotations

import pandas as pd

from ..config import CONFIG


def _goal_diff_multiplier(goal_diff: int) -> float:
    """eloratings.net goal-difference multiplier G."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def compute_elo(matches: pd.DataFrame, cfg=CONFIG) -> dict[str, float]:
    """Sequentially process completed matches in date order and return final ratings.

    `matches` must be played matches with columns home_team, away_team, home_score,
    away_score, neutral, tier. Importance scales K so friendlies move ratings less.
    """
    ratings: dict[str, float] = {}
    init = cfg.elo_initial

    m = matches.sort_values("date")
    for home, away, hs, as_, neutral, tier in zip(
        m["home_team"], m["away_team"], m["home_score"], m["away_score"],
        m["neutral"], m["tier"]
    ):
        ra = ratings.get(home, init)
        rb = ratings.get(away, init)
        hfa = 0.0 if neutral else cfg.elo_hfa
        exp_a = 1.0 / (1.0 + 10 ** (-(ra - rb + hfa) / 400.0))

        gd = int(hs - as_)
        if gd > 0:
            wa = 1.0
        elif gd < 0:
            wa = 0.0
        else:
            wa = 0.5

        k = cfg.elo_k * cfg.importance_weights.get(tier, 0.5) * _goal_diff_multiplier(gd)
        delta = k * (wa - exp_a)
        ratings[home] = ra + delta
        ratings[away] = rb - delta
    return ratings


def win_probability(elo_a: float, elo_b: float, hfa: float = 0.0) -> float:
    """P(team A wins or, loosely, expected score) from Elo difference."""
    return 1.0 / (1.0 + 10 ** (-(elo_a - elo_b + hfa) / 400.0))
