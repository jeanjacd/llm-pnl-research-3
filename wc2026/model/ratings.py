"""
ratings.py
==========
Assembles the final team-strength ratings used by the simulators by combining:

  1. The time-weighted Dixon-Coles attack/defense fit (model/dixon_coles.py).
  2. A transparent Elo prior (model/elo.py), used as TEAM-SPECIFIC shrinkage: a
     data-poor team is pulled toward the strength its Elo implies, not toward the
     league average. The Elo->rating map is itself learned from the well-fit teams,
     so this stays fully data-driven.
  3. The optional soft-factor adjustment layer (model/adjustments.py).

Shrinkage weight per team:  w = n_eff / (n_eff + blend_k), where n_eff is the team's
time-weighted match count. Lots of recent data -> trust the fit; little data -> trust
the Elo prior.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from ..config import CONFIG
from . import elo as elo_mod
from .adjustments import apply_adjustments, load_adjustments
from .dixon_coles import DCRatings
from .dixon_coles import fit as dc_fit


def _elo_to_rating_slopes(att: dict, dfn: dict, elo: dict, n_eff: dict,
                          teams: list[str]) -> tuple[float, float, float]:
    """Weighted least-squares slopes mapping centred Elo -> attack and -> defense,
    fit through the origin on well-observed teams. Returns (slope_att, slope_def,
    elo_mean)."""
    e = np.array([elo.get(t, CONFIG.elo_initial) for t in teams])
    emean = float(e.mean())
    ec = e - emean
    a = np.array([att[t] for t in teams])
    d = np.array([dfn[t] for t in teams])
    wts = np.array([n_eff[t] for t in teams])
    denom = np.sum(wts * ec * ec)
    if denom <= 1e-9:
        return 0.0, 0.0, emean
    slope_a = float(np.sum(wts * ec * a) / denom)
    slope_d = float(np.sum(wts * ec * d) / denom)
    return slope_a, slope_d, emean


def build_team_strength(matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
                        rho: float | None = None, cfg=CONFIG,
                        adjustments_path: str | None = None,
                        max_age_days: float | None = None,
                        verbose: bool = False) -> DCRatings:
    """Return blended, Elo-shrunk, (optionally) adjusted ratings as a DCRatings.

    `matches` should be completed matches (training set). `adjustments_path`
    opts INTO the soft-factor layer for that league; leaving it None (the
    default) keeps historical fits and backtests purely quantitative.
    """
    base = dc_fit(matches, as_of=as_of, rho=rho, cfg=cfg, max_age_days=max_age_days)
    elo = elo_mod.compute_elo(matches, cfg=cfg)

    # Union of teams known to either model (Elo-only teams get a pure-prior rating).
    teams = sorted(set(base.teams) | set(elo))
    att = {t: base.attack.get(t, 0.0) for t in teams}
    dfn = {t: base.defense.get(t, 0.0) for t in teams}
    n_eff = {t: base.n_eff.get(t, 0.0) for t in teams}

    slope_a, slope_d, emean = _elo_to_rating_slopes(att, dfn, elo, n_eff, teams)

    blended_att, blended_def = {}, {}
    for t in teams:
        ec = elo.get(t, emean) - emean
        prior_a = slope_a * ec
        prior_d = slope_d * ec
        w = n_eff[t] / (n_eff[t] + cfg.blend_k)
        blended_att[t] = w * att[t] + (1.0 - w) * prior_a
        blended_def[t] = w * dfn[t] + (1.0 - w) * prior_d

    # Re-centre so contrasts average to zero (keeps the intercept meaningful).
    a_mean = float(np.mean(list(blended_att.values())))
    d_mean = float(np.mean(list(blended_def.values())))
    blended_att = {t: v - a_mean for t, v in blended_att.items()}
    blended_def = {t: v - d_mean for t, v in blended_def.items()}

    if adjustments_path:
        apply_adjustments(blended_att, blended_def,
                          load_adjustments(adjustments_path), verbose=verbose)

    return DCRatings(
        attack=blended_att, defense=blended_def,
        intercept=base.intercept, home_adv=base.home_adv, rho=base.rho,
        n_eff=n_eff, teams=sorted(teams), as_of=base.as_of,
    )


def calibrate_to_tournament(ratings: DCRatings, played_wc: pd.DataFrame,
                            cfg=CONFIG) -> DCRatings:
    """Return ratings with `lambda_mult` set from played tournament games.

    r = exp(w * log(sum actual goals / sum model-predicted goals)), w = n/(n+k):
    an empirical-Bayes nudge of the scoring level toward what THIS tournament is
    actually producing. Only games dated before ratings.as_of enter, so walk-forward
    evaluation stays leakage-free. Note the predicted totals come from the current
    fit (which already saw those games), which shrinks the measured ratio toward 1
    -- i.e. this calibration is conservative by construction.
    """
    k = cfg.tournament_calib_k
    if k is None or played_wc.empty:
        return ratings
    g = played_wc[played_wc["date"] <= ratings.as_of]
    if g.empty:
        return ratings
    pred_total = 0.0
    for _, m in g.iterrows():
        lh, la = ratings.expected_goals(m["home_team"], m["away_team"],
                                        neutral=bool(m["neutral"]), cfg=cfg)
        pred_total += lh + la
    actual_total = float((g["home_score"] + g["away_score"]).sum())
    if pred_total <= 0 or actual_total <= 0:
        return ratings
    w = len(g) / (len(g) + k)
    mult = math.exp(w * math.log(actual_total / pred_total))
    return dataclasses.replace(ratings, lambda_mult=ratings.lambda_mult * mult)


def strength_table(ratings: DCRatings, cfg=CONFIG) -> pd.DataFrame:
    """A readable ranking: each team's expected goals vs a league-average opponent
    on a neutral field. `net` = attacking minus conceding => overall strength."""
    rows = []
    for t in ratings.teams:
        gf, ga = ratings.expected_goals(t, "__AVERAGE__", neutral=True, cfg=cfg)
        rows.append((t, ratings.attack[t], ratings.defense[t], gf, ga, gf - ga,
                     ratings.n_eff.get(t, 0.0)))
    df = pd.DataFrame(rows, columns=["team", "attack", "defense", "xg_for",
                                     "xg_against", "net", "n_eff"])
    return df.sort_values("net", ascending=False).reset_index(drop=True)
