"""
dixon_coles.py
==============
The statistical core: a time-weighted Dixon-Coles bivariate-Poisson model fit by
penalised maximum likelihood on historical international results.

Model
-----
For a match between home team i and away team j:

    log lambda_home = c + h * is_home + attack_i - defense_j
    log lambda_away = c +              attack_j - defense_i

    home_goals ~ Poisson(lambda_home),  away_goals ~ Poisson(lambda_away)

with the Dixon-Coles low-score dependence correction tau(x, y) applied to the four
corner scorelines so that 0-0 and 1-1 are (correctly) more likely and 1-0/0-1 less
likely than independent Poisson predicts. The dependence parameter `rho` is
NEGATIVE -- this is the crux the legacy code got backwards.

Why this design
---------------
* Time weighting (exponential, by half-life) makes recent matches dominate, so
  "recent form" is captured by the fit itself rather than bolted on afterwards.
* Match-importance weights downweight friendlies relative to competitive games.
* An L2 (ridge) penalty on attack/defense both identifies the model (otherwise the
  intercept and the rating levels trade off) and shrinks data-poor teams toward the
  average -- a Bayesian-flavoured regulariser. The Elo prior in model/ratings.py adds
  a second, team-specific shrinkage on top.
* Analytic gradients -> a fast, reliable L-BFGS-B fit even with a few hundred teams.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config import CONFIG


# --- Dixon-Coles tau on the four low-score cells ------------------------------
def dc_tau(x, y, lam_h, lam_a, rho):
    """Vectorised Dixon-Coles correction. x, y are integer goal arrays."""
    t = np.ones_like(lam_h, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t = np.where(m00, 1.0 - lam_h * lam_a * rho, t)
    t = np.where(m01, 1.0 + lam_h * rho, t)
    t = np.where(m10, 1.0 + lam_a * rho, t)
    t = np.where(m11, 1.0 - rho, t)
    return t


class UnknownTeamError(KeyError):
    """A fixture referenced a team with no fitted rating.

    Silently substituting the league average here is the single most dangerous
    failure mode in the system: a promoted club, a provider rename, or a
    name-matching miss all produce a CONFIDENT average-strength forecast with
    no signal that identity resolution failed. Supported predictions therefore
    fail closed; callers that genuinely want the average must opt in.
    """


# Sentinel opponent used to express "versus a league-average side" (strength
# tables, reporting). It is explicitly allowed where a real team is not.
AVERAGE_TEAM = "__AVERAGE__"


@dataclass
class DCRatings:
    """Fitted ratings. attack/defense are per-team contrasts (mean ~ 0)."""
    attack: dict[str, float]
    defense: dict[str, float]
    intercept: float
    home_adv: float
    rho: float
    n_eff: dict[str, float]          # time-weighted match count per team (for shrinkage)
    teams: list[str]
    as_of: pd.Timestamp
    lambda_mult: float = 1.0         # in-tournament goal calibration (ratings.py)

    def knows(self, team: str) -> bool:
        return team in self.attack and team in self.defense

    def expected_goals(self, home: str, away: str, neutral: bool = True,
                       cfg=CONFIG, allow_unknown: bool = False
                       ) -> tuple[float, float]:
        """(lambda_home, lambda_away) for a fixture.

        Raises UnknownTeamError if either side has no fitted rating, unless
        `allow_unknown=True` (which restores the old average-substitution and
        must only be used for exploratory/reporting paths, never for a
        forecast that can reach a market).
        """
        if not allow_unknown:
            for team in (home, away):
                if team != AVERAGE_TEAM and not self.knows(team):
                    raise UnknownTeamError(
                        "no fitted rating for %r (fitted through %s, %d teams). "
                        "Refusing to substitute the league average."
                        % (team, self.as_of.date() if hasattr(self.as_of, "date")
                           else self.as_of, len(self.attack)))
        a_h = self.attack.get(home, 0.0)
        a_a = self.attack.get(away, 0.0)
        d_h = self.defense.get(home, 0.0)
        d_a = self.defense.get(away, 0.0)
        h = 0.0 if neutral else self.home_adv
        lam_h = self.lambda_mult * math.exp(self.intercept + h + a_h - d_a)
        lam_a = self.lambda_mult * math.exp(self.intercept + a_a - d_h)
        lo, hi = cfg.lambda_floor, cfg.lambda_cap
        return min(max(lam_h, lo), hi), min(max(lam_a, lo), hi)


# --- Weights -----------------------------------------------------------------
def match_weights(matches: pd.DataFrame, as_of: pd.Timestamp, cfg=CONFIG) -> np.ndarray:
    """Exponential time-decay x competition-importance weight per match."""
    age_days = (as_of - matches["date"]).dt.total_seconds().to_numpy() / 86400.0
    age_days = np.clip(age_days, 0.0, None)
    decay = np.exp(-math.log(2.0) * age_days / cfg.half_life_days)
    importance = matches["tier"].map(cfg.importance_weights).fillna(0.5).to_numpy()
    return decay * importance


# --- Fit ---------------------------------------------------------------------
def fit(matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
        rho: float | None = None, ridge: float = 1.0, cfg=CONFIG,
        max_age_days: float | None = None) -> DCRatings:
    """Penalised-MLE fit of attack/defense/intercept/home on weighted matches.

    rho is held fixed (cross-validated in eval/tune.py); pass None to use the
    config default. `ridge` is the L2 strength on attack/defense. `max_age_days`
    optionally hard-drops very old matches whose time weight is negligible anyway.
    """
    rho = cfg.rho if rho is None else rho
    if as_of is None:
        as_of = matches["date"].max()

    m = matches
    if max_age_days is not None:
        m = m[(as_of - m["date"]).dt.days <= max_age_days]
    m = m.reset_index(drop=True)

    teams = sorted(set(m["home_team"]) | set(m["away_team"]))
    idx = {t: k for k, t in enumerate(teams)}
    T = len(teams)

    hi = m["home_team"].map(idx).to_numpy()
    ai = m["away_team"].map(idx).to_numpy()
    x = m["home_score"].to_numpy().astype(int)
    y = m["away_score"].to_numpy().astype(int)
    is_home = (~m["neutral"].to_numpy()).astype(float)   # 1 if a real home side
    w = match_weights(m, as_of, cfg)

    # time-weighted match counts (for downstream Elo shrinkage)
    n_eff = np.zeros(T)
    np.add.at(n_eff, hi, w)
    np.add.at(n_eff, ai, w)

    # parameter vector: [attack(T), defense(T), intercept, home_adv]
    def unpack(p):
        return p[:T], p[T:2 * T], p[2 * T], p[2 * T + 1]

    def neg_ll_and_grad(p):
        att, dfn, c, home = unpack(p)
        eta_h = c + home * is_home + att[hi] - dfn[ai]
        eta_a = c + att[ai] - dfn[hi]
        lam_h = np.exp(eta_h)
        lam_a = np.exp(eta_a)

        # Poisson log-likelihood (drop constant -log(x!) -> irrelevant to optimisation)
        ll_pois = x * eta_h - lam_h + y * eta_a - lam_a

        # Dixon-Coles correction
        tau = dc_tau(x, y, lam_h, lam_a, rho)
        tau = np.clip(tau, 1e-12, None)
        ll = ll_pois + np.log(tau)
        nll = -np.sum(w * ll) + ridge * (np.sum(att ** 2) + np.sum(dfn ** 2))

        # --- gradient ---
        # d log tau / d eta_h, d eta_a  (only corner cells contribute)
        dtau_h = np.zeros_like(lam_h)
        dtau_a = np.zeros_like(lam_a)
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        dtau_h = np.where(m00, -lam_h * lam_a * rho / tau, dtau_h)
        dtau_h = np.where(m01, lam_h * rho / tau, dtau_h)
        dtau_a = np.where(m00, -lam_h * lam_a * rho / tau, dtau_a)
        dtau_a = np.where(m10, lam_a * rho / tau, dtau_a)

        g_h = w * ((x - lam_h) + dtau_h)     # dLL/d eta_h
        g_a = w * ((y - lam_a) + dtau_a)     # dLL/d eta_a

        grad = np.zeros_like(p)
        # attack: home team via eta_h, away team via eta_a
        np.add.at(grad, hi, g_h)
        np.add.at(grad, ai, g_a)
        # defense: -d_away in eta_h, -d_home in eta_a
        np.add.at(grad, T + ai, -g_h)
        np.add.at(grad, T + hi, -g_a)
        # intercept and home advantage
        grad[2 * T] = np.sum(g_h + g_a)
        grad[2 * T + 1] = np.sum(g_h * is_home)
        # convert to NEGATIVE-LL gradient and add ridge derivative
        grad = -grad
        grad[:T] += 2.0 * ridge * att
        grad[T:2 * T] += 2.0 * ridge * dfn
        return nll, grad

    p0 = np.zeros(2 * T + 2)
    p0[2 * T] = math.log(1.35)     # sensible base log scoring rate
    p0[2 * T + 1] = 0.25           # ~typical home advantage in log-goals

    res = minimize(neg_ll_and_grad, p0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})

    att, dfn, c, home = unpack(res.x)
    # Centre contrasts for interpretability (ridge already keeps them near 0).
    att = att - att.mean()
    dfn = dfn - dfn.mean()

    return DCRatings(
        attack={t: float(att[idx[t]]) for t in teams},
        defense={t: float(dfn[idx[t]]) for t in teams},
        intercept=float(c), home_adv=float(home), rho=float(rho),
        n_eff={t: float(n_eff[idx[t]]) for t in teams},
        teams=teams, as_of=pd.Timestamp(as_of),
    )
