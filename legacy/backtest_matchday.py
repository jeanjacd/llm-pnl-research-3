"""
backtest_matchday.py — out-of-sample check of the Elo<->attack/defense blend on the most
recent World Cup matchday. Fits attack/defense on everything BEFORE the cutoff date,
predicts that day's games, and scores several blend weights so we can see which the data
prefers. (Small sample -- a spot check, not a definitive tune.)
"""
from __future__ import annotations
import collections
import math
import numpy as np

from fit_team_strength import load_wc_matches, fit_attack_defense
from ratings import (load_database, ELO_PER_GOAL, BASE_TOTAL_GOALS, HOME_FIELD_ADVANTAGE)
from simulate import _joint_scoreline_pmf
from build_database import normalize_team, _norm_name

CUTOFF = "2026-06-24"     # predict games ON this date using only earlier games
RIDGE = 3.0


def probs(lam_h, lam_a):
    j = _joint_scoreline_pmf(lam_h, lam_a)
    n = j.shape[0]; I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (float(j[I > J].sum()), float(j[I == J].sum()), float(j[I < J].sum()),
            float(j[(I + J) >= 3].sum()))


def main():
    db = load_database()
    db_by_norm = {_norm_name(normalize_team(t.name)): t for t in db.teams.values()}

    def dbteam(name):
        return db_by_norm.get(_norm_name(normalize_team(name)))

    allm = load_wc_matches()
    train = [m for m in allm if m["date"] < CUTOFF]
    test = [m for m in allm if m["date"] == CUTOFF]
    print(f"train: {len(train)} games before {CUTOFF}  |  test: {len(test)} games on {CUTOFF}\n")

    teams, att, deff, mu, home = fit_attack_defense(train, RIDGE)
    league_avg = float(np.exp(mu))
    attack = {t: float(np.exp(mu + att[i])) for i, t in enumerate(teams)}
    defense = {t: float(np.exp(mu - deff[i])) for i, t in enumerate(teams)}
    gp = collections.Counter()
    for m in train:
        gp[m["home"]] += 1; gp[m["away"]] += 1

    def lambdas(m, w_override=None, blend_k=6.0):
        th, ta = dbteam(m["home"]), dbteam(m["away"])
        # Elo lambdas (neutral)
        diff = th.elo - ta.elo
        sup = diff / ELO_PER_GOAL
        elo_h = (BASE_TOTAL_GOALS + sup) / 2 * (1 - (ta.gk_quality - 0.5) * 0.30)
        elo_a = (BASE_TOTAL_GOALS - sup) / 2 * (1 - (th.gk_quality - 0.5) * 0.30)
        if m["home"] not in attack or m["away"] not in attack:
            return elo_h, elo_a
        ad_h = attack[m["home"]] * defense[m["away"]] / league_avg
        ad_a = attack[m["away"]] * defense[m["home"]] / league_avg
        if w_override is not None:
            w = w_override
        else:
            g = (gp[m["home"]] + gp[m["away"]]) / 2
            w = g / (g + blend_k)
        return w * ad_h + (1 - w) * elo_h, w * ad_a + (1 - w) * elo_a

    def score(label, lam_fn):
        ll = brier = 0.0
        for m in test:
            lh, la = lam_fn(m)
            ph, pd, pa, po = probs(lh, la)
            hg, ag = m["hg"], m["ag"]
            y = 0 if hg > ag else (1 if hg == ag else 2)
            ll += -math.log(max(1e-9, (ph, pd, pa)[y]))
            yo = 1 if hg + ag >= 3 else 0
            brier += (po - yo) ** 2
        n = len(test)
        print(f"  {label:24s} 1X2 log-loss {ll/n:.3f} | O/U2.5 Brier {brier/n:.3f}")

    print("Scoring yesterday's matchday under different blend settings:")
    score("pure Elo (w=0)", lambda m: lambdas(m, w_override=0.0))
    for k in (3, 6, 10):
        score(f"blend BLEND_K={k}", lambda m, k=k: lambdas(m, blend_k=k))
    score("pure attack/def (w=1)", lambda m: lambdas(m, w_override=1.0))
    print("\n(6 games is a tiny sample -- read this as directional, not a precise tune.)")


if __name__ == "__main__":
    main()
