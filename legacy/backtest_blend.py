"""
backtest_blend.py — walk-forward over ALL 2026 WC games to find the optimal Elo<->
attack/defense blend (BLEND_K). For each matchday it fits attack/defense on only the
PRIOR games, predicts that day, and scores several blend settings. Reports 1X2 and
totals (Over 2.5) accuracy so we can pick a blend that serves both result and goals
markets.
"""
from __future__ import annotations
import collections
import math
import numpy as np

from fit_team_strength import load_wc_matches, fit_attack_defense
from ratings import load_database, ELO_PER_GOAL, BASE_TOTAL_GOALS
from simulate import _joint_scoreline_pmf
from build_database import normalize_team, _norm_name

RIDGE = 3.0
MIN_TRAIN = 4
KS = [1, 2, 4, 6, 8, 10, 15, 25]


def probs(lh, la):
    j = _joint_scoreline_pmf(lh, la)
    n = j.shape[0]; I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (float(j[I > J].sum()), float(j[I == J].sum()), float(j[I < J].sum()),
            float(j[(I + J) >= 3].sum()))


def main():
    db = load_database()
    by_norm = {_norm_name(normalize_team(t.name)): t for t in db.teams.values()}
    dbteam = lambda n: by_norm.get(_norm_name(normalize_team(n)))

    allm = load_wc_matches()
    dates = sorted({m["date"] for m in allm})
    labels = ["pureElo"] + [f"K={k}" for k in KS] + ["pureData"]
    acc = {lab: {"ll": 0.0, "br": 0.0, "oll": 0.0, "n": 0} for lab in labels}

    for d in dates:
        train = [m for m in allm if m["date"] < d]
        test = [m for m in allm if m["date"] == d]
        fit = None
        if len(train) >= MIN_TRAIN:
            teams, att, deff, mu, home = fit_attack_defense(train, RIDGE)
            la_avg = float(np.exp(mu))
            attack = {t: float(np.exp(mu + att[i])) for i, t in enumerate(teams)}
            defense = {t: float(np.exp(mu - deff[i])) for i, t in enumerate(teams)}
            gp = collections.Counter()
            for m in train:
                gp[m["home"]] += 1; gp[m["away"]] += 1
            fit = (attack, defense, gp, la_avg)

        for m in test:
            th, ta = dbteam(m["home"]), dbteam(m["away"])
            if not th or not ta:
                continue
            sup = (th.elo - ta.elo) / ELO_PER_GOAL
            elo_h = (BASE_TOTAL_GOALS + sup) / 2 * (1 - (ta.gk_quality - 0.5) * 0.30)
            elo_a = (BASE_TOTAL_GOALS - sup) / 2 * (1 - (th.gk_quality - 0.5) * 0.30)
            ad = None
            if fit and m["home"] in fit[0] and m["away"] in fit[0]:
                attack, defense, gp, la_avg = fit
                ad_h = attack[m["home"]] * defense[m["away"]] / la_avg
                ad_a = attack[m["away"]] * defense[m["home"]] / la_avg
                g = (gp[m["home"]] + gp[m["away"]]) / 2
                ad = (ad_h, ad_a, g)
            hg, ag = m["hg"], m["ag"]
            y = 0 if hg > ag else (1 if hg == ag else 2)
            yo = 1 if hg + ag >= 3 else 0
            for lab in labels:
                if ad is None:
                    w = 0.0
                elif lab == "pureElo":
                    w = 0.0
                elif lab == "pureData":
                    w = 1.0
                else:
                    w = ad[2] / (ad[2] + float(lab[2:]))
                lh = (w * ad[0] + (1 - w) * elo_h) if ad else elo_h
                la = (w * ad[1] + (1 - w) * elo_a) if ad else elo_a
                ph, pd, pa, po = probs(max(lh, 0.05), max(la, 0.05))
                acc[lab]["ll"] += -math.log(max(1e-9, (ph, pd, pa)[y]))
                acc[lab]["br"] += (po - yo) ** 2
                acc[lab]["oll"] += -math.log(max(1e-9, po if yo else 1 - po))
                acc[lab]["n"] += 1

    n = acc["pureElo"]["n"]
    print(f"Walk-forward over {n} predicted WC games (matchday-by-matchday):\n")
    print(f"  {'blend':10s} {'1X2 logloss':>12} {'O2.5 Brier':>11} {'O2.5 logloss':>13}")
    rows = []
    for lab in labels:
        a = acc[lab]
        rows.append((lab, a["ll"] / a["n"], a["br"] / a["n"], a["oll"] / a["n"]))
        print(f"  {lab:10s} {a['ll']/a['n']:12.4f} {a['br']/a['n']:11.4f} {a['oll']/a['n']:13.4f}")
    best_1x2 = min(rows, key=lambda r: r[1])
    best_tot = min(rows, key=lambda r: r[3])
    print(f"\n  best for RESULT (1X2):  {best_1x2[0]}  (logloss {best_1x2[1]:.4f})")
    print(f"  best for TOTALS (O/U):  {best_tot[0]}  (logloss {best_tot[3]:.4f})")


if __name__ == "__main__":
    main()
