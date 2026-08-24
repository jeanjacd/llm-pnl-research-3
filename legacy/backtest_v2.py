"""
backtest_v2.py — LEAKAGE-FREE evaluation + totals improvement.

Fixes the leakage in backtest_blend.py (which used a post-tournament Elo snapshot) by
rebuilding Elo WALK-FORWARD from the full international history, so each WC game is
predicted using only prior information. It also estimates attack/defense from a longer
RECENT WINDOW of internationals (not just 2-3 WC games) -- the real lever for predicting
total goals -- and sweeps the blend weight to minimize totals Brier.

Reports honest 1X2 and Over-2.5 numbers vs baselines, and the best blend for each.
"""
from __future__ import annotations
import collections
import csv
import math
import os
from datetime import datetime, timedelta

import numpy as np

from backtest import elo_expected, elo_update, INIT_ELO, DEF_K, DEF_HFA
from fit_team_strength import fit_attack_defense
from ratings import ELO_PER_GOAL, BASE_TOTAL_GOALS
from simulate import _joint_scoreline_pmf

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "data", "incoming", "results.csv")
WINDOW_DAYS = 540          # recent-form window for the attack/defense fit
RIDGE = 4.0
KS = [2, 4, 6, 10, 15, 25]


def load_all():
    with open(RESULTS_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            hg, ag = int(float(r["home_score"])), int(float(r["away_score"]))
            d = datetime.strptime(r["date"], "%Y-%m-%d")
        except (ValueError, TypeError, KeyError):
            continue
        out.append({"d": d, "home": r["home_team"].strip(), "away": r["away_team"].strip(),
                    "hg": hg, "ag": ag,
                    "neutral": str(r.get("neutral", "")).strip().upper() in ("TRUE", "1"),
                    "wc": r.get("tournament", "") == "FIFA World Cup" and r["date"].startswith("2026")})
    out.sort(key=lambda m: m["d"])
    return out


def probs(lh, la):
    j = _joint_scoreline_pmf(max(lh, 0.05), max(la, 0.05))
    n = j.shape[0]; I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (float(j[I > J].sum()), float(j[I == J].sum()), float(j[I < J].sum()),
            float(j[(I + J) >= 3].sum()))


def main():
    allm = load_all()
    test = [m for m in allm if m["wc"]]
    print(f"{len(allm)} total matches | {len(test)} WC2026 test games | "
          f"attack/def window {WINDOW_DAYS}d, ridge {RIDGE}\n")

    # 1) Walk-forward Elo over EVERYTHING -> pre-game Elo for each test game (leakage-free).
    elo = {}
    for m in allm:
        if m["wc"]:
            m["pre"] = (elo.get(m["home"], INIT_ELO), elo.get(m["away"], INIT_ELO))
        elo_update(elo, m["home"], m["away"], m["hg"], m["ag"], DEF_K, DEF_HFA)

    # 2) Fit attack/defense on the recent window ending just before each WC matchday.
    fits = {}
    for d in sorted({m["d"] for m in test}):
        win = [m for m in allm if d - timedelta(days=WINDOW_DAYS) <= m["d"] < d]
        teams, att, deff, mu, home = fit_attack_defense(win, RIDGE)
        gp = collections.Counter()
        for m in win:
            gp[m["home"]] += 1; gp[m["away"]] += 1
        fits[d] = ({t: float(np.exp(mu + att[i])) for i, t in enumerate(teams)},
                   {t: float(np.exp(mu - deff[i])) for i, t in enumerate(teams)},
                   float(np.exp(mu)), gp)

    labels = ["pureElo"] + [f"K={k}" for k in KS] + ["pureData"]
    acc = {l: {"ll": 0.0, "br": 0.0, "oll": 0.0} for l in labels}
    over_rate = sum(1 for m in test if m["hg"] + m["ag"] >= 3) / len(test)

    for m in test:
        eh, ea = m["pre"]
        sup = (eh - ea) / ELO_PER_GOAL
        elo_h = (BASE_TOTAL_GOALS + sup) / 2
        elo_a = (BASE_TOTAL_GOALS - sup) / 2
        attack, defense, la_avg, gp = fits[m["d"]]
        ad = None
        if m["home"] in attack and m["away"] in attack:
            ad = (attack[m["home"]] * defense[m["away"]] / la_avg,
                  attack[m["away"]] * defense[m["home"]] / la_avg,
                  (gp[m["home"]] + gp[m["away"]]) / 2)
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
            lh = w * ad[0] + (1 - w) * elo_h if ad else elo_h
            la = w * ad[1] + (1 - w) * elo_a if ad else elo_a
            ph, pd, pa, po = probs(lh, la)
            acc[lab]["ll"] += -math.log(max(1e-9, (ph, pd, pa)[y]))
            acc[lab]["br"] += (po - yo) ** 2
            acc[lab]["oll"] += -math.log(max(1e-9, po if yo else 1 - po))

    n = len(test)
    print("Baselines:   1X2 uniform logloss 1.099 | O2.5 coin-flip Brier 0.250 / logloss 0.693")
    print(f"             O2.5 base-rate (predict {over_rate:.0%}) Brier "
          f"{over_rate*(1-over_rate):.3f}\n")
    print(f"  {'blend':10s} {'1X2 logloss':>12} {'O2.5 Brier':>11} {'O2.5 logloss':>13}")
    rows = []
    for lab in labels:
        a = acc[lab]
        rows.append((lab, a["ll"]/n, a["br"]/n, a["oll"]/n))
        print(f"  {lab:10s} {a['ll']/n:12.4f} {a['br']/n:11.4f} {a['oll']/n:13.4f}")
    b1 = min(rows, key=lambda r: r[1]); bt = min(rows, key=lambda r: r[2])
    print(f"\n  best RESULT (1X2):  {b1[0]} (logloss {b1[1]:.4f})")
    print(f"  best TOTALS (Brier): {bt[0]} (Brier {bt[2]:.4f}, logloss {bt[3]:.4f})")


if __name__ == "__main__":
    main()
