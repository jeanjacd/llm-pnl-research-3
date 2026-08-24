"""
fit_team_strength.py — estimate each team's ATTACK and DEFENSE strength from the
CURRENT World Cup results, and write them into teams.json.

Method (the mathematically correct one for match data):
  A Poisson maximum-likelihood "attack/defense" model (the Dixon-Coles framework):
      log E[goals by team t vs opponent o] = mu + home*I(t home) + att[t] - def[o]
  fit by penalized (ridge) MLE via Newton-Raphson. This is OPPONENT-ADJUSTED (beating
  a strong defense counts more than beating a weak one) and HOME-ADJUSTED -- unlike raw
  goals-for/against averages, which are confounded by schedule. Ridge regularization
  shrinks teams toward the league mean, which is essential with only ~2-3 games each.

Why this powers over/under: the fitted attack x defense gives a *matchup-specific*
expected total, so two even teams that are high-scoring produce a higher total than two
even defensive teams -- exactly what totals/BTTS markets need. ratings.py auto-uses
attack_rating/defense_rating when present.

Usage:
    python fit_team_strength.py                 # fit on 2026 WC, write teams.json
    python fit_team_strength.py --ridge 5 --dry-run
"""

from __future__ import annotations
import argparse
import csv
import json
import os
from datetime import datetime, timedelta

import numpy as np

from ratings import BASE_TOTAL_GOALS
from build_database import normalize_team, _norm_name

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_CSV = os.path.join(DATA_DIR, "incoming", "results.csv")
TEAMS_JSON = os.path.join(DATA_DIR, "teams.json")
DEFAULT_RIDGE = 4.0       # backtest_v2.py-validated shrinkage for the windowed fit
WINDOW_DAYS = 540         # recent-form window (backtest_v2.py: best totals Brier)
TOURNAMENT = "FIFA World Cup"
SEASON_PREFIX = "2026"


def _load_rows():
    with open(RESULTS_CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _parse(r):
    try:
        hg, ag = int(float(r["home_score"])), int(float(r["away_score"]))
        d = datetime.strptime(r["date"], "%Y-%m-%d")
    except (ValueError, TypeError, KeyError):
        return None
    return {"date": r.get("date", ""), "dt": d, "home": r["home_team"].strip(),
            "away": r["away_team"].strip(), "hg": hg, "ag": ag,
            "neutral": str(r.get("neutral", "")).strip().upper() in ("TRUE", "1")}


def load_recent_matches(window_days: int = WINDOW_DAYS) -> list[dict]:
    """All international matches within `window_days` of the most recent result.
    A larger, recent sample makes attack/defense (and thus totals) far more reliable
    than the 2-3 World Cup games alone -- validated leakage-free in backtest_v2.py."""
    ms = [m for m in (_parse(r) for r in _load_rows()) if m]
    if not ms:
        return []
    latest = max(m["dt"] for m in ms)
    cutoff = latest - timedelta(days=window_days)
    return [m for m in ms if m["dt"] >= cutoff]


def load_wc_matches() -> list[dict]:
    """Only the current World Cup games (kept for the matchday spot-check backtest)."""
    out = []
    for r in _load_rows():
        if (r.get("tournament", "").strip() != TOURNAMENT
                or not r.get("date", "").startswith(SEASON_PREFIX)):
            continue
        m = _parse(r)
        if m:
            out.append(m)
    return out


def fit_attack_defense(matches: list[dict], ridge: float):
    """Penalized-MLE Poisson attack/defense. Returns (teams, att, deff, mu, home)."""
    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    p = 2 + 2 * n                      # [mu, home, att(n), def(n)]
    A_OFF, D_OFF = 2, 2 + n

    rows, y = [], []
    for m in matches:
        h, a = idx[m["home"]], idx[m["away"]]
        hf = 0.0 if m["neutral"] else 1.0
        # home team's goals
        x = np.zeros(p); x[0] = 1; x[1] = hf; x[A_OFF + h] = 1; x[D_OFF + a] = -1
        rows.append(x); y.append(m["hg"])
        # away team's goals
        x = np.zeros(p); x[0] = 1; x[A_OFF + a] = 1; x[D_OFF + h] = -1
        rows.append(x); y.append(m["ag"])
    X = np.array(rows); y = np.array(y, dtype=float)

    R = np.zeros((p, p))
    R[1, 1] = ridge              # penalize home advantage (few non-neutral WC games)
    for k in range(A_OFF, p):
        R[k, k] = ridge          # penalize att/def (not the intercept mu)

    theta = np.zeros(p)
    for _ in range(100):         # Newton-Raphson on penalized Poisson NLL (convex)
        eta = np.clip(X @ theta, -10, 10)
        lam = np.exp(eta)
        grad = X.T @ (lam - y) + R @ theta
        H = X.T @ (X * lam[:, None]) + R
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.max(np.abs(step)) < 1e-9:
            break

    return teams, theta[A_OFF:A_OFF + n], theta[D_OFF:D_OFF + n], theta[0], theta[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    matches = load_recent_matches(WINDOW_DAYS)
    print(f"Fitting on {len(matches)} internationals in the last {WINDOW_DAYS} days "
          f"(ridge={args.ridge})")
    if len(matches) < 20:
        print("Too few recent matches -- check results.csv.")
        return

    teams, att, deff, mu, home = fit_attack_defense(matches, args.ridge)
    league_avg = float(np.exp(mu))   # fitted goals per team per match
    # Store TRUE goals-for / goals-against per match vs an average opponent.
    attack = {t: round(float(np.exp(mu + att[i])), 3) for i, t in enumerate(teams)}
    defense = {t: round(float(np.exp(mu - deff[i])), 3) for i, t in enumerate(teams)}
    games = {}
    for m in matches:
        games[m["home"]] = games.get(m["home"], 0) + 1
        games[m["away"]] = games.get(m["away"], 0) + 1

    print(f"home-field effect: x{np.exp(home):.2f} goals  |  league avg/team: {np.exp(mu):.2f}\n")
    print(f"{'team':22s} {'GP':>2}  {'attack':>6} {'defense':>7}  (goals for / against per match)")
    for t in sorted(teams, key=lambda x: -attack[x]):
        print(f"  {t:20s} {games[t]:>2}  {attack[t]:>6.2f} {defense[t]:>7.2f}")

    # Write into teams.json (matched by normalized name).
    with open(TEAMS_JSON, encoding="utf-8") as f:
        tdata = json.load(f)
    tdata.setdefault("_meta", {})["league_avg_gf"] = round(league_avg, 3)
    by_norm = {}
    for t, a in attack.items():
        by_norm[_norm_name(normalize_team(t))] = (a, defense[t], games.get(t, 0))
    written = 0
    for tj in tdata["teams"]:
        hit = by_norm.get(_norm_name(normalize_team(tj["name"])))
        if hit:
            tj["attack_rating"], tj["defense_rating"], tj["wc_games"] = hit
            tj["ratings_source"] = f"poisson {WINDOW_DAYS}d window (ridge={args.ridge})"
            written += 1
    print(f"\nMatched + set ratings on {written} teams in teams.json.")
    if args.dry_run:
        print("[dry-run] not writing.")
        return
    with open(TEAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(tdata, f, indent=2, ensure_ascii=False)
    print("Wrote teams.json. ratings.py now uses real attack/defense for these teams.")


if __name__ == "__main__":
    main()
