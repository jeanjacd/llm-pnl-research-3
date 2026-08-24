"""
backtest.py — measure and sharpen the model.
============================================

Leakage-free WALK-FORWARD backtest: replays matches in date order, predicts each one
using only prior information (a self-maintained Elo), then updates Elo from the result.
This tells you whether the modeling approach is actually calibrated and how to tune it.

Reports:
  * Calibration (reliability) — when it says 30%, does it happen ~30%?
  * Brier score + log-loss vs a base-rate baseline (lower is better)
  * Grid search over ELO_PER_GOAL / BASE_TOTAL_GOALS / K -> best constants for ratings.py
  * If results.csv has market odds, a model-trust (blend) sweep -> best MODEL_TRUST

Data:
  data/incoming/results.csv with columns (header names matched flexibly):
     date, home, away, home_score, away_score
     [optional] mkt_home, mkt_draw, mkt_away  (implied probabilities or decimal odds)
  Or generate a real set:  python backtest.py --from-statsbomb

Probabilities use the same analytic Dixon-Coles model as the live simulator.
"""

from __future__ import annotations
import argparse
import csv
import math
import os

import numpy as np

from simulate import _joint_scoreline_pmf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INCOMING_DIR = os.path.join(DATA_DIR, "incoming")
RESULTS_CSV = os.path.join(INCOMING_DIR, "results.csv")

# Defaults (match ratings.py); the grid search proposes better values.
DEF_ELO_PER_GOAL = 150.0
DEF_BASE_TOTAL = 2.70
DEF_K = 30.0
DEF_HFA = 0.0           # neutral (World Cup); set ~60 for true home games
INIT_ELO = 1500.0


# ---------------------------------------------------------------------------
# model probabilities from Elo (analytic, fast)
# ---------------------------------------------------------------------------
def elo_expected(rh: float, ra: float, hfa: float = 0.0) -> float:
    return 1.0 / (1.0 + 10 ** (-(rh - ra + hfa) / 400.0))


def match_probs(rh: float, ra: float, elo_per_goal: float, base_total: float,
                hfa: float = 0.0) -> dict:
    diff = (rh - ra) + hfa
    sup = diff / elo_per_goal
    lam_h = max((base_total + sup) / 2.0, 0.05)
    lam_a = max((base_total - sup) / 2.0, 0.05)
    joint = _joint_scoreline_pmf(lam_h, lam_a)
    n = joint.shape[0]
    idx = np.arange(n)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    return {
        "p_home": float(joint[I > J].sum()),
        "p_draw": float(joint[I == J].sum()),
        "p_away": float(joint[I < J].sum()),
        "p_over25": float(joint[(I + J) >= 3].sum()),
        "p_btts": float(joint[(I >= 1) & (J >= 1)].sum()),
    }


def elo_update(elo: dict, home: str, away: str, hg: int, ag: int, k: float, hfa: float):
    gd = abs(hg - ag)
    g_mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
    s_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    e_home = elo_expected(elo.get(home, INIT_ELO), elo.get(away, INIT_ELO), hfa)
    delta = k * g_mult * (s_home - e_home)
    elo[home] = elo.get(home, INIT_ELO) + delta
    elo[away] = elo.get(away, INIT_ELO) - delta


def walk_forward(matches: list[dict], elo_per_goal: float, base_total: float,
                 k: float, hfa: float = DEF_HFA) -> list[dict]:
    """Return per-match prediction+outcome records (leakage-free)."""
    elo: dict[str, float] = {}
    records = []
    for m in matches:
        rh = elo.get(m["home"], INIT_ELO)
        ra = elo.get(m["away"], INIT_ELO)
        p = match_probs(rh, ra, elo_per_goal, base_total, hfa)
        hg, ag = m["hg"], m["ag"]
        records.append({
            **p,
            "y_home": int(hg > ag), "y_draw": int(hg == ag), "y_away": int(hg < ag),
            "y_over25": int(hg + ag >= 3), "y_btts": int(hg >= 1 and ag >= 1),
            "mkt": m.get("mkt"),
        })
        elo_update(elo, m["home"], m["away"], hg, ag, k, hfa)
    return records


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _clip(p):
    return min(1 - 1e-12, max(1e-12, p))


def brier_logloss_1x2(records: list[dict]) -> tuple[float, float]:
    brier = ll = 0.0
    for r in records:
        ps = (r["p_home"], r["p_draw"], r["p_away"])
        ys = (r["y_home"], r["y_draw"], r["y_away"])
        brier += sum((p - y) ** 2 for p, y in zip(ps, ys))
        ll += -math.log(_clip(ps[ys.index(1)]))
    n = len(records)
    return brier / n, ll / n


def baseline_logloss(records: list[dict]) -> float:
    n = len(records)
    fh = sum(r["y_home"] for r in records) / n
    fd = sum(r["y_draw"] for r in records) / n
    fa = sum(r["y_away"] for r in records) / n
    ll = 0.0
    for r in records:
        ys = (r["y_home"], r["y_draw"], r["y_away"])
        ll += -math.log(_clip((fh, fd, fa)[ys.index(1)]))
    return ll / n


def calibration_table(records: list[dict], bins: int = 10) -> list[tuple]:
    """Pool all 1X2 outcome probabilities as binary events -> reliability diagram."""
    pairs = []
    for r in records:
        pairs += [(r["p_home"], r["y_home"]), (r["p_draw"], r["y_draw"]),
                  (r["p_away"], r["y_away"])]
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        grp = [(p, y) for p, y in pairs if (lo <= p < hi or (b == bins - 1 and p == 1))]
        if grp:
            pred = sum(p for p, _ in grp) / len(grp)
            obs = sum(y for _, y in grp) / len(grp)
            out.append((lo, hi, len(grp), pred, obs))
    return out


# ---------------------------------------------------------------------------
# tuning
# ---------------------------------------------------------------------------
def grid_search(matches, hfa=DEF_HFA):
    grid_epg = [110, 130, 150, 170, 190]
    grid_total = [2.4, 2.6, 2.8, 3.0]
    grid_k = [20, 30, 40]
    results = []
    for epg in grid_epg:
        for tot in grid_total:
            for k in grid_k:
                recs = walk_forward(matches, epg, tot, k, hfa)
                _, ll = brier_logloss_1x2(recs)
                results.append((ll, epg, tot, k))
    results.sort()
    return results


def blend_sweep(records: list[dict]):
    """Needs market probs. Returns [(w, logloss)] for w in 0..1."""
    have = [r for r in records if r.get("mkt")]
    if not have:
        return []
    out = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        ll = 0.0
        for r in have:
            mh, md, ma = r["mkt"]
            ph = w * r["p_home"] + (1 - w) * mh
            pd = w * r["p_draw"] + (1 - w) * md
            pa = w * r["p_away"] + (1 - w) * ma
            s = ph + pd + pa
            ph, pd, pa = ph / s, pd / s, pa / s
            ys = (r["y_home"], r["y_draw"], r["y_away"])
            ll += -math.log(_clip((ph, pd, pa)[ys.index(1)]))
        out.append((w, ll / len(have)))
    return out


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def _find(fns, *c):
    low = {f.lower().strip(): f for f in fns if f}
    for x in c:
        if x.lower() in low:
            return low[x.lower()]
    for x in c:
        for f in fns:
            if f and x.lower() in f.lower():
                return f
    return None


def _odds_to_prob(h, d, a):
    """Accept implied probs (sum~1) or decimal odds; return normalized (h,d,a) or None."""
    try:
        h, d, a = float(h), float(d), float(a)
    except (ValueError, TypeError):
        return None
    if h <= 0 or d <= 0 or a <= 0:
        return None
    if h > 1.5 or a > 1.5:            # looks like decimal odds
        h, d, a = 1 / h, 1 / d, 1 / a
    s = h + d + a
    return (h / s, d / s, a / s) if s > 0 else None


def load_results(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    fns = list(rows[0].keys())
    c_date = _find(fns, "date", "match_date")
    c_h = _find(fns, "home", "home_team")
    c_a = _find(fns, "away", "away_team")
    c_hg = _find(fns, "home_score", "home_goals", "hg", "fthg")
    c_ag = _find(fns, "away_score", "away_goals", "ag", "ftag")
    c_mh = _find(fns, "mkt_home", "odds_home", "psh", "b365h")
    c_md = _find(fns, "mkt_draw", "odds_draw", "psd", "b365d")
    c_ma = _find(fns, "mkt_away", "odds_away", "psa", "b365a")
    out = []
    for r in rows:
        try:
            hg, ag = int(float(r[c_hg])), int(float(r[c_ag]))
        except (ValueError, TypeError, KeyError):
            continue
        rec = {"date": r.get(c_date, ""), "home": (r.get(c_h) or "").strip(),
               "away": (r.get(c_a) or "").strip(), "hg": hg, "ag": ag, "mkt": None}
        if c_mh and c_md and c_ma:
            rec["mkt"] = _odds_to_prob(r.get(c_mh), r.get(c_md), r.get(c_ma))
        if rec["home"] and rec["away"]:
            out.append(rec)
    out.sort(key=lambda x: x["date"])
    return out


def fetch_statsbomb_results():
    """Build a real international results.csv from StatsBomb World Cups (2018+2022)."""
    import requests
    base = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    rows = []
    for comp, season in [(43, 3), (43, 106)]:  # WC 2018, 2022
        ms = requests.get(f"{base}/matches/{comp}/{season}.json", timeout=60).json()
        for m in ms:
            rows.append({
                "date": m.get("match_date", ""),
                "home": m["home_team"]["home_team_name"],
                "away": m["away_team"]["away_team_name"],
                "home_score": m["home_score"], "away_score": m["away_score"],
            })
    rows.sort(key=lambda r: r["date"])
    os.makedirs(INCOMING_DIR, exist_ok=True)
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "home", "away", "home_score", "away_score"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} matches to {RESULTS_CSV}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-statsbomb", action="store_true",
                    help="generate data/incoming/results.csv from WC 2018+2022")
    ap.add_argument("--hfa", type=float, default=DEF_HFA,
                    help="home-field Elo bonus (0 for neutral World Cup)")
    args = ap.parse_args()

    if args.from_statsbomb:
        fetch_statsbomb_results()

    if not os.path.exists(RESULTS_CSV):
        print(f"No results file at {RESULTS_CSV}. Run with --from-statsbomb or add your own.")
        return
    matches = load_results(RESULTS_CSV)
    n_odds = sum(1 for m in matches if m["mkt"])
    print(f"Loaded {len(matches)} matches ({n_odds} played, with market odds)")
    if n_odds == 0:
        with open(RESULTS_CSV, encoding="utf-8-sig", newline="") as f:
            pend = sum(1 for r in csv.DictReader(f) if (r.get("mkt_home") or "").strip())
        if pend:
            print(f"  note: {pend} rows have odds but no result yet (upcoming games) -- "
                  f"they'll feed the blend sweep automatically once those matches are played.")
    print()
    if len(matches) < 20:
        print("Too few matches for a meaningful backtest (want hundreds).")

    recs = walk_forward(matches, DEF_ELO_PER_GOAL, DEF_BASE_TOTAL, DEF_K, args.hfa)
    brier, ll = brier_logloss_1x2(recs)
    base = baseline_logloss(recs)
    print(f"Default model (EPG={DEF_ELO_PER_GOAL}, total={DEF_BASE_TOTAL}, K={DEF_K}):")
    print(f"  1X2 Brier = {brier:.4f}   log-loss = {ll:.4f}   "
          f"(base-rate baseline = {base:.4f})")
    print(f"  -> model {'BEATS' if ll < base else 'does NOT beat'} the naive baseline\n")

    print("Calibration (predicted vs observed, pooled 1X2):")
    print("  bin        n     pred    obs")
    for lo, hi, n, pred, obs in calibration_table(recs):
        flag = "  <-- off" if abs(pred - obs) > 0.06 else ""
        print(f"  {lo:.1f}-{hi:.1f}  {n:5d}   {pred:.3f}  {obs:.3f}{flag}")

    print("\nGrid search (top 5 by log-loss) -> paste the best into ratings.py:")
    g = grid_search(matches, args.hfa)
    for ll_, epg, tot, k in g[:5]:
        print(f"  log-loss {ll_:.4f}  ELO_PER_GOAL={epg}  BASE_TOTAL_GOALS={tot}  K={k}")
    best = g[0]
    print(f"  BEST -> ELO_PER_GOAL={best[1]}, BASE_TOTAL_GOALS={best[2]} "
          f"(set these in ratings.py; K is a backtest-only param)")

    # Show calibration AT the tuned constants (the table above used the script's
    # internal defaults, so it looks worse than your tuned ratings.py actually is).
    recs_best = walk_forward(matches, best[1], best[2], best[3], args.hfa)
    print("\nCalibration AT BEST constants (what your tuned model does):")
    print("  bin        n     pred    obs")
    for lo, hi, n, pred, obs in calibration_table(recs_best):
        flag = "  <-- off" if abs(pred - obs) > 0.06 else ""
        print(f"  {lo:.1f}-{hi:.1f}  {n:5d}   {pred:.3f}  {obs:.3f}{flag}")

    sweep = blend_sweep(recs)
    if sweep:
        print("\nModel-trust (blend) sweep [needs market odds] -> set MODEL_TRUST in main.py:")
        for w, ll_ in sweep:
            print(f"  MODEL_TRUST={w:.2f}  log-loss={ll_:.4f}")
        market_ll = next((ll_ for w, ll_ in sweep if w == 0.0), 1.0)
        if market_ll < 0.6:
            print(f"  [!] WARNING: pure-market log-loss is {market_ll:.3f}, which is "
                  f"impossibly good for real pre-match odds (~0.95+).")
            print(f"      Your odds look SETTLED / post-match (result-aware), so this "
                  f"sweep is UNRELIABLE -- ignore it and keep your current MODEL_TRUST.")
        else:
            bw = min(sweep, key=lambda x: x[1])[0]
            print(f"  BEST MODEL_TRUST = {bw:.2f}  "
                  f"(1.0=pure model, 0=pure market; lower means trust the market more)")
    else:
        print("\n(No market odds in results.csv -> cannot tune MODEL_TRUST. Add "
              "mkt_home/mkt_draw/mkt_away columns to enable the blend sweep.)")


if __name__ == "__main__":
    main()
