"""
bold_play_shortlist.py
======================
Pull live Kalshi 2026 World Cup contracts, price every mappable market with the
wc2026 model's EXACT scoreline distribution, and rank candidates for a bold-play
strategy (reach a 3x target in the fewest bets):

  Section A -- SINGLE-SHOT ~3x: one contract priced ~25-43c whose model probability
               beats its fee-inclusive cost. One bet, one fee, done.
  Section B -- SEQUENTIAL LEGS ~1.3-1.9x: higher-probability legs on distinct match
               days for a 2-3 step compounding path (re-evaluate between legs).
  Section C -- REAL KALSHI COMBOS (MVE): multi-leg contracts Kalshi prices itself;
               we compute the exact joint (correlation-aware within a game) and
               flag combos where the model joint beats the ask. This is where
               "great payout" and "underpriced" can legitimately coexist.

Math reminder printed with the output: no structure beats
    P(triple) <= (1 + total edge harvested) / 3,
so the ONLY job of this shortlist is to concentrate maximum model-vs-market edge
into the fewest fee payments.

READ-ONLY: this script never places, amends, or cancels an order.
HONEST CAVEATS: the model's edge is proven against a base-rate baseline, NOT yet
against market prices -- treat edges below ~3 points as noise. Player-scorer
markets are excluded (the model deliberately has no player layer). Empty books
(common far from kickoff) simply produce no rows -- re-run near kickoff.

Run:  python bold_play_shortlist.py [--min-edge 0.02]
"""
from __future__ import annotations
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

# legacy/ holds the proven Kalshi REST client (signed auth w/ public-GET fallback)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))
from kalshi_client import KalshiClient, _team_matches  # noqa: E402

from wc2026.cli import _effective_config, FIT_MAX_AGE_DAYS  # noqa: E402
from wc2026.data import loader  # noqa: E402
from wc2026.model.ratings import build_team_strength, calibrate_to_tournament  # noqa: E402
from wc2026.sim.match import predict_match, MatchPrediction  # noqa: E402

TARGET_MULT = 3.0
COMBO_SERIES = ["KXMVESPORTSMULTIGAMEEXTENDED", "KXMVECROSSCATEGORY"]

# dataset name -> the name key the legacy Kalshi alias table expects
KALSHI_NAME = {
    "United States": "USA",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
}


def kalshi_fee(price_dollars: float) -> float:
    """Kalshi taker fee in dollars per contract: 0.07 * P * (1-P)."""
    return 0.07 * price_dollars * (1.0 - price_dollars)


def make_client() -> KalshiClient:
    """Authenticated if the legacy credentials still resolve; else public GETs."""
    try:
        return KalshiClient()
    except Exception as e:
        print(f"(auth unavailable: {e} -> using public market data)")
        return KalshiClient(key_id="", private_key_path="")


# ---------------------------------------------------------------------------
# Model probability for a quoted market name (names as built by the legacy client)
# ---------------------------------------------------------------------------
def model_prob(pred: MatchPrediction, name: str, home: str, away: str) -> float | None:
    """Exact model probability for a market name, or None if unmodeled (players)."""
    g = pred.matrix.shape[0] - 1
    I, J = np.meshgrid(np.arange(g + 1), np.arange(g + 1), indexing="ij")

    if name == f"{home} win":
        return pred.p_home_win
    if name == f"{away} win":
        return pred.p_away_win
    if name == "Draw":
        return pred.p_draw

    mo = re.match(r"(Over|Under) (\d+\.\d+) goals$", name)
    if mo:
        line = float(mo.group(2))
        return pred.prob_over(line) if mo.group(1) == "Over" else pred.prob_under(line)

    if name == "Both teams to score":
        return pred.p_btts
    if name == "Both teams to score - No":
        return 1.0 - pred.p_btts

    mo = re.match(rf"^(.+) over (\d+\.\d+) goals$", name)
    if mo and mo.group(1) in (home, away):
        line = float(mo.group(2))
        team_goals = I if mo.group(1) == home else J
        return float(pred.matrix[team_goals > line].sum())

    mo = re.match(r"^(.+) wins by (\d+)\+ goals$", name)
    if mo and mo.group(1) in (home, away):
        n = int(mo.group(2))
        margin = (I - J) if mo.group(1) == home else (J - I)
        return float(pred.matrix[margin >= n].sum())

    mo = re.match(r"^Score (\d+)-(\d+)$", name)
    if mo:
        i, j = int(mo.group(1)), int(mo.group(2))
        return float(pred.matrix[i, j]) if i <= g and j <= g else 0.0

    return None  # player-scorer and anything else unmodeled


# ---------------------------------------------------------------------------
# Single-leg scan across all upcoming fixtures
# ---------------------------------------------------------------------------
def scan_fixtures(client, ratings, state, cfg) -> tuple[pd.DataFrame, dict]:
    rows, preds = [], {}
    for _, fx in state.upcoming.iterrows():
        home, away, neutral = fx.home_team, fx.away_team, bool(fx.neutral)
        pred = predict_match(ratings, home, away, neutral=neutral, cfg=cfg)
        preds[frozenset((home, away))] = (pred, fx)

        k_home = KALSHI_NAME.get(home, home)
        k_away = KALSHI_NAME.get(away, away)
        try:
            quotes, stats = client.fetch_fixture_quotes(k_home, k_away)
        except Exception as e:
            print(f"  {home} vs {away}: fetch failed ({e})")
            continue
        print(f"  {home} vs {away} ({fx.date.date()}): "
              f"{stats['matched']} markets matched, {stats['quoted']} with live asks")

        for kname, q in quotes.items():
            # translate any Kalshi-alias team names in the market name back to ours
            name = kname.replace(k_home, home).replace(k_away, away)
            p = model_prob(pred, name, home, away)
            if p is None:
                continue
            price = q["price_cents"] / 100.0
            fee = kalshi_fee(price)
            cost = price + fee
            if not 0.01 < cost < 0.99:
                continue
            rows.append({
                "date": fx.date.date(), "match": f"{home} v {away}",
                "market": name, "side": q["side"].upper(), "ticker": q["ticker"],
                "price_c": round(price * 100, 1), "mult": 1.0 / cost,
                "model_p": p, "implied_p": cost, "edge": p - cost,
                "ev": p / cost - 1.0,
            })
    return pd.DataFrame(rows), preds


# ---------------------------------------------------------------------------
# Real Kalshi combo (MVE) contracts, priced with the exact per-game joint
# ---------------------------------------------------------------------------
def _leg_mask(pred: MatchPrediction, series: str, suffix: str, side: str,
              code2ds: dict, home: str, away: str) -> np.ndarray | None:
    """Boolean mask over the scoreline grid for one combo leg, or None if unmappable."""
    g = pred.matrix.shape[0] - 1
    I, J = np.meshgrid(np.arange(g + 1), np.arange(g + 1), indexing="ij")
    mask = None
    if series == "KXWCGAME":
        if suffix == "TIE":
            mask = I == J
        else:
            t = code2ds.get(suffix)
            if t == home:
                mask = I > J
            elif t == away:
                mask = J > I
    elif series == "KXWCTOTAL":
        try:
            mask = (I + J) >= int(suffix)       # suffix n => "Over n-0.5"
        except ValueError:
            mask = None
    elif series == "KXWCSPREAD":
        mo = re.match(r"([A-Z]+?)(\d+)$", suffix)
        if mo:
            t, n = code2ds.get(mo.group(1)), int(mo.group(2))
            if t == home:
                mask = (I - J) >= n
            elif t == away:
                mask = (J - I) >= n
    elif series == "KXWCBTTS":
        mask = (I >= 1) & (J >= 1)
    if mask is None:
        return None
    return ~mask if side == "no" else mask


def scan_combos(client, preds, min_mult: float = 2.0) -> pd.DataFrame:
    """Score Kalshi's real multi-leg contracts against the exact model joint."""
    # game_key -> (home, away) and team-code -> dataset name, from live KXWCGAME
    data = client._get("/markets", params={"series_ticker": "KXWCGAME",
                                           "status": "open", "limit": 1000})
    fixture_teams = {t for key in preds for t in key}
    code2ds, game2key = {}, {}
    for m in data.get("markets", []):
        parts = m.get("ticker", "").split("-")
        if len(parts) < 3:
            continue
        gk, sfx = parts[1], parts[2]
        sub = m.get("yes_sub_title", "")
        if sfx != "TIE" and sub:
            for ds in fixture_teams:
                if _team_matches(KALSHI_NAME.get(ds, ds), sub):
                    code2ds[sfx] = ds
                    break
        mt = re.match(r"(.+?)\s+vs\s+(.+?)\s+Winner", m.get("title", "") or "")
        if mt:
            names = []
            for raw in (mt.group(1).strip(), mt.group(2).strip()):
                hit = next((ds for ds in fixture_teams
                            if _team_matches(KALSHI_NAME.get(ds, ds), raw)), None)
                names.append(hit)
            if all(names) and frozenset(names) in preds:
                game2key[gk] = frozenset(names)

    combos = []
    for s in COMBO_SERIES:
        cursor = None
        for _ in range(20):
            p = {"series_ticker": s, "status": "open", "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = client._get("/markets", params=p)
            combos += d.get("markets", [])
            cursor = d.get("cursor")
            if not cursor:
                break

    rows = []
    for c in combos:
        legs = c.get("mve_selected_legs") or []
        try:
            price = float(c.get("yes_ask_dollars") or 0)
        except (TypeError, ValueError):
            continue
        # sane live ask only (sub-cent asks are ghost liquidity; 1.00 is a placeholder)
        if not legs or not 0.02 <= price <= 0.97:
            continue
        per_game, ok = {}, True
        for leg in legs:
            parts = (leg.get("market_ticker") or "").split("-")
            if len(parts) < 3 or parts[1] not in game2key:
                ok = False
                break
            key = game2key[parts[1]]
            pred, fx = preds[key]
            mask = _leg_mask(pred, parts[0], parts[2], leg.get("side", "yes"),
                             code2ds, fx.home_team, fx.away_team)
            if mask is None:
                ok = False
                break
            per_game.setdefault(parts[1], []).append(mask)
        if not ok:
            continue
        # exact joint: AND masks within a game, multiply across games
        joint = 1.0
        for gk, masks in per_game.items():
            m = masks[0]
            for extra in masks[1:]:
                m = m & extra
            pred, _ = preds[game2key[gk]]
            joint *= float(pred.matrix[m].sum())
        fee = kalshi_fee(price)
        cost = price + fee
        mult = 1.0 / cost
        if mult < min_mult:
            continue
        rows.append({"combo": (c.get("title") or "")[:70], "legs": len(legs),
                     "games": len(per_game), "price_c": round(price * 100, 1),
                     "mult": mult, "model_p": joint, "implied_p": cost,
                     "edge": joint - cost, "ticker": c.get("ticker", "")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "    (none found)"
    out = df[cols].copy()
    for c in ("model_p", "implied_p", "edge", "ev"):
        if c in out:
            out[c] = (out[c] * 100).round(1)
    out["mult"] = out["mult"].round(2)
    return out.to_string(index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-edge", type=float, default=0.02,
                    help="minimum model-minus-cost edge to shortlist (default 0.02)")
    args = ap.parse_args()

    cfg = _effective_config()
    df = loader.load_matches()
    train = loader.training_matches(df)
    state = loader.tournament_state(df)
    ratings = build_team_strength(train, as_of=state.as_of, cfg=cfg,
                                  max_age_days=FIT_MAX_AGE_DAYS, verbose=True)
    ratings = calibrate_to_tournament(ratings, state.played, cfg=cfg)
    if ratings.lambda_mult != 1.0:
        print(f"(in-tournament goal calibration: lambda x{ratings.lambda_mult:.3f})")
    client = make_client()

    print(f"\nScanning Kalshi books for {state.n_upcoming} upcoming fixtures...")
    legs, preds = scan_fixtures(client, ratings, state, cfg)

    n_quoted = len(legs)
    legs = legs[legs["edge"] >= args.min_edge] if n_quoted else legs
    print(f"\n{'='*78}\nBOLD-PLAY SHORTLIST  (target {TARGET_MULT:.0f}x | "
          f"{n_quoted} priced markets, {len(legs)} clear the {args.min_edge*100:.0f}pt edge floor)\n{'='*78}")

    if n_quoted:
        A = legs[(legs["mult"] >= 2.3) & (legs["mult"] <= 5.5)] \
            .sort_values("model_p", ascending=False).head(8)
        print("\n[A] SINGLE-SHOT ~3x  (one bet, one fee -- rank by success prob):")
        print(_fmt(A, ["date", "match", "market", "side", "price_c", "mult",
                       "model_p", "implied_p", "edge"]))

        B = legs[(legs["mult"] >= 1.25) & (legs["mult"] < 2.3)] \
            .sort_values("edge", ascending=False).head(10)
        print("\n[B] SEQUENTIAL LEGS  (compound 2-3 across distinct days -- rank by edge):")
        print(_fmt(B, ["date", "match", "market", "side", "price_c", "mult",
                       "model_p", "implied_p", "edge"]))

        # suggested path: best-prob leg per day (mult>=1.35), first 3 days
        path = (B[B["mult"] >= 1.35].sort_values("model_p", ascending=False)
                .drop_duplicates("date").sort_values("date").head(3))
        if len(path) >= 2:
            pm, pp = float(path["mult"].prod()), float(path["model_p"].prod())
            print(f"\n    suggested sequential path ({len(path)} legs): "
                  f"combined {pm:.2f}x @ model P {pp*100:.1f}%")
            for _, r in path.iterrows():
                print(f"      {r['date']}  {r['match']:28s} {r['market']:28s} "
                      f"{r['mult']:.2f}x @ {r['model_p']*100:.0f}%")
            if pm < TARGET_MULT:
                print(f"      (short of {TARGET_MULT:.0f}x -- extend with a further leg "
                      f"or swap one for a Section-A price)")
    else:
        print("\nNo live asks on any fixture right now (books fill near kickoff) -- "
              "re-run 1-3 hours before a match day's first game.")

    print("\n[C] REAL KALSHI COMBOS  (exact correlation-aware joint vs ask):")
    try:
        C = scan_combos(client, preds)
        if not C.empty:
            C = C[C["edge"] >= args.min_edge].sort_values("edge", ascending=False).head(8)
        print(_fmt(C, ["combo", "legs", "games", "price_c", "mult",
                       "model_p", "implied_p", "edge"]))
    except Exception as e:
        print(f"    (combo scan unavailable: {e})")

    print(f"""
{'-'*78}
Reality check (do not skip):
  * P(reach {TARGET_MULT:.0f}x) <= (1 + total edge) / {TARGET_MULT:.0f} no matter the structure.
  * Model edge is proven vs base rates, NOT vs market prices -- edges under ~3pts
    are within model error. Prefer fewer bets over more.
  * Analysis only: nothing here places orders. Prices move; re-run before acting.""")


if __name__ == "__main__":
    main()
