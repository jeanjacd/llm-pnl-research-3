"""
price_combos.py — price Kalshi's REAL multi-leg (MVE) combo contracts with our model.

Instead of synthesizing a parlay payout from single-leg odds, this reads Kalshi's actual
combo contracts (series KXMVESPORTSMULTIGAMEEXTENDED / KXMVECROSSCATEGORY), parses their
`mve_selected_legs`, evaluates the exact leg-set with our correlation-aware simulation,
and computes a MEASURED edge:

    edge = simulated_joint_probability  -  (combo_price + Kalshi_fee)

This is the only way to know if a combo is genuinely +EV (e.g. if Kalshi prices a combo
as the product of independent legs but the legs are positively correlated, our joint will
be higher -> real edge). Only combos that (a) have a live price and (b) consist entirely
of legs we simulate (result / total / spread / BTTS) are scored; others are skipped.

Run:  python price_combos.py
"""

from __future__ import annotations
import re

import numpy as np

from kalshi_client import KalshiClient
from ratings import load_database
from simulate import simulate_match
from optimizer import kalshi_fee_cents
from build_database import normalize_team, _norm_name

N_ITER = 60_000
COMBO_SERIES = ["KXMVESPORTSMULTIGAMEEXTENDED", "KXMVECROSSCATEGORY"]


def _fp(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def fetch_all_combos(client) -> list[dict]:
    """All open MVE combo contracts (paginated)."""
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
    return combos


def build_combo_index(combos: list[dict]) -> dict:
    """Map frozenset{(market_ticker, side)} -> (best_price_dollars, combo_ticker) so we
    can look up whether a suggested parlay is offered as a real Kalshi combo."""
    idx = {}
    for c in combos:
        legs = c.get("mve_selected_legs") or []
        if not legs:
            continue
        key = frozenset((l.get("market_ticker"), l.get("side")) for l in legs)
        price = _fp(c.get("yes_ask_dollars"))
        if key not in idx or price > idx[key][0]:
            idx[key] = (price, c.get("ticker", ""))
    return idx


def verify_parlay_multiplier(index: dict, legs, synth_multiplier: float) -> str:
    """Compare a suggested parlay's synthesized multiplier to Kalshi's real combo, if
    Kalshi offers that exact leg-set as a single contract."""
    from optimizer import kalshi_fee_cents
    key = frozenset((lg.kalshi_ticker, lg.side) for lg in legs)
    games = {lg.match_key for lg in legs}
    hit = index.get(key)
    if not hit:
        if len(games) == 1:
            return ("NOT a listed Kalshi combo -> this multiplier is unachievable: "
                    "same-game legs settle together, so separate contracts only pay "
                    "ADDITIVELY (analysis only, not a placeable bet).")
        return ("NOT a listed Kalshi combo -> not buyable as one ticket. Separate "
                "contracts pay additively; a true multiplier needs a combo contract, or "
                "(only for different-kickoff games) rolling winnings from one into the next.")
    price, ticker = hit
    if not 0 < price < 1:
        return f"listed on Kalshi ({ticker}) but NO genuine ask -> not tradeable yet"
    net = 1.0 / (price + kalshi_fee_cents(price * 100) / 100.0)
    verdict = "matches" if net >= synth_multiplier * 0.97 else "WORSE than yours (Kalshi margin)"
    return f"Kalshi combo {ticker}: real {net:.2f}x vs your {synth_multiplier:.2f}x -> {verdict}"


def build_wc_maps(client):
    """code->team name and game_key->(home,away) from the live KXWCGAME markets."""
    code2team, game2teams = {}, {}
    d = client._get("/markets", params={"series_ticker": "KXWCGAME",
                                        "status": "open", "limit": 1000})
    for m in d.get("markets", []):
        tk = m.get("ticker", "")
        parts = tk.split("-")
        if len(parts) < 3:
            continue
        game_key, suffix = parts[1], parts[2]
        sub = m.get("yes_sub_title", "")
        if suffix != "TIE" and sub:
            code2team[suffix] = sub
        mt = re.match(r"(.+?)\s+vs\s+(.+?)\s+Winner", m.get("title", "") or "")
        if mt:
            game2teams[game_key] = (mt.group(1).strip(), mt.group(2).strip())
    return code2team, game2teams


def fixture_game_keys(client, fixtures) -> set:
    """The Kalshi game_keys for the user's FIXTURES (so combos can be scoped to them)."""
    from build_database import normalize_team, _norm_name
    _, game2teams = build_wc_maps(client)
    nm = lambda x: _norm_name(normalize_team(x))
    keys = set()
    for fx in fixtures:
        want = frozenset({nm(fx[0]), nm(fx[1])})
        for gk, (h, a) in game2teams.items():
            if frozenset({nm(h), nm(a)}) == want:
                keys.add(gk)
    return keys


def score_live_combos(client, db, combos=None, fixture_keys=None, min_multiplier=1.0):
    """Returns (results, n_combos, n_liquid). results = sorted [(edge, joint, price,
    fee, market_dict)] for liquid, fully-mappable Kalshi combos. No printing.
    fixture_keys: if given, only combos whose legs ALL belong to those game_keys.
    min_multiplier: only combos whose real net payout multiplier >= this."""
    db_by_norm = {_norm_name(normalize_team(t.name)): t.name for t in db.teams.values()}
    code2team, game2teams = build_wc_maps(client)

    def dbname(kalshi_name):
        return db_by_norm.get(_norm_name(normalize_team(kalshi_name or "")))

    def leg_market(market_ticker):
        """market_ticker -> (game_key, our_market_name) for the YES meaning, or None."""
        parts = market_ticker.split("-")
        if len(parts) < 3:
            return None
        series, game_key, suffix = parts[0], parts[1], parts[2]
        if game_key not in game2teams:
            return None
        if series == "KXWCGAME":
            if suffix == "TIE":
                return game_key, "Draw"
            t = dbname(code2team.get(suffix))
            return (game_key, f"{t} win") if t else None
        if series == "KXWCTOTAL":
            try:
                return game_key, f"Over {int(suffix) - 0.5} goals"
            except ValueError:
                return None
        if series == "KXWCSPREAD":
            mo = re.match(r"([A-Z]+)(\d+)", suffix)
            if not mo:
                return None
            t = dbname(code2team.get(mo.group(1)))
            return (game_key, f"{t} wins by {int(mo.group(2))}+ goals") if t else None
        if series == "KXWCBTTS":
            return game_key, "Both teams to score"
        return None  # KXWCGOAL (player codes), other sports, etc. -> unmappable

    book_cache = {}

    def book_for(game_key):
        if game_key in book_cache:
            return book_cache[game_key]
        h, a = (dbname(x) for x in game2teams[game_key])
        if not h or not a:
            book_cache[game_key] = None
            return None
        bk = simulate_match(db.team(h), db.team(a), db.team_players(h),
                            db.team_players(a), n_iter=N_ITER, seed=7)
        book_cache[game_key] = bk
        return bk

    combos = combos if combos is not None else fetch_all_combos(client)
    # A REAL tradeable ask needs: a sane price (sub-cent asks are ghost liquidity that
    # mechanically explode into fake 100x+ multipliers), actual resting size, and book
    # liquidity. yes_ask_dollars==1.00 is a junk placeholder.
    liquid = [c for c in combos
              if 0.02 <= _fp(c.get("yes_ask_dollars")) <= 0.97
              and _fp(c.get("yes_ask_size_fp")) > 0
              and _fp(c.get("liquidity_dollars")) > 0]
    if not liquid:
        return [], len(combos), 0

    results = []
    for c in liquid:
        legs = c.get("mve_selected_legs") or []
        price = _fp(c.get("yes_ask_dollars"))      # dollars
        mapped, ok = {}, True
        for leg in legs:
            lm = leg_market(leg.get("market_ticker", ""))
            if not lm:
                ok = False
                break
            gk, name = lm
            bk = book_for(gk)
            if bk is None or name not in bk.outcomes:
                ok = False
                break
            arr = bk.outcomes[name]
            if leg.get("side") == "no":
                arr = ~arr
            mapped.setdefault(gk, []).append(arr)
        if not ok:
            continue
        if fixture_keys is not None and not set(mapped).issubset(fixture_keys):
            continue  # restrict to combos whose every leg is in the requested fixtures
        fee = kalshi_fee_cents(price * 100) / 100.0
        if 1.0 / (price + fee) < min_multiplier:
            continue  # below the requested payout multiplier
        # joint: AND within a game, multiply across games (independent).
        joint = 1.0
        for gk, arrs in mapped.items():
            mask = np.ones(N_ITER, dtype=bool)
            for a in arrs:
                mask &= a
            joint *= float(mask.mean())
        edge = joint - price - fee
        results.append((edge, joint, price, fee, c))

    results.sort(key=lambda r: -r[0])
    return results, len(combos), len(liquid)


def print_combo_report(results, n_combos, n_liquid, top_n=15):
    print(f"Kalshi combos: {n_combos} listed, {n_liquid} with a live price.")
    if n_liquid == 0:
        print("  No tradeable combos right now (empty order books) -- re-run near kickoff.")
        return
    if not results:
        print("  Liquid combos exist but none map fully to our simulated markets "
              "(they contain goalscorer/other-sport legs).")
        return
    print(f"  {len(results)} priced by our model (edge = model joint - price - fee):\n")
    for edge, joint, price, fee, c in results[:top_n]:
        mult = 1.0 / (price + fee)
        flag = "  <== +EV" if edge > 0 else ("  (ok/neutral)" if edge > -0.03 else "")
        print(f"  {mult:5.2f}x | edge {edge*100:+5.1f}%  | model {joint*100:4.1f}%  vs "
              f"price {price*100:4.1f}c (+{fee*100:.1f}c fee){flag}")
        print(f"     {c.get('title','')[:80]}")
    if all(r[0] <= 0 for r in results):
        print("\n  None are +EV after fees -- skip combos this slate.")


def main():
    client = KalshiClient()
    db = load_database()
    print_combo_report(*score_live_combos(client, db))


if __name__ == "__main__":
    main()
