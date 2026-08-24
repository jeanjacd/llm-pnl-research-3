"""
main.py
=======
End-to-end pipeline:
    database -> ratings -> 100k-iteration simulation -> Kalshi prices ->
    correlation-aware parlay optimization -> ranked recommendations.

Run:
    python main.py            # live Kalshi prices (needs credentials in kalshi_client.py)
    python main.py --demo     # synthetic prices, runs with zero credentials

This script ONLY recommends. It never places a trade. Place bets yourself in Kalshi.
"""

from __future__ import annotations
import argparse
import random
import re
import unicodedata


def _norm_market(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9+ -]", "", s.lower()).strip()

from ratings import load_database
from simulate import simulate_match
from kalshi_client import KalshiClient
from optimizer import (build_candidate_legs, find_best_parlays,
                       find_near_miss_parlays)

# ============================ CONFIG =======================================
N_ITER = 150_000

# Matchups to analyze: (home, away, neutral_venue?)
# Add as many games as you like — the optimizer builds parlays that span them,
# combining legs from different games (independent) with correlation-aware joints
# within each game.
# 3rd element = neutral venue? Use True for normal WC games. Set it to False ONLY for
# host-nation home games (USA / Mexico / Canada playing at home), which applies
# HOME_FIELD_ADVANTAGE to the host — and put the host as the first (home) team.
#   e.g. ("USA", "Paraguay", False)   # USA at home -> gets the home-edge boost
FIXTURES = [
    ("Canada", "South Africa", "True")
]

# Bet selection (YOUR knobs)
# Tuned for HITS, not just edge: lenient leg filter lets solid neutral-EV plays (incl.
# efficiently-priced winners) through, ranked by EV so the best rise without forcing
# longshots. Terrible bets are still rejected by MIN_LEG_EDGE's floor.
TOP_N = 5                  # how many best single bets to recommend
RANK_BY = "confidence"     # surface HITS (highest model prob) | "ev" (longshots) | "kelly"
MIN_CONFIDENCE = 0.0       # optional min hit-rate floor (0 = off)
MIN_MULTIPLIER = 1.5       # decent-payout floor so "hits" aren't near-locks (raise for more)
MIN_LEG_EDGE = -0.02       # lenient: allow ~neutral/slightly-EV bets through, reject worse
MIN_LEG_PROB = 0.05        # sanity floor: ignore sub-5% legs (edges there are unreliable)
COMBO_MIN_MULTIPLIER = 8.0 # min payout multiplier for the Kalshi combo section
MAX_LEGS = 4
BANKROLL = 1000.0          # for converting half-Kelly fraction to a dollar stake
MODEL_TRUST = 0.5          # 1.0=pure model, 0=pure market. Tune via backt.py blend sweep.
MIN_LEGS = 1               # 1 = also suggest strong single bets
MIN_DISTINCT_GAMES = 1     # multi-leg parlays must span >= this many games (singles exempt)
MAX_LEGS_PER_GAME = 4      # cap legs from any single game
MAX_LEGS_PER_CATEGORY = 2  # cap legs per market category (scorer/total/btts/...) so niche
                           # markets stay viable but don't dominate the recommendations
# ===========================================================================


def map_kalshi_prices(match_key, book, client, demo: bool) -> dict:
    """
    Returns {market_name: {ticker, price_cents, decimal_odds, side}}.

    LIVE MODE: you must map each simulated market_name to a real Kalshi ticker.
    Kalshi soccer tickers vary by event, so this is intentionally a manual mapping
    point -- fill in KALSHI_TICKER_MAP for the markets you actually want to bet.
    DEMO MODE: invents a plausible price from the model prob plus a bookmaker margin,
    so you can see the full pipeline work before wiring tickers.
    """
    prices = {}
    if demo:
        rng = random.Random(hash(match_key) & 0xFFFF)
        for name in book.names():
            p = book.market(name).prob
            if p <= 0.02 or p >= 0.98:
                continue
            # Simulate a market that's close to true prob but noisy, with ~5% vig,
            # so some legs show genuine positive edge and others don't.
            noise = rng.uniform(-0.06, 0.06)
            implied = min(0.97, max(0.03, p + noise + 0.025))  # +vig
            price_cents = int(round(implied * 100))
            prices[name] = {
                "ticker": f"DEMO-{match_key}-{abs(hash(name)) % 9999}",
                "price_cents": price_cents,
                "decimal_odds": 100.0 / price_cents,
                "side": "yes",
            }
        return prices

    # ---- LIVE MODE ----
    # Auto-map this fixture to the live KXWC* World Cup series (result/totals/BTTS).
    home, away = match_key.split(" v ", 1)
    try:
        prices, stats = client.fetch_fixture_quotes(home, away)
    except Exception as e:
        print(f"    [kalshi] fetch failed for {match_key}: {e}")
        return {}
    if stats["game_key"] is None:
        print(f"    [kalshi] no KXWCGAME market found matching '{match_key}' "
              f"(check team names / that the game is listed).")
        return {}

    # Reconcile quote keys to the book's exact market names (handles player-name
    # spelling differences); drop any market our model doesn't simulate.
    book_names = set(book.names())
    norm_lookup = {_norm_market(n): n for n in book_names}
    reconciled = {}
    for k, v in prices.items():
        if k in book_names:
            reconciled[k] = v
        else:
            hit = norm_lookup.get(_norm_market(k))
            if hit:
                reconciled[hit] = v
    print(f"    [kalshi] game {stats['game_key']}: matched {stats['matched']} markets, "
          f"{stats['quoted']} quoted, {len(reconciled)} mapped to model markets")
    if stats["quoted"] == 0:
        print(f"    [kalshi] markets exist but order books are empty right now "
              f"(no resting prices) -- re-run near kickoff.")
    return reconciled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic prices")
    args = ap.parse_args()

    db = load_database()
    client = None if args.demo else KalshiClient()

    print(f"Simulating {len(FIXTURES)} matchups at {N_ITER:,} iterations each...\n")
    books_and_prices = []
    for home, away, neutral in FIXTURES:
        h, a = db.team(home), db.team(away)
        book = simulate_match(h, a, db.team_players(home), db.team_players(away),
                              n_iter=N_ITER, neutral=neutral, seed=7)
        match_key = f"{home} v {away}"

        # Show the model's headline numbers for transparency.
        print(f"=== {match_key} ===")
        for nm in (f"{home} win", "Draw", f"{away} win", "Over 2.5 goals",
                   "Both teams to score"):
            if nm in book.outcomes:
                m = book.market(nm)
                print(f"    {nm:26s} {m.prob*100:5.1f}%  "
                      f"[{m.ci_low*100:4.1f},{m.ci_high*100:4.1f}]")
        prices = map_kalshi_prices(match_key, book, client, demo=args.demo)
        books_and_prices.append((match_key, book, prices))
        print()

    legs = build_candidate_legs(books_and_prices, min_edge=MIN_LEG_EDGE,
                                model_trust=MODEL_TRUST, min_leg_prob=MIN_LEG_PROB)
    print(f"Positive-edge candidate legs (edge >= {MIN_LEG_EDGE*100:.0f}%, "
          f"model_trust={MODEL_TRUST}): {len(legs)}")
    if not legs:
        print("No positive-edge legs found. Loosen MIN_LEG_EDGE or check prices.")
        return

    # Single bets only -- these are REAL single Kalshi markets. Multi-leg parlays come
    # exclusively from Kalshi's actual combo contracts (the section below), never from
    # self-built synthetic combos whose multiplier you can't actually buy.
    parlays = find_best_parlays(
        legs,
        min_confidence=MIN_CONFIDENCE,
        min_multiplier=MIN_MULTIPLIER,
        max_legs=1,
        min_legs=1,
        top_n=TOP_N,
        rank_by=RANK_BY,
    )

    # Guardrail: warn only if BOTH floors are active and mutually money-losing.
    if MIN_CONFIDENCE > 0:
        breakeven_mult = 1.0 / MIN_CONFIDENCE
        if MIN_MULTIPLIER > 1.0 and MIN_MULTIPLIER < breakeven_mult:
            print(f"\n[!] WARNING: MIN_MULTIPLIER ({MIN_MULTIPLIER:.2f}x) is below the "
                  f"break-even multiplier for {MIN_CONFIDENCE:.0%} confidence "
                  f"({breakeven_mult:.2f}x) -- a bet could pass both floors and still be -EV.")

    floors = []
    if MIN_CONFIDENCE > 0:
        floors.append(f"confidence>={MIN_CONFIDENCE:.0%}")
    if MIN_MULTIPLIER > 1.0:
        floors.append(f"multiplier>={MIN_MULTIPLIER:.1f}x")
    filt = f" (filters: {', '.join(floors)})" if floors else ""
    print(f"\nTop {TOP_N} SINGLE bets across your fixtures, ranked by {RANK_BY} "
          f"(risk-adjusted){filt}:\n")

    if parlays:
        for i, par in enumerate(parlays, 1):
            stake = par.half_kelly * BANKROLL
            print(f"[{i}] {par.describe()}")
            print(f"     suggested stake (half-Kelly, bankroll ${BANKROLL:.0f}): "
                  f"${stake:.2f}\n")
        print("Reminder: model-based suggestions, not guarantees. Place bets yourself.")
    else:
        print("  No positive-EV single bets cleared your floors "
              "(loosen MIN_LEG_EDGE / floors or check prices).")

    # REAL parlays, SCOPED TO YOUR FIXTURES: Kalshi's actual combo contracts whose every
    # leg is in the games above (genuine ask, real multiplier), priced by our model net
    # of fees. Only buyable multi-leg bets, tied to what you asked for.
    if client is not None:
        print(f"\n=== Real Kalshi combos for your fixtures "
              f"(>= {COMBO_MIN_MULTIPLIER:.1f}x, model-priced) ===")
        try:
            from price_combos import (score_live_combos, print_combo_report,
                                      fixture_game_keys)
            fkeys = fixture_game_keys(client, [(h, a) for h, a, _ in FIXTURES])
            print_combo_report(*score_live_combos(
                client, db, fixture_keys=fkeys, min_multiplier=COMBO_MIN_MULTIPLIER))
        except Exception as e:
            print(f"  [combos] check failed: {e}")


if __name__ == "__main__":
    main()
