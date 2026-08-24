"""
test_live_ready.py — verify the full LIVE path is game-time ready.

Kalshi's WC books are empty until near kickoff, so this exercises the real market
structure (ticker discovery, every category, name reconciliation, leg/parlay
generation) using a SYNTHETIC price fill. If this produces parlays, the pipeline
will work the moment real quotes appear.

Usage:  python test_live_ready.py "Argentina" "Jordan"
"""
from __future__ import annotations
import sys

from ratings import load_database
from simulate import simulate_match
from kalshi_client import KalshiClient
from optimizer import build_candidate_legs, find_best_parlays
from main import _norm_market


def main(home: str, away: str):
    db = load_database()
    c = KalshiClient()

    book = simulate_match(db.team(home), db.team(away),
                          db.team_players(home), db.team_players(away),
                          n_iter=50_000, seed=7)

    quotes, stats = c.fetch_fixture_quotes(home, away, fill_synthetic=True)
    print(f"game_key={stats['game_key']}  matched={stats['matched']}  "
          f"real_quotes={stats['quoted']}  (synthetic fill for the rest)")
    if stats["game_key"] is None:
        print("No KXWCGAME found for that fixture — pick a listed game.")
        return

    # Reconcile to book market names (same logic as main.py).
    book_names = set(book.names())
    norm = {_norm_market(n): n for n in book_names}
    reconciled = {}
    for k, v in quotes.items():
        hit = k if k in book_names else norm.get(_norm_market(k))
        if hit:
            reconciled[hit] = v

    # Coverage by category.
    cats = {"result": 0, "totals": 0, "team totals": 0, "spread": 0,
            "BTTS": 0, "scorer": 0, "correct score": 0}
    for name in reconciled:
        if name.endswith("to score"):
            cats["scorer"] += 1
        elif name.startswith("Score "):
            cats["correct score"] += 1
        elif "wins by" in name:
            cats["spread"] += 1
        elif name.startswith("Both teams"):
            cats["BTTS"] += 1
        elif "over" in name and (home in name or away in name):
            cats["team totals"] += 1
        elif name.startswith(("Over", "Under")):
            cats["totals"] += 1
        elif name.endswith("win") or name == "Draw":
            cats["result"] += 1
    print(f"mapped {len(reconciled)} markets ->", {k: v for k, v in cats.items() if v})

    legs = build_candidate_legs([(f"{home} v {away}", book, reconciled)], min_edge=0.02)
    print(f"\ncandidate +edge legs (synthetic prices): {len(legs)}")
    parlays = find_best_parlays(legs, min_confidence=0.15, min_multiplier=2.0,
                                max_legs=3, top_n=3)
    print(f"example parlays generated: {len(parlays)}\n")
    for i, p in enumerate(parlays, 1):
        print(f"[{i}] {p.describe()}")
    print("\n=> If you see categories mapped and parlays generated, the live path is "
          "ready. Re-run main.py near kickoff for REAL prices.")


if __name__ == "__main__":
    h = sys.argv[1] if len(sys.argv) > 1 else "Argentina"
    a = sys.argv[2] if len(sys.argv) > 2 else "Jordan"
    main(h, a)
