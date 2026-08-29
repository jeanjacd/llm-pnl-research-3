"""Ledger shapes shared by the site tests.

Kept in one place because both the model and the render tests assert against
the SAME portfolio schema, and two drifting copies of it would let a test pass
against a shape the real ledger no longer has.

A plain module rather than `conftest.py`: these are factories called inline,
not fixtures, and pytest puts this directory on `sys.path` for its siblings.
"""


def book(positions=(), boarded=(), orders=(), ledger=(), start=100_000):
    return {
        "starting_cash_cents": start, "cash_cents": start,
        "reserved_cents": 0, "saved_at": "2026-08-29T12:00:00+00:00",
        "positions": {str(i): p for i, p in enumerate(positions)},
        "orders": {str(i): o for i, o in enumerate(orders)},
        "boarded": dict(boarded),
        "ledger": list(ledger),
    }


def pos(claim="draw", home="A", away="B", league="mls", kickoff="2026-08-28",
        pnl=None, clv=None, size=100.0, cost=20.0, venue="kalshi"):
    return {"venue": venue, "instrument_id": "%s-%s" % (claim, home),
            "side": "yes", "size": size, "avg_cost_cents": cost,
            "fees_cents": 0, "league_id": league, "claim": claim,
            "home_team": home, "away_team": away,
            "kickoff_utc": kickoff + "T18:00:00",
            "clv_cents": clv, "settled": pnl is not None,
            "realized_pnl_cents": pnl}


def verdict(league="mls", home="A", away="B", day="2026-08-28",
            action="DEFER", reason="no edge", considered=112):
    return ("%s|%s|%s|%s" % (league, home, away, day),
            {"action": action, "reason": reason, "attempts": 1,
             "decided_by": "quant", "hours_to_kickoff": 8.0,
             "markets_considered": considered, "home": home, "away": away,
             "first_boarded_at": day + "T10:00:00+00:00"})


