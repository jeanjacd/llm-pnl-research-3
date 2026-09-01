"""
paper/rebuild.py
================
Rebuild the paper book from ORDER INTENT, against the venue tape and the
match results.

WHY THIS IS POSSIBLE AT ALL. No order was ever sent to a venue, so nothing
downstream of an order is a fact -- fills, positions, settlements and P&L are
all DERIVED. The primitives are intact: every one of the 1,440 orders on the
live book carries its creation time, expiry, limit price, side, venue,
instrument and fixture. Given those, plus the tape and the results, the book
can be recomputed exactly rather than patched.

WHAT WAS WRONG WITH THE OLD ONE. Two independent defects, both since fixed,
and either alone makes the recorded book unusable:

  1. `replay_fills` read the tape from the last check to NOW rather than to the
     order's expiry, so an order dead at kick-off could still fill on a price
     that only existed once the match was under way -- and a resting bid is
     only reached after kick-off when the match has turned against it. 512 of
     519 fills came from windows running past expiry.

  2. `winning_side` was handed the position's own already-negated claim and
     compared the result to the side, negating twice. Every no-side holding
     was settled backwards: 296 of 447 settled positions, and the `no` book
     showed a 17-point gap between the price it paid and the rate it won,
     which the correctly-settled `yes` book did not.

THE REBUILD IS STRICTER THAN THE ORIGINAL, not just different. The old replay
asked the tape only about the gap since its last check, so a dip through the
limit BETWEEN two cron runs was missed entirely. This asks about the order's
whole life, once.

AN ORDER WHOSE TAPE CANNOT BE FETCHED IS LEFT UNRESOLVED. It is neither filled
nor expired, and it is counted and reported. Guessing would put a fabricated
position in a ledger whose entire purpose is measurement.
"""
from __future__ import annotations

import copy
import datetime as dt

from .broker import PaperPortfolio
from .clv import capture_closing_lines
from .fills import replay_fills
from .settlement import settle_portfolio

# Far enough ahead that `min(now, expires_at)` always resolves to the expiry,
# which is what makes the replay ask about each order's whole life.
_AFTER_EVERYTHING = dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc)


def order_intent(order) -> dict:
    """The fields that describe what was OFFERED, with no outcome in them."""
    get = (order.get if isinstance(order, dict) else
           lambda k, d=None: getattr(order, k, d))
    return {k: get(k) for k in (
        "case_id", "venue", "instrument_id", "side", "limit_price_cents",
        "requested_size", "league_id", "kind", "created_at", "expires_at",
        "claim", "home_team", "away_team", "kickoff_utc",
        "settles_on_regulation")}


def blank_book(portfolio) -> PaperPortfolio:
    """A fresh portfolio: same starting cash, same board record, no positions.

    The BOARD VERDICTS carry over. What the board considered, approved and
    declined, and why, is independent of both defects -- neither the fill
    window nor the settlement sign touches a decision that was reached before
    a single order was placed. Discarding it would throw away the reasoning
    behind the record to fix the arithmetic of it, and the reasoning is the
    part that cannot be recomputed.
    """
    start = int(getattr(portfolio, "starting_cash_cents", 0) or 0)
    book = PaperPortfolio(starting_cash_cents=start, cash_cents=start,
                          path=None)
    book.boarded = copy.deepcopy(getattr(portfolio, "boarded", None) or {})
    return book


def resubmit(book: PaperPortfolio, intents: list) -> dict:
    """Re-place every order, oldest first, exactly as it was offered."""
    stats = {"submitted": 0, "unplaceable": 0, "problems": []}
    for intent in sorted(intents, key=lambda i: str(i.get("created_at") or "")):
        price = intent.get("limit_price_cents")
        size = intent.get("requested_size")
        if not price or not size:
            stats["unplaceable"] += 1
            continue
        try:
            order = book.submit(
                intent["case_id"], intent["venue"], intent["instrument_id"],
                intent["side"], int(price), float(size),
                league_id=intent.get("league_id"),
                expires_at=intent.get("expires_at"),
                claim=intent.get("claim"),
                home_team=intent.get("home_team"),
                away_team=intent.get("away_team"),
                kickoff_utc=intent.get("kickoff_utc"),
                settles_on_regulation=intent.get("settles_on_regulation"))
        except Exception as exc:                              # noqa: BLE001
            stats["unplaceable"] += 1
            stats["problems"].append("%s: %s" % (intent.get("case_id"), exc))
            continue
        # Preserve when it was really offered, so the tape window is the
        # order's own life rather than the moment of the rebuild.
        if intent.get("created_at"):
            order.created_at = intent["created_at"]
        order.last_checked_at = None
        stats["submitted"] += 1
    return stats


def rebuild(portfolio, probes: dict, frames: dict, now=None) -> dict:
    """Replay intent -> fills -> closing lines -> settlement. Returns a report.

    `now` is deliberately pushed past every expiry: with the window clamped to
    `min(now, expires_at)`, that makes each order's replay cover exactly the
    span it was live for.
    """
    intents = [order_intent(o) for o in portfolio.orders.values()]
    book = blank_book(portfolio)
    report = {"orders_on_record": len(intents)}
    report["resubmit"] = resubmit(book, intents)

    # One pass, each order asked about its whole life.
    report["fills"] = replay_fills(book, probes, now=now or _AFTER_EVERYTHING)
    report["expired"] = len(book.expire_due(now=_AFTER_EVERYTHING))
    if probes:
        report["clv"] = capture_closing_lines(book, probes)
    report["settlement"] = settle_portfolio(book, frames)

    report["book"] = book
    report["summary"] = {
        "orders": len(book.orders),
        "filled": sum(1 for o in book.orders.values()
                      if getattr(o, "status", "") == "filled"),
        "unresolved": sum(1 for o in book.orders.values()
                          if getattr(o, "status", "") == "open"),
        "positions": len(book.positions),
        "settled": sum(1 for p in book.positions.values() if p.settled),
        "realized_cents": sum(float(p.realized_pnl_cents or 0)
                              for p in book.positions.values() if p.settled),
        "cash_cents": book.cash_cents,
    }
    return report


def compare(old, new) -> dict:
    """What the correction actually changed, so the swing is auditable."""
    def totals(pf):
        settled = [p for p in _positions(pf) if _get(p, "settled")]
        won = sum(1 for p in settled
                  if float(_get(p, "realized_pnl_cents") or 0) > 0)
        return {
            "positions": len(_positions(pf)),
            "settled": len(settled),
            "won": won,
            "win_rate": (won / len(settled)) if settled else None,
            "realized_cents": sum(float(_get(p, "realized_pnl_cents") or 0)
                                  for p in settled),
        }
    before, after = totals(old), totals(new)
    return {"before": before, "after": after,
            "pnl_swing_cents": after["realized_cents"] - before["realized_cents"]}


def _positions(pf):
    if isinstance(pf, dict):
        return list((pf.get("positions") or {}).values())
    return list(pf.positions.values())


def _get(pos, key):
    return pos.get(key) if isinstance(pos, dict) else getattr(pos, key, None)


def archive(portfolio: dict) -> dict:
    """A deep copy of the old book, kept beside the new one.

    The corrected record is the one to trade on, but the broken one is the
    evidence for why it changed, and a measurement system that quietly
    overwrites its own history is not one.
    """
    return copy.deepcopy(portfolio)
