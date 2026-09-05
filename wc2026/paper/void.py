"""
paper/void.py
=============
Withdraw a position from the MEASUREMENT without withdrawing it from the RECORD.

WHY THERE HAS TO BE SUCH A THING. A book that measures itself has exactly two
bad options when a trade turns out to have been made on a defective input: keep
it, and every rate it feeds is computed over an observation that measures
nothing; or delete it, and the record silently disagrees with what happened.
Both are worse than saying so.

So a void is not a deletion. The position stays where it is, keeps its price,
its size, its closing line and its result, and gains a flag and a reason. What
changes is that nothing which claims to be a MEASUREMENT counts it: not CLV,
not P&L, not the strike rate, not the per-league or per-family tables. It still
appears on its fixture's page, marked, with the reason printed beside it --
because the reader who wants to know whether the record has been tidied is
exactly the reader this system exists for.

THE STAKE COMES BACK. A void in betting means the bet is cancelled and the
stake returned: neither a win nor a loss. Cash is moved by the exact amount the
trade moved it -- the fill and its fees out, the payout in -- so the balance is
what it would have been had the order never been placed.

AND THE ORDER IS VOIDED TOO, or the next rebuild would raise the position from
the dead. `paper.rebuild` replays order intent, and an order whose intent was
defective has to carry that fact with it.

WHAT THIS IS NOT FOR. Not for a losing trade, not for a trade whose reasoning
looks bad in hindsight, and not for one that was merely unlucky. The test is
whether the DECISION was measurable at all: a position priced against a
proposition the model never evaluated is not evidence about the model, in
either direction. Voiding a trade for its outcome rather than its input is the
one use of this that would destroy the thing it protects.
"""
from __future__ import annotations

import datetime as dt


class VoidError(ValueError):
    """The void cannot be applied as asked. Never applied partially."""


def _cash_effect(position: dict) -> int:
    """What this position did to the balance, in cents.

    Out on the fill and its fees, in on the payout. Reversing exactly this is
    what puts the book back where it would have been.
    """
    size = float(position.get("size") or 0)
    cost = float(position.get("avg_cost_cents") or 0)
    fees = float(position.get("fees_cents") or 0)
    payout = float(position.get("payout_cents") or 0) if position.get("settled") else 0.0
    return int(round(payout - (size * cost) - fees))


def plan(portfolio: dict, position_ids, reason: str) -> dict:
    """What voiding these positions would do. Reads only; writes nothing."""
    if not str(reason or "").strip():
        raise VoidError("a void needs a reason -- it is the whole point of it")

    positions = portfolio.get("positions") or {}
    by_id = {}
    for key, pos in positions.items():
        by_id[str(pos.get("position_id") or key)] = (key, pos)

    targets, missing, already = [], [], []
    for wanted in position_ids:
        found = by_id.get(str(wanted))
        if found is None:
            missing.append(str(wanted))
        elif found[1].get("void"):
            already.append(str(wanted))
        else:
            targets.append(found)
    if missing:
        raise VoidError("no such position: %s" % ", ".join(missing))

    orders = portfolio.get("orders") or {}
    entries = portfolio.get("ledger") or []
    steps = []
    for key, pos in targets:
        case_id = pos.get("case_id")
        steps.append({
            "key": key,
            "position_id": pos.get("position_id") or key,
            "case_id": case_id,
            "fixture": "%s v %s" % (pos.get("home_team"), pos.get("away_team")),
            "claim": pos.get("claim"),
            "instrument_id": pos.get("instrument_id"),
            "realized_pnl_cents": float(pos.get("realized_pnl_cents") or 0),
            "cash_delta_cents": -_cash_effect(pos),
            # Everything that has to move with it. Matched on `case_id`, which
            # is what ties an order, its position and its ledger line together.
            "orders": [k for k, o in orders.items()
                       if o.get("case_id") == case_id and not o.get("void")],
            "ledger": [i for i, e in enumerate(entries)
                       if e.get("case_id") == case_id and not e.get("void")],
        })
    return {"steps": steps, "already_void": already,
            "cash_delta_cents": sum(s["cash_delta_cents"] for s in steps),
            "reason": str(reason).strip()}


def apply(portfolio: dict, position_ids, reason: str, now=None) -> dict:
    """Void in place and return the report. Raises rather than half-applying."""
    report = plan(portfolio, position_ids, reason)
    stamp = (now or dt.datetime.now(dt.timezone.utc)).isoformat()
    mark = {"void": True, "void_reason": report["reason"], "voided_at": stamp}

    positions = portfolio["positions"]
    orders = portfolio.get("orders") or {}
    entries = portfolio.get("ledger") or []
    for step in report["steps"]:
        positions[step["key"]].update(mark)
        for key in step["orders"]:
            orders[key].update(mark)
        for i in step["ledger"]:
            entries[i].update(mark)

    portfolio["cash_cents"] = (int(portfolio.get("cash_cents") or 0)
                               + report["cash_delta_cents"])
    return report
