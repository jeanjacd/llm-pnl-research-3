"""
site/model.py
=============
Turn the paper ledger into the numbers the page prints. Pure functions over a
loaded `portfolio.json` -- no I/O, no rendering, no network.

THE FIXTURE IS THE TICKET. `paper/clv.py` establishes that the unit of
independence is the fixture, not the bet: one match produced `away_win`,
`away_over_0.5`, `away_over_1.5`, `away_over_2.5` and two spreads off a single
scoreline grid, which is one directional view written six ways. Averaging over
bets would overstate the sample by roughly 34x. So every headline rate here is
computed over per-fixture means and carries the DISTINCT FIXTURE count as its
sample size, not the bet count.

WHAT A UNIT IS. One unit is 1% of the starting bankroll -- $10 on the $1,000
book. Stated here because a unit is a convention, and an unstated convention is
just a number nobody can check.

THE FORM NOTATION. Racing form adapted to a book that mostly declines to bet:

    .   boarded, no action taken        (the abstention -- most of the record)
    0-9 markets that cashed on a fixture that still finished DOWN
    v   the fixture finished up
    /   month boundary

Reading it answers the question that matters for this system: when it does act,
does it lose with nothing landing, or with several landing and still finish
down? The first is a pricing problem, the second is a sizing problem. The
abstentions are not filler -- 30 of the first 35 boarded fixtures were declined,
and a record that hid that would be describing a different system.
"""
from __future__ import annotations

import collections
import datetime as dt

# One unit is 1% of the starting bankroll.
UNIT_FRACTION = 0.01
FORM_DECLINED = "·"
FORM_CASHED = "✓"
FORM_MONTH = "/"


def unit_cents(portfolio: dict) -> float:
    start = float(portfolio.get("starting_cash_cents") or 0)
    return max(1.0, start * UNIT_FRACTION)


def _parse(value):
    if not value:
        return None
    text = str(value).strip().replace(" ", "T").replace("Z", "+00:00")
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)


def fixture_key(league_id, home, away, kickoff) -> tuple:
    return (str(league_id or ""), str(home or ""), str(away or ""),
            str(kickoff or "")[:10])


def positions(portfolio: dict) -> list:
    return list((portfolio.get("positions") or {}).values())


def orders(portfolio: dict) -> list:
    return list((portfolio.get("orders") or {}).values())


# --- fixtures -----------------------------------------------------------------
def fixtures(portfolio: dict) -> list:
    """Every fixture the book has touched, newest first.

    A fixture appears if it was boarded, or if a position was taken on it, or
    both. A boarded-but-declined fixture is a first-class row: it is the
    decision the system makes most often, and leaving it out would flatter the
    record by hiding everything that was passed over.
    """
    unit = unit_cents(portfolio)
    rows: dict = {}

    for raw, rec in (portfolio.get("boarded") or {}).items():
        # A boarded record carries `home`/`away` but no league and no kickoff.
        # Its KEY does: "league|home|away|YYYY-MM-DD", written by
        # `paper.selection.fixture_key`. Reading the key rather than the record
        # is what keeps a declined fixture joinable to the positions later
        # taken on it -- otherwise every abstention would land in its own row
        # under an empty league and never merge.
        parts = str(raw).split("|")
        if len(parts) == 4:
            key = tuple(parts)
        else:
            key = fixture_key(None, rec.get("home"), rec.get("away"),
                              rec.get("first_boarded_at"))
        rows.setdefault(key, _blank(key))
        row = rows[key]
        row["boarded"] = True
        row["action"] = rec.get("action")
        row["decided_by"] = rec.get("decided_by")
        row["reason"] = rec.get("reason")
        row["attempts"] = rec.get("attempts")
        row["hours_to_kickoff"] = rec.get("hours_to_kickoff")
        row["markets_considered"] = rec.get("markets_considered")
        row["markets_approved"] = rec.get("markets_approved")
        row["boarded_at"] = rec.get("first_boarded_at") or rec.get("ts")

    for pos in positions(portfolio):
        key = fixture_key(pos.get("league_id"), pos.get("home_team"),
                          pos.get("away_team"), pos.get("kickoff_utc"))
        rows.setdefault(key, _blank(key))
        rows[key]["positions"].append(pos)
        if pos.get("kickoff_utc"):
            rows[key]["kickoff_utc"] = pos["kickoff_utc"]

    for row in rows.values():
        _summarise(row, unit)

    out = list(rows.values())
    out.sort(key=lambda r: (r.get("kickoff_utc") or r.get("boarded_at") or ""),
             reverse=True)
    return out


def _blank(key) -> dict:
    league, home, away, day = key
    return {"key": key, "league_id": league, "home": home, "away": away,
            "date": day, "kickoff_utc": None, "positions": [],
            "boarded": False, "action": None, "decided_by": None,
            "reason": None, "attempts": None, "hours_to_kickoff": None,
            "markets_considered": None, "markets_approved": None,
            "boarded_at": None}


def _summarise(row: dict, unit: float) -> None:
    held = row["positions"]
    settled = [p for p in held if p.get("settled")]
    row["n_markets"] = len(held)
    row["n_settled"] = len(settled)
    row["n_open"] = len(held) - len(settled)
    row["n_cashed"] = sum(1 for p in settled
                          if float(p.get("realized_pnl_cents") or 0) > 0)
    row["pnl_cents"] = sum(float(p.get("realized_pnl_cents") or 0)
                           for p in settled)
    row["pnl_units"] = row["pnl_cents"] / unit
    row["staked_cents"] = sum(float(p.get("size") or 0)
                              * float(p.get("avg_cost_cents") or 0)
                              for p in held)
    # What the still-open markets pay if every one of them lands. A binary
    # settles at 100c, so the profit on a contract bought at P is (100 - P)
    # less its fees. This is NOT `pnl_cents`: that sums SETTLED positions, and
    # a live fixture has none by definition, which is why the card's footer
    # read "+0.00u" under "if every open market holds" -- a true statement
    # about a number nobody wanted and a false answer to the question asked.
    still_open = [p for p in held if not p.get("settled")]
    row["open_upside_cents"] = sum(
        float(p.get("size") or 0) * (100.0 - float(p.get("avg_cost_cents") or 0))
        - float(p.get("fees_cents") or 0) for p in still_open)
    row["open_upside_units"] = row["open_upside_cents"] / unit
    row["open_staked_cents"] = sum(float(p.get("size") or 0)
                                   * float(p.get("avg_cost_cents") or 0)
                                   for p in still_open)
    row["open_staked_units"] = row["open_staked_cents"] / unit
    scored = [float(p["clv_cents"]) for p in held
              if p.get("clv_cents") is not None]
    row["clv_cents"] = (sum(scored) / len(scored)) if scored else None
    row["n_clv"] = len(scored)
    # A fixture is "acted on" when money actually went out on it.
    row["acted"] = bool(held)
    row["live"] = row["n_open"] > 0
    row["settled_fixture"] = bool(settled) and row["n_open"] == 0


def form_figure(row: dict) -> str:
    """One character for one fixture -- see the module docstring."""
    if not row["acted"]:
        return FORM_DECLINED
    if row["n_open"] > 0:
        return FORM_DECLINED       # still running; not part of the record yet
    if row["pnl_cents"] > 0:
        return FORM_CASHED
    return str(min(9, row["n_cashed"]))


def form_line(portfolio: dict, limit: int = 48) -> list:
    """The record, oldest first, with month rules inserted.

    Only fixtures that have finished -- or were declined -- appear. A live
    fixture has no result to encode, and inventing one would be the same error
    as settling a match that has not been played.
    """
    done = [r for r in fixtures(portfolio)
            if (not r["acted"]) or r["settled_fixture"]]
    done.sort(key=lambda r: (r.get("kickoff_utc") or r.get("boarded_at") or ""))
    done = done[-limit:]

    out, month = [], None
    for row in done:
        stamp = _parse(row.get("kickoff_utc") or row.get("boarded_at"))
        this = (stamp.year, stamp.month) if stamp else None
        if month is not None and this is not None and this != month:
            out.append({"char": FORM_MONTH, "kind": "brk", "detail": "month"})
        month = this if this is not None else month
        out.append({"char": form_figure(row), "kind": _form_kind(row),
                    "detail": form_detail(row, stamp)})
    return out


def _form_kind(row: dict) -> str:
    if not row["acted"]:
        return "declined"
    if row["pnl_cents"] > 0:
        return "cash"
    # Losing with several markets landing is a different failure from losing
    # with none, and the page distinguishes them by weight.
    return "late" if row["n_cashed"] else "early"


def _clip(text: str, limit: int) -> str:
    """Trim to a word boundary. A readout cut mid-word looks like a bug."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]).rstrip(".,;:") + "…"


def form_detail(row: dict, stamp=None) -> str:
    stamp = stamp or _parse(row.get("kickoff_utc") or row.get("boarded_at"))
    when = stamp.strftime("%d %b") if stamp else "--"
    match = "%s v %s" % (row["home"], row["away"])
    if not row["acted"]:
        why = _clip(row.get("reason") or "", 72)
        head = "%s · %s · %s" % (when, match,
                                 (row.get("action") or "declined").lower())
        return "%s · %s" % (head, why) if why else head
    return "%s · %s · %d of %d cashed · %s" % (
        when, match, row["n_cashed"], row["n_settled"],
        signed_units(row["pnl_units"]))


# --- headline numbers ---------------------------------------------------------
def signed_units(units: float, places: int = 2) -> str:
    sign = "+" if units >= 0 else "−"
    return "%s%.*fu" % (sign, places, abs(units))


def clv(portfolio: dict) -> dict:
    """Closing line value, averaged over FIXTURES rather than bets.

    n is the distinct-fixture count. Reporting the bet count would inflate the
    sample by the number of correlated markets taken on each match, which on
    this book has run as high as 34.
    """
    per_fixture = [r["clv_cents"] for r in fixtures(portfolio)
                   if r.get("clv_cents") is not None]
    scored = [p for p in positions(portfolio) if p.get("clv_cents") is not None]
    if not per_fixture:
        return {"mean_cents": None, "n_fixtures": 0, "n_bets": len(scored),
                "beat": 0, "beat_rate": None}
    beat = sum(1 for v in per_fixture if v > 0)
    return {"mean_cents": sum(per_fixture) / len(per_fixture),
            "n_fixtures": len(per_fixture), "n_bets": len(scored),
            "beat": beat, "beat_rate": beat / len(per_fixture)}


def pnl(portfolio: dict) -> dict:
    unit = unit_cents(portfolio)
    settled = [p for p in positions(portfolio) if p.get("settled")]
    realized = sum(float(p.get("realized_pnl_cents") or 0) for p in settled)
    staked = sum(float(p.get("size") or 0) * float(p.get("avg_cost_cents") or 0)
                 for p in settled)
    fees = sum(float(p.get("fees_cents") or 0) for p in positions(portfolio))
    won = sum(1 for p in settled
              if float(p.get("realized_pnl_cents") or 0) > 0)
    return {
        "realized_cents": realized,
        "realized_units": realized / unit,
        "staked_cents": staked,
        "staked_units": staked / unit,
        "roi": (realized / staked) if staked else None,
        "fees_cents": fees,
        "n_settled": len(settled),
        "n_won": won,
        "strike_rate": (won / len(settled)) if settled else None,
        "unit_cents": unit,
    }


def fills(portfolio: dict) -> dict:
    """How often a resting order was actually reached.

    Reported beside the CLV because the filled set is SELECTED: a resting order
    is reached when the market comes toward it, so fills correlate with adverse
    movement. The rate is the reader's handle on how selected the sample is.
    """
    every = orders(portfolio)
    done = [o for o in every if o.get("status") in ("filled", "expired")]
    filled = [o for o in done if o.get("status") == "filled"]
    limits = [o for o in every if o.get("kind") == "limit"]
    return {"n_orders": len(every), "n_resolved": len(done),
            "n_filled": len(filled), "n_open": len(every) - len(done),
            "n_limit": len(limits),
            "rate": (len(filled) / len(done)) if done else None}


def board(portfolio: dict) -> dict:
    """What the board decided, including everything it declined.

    30 of the first 35 boarded fixtures were declined. That is the system
    working, not the system idle, and it is the single most distinctive thing
    about this record.
    """
    recs = list((portfolio.get("boarded") or {}).values())
    actions = collections.Counter(r.get("action") or "unknown" for r in recs)
    considered = [int(r["markets_considered"]) for r in recs
                  if r.get("markets_considered")]
    acted = sum(n for a, n in actions.items() if str(a).startswith("PAPER_"))
    return {
        "n_fixtures": len(recs),
        "actions": dict(actions),
        "n_acted": acted,
        "n_declined": len(recs) - acted,
        "decline_rate": ((len(recs) - acted) / len(recs)) if recs else None,
        "markets_considered": sum(considered),
        "median_considered": _median(considered),
    }


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def by_league(portfolio: dict) -> list:
    unit = unit_cents(portfolio)
    rows: dict = collections.defaultdict(
        lambda: {"n_markets": 0, "n_settled": 0, "pnl_cents": 0.0,
                 "clv": [], "fixtures": set()})
    for pos in positions(portfolio):
        row = rows[pos.get("league_id") or "?"]
        row["n_markets"] += 1
        row["fixtures"].add(fixture_key(pos.get("league_id"),
                                        pos.get("home_team"),
                                        pos.get("away_team"),
                                        pos.get("kickoff_utc")))
        if pos.get("settled"):
            row["n_settled"] += 1
            row["pnl_cents"] += float(pos.get("realized_pnl_cents") or 0)
        if pos.get("clv_cents") is not None:
            row["clv"].append(float(pos["clv_cents"]))
    out = []
    for league, row in rows.items():
        out.append({
            "league_id": league,
            "n_markets": row["n_markets"],
            "n_fixtures": len(row["fixtures"]),
            "n_settled": row["n_settled"],
            "pnl_units": row["pnl_cents"] / unit,
            "clv_cents": (sum(row["clv"]) / len(row["clv"])
                          if row["clv"] else None),
            "n_clv": len(row["clv"]),
        })
    out.sort(key=lambda r: -r["n_markets"])
    return out


def claim_families(portfolio: dict) -> list:
    """Full-match against first-half, which is the newest thing the book does."""
    rows: dict = collections.defaultdict(
        lambda: {"n": 0, "settled": 0, "pnl_cents": 0.0, "clv": []})
    for pos in positions(portfolio):
        claim = str(pos.get("claim") or "")
        base = claim[4:] if claim.startswith("not_") else claim
        half = base.startswith("1h_")
        if half:
            base = base[3:]
        # `home_over_1.5` is a TEAM TOTAL; `home_wins_by_over_1.5` is a
        # spread. They differ by one word and price completely differently,
        # so the ordering here puts the spread test first -- the reverse
        # silently files every spread under team totals.
        kind = ("total" if base.startswith("total") else
                "score" if base.startswith("score") else
                "btts" if base.startswith("btts") else
                "spread" if "wins_by" in base else
                "team total" if base.startswith(("home_over", "away_over"))
                else "result")
        row = rows[("1H" if half else "FT", kind)]
        row["n"] += 1
        if pos.get("settled"):
            row["settled"] += 1
            row["pnl_cents"] += float(pos.get("realized_pnl_cents") or 0)
        if pos.get("clv_cents") is not None:
            row["clv"].append(float(pos["clv_cents"]))
    unit = unit_cents(portfolio)
    out = [{"period": p, "family": f, "n": r["n"], "settled": r["settled"],
            "pnl_units": r["pnl_cents"] / unit,
            "clv_cents": (sum(r["clv"]) / len(r["clv"])) if r["clv"] else None,
            "n_clv": len(r["clv"])}
           for (p, f), r in rows.items()]
    out.sort(key=lambda r: (r["period"], -r["n"]))
    return out


def equity_curve(portfolio: dict) -> list:
    """Cumulative realised units against cumulative expected, per settlement.

    Expected is accumulated from the price actually paid: a contract bought at
    P cents has an expected return of zero at that price, so the expected curve
    is flat unless CLV says the market moved. Where a closing line exists, the
    expected step is the CLV -- what the position was worth at the number the
    market closed at, which is the only forward-looking value available.
    """
    unit = unit_cents(portfolio)
    ledger = sorted((portfolio.get("ledger") or []),
                    key=lambda e: str(e.get("ts") or ""))
    by_instrument = {}
    for pos in positions(portfolio):
        by_instrument[(pos.get("instrument_id"), pos.get("side"))] = pos

    out, actual, expected = [], 0.0, 0.0
    for entry in ledger:
        actual += float(entry.get("pnl_cents") or 0)
        pos = by_instrument.get((entry.get("instrument_id"),
                                 entry.get("side")))
        if pos is not None and pos.get("clv_cents") is not None:
            expected += float(pos["clv_cents"]) * float(pos.get("size") or 0)
        out.append({"ts": entry.get("ts"),
                    "actual_units": actual / unit,
                    "expected_units": expected / unit,
                    "instrument_id": entry.get("instrument_id"),
                    "won": bool(entry.get("won"))})
    return out


def daily_ledger(portfolio: dict) -> list:
    """One mark per day: result by sign, stake by width."""
    unit = unit_cents(portfolio)
    days: dict = collections.defaultdict(lambda: {"pnl": 0.0, "stake": 0.0})
    by_instrument = {}
    for pos in positions(portfolio):
        by_instrument[(pos.get("instrument_id"), pos.get("side"))] = pos
    for entry in (portfolio.get("ledger") or []):
        day = str(entry.get("ts") or "")[:10]
        if not day:
            continue
        days[day]["pnl"] += float(entry.get("pnl_cents") or 0)
        pos = by_instrument.get((entry.get("instrument_id"),
                                 entry.get("side")))
        if pos is not None:
            days[day]["stake"] += (float(pos.get("size") or 0)
                                   * float(pos.get("avg_cost_cents") or 0))
    return [{"date": day, "pnl_units": row["pnl"] / unit,
             "stake_units": row["stake"] / unit}
            for day, row in sorted(days.items())]


def summary(portfolio: dict) -> dict:
    """Everything the page needs, in one structure."""
    return {
        "saved_at": portfolio.get("saved_at"),
        "cash_cents": portfolio.get("cash_cents"),
        "reserved_cents": portfolio.get("reserved_cents"),
        "starting_cash_cents": portfolio.get("starting_cash_cents"),
        "unit_cents": unit_cents(portfolio),
        "clv": clv(portfolio),
        "pnl": pnl(portfolio),
        "fills": fills(portfolio),
        "board": board(portfolio),
        "fixtures": fixtures(portfolio),
        "form": form_line(portfolio),
        "leagues": by_league(portfolio),
        "families": claim_families(portfolio),
        "equity": equity_curve(portfolio),
        "daily": daily_ledger(portfolio),
    }
