"""
site/ledger.py
==============
The betting ledger: what is open, how the run stands, and the book behind it.

REPLACES THE FORM STRIP. That was 48 fixtures encoded as `0-9 . - v /` with a
legend longer than the data it explained, four channels carrying three facts,
and no representation at all of a bet that had not settled yet. What follows
keeps exactly one channel per fact:

    position above/below the baseline   won or lost
    bar length                          how much, in dollars
    fill style                          settled, open, or in play
    the CLV track                       closing line value, and nothing else

WHAT THIS BOOK HAS, AND WHAT IT DOES NOT. There is no live feed here. The page
is a static build that runs every three hours, so "now" means "as of this
edition" and is printed as such. In-play scores, running win probabilities and
cash-out values are not recorded anywhere in the portfolio, so no card claims
them: an open ticket shows what was paid, what it pays if it lands, where the
market closed when that was captured, and what has to happen -- all facts the
record actually holds. Inventing the rest would put a number on the page that
no part of the system could check.

WHAT HAS TO HAPPEN is a pure function of the claim (`needs`), and it is the one
thing the old strip could never say. A row reading "1st half Real Betis do not
win" is a proposition; "Real Betis must not be ahead at half time" is an
instruction, and it is the same fact.
"""
from __future__ import annotations

import datetime as dt
import math
import re

from .model import _parse, fixture_key, fixtures, is_void, positions

# A fixture is in play from kick-off until the whistle plus a margin for
# stoppage and the walk to the settlement feed. Anything still unsettled after
# that is waiting on a result, not on football.
IN_PLAY_MINUTES = 115
STATE_ORDER = {"live": 0, "awaiting": 1, "pending": 2}

# One outlier should not flatten every other bar, and hiding it would be worse.
# Bars scale to the 95th percentile; anything past it is drawn full and notched.
CLAMP_PERCENTILE = 0.95


def slug(key) -> str:
    """A stable, filesystem-safe id for a fixture, used as its page name."""
    league, home, away, day = key
    text = "-".join(str(part or "") for part in (day, league, home, away))
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text or "fixture"


# --- what has to happen -------------------------------------------------------
def needs(claim: str, home: str, away: str) -> str:
    """One plain sentence saying what would make this position pay.

    Keyed on the market type, never a generic string. The claim already carries
    the side -- `paper.cycle` writes the no side as `"not_" + leg.claim` -- so
    this reads it exactly the way `outcomes.claim_is_true` settles it. If the
    two ever disagree the sentence is a lie about the money, which is why it is
    derived from the claim rather than written per position.
    """
    text = str(claim or "")
    negated = text.startswith("not_")
    base = text[4:] if negated else text
    half = base.startswith("1h_")
    if half:
        base = base[3:]
    when = "at half time" if half else "at full time"
    lead = "by the interval" if half else "by full time"
    line = base.rsplit("_", 1)[-1]

    if base == "home_win":
        body = "%s must %sbe ahead %s" % (home, "not " if negated else "", when)
    elif base == "away_win":
        body = "%s must %sbe ahead %s" % (away, "not " if negated else "", when)
    elif base == "draw":
        body = "the score must %sbe level %s" % ("not " if negated else "", when)
    elif base == "btts":
        body = ("at least one side must fail to score %s" % lead if negated
                else "both sides must score %s" % lead)
    elif base.startswith(("total_over_", "total_under_")):
        over = base.startswith("total_over_") != negated
        body = "%s than %s goals must be scored %s" % (
            "more" if over else "fewer", line, lead)
    elif base.startswith(("home_wins_by_over_", "away_wins_by_over_")):
        who = home if base.startswith("home") else away
        body = "%s must %swin by more than %s" % (
            who, "not " if negated else "", line)
    elif base.startswith(("home_over_", "away_over_")):
        who = home if base.startswith("home_") else away
        body = "%s must score %s than %s %s" % (
            who, "fewer" if negated else "more", line, lead)
    elif base.startswith("score_"):
        try:
            home_goals, away_goals = base.split("_", 1)[1].split("-")
        except ValueError:
            return "The scoreline must settle this market."
        body = ("the score must be anything but %s-%s %s"
                % (home_goals, away_goals, when) if negated else
                "the score must be exactly %s-%s %s"
                % (home_goals, away_goals, when))
    else:
        return "This market settles on the final scoreline."
    return body[0].upper() + body[1:] + "."


# --- zone 1: what is open -----------------------------------------------------
def open_tickets(portfolio: dict, now=None) -> list:
    """Every unsettled position, grouped by fixture, sorted by urgency.

    In play first (nearest the whistle first), then waiting on a result, then
    what has not kicked off, soonest first. That is the order of "tell me the
    thing I might still be able to do something about".
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    rows = {}
    for pos in positions(portfolio):
        if pos.get("settled") or is_void(pos):
            continue
        key = fixture_key(pos.get("league_id"), pos.get("home_team"),
                          pos.get("away_team"), pos.get("kickoff_utc"))
        row = rows.setdefault(key, {
            "key": key, "slug": slug(key), "league_id": key[0],
            "home": pos.get("home_team"), "away": pos.get("away_team"),
            "kickoff_utc": pos.get("kickoff_utc"), "legs": []})
        row["legs"].append(pos)

    out = []
    for row in rows.values():
        kickoff = _parse(row["kickoff_utc"])
        row["kickoff"] = kickoff
        row["state"] = _state(kickoff, now)
        row["minutes_to_kickoff"] = (
            None if kickoff is None
            else (kickoff - now).total_seconds() / 60.0)
        row["staked_cents"] = sum(_stake(p) for p in row["legs"])
        # A binary settles at 100c, so a contract bought at P returns
        # (100 - P) less its fees. This is what is still on the table.
        row["upside_cents"] = sum(
            float(p.get("size") or 0) * (100.0 - _cost(p))
            - float(p.get("fees_cents") or 0) for p in row["legs"])
        scored = [float(p["clv_cents"]) for p in row["legs"]
                  if p.get("clv_cents") is not None]
        row["clv_cents"] = (sum(scored) / len(scored)) if scored else None
        row["n_legs"] = len(row["legs"])
        row["legs"].sort(key=lambda p: -_stake(p))
        for leg in row["legs"]:
            leg["needs"] = needs(leg.get("claim"), row["home"], row["away"])
        out.append(row)

    out.sort(key=lambda r: (STATE_ORDER.get(r["state"], 9), _sort_within(r)))
    return out


def _state(kickoff, now) -> str:
    if kickoff is None or kickoff > now:
        return "pending"
    if (now - kickoff) <= dt.timedelta(minutes=IN_PLAY_MINUTES):
        return "live"
    return "awaiting"


def _sort_within(row):
    """Live: nearest the whistle first. Pending: soonest kick-off first."""
    minutes = row.get("minutes_to_kickoff")
    if minutes is None:
        return 1e9
    return -minutes if row["state"] == "live" else minutes


def _cost(pos) -> float:
    return float(pos.get("avg_cost_cents") or 0)


def _stake(pos) -> float:
    return float(pos.get("size") or 0) * _cost(pos)


def exposure(open_rows: list) -> dict:
    """The one line that answers "how much is riding on this right now"."""
    return {
        "n_fixtures": len(open_rows),
        "n_markets": sum(r["n_legs"] for r in open_rows),
        "n_live": sum(1 for r in open_rows if r["state"] == "live"),
        "n_awaiting": sum(1 for r in open_rows if r["state"] == "awaiting"),
        "n_pending": sum(1 for r in open_rows if r["state"] == "pending"),
        "staked_cents": sum(r["staked_cents"] for r in open_rows),
        "upside_cents": sum(r["upside_cents"] for r in open_rows),
    }


# --- zone 3: the book ---------------------------------------------------------
def columns(portfolio: dict, limit: int = 48, now=None) -> list:
    """One column per fixture, oldest at the left.

    48 of them, which is what fits at the 6px minimum column width on a 380px
    screen without pagination or a horizontal scroll. Everything older lives in
    the settlement archive, one page per fixture.

    Carries every fixture the book has touched -- including the ones it
    declined, which are most of them. A declined fixture is a hairline at the
    baseline: it holds the rhythm of the calendar and claims no attention.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    rows = fixtures(portfolio)
    rows.sort(key=lambda r: (r.get("kickoff_utc") or r.get("boarded_at") or ""))
    rows = rows[-limit:]

    out, month = [], None
    for row in rows:
        stamp = _parse(row.get("kickoff_utc") or row.get("boarded_at"))
        this = (stamp.year, stamp.month) if stamp else None
        out.append({
            "key": row["key"],
            "slug": slug(row["key"]),
            "home": row["home"], "away": row["away"],
            "league_id": row["league_id"],
            "date": row["date"],
            "stamp": stamp,
            "month": stamp.strftime("%b") if stamp else "",
            "new_month": bool(this and month and this != month),
            "outcome": _outcome(row, now),
            "net_cents": row["pnl_cents"],
            "staked_cents": row["staked_cents"],
            "open_staked_cents": row["open_staked_cents"],
            "upside_cents": row["open_upside_cents"],
            "clv_cents": row["clv_cents"],
            "n_markets": row["n_markets"],
            "n_settled": row["n_settled"],
            "n_void": row.get("n_void") or 0,
            "n_cashed": row["n_cashed"],
            "n_unfilled": row["n_unfilled"],
            "n_orders": row["n_orders"],
            "reason": row.get("reason"),
            "action": row.get("action"),
        })
        month = this if this is not None else month
    return out


def _outcome(row, now) -> str:
    """The single fact the column's shape encodes."""
    if row["n_open"] > 0:
        kickoff = _parse(row.get("kickoff_utc"))
        return "live" if (kickoff and kickoff <= now) else "open"
    if not row["acted"]:
        return "unfilled" if row["ordered"] else "declined"
    if not row["n_settled"]:
        return "void" if row.get("n_void") else "open"
    net = row["pnl_cents"]
    return "won" if net > 0 else "lost" if net < 0 else "push"


def scale(cols: list) -> dict:
    """One linear scale for every bar, clamped so an outlier cannot flatten it.

    The clamp is the 95th percentile of the magnitudes actually present. A bar
    past it is drawn full height with a notch, which says "this one is off the
    scale" without letting it set the scale for everything else.
    """
    pool = [abs(float(c["net_cents"] or 0)) for c in cols
            if c["outcome"] in ("won", "lost")]
    # An open bar is drawn at what it PAYS, not what it cost, so the scale has
    # to see the same number or the two would be measured against each other
    # in different units.
    pool += [abs(float(c["open_staked_cents"] or 0))
             + abs(float(c["upside_cents"] or 0)) for c in cols
             if c["outcome"] in ("open", "live")]
    pool = sorted(v for v in pool if v > 0)
    if not pool:
        return {"top_cents": 1.0, "clamped": 0}
    # Nearest-rank, so the clamp behaves at small n. Interpolating on
    # `len - 1` truncated to index 0 for a two-fixture book and scaled every
    # bar off the SMALLER of the two, which is the opposite of a clamp.
    idx = min(len(pool) - 1,
              max(0, math.ceil(CLAMP_PERCENTILE * len(pool)) - 1))
    top = pool[idx] or pool[-1]
    return {"top_cents": float(top or 1.0),
            "clamped": sum(1 for v in pool if v > top)}


def height(value_cents, scale_info) -> tuple:
    """(0..1 height, overflowed) for a magnitude on the shared scale."""
    top = float(scale_info.get("top_cents") or 1.0)
    magnitude = abs(float(value_cents or 0))
    if magnitude <= 0:
        return 0.0, False
    return min(1.0, magnitude / top), magnitude > top


# --- zone 2: the run ----------------------------------------------------------
def run(cols: list) -> dict:
    """Cumulative dollars across the same columns the book draws.

    Shares the book's x-positions exactly, so the two read as one object. The
    cone at the right edge is the open position: its floor is every open market
    losing its stake, its ceiling every one of them landing. That is exposure
    stated as a range, not a forecast, and it is the honest way to show an
    unfinished position beside a finished record.
    """
    points, total = [], 0.0
    for i, col in enumerate(cols):
        if col["outcome"] in ("won", "lost", "push"):
            total += float(col["net_cents"] or 0)
        points.append({"i": i, "cents": total,
                       "settled": col["outcome"] in ("won", "lost", "push")})
    open_stake = sum(float(c["open_staked_cents"] or 0) for c in cols)
    open_upside = sum(float(c["upside_cents"] or 0) for c in cols)
    return {"points": points, "final_cents": total,
            "floor_cents": total - open_stake,
            "ceiling_cents": total + open_upside,
            "has_open": open_stake > 0 or open_upside > 0}


# --- the settlement archive ---------------------------------------------------
def settlements(portfolio: dict) -> list:
    """Every settled market ever, newest first, joined to its fixture.

    The front page shows a rolling window; this is the whole record, and it is
    what the archive pages are built from. A measurement system that publishes
    only its recent history is asking to be taken on trust.
    """
    out = []
    for pos in positions(portfolio):
        if not pos.get("settled"):
            continue
        void = is_void(pos)
        key = fixture_key(pos.get("league_id"), pos.get("home_team"),
                          pos.get("away_team"), pos.get("kickoff_utc"))
        out.append({
            "key": key, "slug": slug(key),
            "league_id": key[0],
            "home": pos.get("home_team"), "away": pos.get("away_team"),
            "kickoff_utc": pos.get("kickoff_utc"),
            "date": key[3],
            "claim": pos.get("claim"),
            "venue": pos.get("venue"),
            "instrument_id": pos.get("instrument_id"),
            "size": float(pos.get("size") or 0),
            "cost_cents": _cost(pos),
            "stake_cents": _stake(pos),
            "fees_cents": float(pos.get("fees_cents") or 0),
            "closing_price_cents": pos.get("closing_price_cents"),
            "clv_cents": pos.get("clv_cents"),
            "result": pos.get("result"),
            "payout_cents": float(pos.get("payout_cents") or 0),
            "pnl_cents": float(pos.get("realized_pnl_cents") or 0),
            "won": (not void) and float(pos.get("realized_pnl_cents") or 0) > 0,
            "void": void,
            "void_reason": pos.get("void_reason"),
            "opened_at": pos.get("opened_at"),
        })
    out.sort(key=lambda r: (str(r["kickoff_utc"] or ""), str(r["claim"] or "")),
             reverse=True)
    return out


def settled_fixtures(portfolio: dict) -> list:
    """The archive index: one row per fixture that has settled anything."""
    rows = {}
    for market in settlements(portfolio):
        row = rows.setdefault(market["key"], {
            "key": market["key"], "slug": market["slug"],
            "league_id": market["league_id"], "home": market["home"],
            "away": market["away"], "kickoff_utc": market["kickoff_utc"],
            "date": market["date"], "markets": []})
        row["markets"].append(market)
    out = []
    for row in rows.values():
        counted = [m for m in row["markets"] if not m["void"]]
        row["n_markets"] = len(counted)
        row["n_void"] = len(row["markets"]) - len(counted)
        row["n_won"] = sum(1 for m in counted if m["won"])
        row["pnl_cents"] = sum(m["pnl_cents"] for m in counted)
        row["staked_cents"] = sum(m["stake_cents"] for m in counted)
        scored = [float(m["clv_cents"]) for m in counted
                  if m["clv_cents"] is not None]
        row["clv_cents"] = (sum(scored) / len(scored)) if scored else None
        # Voids sink to the bottom: they are part of the record and not part
        # of the result, and the reader should meet them in that order.
        row["markets"].sort(key=lambda m: (m["void"], -m["pnl_cents"]))
        out.append(row)
    out.sort(key=lambda r: str(r["kickoff_utc"] or ""), reverse=True)
    return out
