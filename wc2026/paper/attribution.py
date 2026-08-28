"""
paper/attribution.py
====================
Split what the book earned into FORECAST and SPREAD CAPTURE.

The board rests orders on both sides of the same claim -- 83 such pairs in one
live run, at a combined 83-87c for a guaranteed 100c. That is not a mistake:
if both legs fill it is locked profit, and quoting both sides to capture the
width is a real strategy. But it is a DIFFERENT strategy from the one being
measured, and mixing them destroys the measurement.

The arithmetic is decisive. Hold `yes` at Py and `no` at Pn on one claim, and
let the closing price of the yes side be c:

    CLV(yes) = c - Py
    CLV(no)  = (100 - c) - Pn
    sum      = 100 - (Py + Pn)          <- c cancels

The fixture contributes a CONSTANT, carrying no information about whether the
forecast beat the market. Left in the headline it would look like edge while
measuring nothing but the spread.

So each claim is decomposed:

    matched  = min(size_yes, size_no)   -> spread capture, locked, no view
    residual = |size_yes - size_no|     -> forecast, the only directional part

Only the residual reaches the forecast numbers. Nothing about trading changes.

CROSS-VENUE MATCHING IS REFUSED UNLESS THE SETTLEMENT BASIS IS KNOWN AND EQUAL.
A `yes` on Kalshi and a `no` on Polymarket offset only if both settle on the
same event definition. `venues.base.equivalent` already refuses an unknown
basis, on the grounds that "probably the same rules" is what produces a losing
trade; the same rule applies here, and for the same reason. A pair that is not
genuinely offsetting is live risk, and calling it captured spread would book a
riskless profit that does not exist. Unmatched legs stay in the forecast
bucket, where their risk is counted honestly.

A CAVEAT THAT BELONGS NEXT TO THE NUMBER. The residual is a SELECTED sample:
which leg fills depends on which way the market moved, and resting orders are
reached when the market comes toward them. So forecast CLV is conditional on
having been filled, and fills correlate with adverse movement. That is inherent
to measuring any resting-order strategy, not something this decomposition
introduces or can remove -- which is why fill rate is reported beside it.
"""
from __future__ import annotations

import collections


def _claim_key(position) -> tuple:
    """The proposition, independent of which side of it is held."""
    claim = str(position.claim or "")
    base = claim[4:] if claim.startswith("not_") else claim
    return (position.league_id, position.home_team, position.away_team,
            str(position.kickoff_utc or "")[:10], base)


def _held_side(position) -> str:
    """"yes" or "no" for the PROPOSITION, folding the not_ prefix in.

    Holding `no` on `draw` and holding `yes` on `not_draw` are the same
    economic position, and both must fold to the same side or an offsetting
    pair would look like two bets in the same direction.
    """
    negated = str(position.claim or "").startswith("not_")
    side = position.side == "yes"
    return "yes" if side != negated else "no"


def _matchable(a, b) -> bool:
    """May these two positions be netted against each other?

    Same venue: always -- one market, one settlement rule.
    Across venues: only when BOTH settlement bases are known and identical.
    """
    if a.venue == b.venue:
        return True
    if a.settles_on_regulation is None or b.settles_on_regulation is None:
        return False
    return a.settles_on_regulation == b.settles_on_regulation


def decompose(portfolio) -> dict:
    """Per claim: the matched (spread) and residual (forecast) components.

    Returns {claim_key: {"matched": n, "spread_cost_cents": c,
                         "residual": n, "residual_side": "yes"|"no",
                         "residual_positions": [(position, size), ...],
                         "unmatched_reason": str|None}}.
    """
    grouped = collections.defaultdict(lambda: {"yes": [], "no": []})
    for pos in portfolio.positions.values():
        if not pos.claim or pos.size <= 0:
            continue
        grouped[_claim_key(pos)][_held_side(pos)].append(pos)

    out = {}
    for key, sides in grouped.items():
        yes, no = list(sides["yes"]), list(sides["no"])
        matched = 0
        spread_cost = 0.0
        refused = False
        # Net greedily, cheapest legs first, so the pairs booked as locked are
        # the ones that genuinely were.
        yes.sort(key=lambda p: p.avg_cost_cents)
        no.sort(key=lambda p: p.avg_cost_cents)
        left = {id(p): p.size for p in yes + no}
        for a in yes:
            for b in no:
                if left[id(a)] <= 0:
                    break
                if left[id(b)] <= 0:
                    continue
                if not _matchable(a, b):
                    refused = True
                    continue
                take = min(left[id(a)], left[id(b)])
                matched += take
                spread_cost += take * (a.avg_cost_cents + b.avg_cost_cents)
                left[id(a)] -= take
                left[id(b)] -= take

        residual = [(p, left[id(p)]) for p in yes + no if left[id(p)] > 1e-9]
        side = None
        if residual:
            side = _held_side(residual[0][0])
        out[key] = {
            "matched": matched,
            "spread_cost_cents": spread_cost,
            # Each matched pair pays exactly 100c whatever happens.
            "spread_pnl_cents": matched * 100.0 - spread_cost,
            "residual": sum(size for _, size in residual),
            "residual_side": side,
            "residual_positions": residual,
            "unmatched_reason": ("settlement basis differs or is unknown"
                                 if refused else None),
        }
    return out


def residual_share(portfolio) -> dict:
    """position key -> the fraction of that position which is directional.

    A position can be partly netted and partly not, so attribution is by SIZE
    rather than all-or-nothing. 1.0 means the whole holding expresses a view;
    0.0 means it is entirely one leg of a locked pair.
    """
    shares = {}
    for claim in decompose(portfolio).values():
        residual = {id(pos): size for pos, size in claim["residual_positions"]}
        for pos, _ in claim["residual_positions"]:
            shares[id(pos)] = residual[id(pos)] / pos.size if pos.size else 0.0
    for pos in portfolio.positions.values():
        shares.setdefault(id(pos), 0.0)
    return shares


def attribute(portfolio) -> dict:
    """Realised P&L split into forecast and spread capture, by fixture.

    Conservation is exact and asserted by the tests: for every fixture,
    forecast + spread == the realised total. The split can therefore neither
    lose money nor invent it, which matters because nothing downstream would
    catch a discrepancy.

    Fees follow their contracts: a position that is 40% directional carries 40%
    of its fees into the forecast bucket and 60% into spread capture. Charging
    them all to one side would flatter it.
    """
    shares = residual_share(portfolio)
    positions = {(p.instrument_id, p.side): p
                 for p in portfolio.positions.values()}
    out = collections.defaultdict(
        lambda: {"forecast_pnl_cents": 0.0, "spread_pnl_cents": 0.0,
                 "total_pnl_cents": 0.0, "forecast_contracts": 0.0,
                 "matched_contracts": 0.0, "n_settled": 0})

    for entry in portfolio.ledger or []:
        pos = positions.get((entry.get("instrument_id"), entry.get("side")))
        pnl = float(entry.get("pnl_cents") or 0.0)
        if pos is None:
            key = ("?", str(entry.get("instrument_id")), "", "")
            out[key]["forecast_pnl_cents"] += pnl
            out[key]["total_pnl_cents"] += pnl
            out[key]["n_settled"] += 1
            continue
        key = (pos.league_id, pos.home_team, pos.away_team,
               str(pos.kickoff_utc or "")[:10])
        share = shares.get(id(pos), 1.0)
        bucket = out[key]
        bucket["forecast_pnl_cents"] += pnl * share
        bucket["spread_pnl_cents"] += pnl * (1.0 - share)
        bucket["total_pnl_cents"] += pnl
        bucket["forecast_contracts"] += pos.size * share
        bucket["matched_contracts"] += pos.size * (1.0 - share)
        bucket["n_settled"] += 1
    return dict(out)


def forecast_clv(portfolio) -> list:
    """(fixture_key, clv_cents, contracts) for the DIRECTIONAL part only.

    An offsetting pair's CLV sums to a constant regardless of the closing
    price, so including it would report the captured spread as forecasting
    skill. Only the residual carries a view, and only it is scored.
    """
    shares = residual_share(portfolio)
    rows = []
    for pos in portfolio.positions.values():
        if pos.clv_cents is None:
            continue
        share = shares.get(id(pos), 1.0)
        if share <= 0:
            continue
        rows.append(((pos.league_id, pos.home_team, pos.away_team,
                      str(pos.kickoff_utc or "")[:10]),
                     pos.clv_cents, pos.size * share))
    return rows
