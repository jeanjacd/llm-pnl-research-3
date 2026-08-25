"""
paper/settlement.py
===================
Resolve open paper positions against finished matches.

Until this ran, the paper loop could not produce a P&L at any cadence: orders
were submitted and either expired or sat forever, `PaperPortfolio.settle` had
no production caller, and `realized_pnl_usd` was structurally 0.00.

Every path that cannot produce a trustworthy result SKIPS and is counted, never
guessed. A position that cannot be settled stays open and visible; it is never
quietly resolved to a loss, which would read as model error rather than as a
gap in the plumbing.
"""
from __future__ import annotations

import datetime as dt

from .outcomes import UnsettleableClaim, regulation_score, winning_side

# A postponed match reappears on a later date, so the fixture is looked up in a
# window around the stored kick-off rather than on the exact day. The window is
# narrow and the match must be UNIQUE within it: two teams meeting twice inside
# three days would be refused rather than settled against a guess.
LOOKUP_WINDOW_DAYS = 3


def _as_date(value):
    if value is None:
        return None
    try:
        import pandas as pd
        stamp = pd.Timestamp(value)
    except Exception:                                         # noqa: BLE001
        return None
    if stamp is None or (hasattr(stamp, "tz") and stamp.tz is not None):
        stamp = stamp.tz_convert(None) if stamp.tzinfo else stamp
    return stamp.tz_localize(None).normalize() if stamp.tzinfo else stamp.normalize()


def find_fixture(frame, home_team, away_team, kickoff_utc):
    """The unique played row for this fixture, or None.

    Matched on OUR canonical team names -- the ones `resolve_fixture` wrote
    when the position was opened -- so settlement never re-runs venue name
    matching, which could resolve differently weeks later.
    """
    if frame is None or len(frame) == 0 or not home_team or not away_team:
        return None
    import pandas as pd
    same = frame[(frame["home_team"] == home_team)
                 & (frame["away_team"] == away_team)]
    if not len(same):
        return None
    day = _as_date(kickoff_utc)
    if day is not None and "date" in same.columns:
        window = pd.Timedelta(days=LOOKUP_WINDOW_DAYS)
        same = same[(same["date"] >= day - window)
                    & (same["date"] <= day + window)]
    if len(same) != 1:
        return None
    return same.iloc[0]


def settle_portfolio(portfolio, frames: dict, now=None) -> dict:
    """Settle every open position whose match has finished in regulation.

    `frames` maps league_id -> that league's full match table (played rows
    included). Returns a counted breakdown; the counts are the point, because
    a position that silently fails to settle is indistinguishable from a
    losing bet in the headline P&L.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    stats = {"settled": 0, "pnl_cents": 0.0, "still_open": 0,
             "no_result_yet": 0, "missing_identity": 0,
             "fixture_not_found": 0, "unsettleable_claim": 0,
             "problems": []}

    for pos in list(portfolio.positions.values()):
        if pos.settled:
            continue
        if not pos.claim or not pos.home_team or not pos.away_team:
            # Written before positions carried their settlement identity.
            stats["missing_identity"] += 1
            stats["still_open"] += 1
            continue
        row = find_fixture(frames.get(pos.league_id), pos.home_team,
                           pos.away_team, pos.kickoff_utc)
        if row is None:
            stats["fixture_not_found"] += 1
            stats["still_open"] += 1
            continue
        score = regulation_score(row)
        if score is None:
            stats["no_result_yet"] += 1
            stats["still_open"] += 1
            continue
        try:
            result = winning_side(pos.claim, score[0], score[1])
        except UnsettleableClaim as exc:
            stats["unsettleable_claim"] += 1
            stats["still_open"] += 1
            stats["problems"].append("%s: %s" % (pos.instrument_id, exc))
            continue
        pnl = portfolio.settle(pos.instrument_id, pos.side, result)
        stats["settled"] += 1
        stats["pnl_cents"] += pnl
    return stats
