"""
paper/clv.py
============
Closing line value: did the market move toward our price after we bought?

CLV is the primary measurement instrument here, and P&L the secondary one,
because CLV converges roughly an order of magnitude faster. Detecting a 0.5c
edge needs ~246 independent bets at the measured std of 2.8c; detecting a 4c
edge in P&L on a 20c binary, where the outcome is 100c or nothing and the std
is nearer 40c, needs ~785. At ~48 fixtures a week that is about five weeks
against sixteen.

THE UNIT OF INDEPENDENCE IS THE FIXTURE, NOT THE BET. Measured 2026-08-24, the
307 placeable cases across both venues covered 9 distinct matches -- 34 bets per
fixture, and not merely correlated ones: a single fixture produced `away_win`,
`away_over_0.5`, `away_over_1.5`, `away_over_2.5`, `away_wins_by_over_1.5` and
`away_wins_by_over_2.5`, which is one directional view written six ways off one
scoreline grid. Averaging CLV over bets would therefore overstate the sample by
roughly 34x, so `clv_summary` reports the distinct-fixture count alongside it
and computes the headline over per-fixture means.

THE CLOSING LINE IS THE PRICE AT KICK-OFF, not any field read afterwards. Both
venues keep trading through the match, so a post-match read is contaminated by
in-play flow. An earlier version of the legacy tracker used a live price field
and produced a CLV of +7c with a std of 38c; read from history at kick-off the
same bets gave -0.2c with a std of 2.8c.
"""
from __future__ import annotations

import datetime as dt


def _parse(value):
    if value is None:
        return None
    text = str(value).strip().replace(" ", "T").replace("Z", "+00:00")
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)


def capture_closing_lines(portfolio, probes: dict, now=None) -> dict:
    """Record the closing line for every position whose match has kicked off.

    Independent of settlement: the closing line exists at kick-off, while the
    result may not be ingested for hours. Recording them separately means a
    slow data refresh delays P&L without also losing the CLV reading, which is
    the faster-converging measurement of the two.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    stats = {"captured": 0, "already_had": 0, "not_kicked_off": 0,
             "no_probe": 0, "no_history": 0}
    for pos in portfolio.positions.values():
        if pos.closing_price_cents is not None:
            stats["already_had"] += 1
            continue
        kickoff = _parse(pos.kickoff_utc)
        if kickoff is None or kickoff > now:
            stats["not_kicked_off"] += 1
            continue
        probe = probes.get(pos.venue)
        if probe is None:
            stats["no_probe"] += 1
            continue
        price = probe.closing_price_cents(pos.instrument_id, pos.side, kickoff)
        if price is None:
            stats["no_history"] += 1
            continue
        pos.closing_price_cents = round(float(price), 2)
        # Positive means the market moved toward us: we bought cheaper than
        # the market's final pre-game price. Fees are excluded by convention,
        # so CLV measures the FORECAST, not the round trip.
        pos.clv_cents = round(pos.closing_price_cents - pos.avg_cost_cents, 2)
        pos.clv_source = getattr(probe, "venue", None)
        stats["captured"] += 1
    return stats


def clv_summary(portfolio) -> dict:
    """Headline CLV, computed per FIXTURE so 34 bets on one match count once.

    Reports the naive per-bet mean too, but only alongside the fixture count,
    so the gap between them stays visible rather than being quietly presented
    as a larger sample than it is.
    """
    scored = [p for p in portfolio.positions.values()
              if p.clv_cents is not None]
    if not scored:
        return {"n_bets": 0, "n_fixtures": 0, "clv_cents": None,
                "clv_cents_per_bet": None, "bets_per_fixture": None}

    by_fixture = {}
    for pos in scored:
        key = (pos.league_id, pos.home_team, pos.away_team, pos.kickoff_utc)
        by_fixture.setdefault(key, []).append(pos.clv_cents)
    per_fixture = [sum(v) / len(v) for v in by_fixture.values()]

    mean = sum(per_fixture) / len(per_fixture)
    if len(per_fixture) > 1:
        var = sum((x - mean) ** 2 for x in per_fixture) / (len(per_fixture) - 1)
        std = var ** 0.5
        stderr = std / (len(per_fixture) ** 0.5)
    else:
        std = stderr = None
    return {
        "n_bets": len(scored),
        "n_fixtures": len(by_fixture),
        "bets_per_fixture": round(len(scored) / len(by_fixture), 1),
        "clv_cents": round(mean, 3),
        "clv_cents_per_bet": round(sum(p.clv_cents for p in scored)
                                   / len(scored), 3),
        "clv_std_cents": None if std is None else round(std, 3),
        "clv_stderr_cents": None if stderr is None else round(stderr, 3),
        # t against zero, on FIXTURES. Below about |2| this is noise, and at
        # these sample sizes it usually is -- which is the honest reading.
        "t_stat": (None if not stderr
                   else round(mean / stderr, 2) if stderr > 0 else None),
    }
