"""
paper/fills.py
==============
Would a resting paper order have been filled between two moments?

Nothing this system submits reaches a venue -- `paper/broker.py` imports no
network library at all -- so no matching engine knows our orders exist and
there is no fill to observe. A fill is a COUNTERFACTUAL and has to be
reconstructed from what the market did.

The obvious way, comparing order-book snapshots each time the scheduler runs,
makes the answer depend on the schedule: a market that dipped through our limit
at 10:40 and recovered by noon looks like no fill to a cycle that ran at 09:17
and 12:17. Measured fill rate would then be a property of the cron expression.

Both venues publish price history, so instead we replay the tape across the
whole gap. Cadence stops affecting the answer.

WHAT EACH VENUE LETS US SEE -- they are not equivalent, and the difference is
recorded on every fill rather than averaged away:

  Kalshi     per-minute candlesticks carrying `yes_ask` and `yes_bid` OHLC.
             The low of the ask is exactly what a resting buy needs, so the
             test is exact.

  Polymarket `/prices-history` returns ONE price per sample and, verified
             against a live book on 2026-08-25 (bid 0.600 / ask 0.610 ->
             history 0.605), it is the MIDPOINT, not the ask and not the last
             trade. The ask is therefore unobservable, and is bounded instead:
             with a 1c minimum tick the ask is at least half a cent above the
             mid, so the mid must reach `limit - 1.5c` for the ask to have
             reached `limit - 1c`. This is deliberately stricter than Kalshi's
             test, so Polymarket fill rates are biased low relative to Kalshi
             and the two must not be compared naively.

SIZE IS ASSUMED, NOT OBSERVED. History gives prices, not depth, so a fill takes
the order's whole remaining size. Every fill records `basis="history"` so these
can be separated from snapshot fills, where depth was real.
"""
from __future__ import annotations

import datetime as dt

import requests

from ..venues.base import Book

CLOB = "https://clob.polymarket.com"

# A resting buy fills only when the market trades a FULL TICK through it: at
# our exact price we cannot know our place in the queue. Same rule the
# snapshot path uses (`broker.try_fill_resting`).
TICK_CENTS = 1
# Half of Polymarket's 1c minimum spread -- the smallest gap that can separate
# its published mid from the ask we cannot see.
POLY_MID_TO_ASK_CENTS = 0.5


def _ts(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().replace(" ", "T").replace("Z", "+00:00")
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return int(stamp.timestamp())


def _dollars(entry, *path):
    node = entry
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


class FillProbe:
    """A venue's tape: what a market did between two moments.

    Two questions are asked of it, and both need the same venue plumbing --
    would a resting order have filled (`best_executable_cents`), and what was
    the market's closing price (`closing_price_cents`, used for CLV).
    """

    venue = "abstract"

    def best_executable_cents(self, order, since, until):
        raise NotImplementedError

    def closing_price_cents(self, venue_id: str, side: str, kickoff):
        """Price of `side` at kick-off -- the closing line -- or None."""
        raise NotImplementedError


class KalshiFillProbe(FillProbe):
    venue = "kalshi"

    def __init__(self, client=None):
        from ..betting.kalshi import KalshiClient
        self.client = client or KalshiClient()

    def best_executable_cents(self, order, since, until):
        """Lowest price our side could have been bought at, or None.

        Buying NO is selling YES, so the NO ask is `100 - yes_bid`: a resting
        NO buy is reached when the YES bid rises, not when it falls.
        """
        start, end = _ts(since), _ts(until)
        if start is None or end is None or end <= start:
            return None
        try:
            candles = self.client.get_candlesticks(
                order.instrument_id, start, end, period_interval=1)
        except Exception:                                     # noqa: BLE001
            return None
        best = None
        for candle in candles or ():
            if order.side == "yes":
                low = _dollars(candle, "yes_ask", "low_dollars")
                price = None if low is None else low * 100.0
            else:
                high = _dollars(candle, "yes_bid", "high_dollars")
                price = None if high is None else 100.0 - high * 100.0
            if price is None or not 0 < price < 100:
                continue
            best = price if best is None else min(best, price)
        return best

    def closing_price_cents(self, venue_id: str, side: str, kickoff,
                            lookback_hours: int = 6):
        """Last traded price at or before kick-off.

        NOT any live field on the market: Kalshi trades right through the
        match, so a price read afterwards is contaminated by in-play flow. The
        closing line is the final PRE-GAME consensus and only the candlestick
        history can pin it to a moment.
        """
        end = _ts(kickoff)
        if end is None:
            return None
        try:
            candles = self.client.get_candlesticks(
                venue_id, end - lookback_hours * 3600, end, period_interval=1)
        except Exception:                                     # noqa: BLE001
            return None
        last = None
        for candle in candles or ():
            if int(candle.get("end_period_ts") or 0) > end:
                continue
            close = _dollars(candle, "yes_ask", "close_dollars")
            bid = _dollars(candle, "yes_bid", "close_dollars")
            if close is None or bid is None:
                continue
            mid = (close + bid) / 2.0 * 100.0
            last = mid if side == "yes" else 100.0 - mid
        return last


class PolymarketFillProbe(FillProbe):
    venue = "polymarket"

    def __init__(self, session=None, timeout: int = 30):
        self.session = session or requests.Session()
        self.timeout = timeout
        self._tokens: dict = {}

    def token_for(self, condition_id: str, side: str):
        """The CLOB token id for our side of a market, or None.

        `instrument_id` is the conditionId, not a token, so the pair has to be
        looked up. Index 0 is the Yes token and index 1 the No token, matching
        the order the claim parser reads outcomes in.
        """
        key = str(condition_id or "")
        if key not in self._tokens:
            ids = []
            try:
                resp = self.session.get("%s/markets/%s" % (CLOB, key),
                                        timeout=self.timeout)
                if resp.status_code == 200:
                    ids = [str(t.get("token_id"))
                           for t in (resp.json().get("tokens") or [])
                           if t.get("token_id")]
            except (requests.RequestException, ValueError):
                ids = []
            self._tokens[key] = ids
        ids = self._tokens[key]
        index = 0 if side == "yes" else 1
        return ids[index] if len(ids) > index else None

    def best_executable_cents(self, order, since, until):
        start, end = _ts(since), _ts(until)
        if start is None or end is None or end <= start:
            return None
        token = self.token_for(order.instrument_id, order.side)
        if not token:
            return None
        try:
            resp = self.session.get(
                "%s/prices-history" % CLOB,
                params={"market": token, "startTs": start, "endTs": end,
                        "fidelity": 1},
                timeout=self.timeout)
            history = resp.json().get("history", []) if resp.status_code == 200 else []
        except (requests.RequestException, ValueError):
            return None
        lowest_mid = None
        for point in history or ():
            try:
                mid = float(point.get("p")) * 100.0
            except (TypeError, ValueError):
                continue
            if not 0 < mid < 100:
                continue
            lowest_mid = mid if lowest_mid is None else min(lowest_mid, mid)
        if lowest_mid is None:
            return None
        # The published price is the mid; the ask we would have paid sits at
        # least half a tick above it.
        return lowest_mid + POLY_MID_TO_ASK_CENTS

    def closing_price_cents(self, venue_id: str, side: str, kickoff,
                            lookback_hours: int = 6):
        """Last published mid at or before kick-off, for our side's token."""
        end = _ts(kickoff)
        if end is None:
            return None
        token = self.token_for(venue_id, side)
        if not token:
            return None
        try:
            resp = self.session.get(
                "%s/prices-history" % CLOB,
                params={"market": token, "startTs": end - lookback_hours * 3600,
                        "endTs": end, "fidelity": 1},
                timeout=self.timeout)
            history = resp.json().get("history", []) if resp.status_code == 200 else []
        except (requests.RequestException, ValueError):
            return None
        last = None
        for point in history or ():
            try:
                if int(point.get("t") or 0) > end:
                    continue
                price = float(point.get("p")) * 100.0
            except (TypeError, ValueError):
                continue
            if 0 < price < 100:
                last = price
        return last


def synthetic_book(side: str, price_cents: float, size: float) -> Book:
    """A one-level book standing in for the moment history says we traded.

    Feeding it back through `try_fill_resting` keeps ONE fill path -- the
    tested, conservative one -- rather than a second implementation of the
    trade-through rule that could drift from it.

    Quoted at OUR LIMIT, not at the price the tape printed. A resting order is
    the passive side: when an aggressive sell at 29c meets our resting bid at
    30c, the trade prints at 30c and the aggressor takes the improvement, not
    us. Quoting the synthetic book at 29c would hand us a free cent per
    contract on every history fill and inflate measured P&L.
    """
    level = ((int(round(price_cents)), float(size)),)
    return (Book(yes_asks=level, observed_at="") if side == "yes"
            else Book(no_asks=level, observed_at=""))


def replay_fills(portfolio, probes: dict, now=None) -> dict:
    """Fill every resting order the tape says traded through. Counted, never guessed."""
    now = now or dt.datetime.now(dt.timezone.utc)
    stats = {"checked": 0, "filled": 0, "no_history": 0, "not_through": 0,
             "no_probe": 0}
    for order in list(portfolio.orders.values()):
        if order.terminal or order.remaining <= 0:
            continue
        stats["checked"] += 1
        probe = probes.get(order.venue)
        if probe is None:
            stats["no_probe"] += 1
            continue
        since = order.last_checked_at or order.created_at
        best = probe.best_executable_cents(order, since, now)
        order.last_checked_at = now.isoformat()
        if best is None:
            stats["no_history"] += 1
            continue
        if best > order.limit_price_cents - TICK_CENTS:
            stats["not_through"] += 1
            continue
        # `replay_fills` has already applied the trade-through rule against
        # the tape, so the broker must not re-apply it to a book quoted at our
        # own limit -- it would reject every fill.
        book = synthetic_book(order.side, order.limit_price_cents,
                              order.remaining)
        before = order.filled_size
        portfolio.try_fill_resting(order.order_id, book,
                                   require_trade_through=False)
        if order.filled_size > before:
            stats["filled"] += 1
            order.log("fill_basis", basis="history", venue=order.venue,
                      observed_cents=round(best, 2), window_start=since,
                      window_end=now.isoformat())
    return stats
