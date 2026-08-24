"""
paper/broker.py
===============
Paper trading: an explicit order lifecycle, not recommendation accounting.

This is strictly simulation. There is no code path here that can reach a real
venue, by design -- the providers are read-only and the broker writes only to
its own ledger.

Fill rules, deliberately conservative
-------------------------------------
  * A PAPER_BUY_NOW may fill ONLY against the captured executable book, and
    only up to the depth that was actually visible at or below its limit.
  * A resting PAPER_PLACE_LIMIT does NOT fill merely because a later trade
    printed at its price. Being last in a queue is the normal case, so a
    resting order requires a later observation where the executable ASK trades
    THROUGH the limit (strictly better than it) with real size behind it.
  * Cash is reserved when an order is submitted and released on cancel,
    expiry or fill, so the same dollar cannot back two orders.
  * Settlement is idempotent: a position can be settled exactly once.

Every fill records the ACTUAL simulated size and price, never the proposed
size, so realized P&L cannot flatter itself.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from dataclasses import asdict, dataclass, field

from ..decision.calculator import fee_cents

PAPER_DIR = os.path.join("data", "paper")

# order states
OPEN = "open"
FILLED = "filled"
PARTIAL = "partially_filled"
CANCELLED = "cancelled"
EXPIRED = "expired"
REJECTED = "rejected"

TERMINAL = (FILLED, CANCELLED, EXPIRED, REJECTED)


class BrokerError(RuntimeError):
    pass


def _utcnow():
    return dt.datetime.now(dt.timezone.utc)


def _iso(stamp=None):
    return (stamp or _utcnow()).isoformat()


@dataclass
class PaperOrder:
    order_id: str
    case_id: str
    venue: str
    instrument_id: str
    side: str
    limit_price_cents: int
    requested_size: float
    league_id: str | None = None
    kind: str = "limit"
    status: str = OPEN
    filled_size: float = 0.0
    avg_fill_price_cents: float = 0.0
    fees_cents: int = 0
    reserved_cents: int = 0
    created_at: str = field(default_factory=_iso)
    expires_at: str | None = None
    events: list = field(default_factory=list)
    cancel_triggers: list = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.requested_size - self.filled_size)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def log(self, event: str, **payload):
        self.events.append({"ts": _iso(), "event": event, **payload})


@dataclass
class PaperPosition:
    position_id: str
    venue: str
    instrument_id: str
    side: str
    size: float
    avg_cost_cents: float
    fees_cents: int
    league_id: str | None = None
    case_id: str | None = None
    opened_at: str = field(default_factory=_iso)
    settled: bool = False
    result: str | None = None
    payout_cents: float = 0.0
    realized_pnl_cents: float = 0.0


@dataclass
class PaperPortfolio:
    starting_cash_cents: int = 100_000
    cash_cents: int = 100_000
    reserved_cents: int = 0
    orders: dict = field(default_factory=dict)
    positions: dict = field(default_factory=dict)
    ledger: list = field(default_factory=list)
    path: str = os.path.join(PAPER_DIR, "portfolio.json")

    # ---- cash ----
    @property
    def available_cents(self) -> int:
        return self.cash_cents - self.reserved_cents

    def _reserve(self, amount: int) -> None:
        if amount > self.available_cents:
            raise BrokerError("insufficient available cash: need %d, have %d"
                              % (amount, self.available_cents))
        self.reserved_cents += amount

    def _release(self, amount: int) -> None:
        self.reserved_cents = max(0, self.reserved_cents - amount)

    # ---- orders ----
    def submit(self, case_id: str, venue: str, instrument_id: str, side: str,
               limit_price_cents: int, size: float, league_id=None,
               expires_at=None, cancel_triggers=None,
               idempotency_key: str | None = None) -> PaperOrder:
        """Submit a paper limit order, reserving the cash it could consume."""
        if not 0 < limit_price_cents < 100:
            raise BrokerError("limit price must be 1..99")
        if size <= 0:
            raise BrokerError("size must be positive")
        key = idempotency_key or "%s|%s|%s|%s" % (case_id, instrument_id, side,
                                                  limit_price_cents)
        # Idempotent across the order's WHOLE life, terminal states included.
        # A scheduled cycle re-runs over the same markets; if this exact case,
        # instrument, side and price has already been acted on -- even if it
        # has since filled or expired -- submitting again would double the
        # position. A genuinely new decision has a different price, and so a
        # different key.
        for existing in self.orders.values():
            if existing.events and existing.events[0].get("key") == key:
                return existing

        worst_fee = fee_cents(venue, int(size) or 1, limit_price_cents)
        need = int(limit_price_cents * size + worst_fee)
        self._reserve(need)
        order = PaperOrder(order_id=str(uuid.uuid4()), case_id=case_id,
                           venue=venue, instrument_id=instrument_id, side=side,
                           limit_price_cents=limit_price_cents,
                           requested_size=size, league_id=league_id,
                           reserved_cents=need, expires_at=expires_at,
                           cancel_triggers=list(cancel_triggers or []))
        order.log("submitted", key=key, reserved_cents=need)
        self.orders[order.order_id] = order
        return order

    def cancel(self, order_id: str, reason: str = "manual") -> PaperOrder:
        order = self._order(order_id)
        if order.terminal:
            return order
        self._release(order.reserved_cents)
        order.reserved_cents = 0
        order.status = CANCELLED if order.filled_size == 0 else PARTIAL
        order.log("cancelled", reason=reason)
        return order

    def expire_due(self, now=None) -> list:
        """Expire any resting order past its expiry, releasing its cash."""
        moment = now or _utcnow()
        expired = []
        for order in list(self.orders.values()):
            if order.terminal or not order.expires_at:
                continue
            try:
                due = dt.datetime.fromisoformat(
                    str(order.expires_at).replace("Z", "+00:00"))
            except ValueError:
                continue
            if due.tzinfo is None:
                due = due.replace(tzinfo=dt.timezone.utc)
            if moment >= due:
                self._release(order.reserved_cents)
                order.reserved_cents = 0
                order.status = EXPIRED
                order.log("expired")
                expired.append(order)
        return expired

    def _order(self, order_id: str) -> PaperOrder:
        try:
            return self.orders[order_id]
        except KeyError:
            raise BrokerError("unknown order %r" % order_id) from None

    # ---- fills ----
    def fill_marketable(self, order_id: str, book) -> PaperOrder:
        """Fill a BUY_NOW against the CAPTURED book, capped by real depth."""
        order = self._order(order_id)
        if order.terminal:
            return order
        available = book.max_size_at_or_below(order.side,
                                              order.limit_price_cents)
        take = min(order.remaining, available)
        if take <= 0:
            order.log("no_fill", reason="no depth at or below the limit")
            return order
        walked = book.walk(order.side, take)
        if walked is None:
            order.log("no_fill", reason="book empty")
            return order
        avg, filled, _worst = walked
        self._book_fill(order, filled, avg)
        return order

    def try_fill_resting(self, order_id: str, later_book,
                         require_trade_through: bool = True) -> PaperOrder:
        """Attempt to fill a RESTING order against a LATER observation.

        Conservative by design: a resting buy is filled only when the market
        trades strictly THROUGH the limit -- i.e. the executable ask is at least
        one tick BETTER than our price -- with real size behind it. A print at
        exactly our price does not fill us, because we cannot know our place in
        the queue.
        """
        order = self._order(order_id)
        if order.terminal:
            return order
        touch = later_book.touch(order.side)
        if touch is None:
            order.log("no_fill", reason="no executable quote")
            return order
        threshold = (order.limit_price_cents - 1 if require_trade_through
                     else order.limit_price_cents)
        if touch > threshold:
            order.log("no_fill", reason="ask %dc did not trade through %dc"
                      % (touch, order.limit_price_cents))
            return order
        available = later_book.max_size_at_or_below(order.side, threshold)
        take = min(order.remaining, available)
        if take <= 0:
            order.log("no_fill", reason="no size through the limit")
            return order
        walked = later_book.walk(order.side, take)
        avg, filled, _worst = walked
        # never worse than our limit
        avg = min(avg, order.limit_price_cents)
        self._book_fill(order, filled, avg)
        return order

    def _book_fill(self, order: PaperOrder, size: float, avg_price: float):
        fee = fee_cents(order.venue, max(int(size), 1), int(round(avg_price)))
        cost = int(avg_price * size + fee)
        # release the proportional reservation, then pay actual cost
        release = min(order.reserved_cents, cost)
        self._release(release)
        order.reserved_cents = max(0, order.reserved_cents - release)
        self.cash_cents -= cost

        total = order.filled_size + size
        order.avg_fill_price_cents = (
            (order.avg_fill_price_cents * order.filled_size + avg_price * size)
            / total) if total else 0.0
        order.filled_size = total
        order.fees_cents += fee
        order.status = FILLED if order.remaining <= 1e-9 else PARTIAL
        if order.status == FILLED and order.reserved_cents:
            self._release(order.reserved_cents)
            order.reserved_cents = 0
        order.log("fill", size=size, price_cents=avg_price, fee_cents=fee)

        key = "%s|%s" % (order.instrument_id, order.side)
        pos = self.positions.get(key)
        if pos is None:
            self.positions[key] = PaperPosition(
                position_id=str(uuid.uuid4()), venue=order.venue,
                instrument_id=order.instrument_id, side=order.side, size=size,
                avg_cost_cents=avg_price, fees_cents=fee,
                league_id=order.league_id, case_id=order.case_id)
        else:
            grand = pos.size + size
            pos.avg_cost_cents = ((pos.avg_cost_cents * pos.size
                                   + avg_price * size) / grand)
            pos.size = grand
            pos.fees_cents += fee

    # ---- settlement ----
    def settle(self, instrument_id: str, side: str, result: str) -> float:
        """Settle one position exactly once. `result` is the winning side."""
        key = "%s|%s" % (instrument_id, side)
        pos = self.positions.get(key)
        if pos is None:
            return 0.0
        if pos.settled:
            raise BrokerError("position %s already settled" % key)
        won = (result == side)
        payout = 100.0 * pos.size if won else 0.0
        cost = pos.avg_cost_cents * pos.size + pos.fees_cents
        pnl = payout - cost
        pos.settled, pos.result = True, result
        pos.payout_cents, pos.realized_pnl_cents = payout, pnl
        self.cash_cents += int(payout)
        self.ledger.append({"ts": _iso(), "instrument_id": instrument_id,
                            "side": side, "result": result, "won": won,
                            "size": pos.size,
                            "avg_cost_cents": pos.avg_cost_cents,
                            "fees_cents": pos.fees_cents,
                            "payout_cents": payout, "pnl_cents": pnl,
                            "league_id": pos.league_id,
                            "case_id": pos.case_id})
        return pnl

    # ---- reporting ----
    def summary(self) -> dict:
        settled = [e for e in self.ledger]
        realized = sum(e["pnl_cents"] for e in settled)
        open_positions = [p for p in self.positions.values() if not p.settled]
        return {
            "starting_cash_usd": self.starting_cash_cents / 100,
            "cash_usd": self.cash_cents / 100,
            "reserved_usd": self.reserved_cents / 100,
            "available_usd": self.available_cents / 100,
            "n_orders": len(self.orders),
            "n_open_orders": sum(1 for o in self.orders.values()
                                 if not o.terminal),
            "n_positions_open": len(open_positions),
            "n_settled": len(settled),
            "realized_pnl_usd": realized / 100,
            "fees_paid_usd": sum(o.fees_cents for o in self.orders.values()) / 100,
            "fill_rate": (sum(1 for o in self.orders.values()
                              if o.filled_size > 0) / len(self.orders))
                         if self.orders else 0.0,
        }

    # ---- persistence ----
    def save(self, path: str | None = None) -> str:
        target = path or self.path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        payload = {
            "starting_cash_cents": self.starting_cash_cents,
            "cash_cents": self.cash_cents,
            "reserved_cents": self.reserved_cents,
            "orders": {k: asdict(v) for k, v in self.orders.items()},
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "ledger": self.ledger,
            "saved_at": _iso(),
        }
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, target)
        return target

    @classmethod
    def load(cls, path: str | None = None) -> "PaperPortfolio":
        target = path or os.path.join(PAPER_DIR, "portfolio.json")
        if not os.path.exists(target):
            return cls(path=target)
        with open(target, encoding="utf-8") as fh:
            payload = json.load(fh)
        portfolio = cls(
            starting_cash_cents=payload.get("starting_cash_cents", 100_000),
            cash_cents=payload["cash_cents"],
            reserved_cents=payload.get("reserved_cents", 0),
            ledger=payload.get("ledger", []), path=target)
        for key, raw in (payload.get("orders") or {}).items():
            portfolio.orders[key] = PaperOrder(**raw)
        for key, raw in (payload.get("positions") or {}).items():
            portfolio.positions[key] = PaperPosition(**raw)
        return portfolio
