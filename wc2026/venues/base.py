"""
venues/base.py
==============
Venue-independent market representation.

The three instrument kinds are deliberately NOT interchangeable:

  BINARY        one contract settling at 1 or 0 on a single claim.
  NATIVE_COMBO  one venue-defined payout function over several legs, with a
                single execution path (one fill, one fee, one settlement).
  BUNDLE        several separately executed contracts. There is no venue
                parlay payout: each leg has its own price, its own fee, and
                its own chance of not filling. Treating a bundle as a combo
                invents a payout that the venue never offered.

A claim is only usable if the model can value it. `Leg.supported` is False for
anything outside the validated scoreline-derived family, and an unsupported leg
poisons its whole instrument -- the instrument is recorded, explained, and
abstained from, never priced from qualitative confidence.

Snapshots carry a `decision_hash` over the fields that can change a decision
(prices, depth, rules, status, settlement basis). Unchanged markets are not
re-sent through the expensive board; a changed hash always produces a fresh
case. Deduplicating permanently by ticker would be wrong -- the same market at
a different price is a different decision.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field

# --- instrument kinds ---------------------------------------------------------
KIND_BINARY = "binary"
KIND_NATIVE_COMBO = "native_combo"
KIND_BUNDLE = "bundle"

# --- claims the model can value (derived from the regulation scoreline grid) --
SUPPORTED_CLAIM_PREFIXES = (
    "home_win", "away_win", "draw",
    "total_over_", "total_under_",
    "btts",
    "home_over_", "away_over_",
    "home_wins_by_over_", "away_wins_by_over_",
    "score_",
)


def claim_supported(claim: str) -> bool:
    """True only for claims the frozen engine can value exactly."""
    if not claim:
        return False
    stripped = claim[4:] if claim.startswith("not_") else claim
    return any(stripped.startswith(p) for p in SUPPORTED_CLAIM_PREFIXES)


class UnsupportedInstrument(ValueError):
    """Raised when an instrument cannot be valued by any validated model."""


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


# --- order book ---------------------------------------------------------------
@dataclass(frozen=True)
class Book:
    """Executable depth, normalised across venues.

    `yes_asks` / `no_asks` are the levels you would PAY to buy that side, best
    (cheapest) first, as (price_cents, size). Bids are what you could sell into.
    A venue that only publishes one side is normalised by complementing it.
    """

    yes_asks: tuple = ()
    no_asks: tuple = ()
    yes_bids: tuple = ()
    no_bids: tuple = ()
    observed_at: str = ""

    def asks(self, side: str) -> tuple:
        return self.yes_asks if side == "yes" else self.no_asks

    def bids(self, side: str) -> tuple:
        return self.yes_bids if side == "yes" else self.no_bids

    def touch(self, side: str):
        levels = self.asks(side)
        return levels[0][0] if levels else None

    def depth_at_touch(self, side: str) -> float:
        levels = self.asks(side)
        return levels[0][1] if levels else 0.0

    def best_bid(self, side: str):
        levels = self.bids(side)
        return levels[0][0] if levels else None

    def spread_cents(self, side: str):
        ask, bid = self.touch(side), self.best_bid(side)
        return None if ask is None or bid is None else ask - bid

    def walk(self, side: str, count: float):
        """Walk the book for `count` contracts.

        Returns (average_price_cents, filled_count, worst_price_cents) using
        REAL depth, so size beyond the touch is priced at what it would
        actually cost. Returns None when nothing is executable.
        """
        levels = self.asks(side)
        if not levels or count <= 0:
            return None
        remaining, cost, filled, worst = count, 0.0, 0.0, None
        for price, size in levels:
            take = min(remaining, size)
            if take <= 0:
                continue
            cost += take * price
            filled += take
            worst = price
            remaining -= take
            if remaining <= 0:
                break
        if filled <= 0:
            return None
        return cost / filled, filled, worst

    def max_size_at_or_below(self, side: str, limit_cents: int) -> float:
        """Contracts obtainable without paying more than `limit_cents`."""
        return float(sum(size for price, size in self.asks(side)
                         if price <= limit_cents))


# --- legs and instruments -----------------------------------------------------
@dataclass(frozen=True)
class Leg:
    claim: str
    market_ref: str
    description: str = ""
    home: str | None = None
    away: str | None = None
    league_id: str | None = None
    kickoff_utc: str | None = None
    supported: bool = True
    unsupported_reason: str | None = None

    @staticmethod
    def build(claim: str, market_ref: str, **kw) -> "Leg":
        ok = claim_supported(claim)
        return Leg(claim=claim, market_ref=market_ref, supported=ok,
                   unsupported_reason=None if ok else
                   "no validated model for claim %r" % claim, **kw)


@dataclass
class MarketInstrument:
    venue: str
    instrument_id: str
    kind: str
    title: str
    legs: tuple
    rules_text: str = ""
    settlement_source: str = ""
    settles_on_regulation: bool | None = None
    event_ref: str = ""
    league_id: str | None = None
    status: str = ""
    close_time: str | None = None
    kickoff_utc: str | None = None
    tick_cents: int = 1
    min_size: float = 1.0
    fee_model: dict = field(default_factory=dict)
    book: Book | None = None
    observed_at: str = field(default_factory=utcnow_iso)
    raw: dict = field(default_factory=dict, repr=False)

    # ---- support ----
    @property
    def supported(self) -> bool:
        """An instrument is valuable only if EVERY leg is valuable."""
        return bool(self.legs) and all(leg.supported for leg in self.legs)

    @property
    def unsupported_reasons(self) -> list:
        return [leg.unsupported_reason for leg in self.legs
                if not leg.supported and leg.unsupported_reason]

    @property
    def is_multi_leg(self) -> bool:
        return len(self.legs) > 1

    @property
    def shares_a_match(self) -> bool:
        """True when legs reference the same fixture -- their outcomes are
        dependent and must never be multiplied as if independent."""
        keys = {(leg.home, leg.away, leg.kickoff_utc) for leg in self.legs}
        return len(self.legs) > 1 and len(keys) == 1

    def require_valuable(self) -> None:
        if not self.supported:
            raise UnsupportedInstrument(
                "%s %s: %s" % (self.venue, self.instrument_id,
                               "; ".join(self.unsupported_reasons)
                               or "unsupported"))

    # ---- hashing ----
    def decision_fields(self) -> dict:
        """Exactly the fields that can change a decision."""
        book = self.book or Book()
        return {
            "venue": self.venue,
            "instrument_id": self.instrument_id,
            "kind": self.kind,
            "status": self.status,
            "rules_text": self.rules_text,
            "settlement_source": self.settlement_source,
            "settles_on_regulation": self.settles_on_regulation,
            "close_time": self.close_time,
            "kickoff_utc": self.kickoff_utc,
            "tick_cents": self.tick_cents,
            "min_size": self.min_size,
            "fee_model": self.fee_model,
            "legs": [(leg.claim, leg.market_ref, leg.supported)
                     for leg in self.legs],
            "yes_asks": list(book.yes_asks),
            "no_asks": list(book.no_asks),
            "yes_bids": list(book.yes_bids),
            "no_bids": list(book.no_bids),
        }

    def decision_hash(self) -> str:
        payload = json.dumps(self.decision_fields(), sort_keys=True,
                             default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_record(self) -> dict:
        rec = asdict(self)
        rec.pop("raw", None)
        rec["decision_hash"] = self.decision_hash()
        rec["supported"] = self.supported
        return rec


# --- provider interface -------------------------------------------------------
class MarketDataProvider:
    """Read-only market data. No provider may place, amend or cancel an order.

    Implementations must return normalised MarketInstruments and must never
    fabricate a price: a market with no executable depth returns a Book with
    empty levels, which downstream becomes WAIT_FOR_QUOTE, not a midpoint.
    """

    venue = "abstract"

    def discover(self, spec, **kw) -> list:
        raise NotImplementedError

    def fetch_book(self, instrument: MarketInstrument) -> Book:
        raise NotImplementedError

    # Placing orders is deliberately absent from this interface.


def equivalent(a: "MarketInstrument", b: "MarketInstrument",
               kickoff_tolerance_minutes: int = 90) -> bool:
    """True only when two instruments are GENUINELY the same proposition.

    Used for relative-value context across venues -- never as an arbitrage
    signal, because two contracts that merely look alike can settle
    differently. Requires: same instrument kind, same fixture, same claim set,
    and a settlement basis that is KNOWN and identical on both sides. An
    unknown settlement basis on either side returns False, because "probably
    the same rules" is exactly the assumption that produces a losing trade.
    """
    if a.venue == b.venue:
        return False
    if a.kind != b.kind or len(a.legs) != len(b.legs):
        return False
    if a.settles_on_regulation is None or b.settles_on_regulation is None:
        return False
    if a.settles_on_regulation != b.settles_on_regulation:
        return False
    if {leg.claim for leg in a.legs} != {leg.claim for leg in b.legs}:
        return False
    fixtures_a = {(str(leg.home).lower(), str(leg.away).lower())
                  for leg in a.legs}
    fixtures_b = {(str(leg.home).lower(), str(leg.away).lower())
                  for leg in b.legs}
    if fixtures_a != fixtures_b:
        return False
    ka, kb = a.kickoff_utc, b.kickoff_utc
    if ka and kb:
        try:
            ta = dt.datetime.fromisoformat(str(ka).replace("Z", "+00:00"))
            tb = dt.datetime.fromisoformat(str(kb).replace("Z", "+00:00"))
        except ValueError:
            return False
        if abs((ta - tb).total_seconds()) > kickoff_tolerance_minutes * 60:
            return False
    return True


def snapshot_record(instruments: list, league_id: str, venue: str) -> dict:
    """A timestamped, hashed snapshot of everything observed in one sweep."""
    records = [inst.to_record() for inst in instruments]
    body = json.dumps([r["decision_hash"] for r in records], sort_keys=True)
    return {
        "venue": venue,
        "league_id": league_id,
        "observed_at": utcnow_iso(),
        "n_instruments": len(records),
        "n_supported": sum(1 for r in records if r["supported"]),
        "snapshot_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "instruments": records,
    }


def changed_since(previous: dict | None, instruments: list) -> list:
    """Instruments whose decision-relevant state changed since a snapshot.

    Everything is returned when there is no previous snapshot. This is what
    prevents both wasteful re-review of static markets and the opposite error
    of permanently suppressing a market whose price has moved.
    """
    if not previous:
        return list(instruments)
    seen = {r["instrument_id"]: r["decision_hash"]
            for r in previous.get("instruments", [])}
    return [inst for inst in instruments
            if seen.get(inst.instrument_id) != inst.decision_hash()]
