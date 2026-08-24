"""
kalshi.py
=========
Minimal, typed Kalshi REST client for the betting layer.

Auth follows the platform's RSA-PSS request-signing scheme (the same pattern
proven in legacy/kalshi_client.py): signature = RSA-PSS-SHA256 over
timestamp_ms + METHOD + path, sent with the API key id and timestamp headers.

Credentials come from environment variables ONLY (config.ENV_*):
    KALSHI_API_KEY_ID       the API key UUID
    KALSHI_PRIVATE_KEY_PATH path to the RSA private key .pem
No key material or key path ever appears in source or in any config file.
Public market-data GETs work unauthenticated; portfolio and order endpoints
require credentials and raise ConfigError without them.

Order placement is limit-only by construction -- there is deliberately no
market-order code path in this client. POSTs are never retried: any error or
ambiguity raises and the caller must treat the order as unknown/aborted.
"""
from __future__ import annotations

import base64
import os
import time
import uuid
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from .config import ENV_KEY_ID, ENV_PRIVATE_KEY_PATH, KALSHI_BASE


class ConfigError(RuntimeError):
    pass


class OrderError(RuntimeError):
    """An order POST failed or returned an unexpected state. Never retried."""


@dataclass
class OrderBookSide:
    """Resting bids on one side, ascending by price. Kalshi books quote BIDS on
    each side; the executable ask to BUY this side is 100 - best opposite bid."""
    levels: list[tuple[int, int]] = field(default_factory=list)  # (price_cents, count)

    @property
    def best(self) -> tuple[int, int] | None:
        return self.levels[-1] if self.levels else None


@dataclass
class OrderBook:
    ticker: str
    yes_bids: OrderBookSide
    no_bids: OrderBookSide

    def executable(self, side: str) -> list[tuple[int, int]]:
        """Levels available to BUY `side`, best first, as (ask_cents, count).

        Buying YES fills against resting NO bids at ask = 100 - no_bid;
        buying NO fills against resting YES bids at ask = 100 - yes_bid.
        """
        opp = self.no_bids if side == "yes" else self.yes_bids
        return [(100 - price, count) for price, count in reversed(opp.levels)]

    def depth_at_touch(self, side: str) -> int:
        levels = self.executable(side)
        return levels[0][1] if levels else 0

    def touch(self, side: str) -> int | None:
        levels = self.executable(side)
        return levels[0][0] if levels else None

    def best_bid(self, side: str) -> int | None:
        b = (self.yes_bids if side == "yes" else self.no_bids).best
        return b[0] if b else None

    def avg_fill_price(self, side: str, count: int) -> tuple[float, int] | None:
        """Walk the book: (average price in cents, fillable count) for buying
        up to `count` contracts of `side`. None if the book is empty."""
        levels = self.executable(side)
        if not levels:
            return None
        remaining, cost, filled = count, 0, 0
        for price, avail in levels:
            take = min(remaining, avail)
            cost += take * price
            filled += take
            remaining -= take
            if remaining == 0:
                break
        if filled == 0:
            return None
        return cost / filled, filled


# --------------------------------------------------------------------------- #
# V2 order construction & parsing (pure, unit-tested -- SAFETY CRITICAL)
# --------------------------------------------------------------------------- #
# Kalshi deprecated POST /portfolio/orders (returns 410) in favour of
# POST /portfolio/events/orders, a single-book model quoted from the YES leg:
#   * side "bid"  = BUY YES   at the given YES price
#   * side "ask"  = SELL YES  = economically BUY NO at (1 - YES price)
# So our internal ("yes"/"no", price we pay) maps as:
#   buy YES at p  -> {side: "bid", price: p/100}
#   buy NO  at p  -> {side: "ask", price: (100 - p)/100}   (sell YES at 100-p)
# Prices/counts are fixed-point dollar strings. Getting this mapping wrong buys
# the opposite side, so it is isolated here and tested for both sides.
def build_order_body(ticker: str, side: str, count: int, price_cents: int,
                     time_in_force: str, self_trade_prevention: str,
                     client_order_id: str) -> dict:
    if side not in ("yes", "no"):
        raise OrderError(f"invalid side {side!r}")
    price_cents = int(price_cents)
    if not (0 < price_cents < 100) or int(count) <= 0:
        raise OrderError(f"invalid order: {count} @ {price_cents}c")
    if side == "yes":
        book_side, yes_leg_cents = "bid", price_cents
    else:
        book_side, yes_leg_cents = "ask", 100 - price_cents
    return {
        "ticker": ticker,
        "side": book_side,
        "count": str(int(count)),
        "price": f"{yes_leg_cents / 100:.4f}",
        "time_in_force": time_in_force,
        "self_trade_prevention_type": self_trade_prevention,
        "client_order_id": client_order_id,
    }


def _dec(x):
    try:
        return Decimal(str(x)) if x not in (None, "") else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def parse_order_response(resp: dict, side: str, requested_price_cents: int) -> dict:
    """Normalise the V2 order response into ACTUAL fill terms for OUR side.
    average_fill_price is the YES-leg price; convert back to our side's cents."""
    if not isinstance(resp, dict) or not resp.get("order_id"):
        raise OrderError(f"unexpected order response: {resp!r}")
    filled = _dec(resp.get("fill_count"))
    filled = int(filled) if filled is not None else 0
    avg_yes = _dec(resp.get("average_fill_price"))
    if avg_yes is not None:
        yes_cents = int((avg_yes * 100).to_integral_value(rounding=ROUND_HALF_UP))
        our_cents = yes_cents if side == "yes" else 100 - yes_cents
    else:
        our_cents = int(requested_price_cents)
    fee = _dec(resp.get("average_fee_paid"))
    fee_cents = (int((fee * 100).to_integral_value(rounding=ROUND_HALF_UP)) * filled
                 if fee is not None else 0)
    return {"order_id": resp["order_id"], "filled": filled,
            "fill_price_cents": our_cents, "fee_cents": fee_cents, "raw": resp}


class KalshiClient:
    def __init__(self, base_url: str = KALSHI_BASE, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.key_id = os.environ.get(ENV_KEY_ID)
        key_path = os.environ.get(ENV_PRIVATE_KEY_PATH)
        self._private_key = None
        if self.key_id and key_path:
            with open(key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(), password=None)

    # ---------- auth ----------
    @property
    def authenticated(self) -> bool:
        return self._private_key is not None

    def require_auth(self, why: str):
        if not self.authenticated:
            raise ConfigError(
                f"{why} requires Kalshi credentials. Set {ENV_KEY_ID} and "
                f"{ENV_PRIVATE_KEY_PATH} in the environment (never in source).")

    def _sign(self, ts_ms: str, method: str, path: str) -> str:
        msg = (ts_ms + method.upper() + path).encode("utf-8")
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())
        return base64.b64encode(sig).decode("utf-8")

    def _headers(self, method: str, path: str) -> dict:
        if not self.authenticated:
            return {}
        ts = str(int(time.time() * 1000))
        return {"KALSHI-ACCESS-KEY": self.key_id,
                "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "Content-Type": "application/json"}

    # polite pacing for public endpoints; GETs (idempotent) retry on 429/5xx
    # with backoff. POSTs are NEVER retried (see place_limit_order).
    _MIN_REQUEST_INTERVAL = 0.15
    _GET_RETRIES = 4

    def _request(self, method: str, path: str, params=None, json_body=None) -> dict:
        full_path = "/trade-api/v2" + path
        url = self.base_url.replace("/trade-api/v2", "") + full_path
        attempts = self._GET_RETRIES if method.upper() == "GET" else 1
        for attempt in range(attempts):
            wait = time.time() - getattr(self, "_last_request_ts", 0.0)
            if wait < self._MIN_REQUEST_INTERVAL:
                time.sleep(self._MIN_REQUEST_INTERVAL - wait)
            self._last_request_ts = time.time()
            resp = self.session.request(method, url, params=params, json=json_body,
                                        headers=self._headers(method, full_path),
                                        timeout=self.timeout)
            if resp.status_code in (429, 502, 503) and attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("unreachable")

    # ---------- market data (public) ----------
    def get_markets(self, series_ticker: str | None = None,
                    event_ticker: str | None = None, status: str = "open",
                    limit: int = 200) -> list[dict]:
        """All matching markets, following pagination cursors."""
        out, cursor = [], None
        while True:
            params = {"limit": limit, "status": status}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if event_ticker:
                params["event_ticker"] = event_ticker
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", "/markets", params=params)
            out.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                return out

    def get_market(self, ticker: str) -> dict:
        return self._request("GET", f"/markets/{ticker}").get("market", {})

    def get_candlesticks(self, ticker: str, start_ts: int, end_ts: int,
                         period_interval: int = 1,
                         series_ticker: str | None = None) -> list[dict]:
        """Price-history candlesticks for a market (public). `period_interval`
        is in minutes; timestamps are unix seconds. The series ticker defaults
        to the prefix of `ticker` (e.g. KXMLSGAME-...-MIN -> KXMLSGAME). Each
        candle carries OHLC under `price` (`close_dollars` etc.) plus `yes_bid`
        / `yes_ask`. Returns [] on any error rather than raising, so tracking
        degrades gracefully to a lower-quality closing line."""
        series = series_ticker or ticker.split("-")[0]
        try:
            data = self._request(
                "GET", f"/series/{series}/markets/{ticker}/candlesticks",
                params={"start_ts": int(start_ts), "end_ts": int(end_ts),
                        "period_interval": int(period_interval)})
        except Exception:                                     # noqa: BLE001
            return []
        return data.get("candlesticks", []) or []

    def get_orderbook(self, ticker: str, depth: int = 10) -> OrderBook:
        """Parse the live `orderbook_fp` format (verified 2026-07-20): each side
        is a list of [price_dollars_str, count_str] resting BIDS, ascending."""
        from decimal import Decimal
        data = self._request("GET", f"/markets/{ticker}/orderbook",
                             params={"depth": depth})
        ob = data.get("orderbook_fp")
        if ob is None:                      # tolerate the older integer format
            ob = data.get("orderbook") or {}
            def side_int(key):
                levels = [(int(p), int(c)) for p, c in (ob.get(key) or [])]
                return OrderBookSide(levels=sorted(levels))
            return OrderBook(ticker=ticker, yes_bids=side_int("yes"),
                             no_bids=side_int("no"))

        def side(key):
            levels = []
            for price_str, count_str in (ob.get(key) or []):
                cents = int(Decimal(price_str) * 100)   # "0.9500" -> 95, exact
                count = int(Decimal(count_str))          # floor fractional lots
                if count > 0:
                    levels.append((cents, count))
            return OrderBookSide(levels=sorted(levels))
        return OrderBook(ticker=ticker, yes_bids=side("yes_dollars"),
                         no_bids=side("no_dollars"))

    # ---------- portfolio (authenticated) ----------
    def get_balance_cents(self) -> int:
        """Live account balance in cents. Handles both known payload shapes
        ('balance' in integer cents; 'balance_dollars' as a dollar string) and
        refuses to guess on anything else -- sizing must not run on a
        misparsed balance."""
        from decimal import Decimal
        self.require_auth("Reading the account balance")
        data = self._request("GET", "/portfolio/balance")
        if isinstance(data.get("balance"), int):
            return data["balance"]
        if data.get("balance_dollars") is not None:
            return int(Decimal(str(data["balance_dollars"])) * 100)
        raise RuntimeError(f"unrecognised balance payload: {data!r}")

    def get_positions(self) -> list[dict]:
        self.require_auth("Reading positions")
        out, cursor = [], None
        while True:
            params = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", "/portfolio/positions", params=params)
            out.extend(data.get("market_positions", []))
            cursor = data.get("cursor")
            if not cursor:
                return out

    def open_position_tickers(self) -> set[str]:
        """Tickers with a non-zero net position on Kalshi -- the source of truth
        for what is actually still open. `position_fp` is a signed fixed-point
        count (positive=YES, negative=NO); 0/absent means flat (settled/closed)."""
        out = set()
        for mp in self.get_positions():
            pos = _dec(mp.get("position_fp"))
            if pos is not None and pos != 0 and mp.get("ticker"):
                out.add(mp["ticker"])
        return out

    # ---------- orders (authenticated; LIMIT ONLY; never retried) ----------
    def place_limit_order(self, ticker: str, side: str, count: int,
                          price_cents: int,
                          time_in_force: str = "immediate_or_cancel",
                          self_trade_prevention: str = "taker_at_cross") -> dict:
        """Buy `count` contracts of OUR `side` ('yes'/'no') at limit
        `price_cents` via the Kalshi V2 order endpoint. Returns a normalised
        fill dict (order_id, filled, fill_price_cents, fee_cents, raw). Raises
        OrderError on ANY failure or unexpected response; callers must abort,
        not retry.

        Default IOC: take what rests at our limit now and cancel the remainder
        -- no surprise resting orders, never fills worse than our price.
        """
        self.require_auth("Placing orders")
        body = build_order_body(ticker, side, count, price_cents,
                                time_in_force, self_trade_prevention,
                                str(uuid.uuid4()))
        try:
            resp = self._request("POST", "/portfolio/events/orders", json_body=body)
        except Exception as e:                     # noqa: BLE001 -- deliberate
            raise OrderError(f"order POST failed ({e}); NOT retrying") from e
        return parse_order_response(resp, side, int(price_cents))
