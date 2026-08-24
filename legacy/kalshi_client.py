"""
kalshi_client.py
================
Minimal Kalshi REST client.

Auth: Kalshi uses RSA request signing. You create an API key in the Kalshi web UI,
which gives you (1) an API Key ID (a UUID) and (2) an RSA private key (downloaded as
a .pem file). Each request is signed: signature = RSA-PSS-SHA256(timestamp+METHOD+path).

  >>> FILL THESE IN <<<
  KALSHI_API_KEY_ID  = "your-api-key-id-uuid"
  KALSHI_PRIVATE_KEY = r"C:/Users/you/Downloads/privkey.pem"   # use a raw string or
                                                               # forward slashes on Windows

Endpoints used here are GET market data. The client never places orders -- placing
trades is left entirely to you in the Kalshi UI.

Docs: https://trading-api.readme.io/  | Base: https://api.elections.kalshi.com/trade-api/v2
"""

from __future__ import annotations
import base64
import datetime as dt
import re
import time
from dataclasses import dataclass

import requests

# Our teams.json name -> alternative spellings Kalshi may use in titles/subtitles.
_KALSHI_NAME_ALIASES = {
    "Bosnia & Herzegovina": ["bosnia and herzegovina", "bosnia & herzegovina", "bosnia"],
    "DR Congo": ["congo dr", "dr congo", "congo"],
    "South Korea": ["korea republic", "south korea", "korea"],
    "USA": ["usa", "united states"],
    "Turkiye": ["turkiye", "turkey", "türkiye"],
    "Ivory Coast": ["ivory coast", "cote d'ivoire", "côte d'ivoire"],
    "Curacao": ["curacao", "curaçao"],
}


def _team_matches(our_name: str, text: str) -> bool:
    """True if `text` (a Kalshi title/subtitle) refers to our team `our_name`."""
    if not text:
        return False
    t = text.lower()
    for alias in _KALSHI_NAME_ALIASES.get(our_name, [our_name.lower()]):
        if alias in t:
            return True
    return our_name.lower() in t
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

# ============================================================================
# >>>>>>>>>>>>>>>>>>>>>>>  FILL IN YOUR CREDENTIALS  <<<<<<<<<<<<<<<<<<<<<<<<<<<
KALSHI_API_KEY_ID = "91c7848e-027c-4dae-bced-63981748a007"
KALSHI_PRIVATE_KEY_PATH = r"C:\Users\jhjoh\Downloads\privkey.prem"
# ============================================================================

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
# Demo/sandbox environment (paper trading):
# KALSHI_BASE = "https://demo-api.kalshi.co/trade-api/v2"


@dataclass
class KalshiMarket:
    ticker: str
    title: str
    yes_bid: int       # cents (1-99)
    yes_ask: int       # cents you'd pay to buy YES
    no_bid: int
    no_ask: int        # cents you'd pay to buy NO
    volume: int
    status: str
    raw: dict

    # ---- helpers (YES side) ----
    @property
    def yes_price(self) -> int:
        """Effective cost in cents to take YES (the ask)."""
        return self.yes_ask if self.yes_ask else self.yes_bid

    @property
    def no_price(self) -> int:
        return self.no_ask if self.no_ask else self.no_bid

    def implied_prob(self, side: str = "yes") -> float:
        price = self.yes_price if side == "yes" else self.no_price
        return price / 100.0

    def decimal_odds(self, side: str = "yes") -> float:
        """Decimal odds for a 1-unit stake: pays 100c per contract bought at `price`c."""
        price = self.yes_price if side == "yes" else self.no_price
        if price <= 0:
            return float("inf")
        return 100.0 / price


class KalshiClient:
    def __init__(self, key_id: str = KALSHI_API_KEY_ID,
                 private_key_path: str = KALSHI_PRIVATE_KEY_PATH,
                 base_url: str = KALSHI_BASE):
        self.key_id = key_id
        self.base_url = base_url.rstrip("/")
        self._private_key = self._load_key(private_key_path) if _looks_real(key_id) else None
        self.session = requests.Session()

    # ---------- auth ----------
    @staticmethod
    def _load_key(path: str):
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, ts_ms: str, method: str, path: str) -> str:
        msg = (ts_ms + method.upper() + path).encode("utf-8")
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def _headers(self, method: str, path: str) -> dict:
        if self._private_key is None:
            return {}  # unauthenticated; public GETs still work for market data
        ts = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        # path must be the full request path used in the signature, e.g. /trade-api/v2/markets
        full_path = "/trade-api/v2" + path
        url = self.base_url.replace("/trade-api/v2", "") + full_path
        r = self.session.get(url, headers=self._headers("GET", full_path),
                             params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    # ---------- market data ----------
    def get_markets(self, series_ticker: str | None = None, event_ticker: str | None = None,
                    status: str = "open", limit: int = 200, search: str | None = None
                    ) -> list[KalshiMarket]:
        params = {"limit": limit, "status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        data = self._get("/markets", params=params)
        out = [_to_market(m) for m in data.get("markets", [])]
        if search:
            s = search.lower()
            out = [m for m in out if s in m.title.lower() or s in m.ticker.lower()]
        return out

    def get_event_markets(self, event_ticker: str) -> list[KalshiMarket]:
        return self.get_markets(event_ticker=event_ticker, status="open", limit=200)

    def orderbook_prices(self, ticker: str) -> dict:
        """
        Real executable prices from the order book (the /markets LIST endpoint returns
        null prices here). Kalshi resting orders are BIDS on each side:
            best_no_bid  -> cost to BUY YES  = 100 - best_no_bid   (yes_ask)
            best_yes_bid -> cost to BUY NO   = 100 - best_yes_bid  (no_ask)
        Quantities are ignored here (we only need the touch price). Returns cents.
        """
        try:
            ob = self._get(f"/markets/{ticker}/orderbook").get("orderbook_fp", {}) or {}
        except Exception:
            return {"yes_ask": None, "no_ask": None}
        yd = ob.get("yes_dollars") or []   # resting YES bids, ascending by price
        nd = ob.get("no_dollars") or []    # resting NO bids, ascending by price
        best_yes_bid = float(yd[-1][0]) if yd else None
        best_no_bid = float(nd[-1][0]) if nd else None
        yes_ask = int(round(100 * (1 - best_no_bid))) if best_no_bid is not None else None
        no_ask = int(round(100 * (1 - best_yes_bid))) if best_yes_bid is not None else None
        yes_bid = int(round(100 * best_yes_bid)) if best_yes_bid is not None else None
        no_bid = int(round(100 * best_no_bid)) if best_no_bid is not None else None
        return {"yes_ask": yes_ask, "no_ask": no_ask, "yes_bid": yes_bid, "no_bid": no_bid}

    def find_world_cup_markets(self, search: str = "world cup") -> list[KalshiMarket]:
        """Convenience scan. Kalshi's soccer series tickers change; adjust as needed."""
        markets = []
        for status in ("open",):
            try:
                markets += self.get_markets(status=status, limit=1000, search=search)
            except requests.HTTPError:
                pass
        return markets

    def fetch_fixture_quotes(self, home: str, away: str,
                             fill_synthetic: bool = False) -> tuple[dict, dict]:
        """
        Map a single World Cup match to our model's market names, using the live
        KXWC* series:
            KXWCGAME  -> result (home/away/draw)
            KXWCTOTAL -> Over/Under X.5 goals
            KXWCBTTS  -> Both teams to score (Yes/No)

        Returns (quotes, stats) where:
            quotes[market_name] = {ticker, price_cents, decimal_odds, side}
            stats = {"matched": int, "quoted": int, "game_key": str|None}
        Markets with no live ask (empty order book) are skipped.
        """
        quotes: dict[str, dict] = {}
        stats = {"matched": 0, "quoted": 0, "game_key": None}
        if fill_synthetic:
            print("    [WARN] fill_synthetic=True -> using FAKE prices for empty books "
                  "(TEST MODE ONLY). Do NOT trade on these.")

        # 1) Locate the game via KXWCGAME by matching both team names in the title.
        games = self.get_markets(series_ticker="KXWCGAME", status="open", limit=1000)
        game_key = None
        for m in games:
            title = m.raw.get("title", "")
            if _team_matches(home, title) and _team_matches(away, title):
                game_key = m.ticker.split("-")[1] if "-" in m.ticker else None
                break
        if not game_key:
            return quotes, stats
        stats["game_key"] = game_key

        ob_cache: dict[str, dict] = {}

        def add(name, m, side):
            stats["matched"] += 1
            if m.ticker not in ob_cache:
                ob_cache[m.ticker] = self.orderbook_prices(m.ticker)
            obp = ob_cache[m.ticker]
            price = obp["yes_ask"] if side == "yes" else obp["no_ask"]
            if price and 0 < price < 100:
                stats["quoted"] += 1
            elif fill_synthetic:  # test hook: fake a price only when the book is empty
                price = 25 + (abs(hash(m.ticker)) % 60)  # 25..84c
            else:
                return  # no resting orders on this side
            quotes[name] = {"ticker": m.ticker, "price_cents": price,
                            "decimal_odds": 100.0 / price, "side": side}

        # 2) Result markets (same series, this event).
        for m in games:
            if game_key not in m.ticker:
                continue
            sub = m.raw.get("yes_sub_title", "")
            if _team_matches(home, sub):
                add(f"{home} win", m, "yes")
            elif _team_matches(away, sub):
                add(f"{away} win", m, "yes")
            elif "tie" in sub.lower() or "draw" in sub.lower():
                add("Draw", m, "yes")

        # 3) Totals: Over X.5 (yes) and Under X.5 (no side of the same contract).
        for m in self.get_markets(event_ticker=f"KXWCTOTAL-{game_key}", status="open"):
            sub = m.raw.get("yes_sub_title", "")           # "Over 2.5 goals scored"
            mo = re.search(r"(\d+\.\d+)", sub)
            if not mo:
                continue
            line = mo.group(1)
            add(f"Over {line} goals", m, "yes")
            add(f"Under {line} goals", m, "no")

        # 4) Both teams to score.
        for m in self.get_markets(event_ticker=f"KXWCBTTS-{game_key}", status="open"):
            add("Both teams to score", m, "yes")
            add("Both teams to score - No", m, "no")

        # 5) Team totals: "{Team} over X.5 goals".
        for m in self.get_markets(event_ticker=f"KXWCTEAMTOTAL-{game_key}", status="open"):
            sub = m.raw.get("yes_sub_title", "")
            mo = re.search(r"(\d+\.\d+)", sub)
            if not mo:
                continue
            team = home if _team_matches(home, sub) else (away if _team_matches(away, sub) else None)
            if team:
                add(f"{team} over {mo.group(1)} goals", m, "yes")

        # 6) Spread / handicap: "{Team} wins by more than X.5 goals" -> (X+1)+ goals.
        for m in self.get_markets(event_ticker=f"KXWCSPREAD-{game_key}", status="open"):
            sub = m.raw.get("yes_sub_title", "")
            mo = re.search(r"(\d+)\.5", sub)
            team = home if _team_matches(home, sub) else (away if _team_matches(away, sub) else None)
            if team and mo:
                add(f"{team} wins by {int(mo.group(1)) + 1}+ goals", m, "yes")

        # 7) Anytime goalscorer: "{Player}: 1+". (Higher thresholds skipped for now.)
        for m in self.get_markets(event_ticker=f"KXWCGOAL-{game_key}", status="open"):
            sub = m.raw.get("yes_sub_title", "")
            mo = re.match(r"(.+?):\s*1\+", sub)
            if mo:
                add(f"{mo.group(1).strip()} to score", m, "yes")

        # 8) Correct score: "{Winner} wins A-B" or "Draw A-A" -> "Score {home}-{away}".
        for m in self.get_markets(event_ticker=f"KXWCSCORE-{game_key}", status="open"):
            sub = m.raw.get("yes_sub_title", "")
            dm = re.search(r"draw\s+(\d+)-(\d+)", sub.lower())
            wm = re.search(r"wins\s+(\d+)-(\d+)", sub.lower())
            if dm:
                add(f"Score {dm.group(1)}-{dm.group(2)}", m, "yes")
            elif wm:
                a, b = wm.group(1), wm.group(2)
                if _team_matches(home, sub):
                    add(f"Score {a}-{b}", m, "yes")
                elif _team_matches(away, sub):
                    add(f"Score {b}-{a}", m, "yes")

        return quotes, stats


def _to_market(m: dict) -> KalshiMarket:
    # Kalshi returns null (None) for these on untraded markets; coerce to 0.
    return KalshiMarket(
        ticker=m.get("ticker", ""),
        title=m.get("title", m.get("yes_sub_title", "")),
        yes_bid=int(m.get("yes_bid") or 0),
        yes_ask=int(m.get("yes_ask") or 0),
        no_bid=int(m.get("no_bid") or 0),
        no_ask=int(m.get("no_ask") or 0),
        volume=int(m.get("volume") or 0),
        status=m.get("status", ""),
        raw=m,
    )


def _looks_real(key_id: str) -> bool:
    return bool(key_id) and "PASTE_" not in key_id


if __name__ == "__main__":
    client = KalshiClient()
    if client._private_key is None:
        print("No credentials filled in -- running in public/unauthenticated mode.")
    try:
        mkts = client.find_world_cup_markets("soccer")
        print(f"Fetched {len(mkts)} markets.")
        for m in mkts[:10]:
            print(f"  {m.ticker:30s} YES {m.yes_price}c "
                  f"(impl {m.implied_prob()*100:.0f}%, odds {m.decimal_odds():.2f})  {m.title}")
    except Exception as e:
        print(f"Live fetch failed ({e}). Fill credentials / check network.")
