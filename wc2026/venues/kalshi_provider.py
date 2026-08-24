"""
venues/kalshi_provider.py
=========================
Read-only Kalshi provider: single contracts and multivariate (parlay) combos.

Everything below was enumerated live against the API on 2026-08-24, not
recalled.

SINGLE CONTRACTS
----------------
Every league exposes its families behind a league prefix. Full observed sets:

  KXEPL*         38 series   KXLALIGA*  36   KXBUNDESLIGA* 30
  KXLIGUE1*      25 series   KXMLS*     28

Of those, exactly six map onto the regulation scoreline grid and are SUPPORTED:
GAME (1X2), TOTAL, SPREAD, BTTS, TEAMTOTAL, SCORE.

Everything else is recorded and abstained from, including families that are
easy to mistake for supported ones:
  * 1H / 1HBTTS / 1HSPREAD / 1HTOTAL / 1HSCORE / 2H*  -- half-time markets; the
    model has no first-half goal process.
  * GOAL / ANYGOAL / FIRSTGOAL / SOA / AST            -- player props.
  * CORNERS / TCORNERS                                -- no corner model.
  * MOV / FTTS / ADVANCE                              -- method of victory,
    first team to score, tie-break/advancement rules.
  * LEADER / LAST / TOP / TOP2 / TOP4 / TOP6 / RELEGATION / POINTMARGIN /
    TEAMPOINTS / SEASONSTAT / POY / H2H / H2HFINISH   -- season futures.

TWO TRAPS, both handled by constructing exact tickers rather than prefix-matching:
  * KXLALIGA2*, KXBUNDESLIGA2*  are the SECOND DIVISION (LaLiga 2, 2. Bundesliga).
  * KXMLSAST*                   is the ALL-STAR game (KXMLSASTGAME, ...).
A prefix scan would silently pull both into a top-flight league's slate.

COMBOS / PARLAYS
----------------
Kalshi parlays are "multivariate event collections". Observed open collections:
KXMVESPORTSMULTIGAMEEXTENDED-R, KXMVECROSSCATEGORY-R, KXMVECROSSCATEGORY-SHARD1-R
-- each with size_min=2, size_max=0 (unbounded), is_all_yes=False,
is_single_market_per_event=False, and the functional description "the resulting
market will only resolve to YES if every associated market resolves to YES;
scalar outcomes are multiplied".

`GET /multivariate_event_collections/{ticker}` returns `associated_event_tickers`
-- the ELIGIBLE legs. Of 1,462 eligible legs, 217 belong to our five leagues
across 58 fixtures (EPL 10, La Liga 14, Bundesliga 9, Ligue 1 9, MLS 16), and
they include unsupported families (1H*, CORNERS) alongside supported ones.

Because `is_single_market_per_event` is False, two legs from the SAME fixture
may be combined. Their outcomes are strongly dependent, so a joint probability
must come from the shared scoreline grid -- never from multiplying marginals.

CRITICAL LIMITATION (verified, and the reason combos cannot be traded here):
pre-created combo markets (e.g.
KXMVESPORTSMULTIGAMEEXTENDED-S20263BE0E27BA60-9AED4F77C19) expose only repeated
leg DESCRIPTIONS ("no Over 5.5 goals scored" x36) with NO fixture identifiers --
absent from the market payload, the event payload (`with_nested_markets`), and
`rules_primary` (empty). Their order books are also empty. Pricing a chosen
combination requires the lookup/RFQ path, which creates or confirms a quote and
is forbidden in paper mode.

Therefore this provider DISCOVERS combos and can value ones WE construct from
eligible legs, but never claims an executable combo price: an unresolvable or
unquoted combo is reported as DEFER/UNSUPPORTED, never as a fill.
"""
from __future__ import annotations

import re

from ..betting.kalshi import KalshiClient
from ..leagues import SUPPORTED_KALSHI_SUFFIXES
from .base import (
    KIND_BINARY,
    KIND_NATIVE_COMBO,
    Book,
    Leg,
    MarketDataProvider,
    MarketInstrument,
    utcnow_iso,
)

COLLECTIONS_PATH = "/multivariate_event_collections"


class DiscoveryError(RuntimeError):
    """A market sweep could not be completed.

    Raised rather than swallowed: a silently-dropped series is
    indistinguishable from a league that genuinely has no such market, and
    that ambiguity has already caused one wrong coverage report.
    """

# Families present on-venue with no validated model. Recorded, never priced.
UNSUPPORTED_SUFFIXES = (
    "1H", "1HBTTS", "1HSPREAD", "1HTOTAL", "1HSCORE",
    "2H", "2HBTTS", "2HSPREAD", "2HTOTAL",
    "GOAL", "ANYGOAL", "FIRSTGOAL", "SOA", "AST",
    "CORNERS", "TCORNERS", "MOV", "FTTS", "ADVANCE",
    "LEADER", "LAST", "TOP", "TOP2", "TOP4", "TOP6",
    "RELEGATION", "POINTMARGIN", "TEAMPOINTS", "SEASONSTAT", "POY",
    "H2H", "H2HFINISH", "SKILLS", "JOIN", "CUP", "EAST", "WEST",
)

_REGULATION_RE = re.compile(
    r"90\s+minutes\s+plus\s+stoppage\s+time\s*\(does\s+not\s+include\s+extra\s+time",
    re.IGNORECASE)


def league_prefix(spec) -> str | None:
    """The Kalshi prefix for a league, derived from its verified series."""
    series = spec.venue_series.get("kalshi") or ()
    if not series:
        return None
    game = next((t for t in series if t.endswith("GAME")), series[0])
    return game[:-4] if game.endswith("GAME") else game


def family_of(ticker: str, prefix: str) -> str:
    """The family suffix of a series ticker under a league prefix."""
    rest = ticker[len(prefix):] if ticker.startswith(prefix) else ticker
    return rest.split("-")[0]


def is_supported_family(family: str) -> bool:
    return family in set(SUPPORTED_KALSHI_SUFFIXES)


class KalshiProvider(MarketDataProvider):
    venue = "kalshi"

    def __init__(self, client: KalshiClient | None = None):
        self.client = client or KalshiClient()
        self.last_errors = []

    # ---- single contracts ----
    def discover(self, spec, status: str = "open", with_books: bool = True,
                 include_unsupported: bool = True, strict: bool = True) -> list:
        """Every market in this league's verified series.

        `include_unsupported` keeps out-of-model families in the returned set so
        they can be RECORDED and explicitly abstained from, which the mission
        requires, rather than silently disappearing from coverage counts.

        `strict` (default) RAISES if any series fetch fails. Swallowing the
        error would let a transient network blip masquerade as "this league has
        no 1X2 markets today" -- an error that already produced a wrong
        coverage report once. Callers that genuinely tolerate partial data must
        opt out explicitly and inspect `last_errors`.
        """
        prefix = league_prefix(spec)
        if not prefix:
            return []
        out = []
        self.last_errors = []
        wanted = list(spec.venue_series.get("kalshi", ()))
        if include_unsupported:
            wanted += [prefix + suffix for suffix in UNSUPPORTED_SUFFIXES]
        for series in wanted:
            try:
                markets = self.client.get_markets(series_ticker=series,
                                                  status=status)
            except Exception as exc:                          # noqa: BLE001
                self.last_errors.append((series, str(exc)))
                if strict:
                    raise DiscoveryError(
                        "%s: fetching %s failed (%s). Refusing to report "
                        "partial coverage as complete."
                        % (spec.league_id, series, exc)) from exc
                continue
            for market in markets:
                out.append(self._instrument(market, spec, prefix,
                                            with_books=with_books))
        return out

    def _instrument(self, market: dict, spec, prefix: str,
                    with_books: bool) -> MarketInstrument:
        event = market.get("event_ticker", "") or ""
        family = family_of(event, prefix)
        supported_family = is_supported_family(family)
        rules = ((market.get("rules_primary") or "") + " "
                 + (market.get("rules_secondary") or ""))
        regulation = bool(_REGULATION_RE.search(rules))
        sub = market.get("yes_sub_title") or ""

        if not supported_family:
            leg = Leg(claim="family_%s" % family,
                      market_ref=market.get("ticker", ""), description=sub,
                      league_id=spec.league_id, supported=False,
                      unsupported_reason="family %s has no validated model"
                                         % family)
        elif not regulation:
            leg = Leg(claim="family_%s" % family,
                      market_ref=market.get("ticker", ""), description=sub,
                      league_id=spec.league_id, supported=False,
                      unsupported_reason="settlement basis not confirmed as "
                                         "regulation time")
        else:
            # The exact claim (side/line) is resolved by betting/markets.py
            # against the fixture; here the family is enough to mark support.
            leg = Leg(claim="family_%s_ok" % family,
                      market_ref=market.get("ticker", ""), description=sub,
                      league_id=spec.league_id, supported=True)

        book = Book(observed_at=utcnow_iso())
        if with_books and leg.supported:
            book = self.fetch_book_by_ticker(market.get("ticker", ""))
        return MarketInstrument(
            venue=self.venue, instrument_id=market.get("ticker", ""),
            kind=KIND_BINARY,
            title="%s | %s" % (market.get("title", ""), sub),
            legs=(leg,), rules_text=rules.strip(),
            settlement_source="kalshi", settles_on_regulation=regulation,
            event_ref=event, league_id=spec.league_id,
            status=market.get("status", ""),
            close_time=market.get("close_time"),
            kickoff_utc=market.get("expected_expiration_time"),
            tick_cents=1, min_size=1.0,
            fee_model={"venue": "kalshi", "taker_factor": 0.07,
                       "maker_factor": 0.0175},
            book=book, raw=market)

    def fetch_book_by_ticker(self, ticker: str) -> Book:
        try:
            ob = self.client.get_orderbook(ticker)
        except Exception:                                     # noqa: BLE001
            return Book(observed_at=utcnow_iso())
        return Book(
            yes_asks=tuple(ob.executable("yes")),
            no_asks=tuple(ob.executable("no")),
            yes_bids=tuple(reversed(ob.yes_bids.levels)),
            no_bids=tuple(reversed(ob.no_bids.levels)),
            observed_at=utcnow_iso())

    def fetch_book(self, instrument: MarketInstrument) -> Book:
        return self.fetch_book_by_ticker(instrument.instrument_id)

    # ---- combos / parlays ----
    def collections(self, status: str = "open") -> list:
        data = self.client._request("GET", COLLECTIONS_PATH,
                                    params={"limit": 100, "status": status})
        return data.get("multivariate_contracts", []) or []

    def eligible_combo_legs(self, spec) -> dict:
        """Parlay-eligible legs for one league, split by support.

        Returns {collection_ticker: {"supported": [...], "unsupported": [...],
        "fixtures": [...]}}. This is read-only discovery: it never creates or
        confirms an RFQ.
        """
        prefix = league_prefix(spec)
        out = {}
        if not prefix:
            return out
        for col in self.collections():
            ticker = col.get("collection_ticker")
            if not ticker:
                continue
            detail = self.client._request("GET", "%s/%s" % (COLLECTIONS_PATH,
                                                            ticker))
            contract = detail.get("multivariate_contract", {}) or {}
            legs = contract.get("associated_event_tickers") or []
            mine = [e for e in legs if e.startswith(prefix)]
            supported, unsupported, fixtures = [], [], set()
            for event in mine:
                family = family_of(event, prefix)
                fixture = event.split("-", 1)[1] if "-" in event else event
                fixtures.add(fixture)
                (supported if is_supported_family(family)
                 else unsupported).append(event)
            if mine:
                out[ticker] = {
                    "supported": supported, "unsupported": unsupported,
                    "fixtures": sorted(fixtures),
                    "size_min": contract.get("size_min"),
                    "size_max": contract.get("size_max"),
                    "is_all_yes": contract.get("is_all_yes"),
                    "single_market_per_event":
                        contract.get("is_single_market_per_event"),
                    "functional_description":
                        contract.get("functional_description", ""),
                }
        return out

    def discover_combo_markets(self, spec, status: str = "open") -> list:
        """Pre-created combo markets, recorded as UNRESOLVABLE.

        Verified: these expose repeated leg descriptions with no fixture
        identifiers in the market, the event (even with nested markets) or the
        rules, and carry empty books. They are therefore returned as
        unsupported instruments so that coverage reporting can count them
        honestly instead of implying they were evaluated.
        """
        out = []
        for col in self.collections(status=status):
            series = (col.get("collection_ticker") or "").rsplit("-", 1)[0]
            if not series:
                continue
            try:
                markets = self.client.get_markets(series_ticker=series,
                                                  status=status)
            except Exception:                                 # noqa: BLE001
                continue
            for market in markets:
                sub = market.get("yes_sub_title") or ""
                n_legs = len([p for p in sub.split(",") if p.strip()])
                out.append(MarketInstrument(
                    venue=self.venue, instrument_id=market.get("ticker", ""),
                    kind=KIND_NATIVE_COMBO,
                    title="Kalshi combo (%d legs)" % n_legs,
                    legs=tuple(
                        Leg(claim="combo_leg_unresolved",
                            market_ref=market.get("ticker", ""),
                            description=part.strip(), supported=False,
                            unsupported_reason=(
                                "combo leg cannot be resolved to a fixture "
                                "from public read-only data"))
                        for part in sub.split(",") if part.strip()),
                    rules_text=market.get("rules_primary") or "",
                    settlement_source="kalshi", settles_on_regulation=None,
                    event_ref=market.get("event_ticker", ""),
                    league_id=None, status=market.get("status", ""),
                    close_time=market.get("close_time"),
                    fee_model={"venue": "kalshi", "taker_factor": 0.07},
                    book=Book(observed_at=utcnow_iso()), raw=market))
        return out
