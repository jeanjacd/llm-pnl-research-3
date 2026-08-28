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

import datetime as dt
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
from .naming import name_similarity, resolve_fixture

COLLECTIONS_PATH = "/multivariate_event_collections"


class DiscoveryError(RuntimeError):
    """A market sweep could not be completed.

    Raised rather than swallowed: a silently-dropped series is
    indistinguishable from a league that genuinely has no such market, and
    that ambiguity has already caused one wrong coverage report.
    """

# Families present on-venue with no validated model. Recorded, never priced.
UNSUPPORTED_SUFFIXES = (
    "1HBTTS", "1HSPREAD",
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
# First-half markets settle on the interval, not on regulation, so they carry a
# different phrase and would otherwise be refused as "settlement basis not
# confirmed". Verified against live rules text for 1H, 1HTOTAL and 1HSCORE.
_FIRST_HALF_RE = re.compile(
    r"(45\s+minutes\s+plus\s+stoppage\s+time|1st\s+Half|first\s+half)",
    re.IGNORECASE)
FIRST_HALF_FAMILIES = ("1H", "1HTOTAL", "1HSCORE")


def settlement_confirmed(family: str, rules: str) -> bool:
    """Is this market's settlement basis one we actually model?

    A family is only tradeable when its rules SAY which period settles it. An
    unstated basis is refused rather than assumed, exactly as for full-match
    markets -- the failure mode is paying out on a number nobody observed.
    """
    if family in FIRST_HALF_FAMILIES:
        return bool(_FIRST_HALF_RE.search(rules))
    return bool(_REGULATION_RE.search(rules))


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


# --- fixture identity ---------------------------------------------------------
# The event ticker (KXEPLGAME-26SEP06ARSCFC) encodes the two clubs as variable
# length codes -- RSLLAFC is RSL+LAFC, not RSL+LAF -- so it CANNOT be split
# reliably. The rules text names both clubs in full and is used instead.
#
# Ordering: the rules list the HOME side first. Verified two independent ways
# on 2026-08-24: (a) against Kalshi's own structured `custom_strike`
# home_team_id/away_team_id on SCORE markets, 4/4 agree; (b) against our ESPN
# fixture table across all five leagues, 255/255 agree with zero counter-
# examples. `resolve_fixture` still reports `flipped` and callers still honour
# it, so a future convention change degrades to a corrected claim, not a wrong
# one.
KALSHI_LEAGUE_LABEL = {"premier_league": "EPL", "mls": "MLS",
                       "bundesliga": "Bundesliga", "la_liga": "La Liga",
                       "ligue_1": "Ligue 1"}

# Note the leading greedy `.*`: it forces `the` to bind to the LAST occurrence
# before "vs", because the sentence usually opens "If Tie is the result of the
# <A> vs <B>...". Anchoring on the first `the` captures "result of the <A>".
_RULES_CACHE: dict = {}


def _rules_re(label: str):
    if label not in _RULES_CACHE:
        _RULES_CACHE[label] = re.compile(
            r".*\bthe\s+(?P<home>.+?)\s+vs\.?\s+(?P<away>.+?)\s+"
            r"(?:professional\s+)?" + re.escape(label) +
            r"\s+(?:soccer\s+)?(?:game|match)\s+originally scheduled for\s+"
            r"(?P<date>[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})", re.IGNORECASE)
    return _RULES_CACHE[label]


def rules_fixture(rules_text: str, league_id: str):
    """(home, away, date) as the venue's own rules text names them, or None.

    Parsed rather than assumed: this is the only field on a Kalshi soccer
    market that carries full club names and the scheduled date together.
    """
    label = KALSHI_LEAGUE_LABEL.get(league_id)
    if not label or not rules_text:
        return None
    mo = _rules_re(label).search(rules_text)
    if not mo:
        return None
    try:
        when = dt.datetime.strptime(mo.group("date"), "%b %d, %Y")
    except ValueError:
        return None
    return mo.group("home").strip(), mo.group("away").strip(), when


# --- claims -------------------------------------------------------------------
_TIE_RE = re.compile(r"^tie$", re.I)
_TOTAL_RE = re.compile(r"^over\s+(?P<line>[\d.]+)\s+goals?\s+scored$", re.I)
_SPREAD_RE = re.compile(r"^(?P<team>.+?)\s+wins?\s+by\s+more\s+than\s+"
                        r"(?P<line>[\d.]+)\s+goals?$", re.I)
_TEAMTOTAL_RE = re.compile(r"^(?P<team>.+?)\s+over\s+(?P<line>[\d.]+)\s+goals?$", re.I)
_BTTS_RE = re.compile(r"^both\s+teams\s+to\s+score$", re.I)
_SCORE_RE = re.compile(r"^(?P<team>.+?)\s+wins?\s+(?P<a>\d+)\s*-\s*(?P<b>\d+)$", re.I)
# First-half wordings, read from live markets 2026-08-28:
#   1H        "Tottenham wins 1st Half" / "Tie 1st Half"
#   1HTOTAL   "Over 2.5 1H goals scored"
#   1HSCORE   "Tottenham Hotspur wins 1H 3-2"   (winner's goals first)
_1H_TIE_RE = re.compile(r"^tie\s+1st\s+half$", re.I)
_1H_WIN_RE = re.compile(r"^(?P<team>.+?)\s+wins?\s+1st\s+half$", re.I)
_1H_TOTAL_RE = re.compile(
    r"^over\s+(?P<line>[\d.]+)\s+1h\s+goals?\s+scored$", re.I)
_1H_SCORE_RE = re.compile(
    r"^(?P<team>.+?)\s+wins?\s+1h\s+(?P<a>\d+)\s*-\s*(?P<b>\d+)$", re.I)


def claim_for(family: str, yes_sub_title: str, home: str, away: str,
              floor_strike=None):
    """The claim a market's YES side pays on, in the model's vocabulary.

    Returns None for anything the exact-scoreline grid cannot price, which the
    caller records as an explicit abstention rather than dropping.
    """
    sub = (yes_sub_title or "").strip()
    if not sub:
        return None

    def side(team):
        if name_similarity(team, home) >= 0.6:
            return "home"
        if name_similarity(team, away) >= 0.6:
            return "away"
        return None

    if family == "GAME":
        if _TIE_RE.match(sub):
            return "draw"
        which = side(sub)
        return "%s_win" % which if which else None
    if family == "TOTAL":
        mo = _TOTAL_RE.match(sub)
        line = mo.group("line") if mo else floor_strike
        return "total_over_%s" % line if line is not None else None
    if family == "SPREAD":
        mo = _SPREAD_RE.match(sub)
        if not mo:
            return None
        which = side(mo.group("team"))
        return ("%s_wins_by_over_%s" % (which, mo.group("line"))
                if which else None)
    if family == "BTTS":
        return "btts" if _BTTS_RE.match(sub) else None
    if family == "TEAMTOTAL":
        mo = _TEAMTOTAL_RE.match(sub)
        if not mo:
            return None
        which = side(mo.group("team"))
        return "%s_over_%s" % (which, mo.group("line")) if which else None
    if family == "1H":
        if _1H_TIE_RE.match(sub):
            return "1h_draw"
        mo = _1H_WIN_RE.match(sub)
        if not mo:
            return None
        which = side(mo.group("team"))
        return "1h_%s_win" % which if which else None
    if family == "1HTOTAL":
        mo = _1H_TOTAL_RE.match(sub)
        line = mo.group("line") if mo else floor_strike
        return "1h_total_over_%s" % line if line is not None else None
    if family == "1HSCORE":
        mo = _1H_SCORE_RE.match(sub)
        if not mo:
            return None
        which = side(mo.group("team"))
        a, b = int(mo.group("a")), int(mo.group("b"))
        # As with full-match SCORE, the named team's goals come first.
        if which == "home":
            return "1h_score_%d-%d" % (a, b)
        if which == "away":
            return "1h_score_%d-%d" % (b, a)
        return None
    if family == "SCORE":
        mo = _SCORE_RE.match(sub)
        if not mo:
            return None
        which = side(mo.group("team"))
        a, b = int(mo.group("a")), int(mo.group("b"))
        # "Fulham FC wins 4-3" states the NAMED team's goals first.
        if which == "home":
            return "score_%d-%d" % (a, b)
        if which == "away":
            return "score_%d-%d" % (b, a)
        return None
    return None


class KalshiProvider(MarketDataProvider):
    venue = "kalshi"

    def __init__(self, client: KalshiClient | None = None):
        self.client = client or KalshiClient()
        self.last_errors = []

    # ---- single contracts ----
    def discover(self, spec, status: str = "open", with_books: bool = True,
                 include_unsupported: bool = True, strict: bool = True,
                 fixtures=None) -> list:
        """Every market in this league's verified series.

        `include_unsupported` keeps out-of-model families in the returned set so
        they can be RECORDED and explicitly abstained from, which the mission
        requires, rather than silently disappearing from coverage counts.

        `strict` (default) RAISES if any series fetch fails. Swallowing the
        error would let a transient network blip masquerade as "this league has
        no 1X2 markets today" -- an error that already produced a wrong
        coverage report once. Callers that genuinely tolerate partial data must
        opt out explicitly and inspect `last_errors`.

        `fixtures` is this league's own fixture table. Without it a leg
        carries no home/away and therefore no priceable claim, so the
        instrument is returned marked unsupported rather than silently
        looking tradeable -- omitting fixtures previously produced 223
        "supported" Kalshi instruments that could not yield a single case.
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
                                            with_books=with_books,
                                            fixtures=fixtures))
        return out

    def _instrument(self, market: dict, spec, prefix: str,
                    with_books: bool, fixtures=None) -> MarketInstrument:
        event = market.get("event_ticker", "") or ""
        family = family_of(event, prefix)
        supported_family = is_supported_family(family)
        rules = ((market.get("rules_primary") or "") + " "
                 + (market.get("rules_secondary") or ""))
        regulation = settlement_confirmed(family, rules)
        sub = market.get("yes_sub_title") or ""
        ticker = market.get("ticker", "")

        parsed = rules_fixture(rules, spec.league_id)
        resolved = None
        if parsed and fixtures is not None:
            resolved = resolve_fixture(parsed[0], parsed[1], parsed[2], fixtures)
        home = resolved["home"] if resolved else None
        away = resolved["away"] if resolved else None
        # NOT `expected_expiration_time`: measured 2026-08-24 across 63 EPL
        # markets, that field equals `occurrence_datetime` and sits exactly 3h
        # AFTER kick-off -- it is when the market settles. Reading it as the
        # kick-off offered a match already 72 minutes old as a pre-match trade.
        # Our fixture table carries the real kick-off, so it governs.
        kickoff = None
        if resolved and resolved.get("kickoff_utc") is not None:
            stamp = resolved["kickoff_utc"]
            kickoff = (stamp.isoformat() if hasattr(stamp, "isoformat")
                       else str(stamp))

        claim = None
        if supported_family and regulation and home and away:
            claim = claim_for(family, sub, home, away,
                              market.get("floor_strike"))

        def abstain(reason):
            return Leg(claim="family_%s" % family, market_ref=ticker,
                       description=sub, home=home, away=away,
                       league_id=spec.league_id, kickoff_utc=kickoff,
                       supported=False, unsupported_reason=reason)

        if not supported_family:
            leg = abstain("family %s has no validated model" % family)
        elif not regulation:
            leg = abstain("settlement basis not confirmed as regulation time")
        elif parsed is None:
            leg = abstain("rules text does not name both clubs and a date")
        elif fixtures is None:
            leg = abstain("no fixture table supplied; cannot identify the match")
        elif resolved is None:
            leg = abstain("no unique fixture matches %r vs %r on %s"
                          % (parsed[0], parsed[1], parsed[2].date()))
        elif claim is None:
            leg = abstain("no validated model for %r in family %s"
                          % (sub[:60], family))
        else:
            leg = Leg.build(claim, ticker, description=sub, home=home,
                            away=away, league_id=spec.league_id,
                            kickoff_utc=kickoff)

        book = Book(observed_at=utcnow_iso())
        if with_books and leg.supported:
            book = self.fetch_book_by_ticker(ticker)
        return MarketInstrument(
            venue=self.venue, instrument_id=ticker,
            kind=KIND_BINARY,
            title="%s | %s" % (market.get("title", ""), sub),
            legs=(leg,), rules_text=rules.strip(),
            settlement_source="kalshi", settles_on_regulation=regulation,
            event_ref=event, league_id=spec.league_id,
            status=market.get("status", ""),
            close_time=market.get("close_time"),
            kickoff_utc=kickoff,
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
