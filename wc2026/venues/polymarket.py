"""
venues/polymarket.py
====================
Read-only Polymarket provider (Gamma metadata + CLOB order books).

Verified live on 2026-08-24:
  * GET https://gamma-api.polymarket.com/sports            -- the league
    registry. This is where league tag IDs come from.
  * GET https://gamma-api.polymarket.com/events?tag_id=..  -- per-league events
    with nested markets.
  * GET https://clob.polymarket.com/book?token_id=..       -- real depth,
    {"bids": [{"price","size"}...], "asks": [...]}.

MEASURED COVERAGE, corrected 2026-08-24. An earlier version of this file used
`tag_slug=soccer` plus a naive team-name substring match and reported near-zero
liquidity for these leagues. That was WRONG on both counts: the generic soccer
tag returns a different, largely obscure population, and ESPN display names
("Manchester City") do not substring-match venue names ("Man City").

Re-measured with the registry tag IDs, counting only fixtures whose
`gameStartTime` is still in the future and whose markets are `acceptingOrders`:

    league          upcoming  markets  median liq  ask depth (soonest 9)  spread
    premier_league        95    1,074      $1,329               $93,605      1c
    la_liga               96    1,124        $950               $23,418      1c
    mls                   82      516        $385              $173,688      6c
    bundesliga            93      975         $76                $7,791      4c
    ligue_1               89    1,035         $39                $5,762      4c

All five leagues are genuinely liquid here, with EPL/La Liga at 1-cent spreads.
Median liquidity across ALL upcoming markets looks small only because markets
open weeks early and fill up near kickoff -- which is exactly why depth is read
per fixture from the CLOB book and never inferred from an aggregate field.

FIELD TRAPS (each verified; each one produced a wrong answer before):
  * `startDate` is when the MARKET WAS CREATED, not kickoff. Kickoff is
    `gameStartTime` on the market (the event also carries `startTime`).
  * `liquidityNum` / `liquidity` / `liquidityClob` are venue aggregates, not
    executable depth. Cost basis comes from the CLOB book or it does not exist.
  * `orderMinSize` is 5 on these markets, not 1; `orderPriceMinTickSize` = 0.01.
  * `acceptingOrders` is the string "True"/"False", not a bool.
"""
from __future__ import annotations

import datetime as dt
import difflib
import json
import re
import unicodedata

import requests

from .base import (
    KIND_BINARY,
    Book,
    Leg,
    MarketDataProvider,
    MarketInstrument,
    utcnow_iso,
)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# --- question wordings, enumerated from the live venue (2026-08-24) ----------
# Per fixture Polymarket splits categories across SEPARATE events:
#   "<A> vs. <B>"                  1X2                       (3 markets)
#   "<A> vs. <B> - Exact Score"    exact scorelines          (~17)
#   "<A> vs. <B> - More Markets"   totals, team totals,      (33)
#                                  spreads, BTTS, half markets
#   "- Halftime Result", "- Second Half Result", "- First Team to Score",
#   "- Total Corners", "- Player Props"   -> no validated model
#
# Full-match goal totals are spelled bare ("O/U 2.5"), NOT "Total Goals";
# searching for the word "goals" finds nothing and wrongly concludes the venue
# has no totals. Corner totals use the same O/U wording plus "Total Corners",
# so the corner and half-specific forms must be excluded explicitly.
_WIN_RE = re.compile(r"^will (?P<team>.+?) win\b", re.I)
_DRAW_RE = re.compile(r"end in a draw", re.I)
_BTTS_RE = re.compile(r":\s*both teams to score\s*$", re.I)
_EXACT_RE = re.compile(
    r"^exact score:\s*(?P<a>.+?)\s+(?P<hg>\d+)\s*-\s*(?P<ag>\d+)\s+(?P<b>.+?)\s*\??$",
    re.I)
_SPREAD_RE = re.compile(r"^spread:\s*(?P<team>.+?)\s*\(\s*-(?P<line>\d+(?:\.\d+)?)\s*\)",
                        re.I)
_TEAM_TOTAL_RE = re.compile(
    r"^(?P<a>.+?)\s+vs\.?\s+(?P<b>.+?):\s*(?P<team>.+?)\s+o/u\s+(?P<line>\d+(?:\.\d+)?)\s*$",
    re.I)
_TOTAL_RE = re.compile(
    r"^(?P<a>.+?)\s+vs\.?\s+(?P<b>.+?):\s*o/u\s+(?P<line>\d+(?:\.\d+)?)\s*$", re.I)
# Anything matching these is out of model scope regardless of shape.
_EXCLUDE_RE = re.compile(
    r"corner|1st half|2nd half|first half|second half|halftime|half-time|"
    r"any other score|first team to score|player|assist|card|booking", re.I)

# Club-name affixes carrying no identifying information across venues.
_AFFIXES = {"fc", "cf", "afc", "sc", "ac", "cd", "rc", "ud", "ca", "sv", "vfb",
            "vfl", "bsc", "tsg", "sd", "as", "ss", "us", "rcd", "club", "de",
            "the", "1", "04", "05", "96"}


def normalise_team(name: str) -> set:
    """Accent-folded, affix-stripped token set for cross-venue comparison."""
    folded = unicodedata.normalize("NFKD", name or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = re.sub(r"[^a-z0-9 ]", " ", folded.lower())
    return {t for t in folded.split() if t and t not in _AFFIXES}


def name_similarity(a: str, b: str) -> float:
    """0..1 similarity of two club names, robust to affixes and accents."""
    ta, tb = normalise_team(a), normalise_team(b)
    if not ta or not tb:
        return 0.0
    if ta & tb:
        return 0.5 + 0.5 * len(ta & tb) / len(ta | tb)
    best = 0.0
    for x in ta:
        for y in tb:
            best = max(best, difflib.SequenceMatcher(None, x, y).ratio())
    return 0.5 * best


def parse_ts(value):
    """Parse the venue's several timestamp spellings to aware UTC."""
    if not value:
        return None
    text = str(value).strip().replace(" ", "T").replace("Z", "+00:00")
    if re.search(r"[+-]\d{2}$", text):
        text += ":00"
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)


def _levels(entries, best_first_high: bool) -> tuple:
    """Normalise CLOB levels to ((price_cents, size), ...), best first."""
    out = []
    for entry in entries or []:
        try:
            price = int(round(float(entry.get("price")) * 100))
            size = float(entry.get("size") or 0.0)
        except (TypeError, ValueError):
            continue
        if 0 < price < 100 and size > 0:
            out.append((price, size))
    out.sort(key=lambda t: -t[0] if best_first_high else t[0])
    return tuple(out)


def resolve_fixture(home_raw, away_raw, kickoff, fixtures, min_score=0.6):
    """Resolve a venue event to exactly one of OUR fixtures, or None.

    Matching uses the kickoff date (+/- 1 day) AND both club names, and demands
    a unique winner clearly ahead of the runner-up. Name similarity alone is
    never sufficient: "Manchester City" and "Manchester United" score highly
    against each other, so an ambiguous pair is refused rather than guessed.
    """
    if kickoff is None or fixtures is None or len(fixtures) == 0:
        return None
    import pandas as pd
    day = pd.Timestamp(kickoff).tz_localize(None).normalize()
    window = fixtures[(fixtures["date"] >= day - pd.Timedelta(days=1))
                      & (fixtures["date"] <= day + pd.Timedelta(days=1))]
    scored = []
    for _, row in window.iterrows():
        straight = min(name_similarity(home_raw, row["home_team"]),
                       name_similarity(away_raw, row["away_team"]))
        flipped = min(name_similarity(home_raw, row["away_team"]),
                      name_similarity(away_raw, row["home_team"]))
        if straight >= flipped:
            scored.append((straight, row["home_team"], row["away_team"], False))
        else:
            scored.append((flipped, row["home_team"], row["away_team"], True))
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    best = scored[0]
    if best[0] < min_score:
        return None
    if len(scored) > 1 and scored[1][0] > best[0] - 0.05:
        return None
    return {"home": best[1], "away": best[2], "flipped": best[3],
            "score": best[0]}


class PolymarketProvider(MarketDataProvider):
    venue = "polymarket"

    def __init__(self, session=None, timeout: int = 45):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "wc2026-research/1.0")
        self.timeout = timeout

    def _get(self, url: str, params: dict | None = None):
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ---- discovery ----
    def league_events(self, spec, max_events: int | None = None) -> list:
        """Open events for one league, via the venue's own league tag IDs.

        Exhaustive by default: paging stops only on a short page. A fixed
        cap silently truncated MLS (241 open events against a 200 cap) and
        made its coverage look smaller than it is.
        """
        tags = spec.venue_series.get("polymarket") or ()
        events, seen = [], set()
        for tag_id in tags:
            offset = 0
            while max_events is None or len(events) < max_events:
                page = self._get(GAMMA + "/events",
                                 {"limit": 100, "offset": offset,
                                  "closed": "false", "tag_id": tag_id})
                if not page:
                    break
                for event in page:
                    key = str(event.get("id"))
                    if key not in seen:
                        seen.add(key)
                        events.append(event)
                if len(page) < 100:
                    break
                offset += 100
        return events

    @staticmethod
    def teams_from_title(title: str):
        parts = re.split(r"\s+vs\.?\s+", title or "", flags=re.I)
        if len(parts) != 2:
            return None, None
        return parts[0].strip(), parts[1].strip()

    @staticmethod
    def claim_for(question: str, home: str, away: str):
        """Map a question to the claim that this market's FIRST outcome pays on.

        Polymarket's outcome labels differ by category -- ["Yes","No"],
        ["Over","Under"], or [teamA, teamB] -- and the first CLOB token always
        corresponds to outcomes[0]. Every claim returned here is therefore
        written from outcomes[0]'s point of view, so the token and the claim
        cannot drift apart.

        Returns None for anything without a validated model.
        """
        q = (question or "").strip()
        if _EXCLUDE_RE.search(q):
            return None

        def side(team):
            if name_similarity(team, home) >= 0.6:
                return "home"
            if name_similarity(team, away) >= 0.6:
                return "away"
            return None

        mo = _EXACT_RE.match(q)
        if mo:                       # outcomes ["Yes","No"]
            first, hg, ag = mo.group("a"), int(mo.group("hg")), int(mo.group("ag"))
            which = side(first)
            if which == "home":
                return "score_%d-%d" % (hg, ag)
            if which == "away":
                return "score_%d-%d" % (ag, hg)
            return None
        mo = _SPREAD_RE.match(q)
        if mo:                       # outcomes [named team, other team]
            which = side(mo.group("team"))
            return ("%s_wins_by_over_%s" % (which, mo.group("line"))
                    if which else None)
        if _DRAW_RE.search(q):       # outcomes ["Yes","No"]
            return "draw"
        if _BTTS_RE.search(q):       # outcomes ["Yes","No"]
            return "btts"
        mo = _TEAM_TOTAL_RE.match(q)
        if mo:                       # outcomes ["Over","Under"]
            which = side(mo.group("team"))
            return "%s_over_%s" % (which, mo.group("line")) if which else None
        mo = _TOTAL_RE.match(q)
        if mo:                       # outcomes ["Over","Under"]
            return "total_over_%s" % mo.group("line")
        mo = _WIN_RE.match(q)
        if mo:                       # outcomes ["Yes","No"]
            which = side(mo.group("team"))
            return "%s_win" % which if which else None
        return None

    def discover(self, spec, fixtures=None, with_books: bool = True,
                 max_events: int | None = None,
                 upcoming_only: bool = True) -> list:
        """Normalised instruments for one league.

        `fixtures` is that league's own fixture table. An event that cannot be
        resolved to exactly one fixture is skipped, never attached to a guess.
        """
        now = dt.datetime.now(dt.timezone.utc)
        out = []
        for event in self.league_events(spec, max_events=max_events):
            home_raw, away_raw = self.teams_from_title(event.get("title") or "")
            if not home_raw or not away_raw:
                continue
            for market in event.get("markets") or []:
                kickoff = parse_ts(market.get("gameStartTime")
                                   or event.get("startTime"))
                if upcoming_only and (kickoff is None or kickoff <= now):
                    continue
                if fixtures is not None:
                    resolved = resolve_fixture(home_raw, away_raw, kickoff,
                                               fixtures)
                else:
                    resolved = {"home": home_raw, "away": away_raw,
                                "flipped": False}
                if resolved is None:
                    continue
                out.append(self._instrument(event, market, spec, resolved,
                                            kickoff, with_books=with_books))
        return out

    def _instrument(self, event, market, spec, resolved, kickoff, with_books):
        question = market.get("question") or ""
        home, away = resolved["home"], resolved["away"]
        claim = self.claim_for(question, home, away)
        if resolved.get("flipped") and claim in ("home_win", "away_win"):
            claim = "away_win" if claim == "home_win" else "home_win"
        ref = str(market.get("conditionId") or market.get("id") or "")
        kickoff_iso = kickoff.isoformat() if kickoff else None
        if claim:
            leg = Leg.build(claim, ref, description=question, home=home,
                            away=away, league_id=spec.league_id,
                            kickoff_utc=kickoff_iso)
        else:
            leg = Leg(claim="unmapped", market_ref=ref, description=question,
                      home=home, away=away, league_id=spec.league_id,
                      kickoff_utc=kickoff_iso, supported=False,
                      unsupported_reason="no validated model for question %r"
                                         % question[:80])
        accepting = str(market.get("acceptingOrders")).lower() == "true"
        book = Book(observed_at=utcnow_iso())
        if with_books and leg.supported and accepting:
            book = self.fetch_book_for_market(market)
        return MarketInstrument(
            venue=self.venue, instrument_id=ref, kind=KIND_BINARY,
            title="%s | %s" % (event.get("title"), question),
            legs=(leg,), rules_text=market.get("description") or "",
            settlement_source=str(market.get("resolutionSource") or ""),
            settles_on_regulation=None,
            event_ref=str(event.get("id") or ""), league_id=spec.league_id,
            status="open" if accepting else "not_accepting",
            close_time=event.get("endDate"), kickoff_utc=kickoff_iso,
            tick_cents=max(1, int(round(float(
                market.get("orderPriceMinTickSize") or 0.01) * 100))),
            min_size=float(market.get("orderMinSize") or 1.0),
            fee_model={"venue": "polymarket",
                       "maker_base_fee": market.get("makerBaseFee"),
                       "taker_base_fee": market.get("takerBaseFee")},
            book=book,
            raw={"clobTokenIds": market.get("clobTokenIds"),
                 "liquidityClob": market.get("liquidityClob"),
                 "volume24hr": market.get("volume24hr"),
                 "bestBid": market.get("bestBid"),
                 "bestAsk": market.get("bestAsk")})

    # ---- books ----
    def fetch_book_for_market(self, market: dict) -> Book:
        tokens = market.get("clobTokenIds")
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except json.JSONDecodeError:
                tokens = []
        if not tokens:
            return Book(observed_at=utcnow_iso())
        yes = self._token_book(tokens[0])
        no = self._token_book(tokens[1]) if len(tokens) > 1 else {}
        return Book(yes_asks=_levels(yes.get("asks"), False),
                    yes_bids=_levels(yes.get("bids"), True),
                    no_asks=_levels(no.get("asks"), False),
                    no_bids=_levels(no.get("bids"), True),
                    observed_at=utcnow_iso())

    def _token_book(self, token_id: str) -> dict:
        try:
            return self._get(CLOB + "/book", {"token_id": token_id}) or {}
        except Exception:                                     # noqa: BLE001
            # No book is a legitimate state. It must surface as "no executable
            # price", never as a fabricated one.
            return {}

    def fetch_book(self, instrument: MarketInstrument) -> Book:
        return self.fetch_book_for_market(instrument.raw or {})
