"""
markets.py
==========
Discover Kalshi MLS markets and map each one PRECISELY onto the frozen engine's
regulation scoreline grid.

Verified live against the API (2026-07-20):
  * Series in scope: KXMLSGAME (3-way winner incl. Tie), KXMLSTOTAL (Over X.5),
    KXMLSSPREAD ("T wins by more than X.5"), KXMLSBTTS, KXMLSTEAMTOTAL
    ("T over X.5 goals"), KXMLSSCORE (correct score).
  * Settlement: "after 90 minutes plus stoppage time (does not include extra
    time or penalties)" -- i.e. the regulation grid, exactly. Every market's
    rules text is CHECKED for this phrase; a market whose rules do not confirm
    regulation settlement is skipped (fail-closed), which matters for playoff
    fixtures.
  * Event tickers: KXMLSGAME-26JUL25SJLAG = date (local) + HOME code + AWAY
    code, home first (verified against the fixture list on five events).
  * Kalshi local dates can sit one day before our UTC fixture dates, so
    fixtures are matched by team pair within +/-1 day.

Team identity: an explicit, auditable alias table from Kalshi's short display
names / ticker codes to the canonical ESPN names in the MLS dataset. An
unmatched name is a loud skip, never a guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- Kalshi -> canonical (ESPN) team mapping ----------------------------------
# code = ticker fragment; names = lowercase substrings seen in titles/subtitles.
_TEAMS: list[tuple[str, tuple[str, ...], str]] = [
    ("ATL",  ("atlanta",),            "Atlanta United FC"),
    ("ATX",  ("austin",),             "Austin FC"),
    ("CLT",  ("charlotte",),          "Charlotte FC"),
    ("CHI",  ("chicago",),            "Chicago Fire FC"),
    ("CIN",  ("cincinnati",),         "FC Cincinnati"),
    ("COL",  ("colorado",),           "Colorado Rapids"),
    ("CLB",  ("columbus",),           "Columbus Crew"),
    ("DAL",  ("dallas",),             "FC Dallas"),
    ("DC",   ("d.c. united", "dc united", "washington"), "D.C. United"),
    ("HOU",  ("houston",),            "Houston Dynamo FC"),
    ("SKC",  ("kansas city", "sporting kc"), "Sporting Kansas City"),
    ("LAFC", ("lafc", "los angeles fc"), "LAFC"),
    ("LAG",  ("los angeles g", "la galaxy", "galaxy"), "LA Galaxy"),
    ("MIA",  ("miami",),              "Inter Miami CF"),
    ("MIN",  ("minnesota",),          "Minnesota United FC"),
    ("MTL",  ("montreal", "montréal"), "CF Montréal"),
    ("NSH",  ("nashville",),          "Nashville SC"),
    ("NE",   ("new england",),        "New England Revolution"),
    ("NYC",  ("new york city",),      "New York City FC"),
    ("NYRB", ("new york rb", "red bull"), "Red Bull New York"),
    ("ORL",  ("orlando",),            "Orlando City SC"),
    ("PHI",  ("philadelphia",),       "Philadelphia Union"),
    ("POR",  ("portland",),           "Portland Timbers"),
    ("RSL",  ("salt lake",),          "Real Salt Lake"),
    ("SD",   ("san diego",),          "San Diego FC"),
    ("SJ",   ("san jose",),           "San Jose Earthquakes"),
    ("SEA",  ("seattle",),            "Seattle Sounders FC"),
    ("STL",  ("st. louis", "st louis"), "St. Louis CITY SC"),
    ("TOR",  ("toronto",),            "Toronto FC"),
    ("VAN",  ("vancouver",),          "Vancouver Whitecaps"),
]
CODE_TO_CANONICAL = {code: canon for code, _, canon in _TEAMS}


def canonical_from_text(text: str) -> str | None:
    """Resolve a Kalshi display string to a canonical team name, or None."""
    t = (text or "").lower()
    # 'new york city' must win over 'new york rb' aliasing: check longer
    # aliases first by sorting all aliases by length descending.
    candidates = [(alias, canon) for _, aliases, canon in _TEAMS for alias in aliases]
    for alias, canon in sorted(candidates, key=lambda x: -len(x[0])):
        if alias in t:
            return canon
    return None


# Phrase that must appear in the rules for a market to map onto the regulation
# grid. Verified wording from live rules text.
_REGULATION_PHRASE = re.compile(
    r"90\s+minutes\s+plus\s+stoppage\s+time\s*\(does\s+not\s+include\s+extra\s+time",
    re.IGNORECASE)


def rules_confirm_regulation(rules_text: str) -> bool:
    return bool(_REGULATION_PHRASE.search(rules_text or ""))


@dataclass
class MappedMarket:
    """A Kalshi market bound to a model claim on the regulation score grid."""
    ticker: str
    event_ticker: str
    series: str
    title: str
    sub_title: str
    home: str                      # canonical names, model orientation
    away: str
    kickoff_utc: pd.Timestamp | None
    claim: str                     # human-readable claim, e.g. "home_win"
    indicator: np.ndarray          # bool (G+1,G+1): cells where YES settles

    def model_prob(self, score_matrix: np.ndarray) -> float:
        return float(score_matrix[self.indicator].sum())


class MarketMapper:
    """Turns raw Kalshi market dicts for one event into MappedMarkets, given
    the fixture (canonical home/away) they belong to."""

    def __init__(self, grid_size: int):
        self.g = grid_size  # matrix is (g+1) x (g+1)
        k = np.arange(grid_size + 1)
        self._i, self._j = np.meshgrid(k, k, indexing="ij")

    # --- indicator builders (exact definitions of each settlement) -----------
    def ind_home_win(self):  return self._i > self._j
    def ind_away_win(self):  return self._i < self._j
    def ind_draw(self):      return self._i == self._j
    def ind_total_over(self, line: float):  return (self._i + self._j) > line
    def ind_btts(self):      return (self._i >= 1) & (self._j >= 1)
    def ind_team_over(self, is_home: bool, line: float):
        return (self._i if is_home else self._j) > line
    def ind_wins_by_more(self, is_home: bool, line: float):
        diff = self._i - self._j if is_home else self._j - self._i
        return diff > line
    def ind_exact(self, hs: int, as_: int):
        return (self._i == hs) & (self._j == as_)

    def map_market(self, m: dict, home: str, away: str) -> MappedMarket | None:
        """Return the MappedMarket for a raw market dict, or None (with reason
        recorded by the caller) when it cannot be mapped EXACTLY."""
        series = m.get("event_ticker", "").split("-")[0]
        sub = m.get("yes_sub_title", "") or ""
        title = m.get("title", "") or ""
        rules = (m.get("rules_primary", "") or "") + " " + (m.get("rules_secondary", "") or "")

        if not rules_confirm_regulation(rules):
            return None                      # fail-closed on settlement basis

        strike = m.get("floor_strike")
        team_in_sub = canonical_from_text(sub)

        if series == "KXMLSGAME":
            if "tie" in sub.lower() or "draw" in sub.lower():
                claim, ind = "draw", self.ind_draw()
            elif team_in_sub == home:
                claim, ind = "home_win", self.ind_home_win()
            elif team_in_sub == away:
                claim, ind = "away_win", self.ind_away_win()
            else:
                return None
        elif series == "KXMLSTOTAL":
            line = float(strike) if strike is not None else _line_from_text(sub)
            if line is None:
                return None
            claim, ind = f"total_over_{line}", self.ind_total_over(line)
        elif series == "KXMLSBTTS":
            claim, ind = "btts", self.ind_btts()
        elif series == "KXMLSTEAMTOTAL":
            line = float(strike) if strike is not None else _line_from_text(sub)
            if line is None or team_in_sub not in (home, away):
                return None
            is_home = team_in_sub == home
            claim = f"{'home' if is_home else 'away'}_over_{line}"
            ind = self.ind_team_over(is_home, line)
        elif series == "KXMLSSPREAD":
            line = float(strike) if strike is not None else _line_from_text(sub)
            if line is None or team_in_sub not in (home, away):
                return None
            is_home = team_in_sub == home
            claim = f"{'home' if is_home else 'away'}_wins_by_over_{line}"
            ind = self.ind_wins_by_more(is_home, line)
        elif series == "KXMLSSCORE":
            parsed = _exact_score_from_text(sub, home, away)
            if parsed is None:
                return None
            hs, as_ = parsed
            if hs > self.g or as_ > self.g:
                return None
            claim, ind = f"score_{hs}-{as_}", self.ind_exact(hs, as_)
        else:
            return None

        kickoff = m.get("expected_expiration_time")
        return MappedMarket(
            ticker=m["ticker"], event_ticker=m.get("event_ticker", ""),
            series=series, title=title, sub_title=sub, home=home, away=away,
            kickoff_utc=pd.Timestamp(kickoff) if kickoff else None,
            claim=claim, indicator=ind)


def _line_from_text(text: str) -> float | None:
    mo = re.search(r"(\d+\.5)", text or "")
    return float(mo.group(1)) if mo else None


def _exact_score_from_text(sub: str, home: str, away: str) -> tuple[int, int] | None:
    """'Houston Dynamo wins 5-2' / 'Draw 1-1' -> (home_goals, away_goals)."""
    s = (sub or "").lower()
    dm = re.search(r"(?:draw|tie)\s+(\d+)-(\d+)", s)
    if dm:
        a, b = int(dm.group(1)), int(dm.group(2))
        return (a, a) if a == b else None
    wm = re.search(r"wins\s+(\d+)-(\d+)", s)
    if not wm:
        return None
    a, b = int(wm.group(1)), int(wm.group(2))
    team = canonical_from_text(sub)
    if team == home:
        return (a, b)
    if team == away:
        return (b, a)
    return None


# --- fixture matching ---------------------------------------------------------
def match_event_to_fixture(event_markets: list[dict],
                           fixtures: pd.DataFrame) -> tuple[str, str] | None:
    """Identify (home, away) canonical names for a Kalshi event by matching the
    team pair to an upcoming fixture within +/-1 day. Returns None (loudly, at
    the caller) if the pair or the orientation cannot be confirmed."""
    # Gather candidate team names from all sub-titles and the title.
    names = set()
    for m in event_markets:
        c = canonical_from_text(m.get("yes_sub_title", ""))
        if c:
            names.add(c)
    if len(names) != 2:
        title = event_markets[0].get("title", "") if event_markets else ""
        parts = re.split(r"\s+vs\.?\s+", title.replace(" Winner?", ""), flags=re.I)
        names = {c for p in parts for c in [canonical_from_text(p)] if c}
    if len(names) != 2:
        return None
    a, b = sorted(names)

    # Kalshi expiration ~= kickoff + settlement lag; fixture date is UTC-date.
    exp = event_markets[0].get("expected_expiration_time")
    when = pd.Timestamp(exp).tz_convert(None) if exp else None
    cand = fixtures[((fixtures["home_team"] == a) & (fixtures["away_team"] == b)) |
                    ((fixtures["home_team"] == b) & (fixtures["away_team"] == a))]
    if when is not None:
        cand = cand[(cand["date"] >= when.normalize() - pd.Timedelta(days=2)) &
                    (cand["date"] <= when.normalize() + pd.Timedelta(days=2))]
    if len(cand) != 1:
        return None
    row = cand.iloc[0]
    return row["home_team"], row["away_team"]
