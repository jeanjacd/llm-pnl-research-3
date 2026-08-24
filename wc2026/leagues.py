"""
leagues.py
==========
The league registry: one LeagueSpec per independently-modelled league.

This replaces the single-active-league globals (RESULTS_CSV,
FITTED_PARAMS_JSON, ADJUSTMENTS_JSON, MLS_CALIBRATION_JSON). Every league owns
its own data directory, fitted parameters, calibration artifacts and evaluation
report, so nothing can bleed between leagues.

Provider facts here were VERIFIED live against the ESPN scoreboard API on
2026-08-01 (see docs/BASELINE.md), not assumed:

    eng.1  English Premier League  380 matches / 20 teams  (double round-robin)
    esp.1  Spanish LALIGA          380 matches / 20 teams
    ger.1  German Bundesliga       306 matches / 18 teams
    fra.1  French Ligue 1          306 matches / 18 teams  (20 teams pre-2023-24)
    usa.1  MLS                     34 games per team + playoffs

Anything NOT verified -- notably which Kalshi/Polymarket series cover the
European leagues -- is left empty rather than guessed. An empty venue_series
means "no verified trading coverage", and the betting layer must treat that
league as research-only until it is populated.
"""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field

from .config import DATA_DIR, ModelConfig

LEAGUES_DIR = os.path.join(DATA_DIR, "leagues")

# --- competition roles --------------------------------------------------------
ROLE_SCORING = "scoring"              # eligible for evaluation AND trading
ROLE_TRAINING_ONLY = "training_only"  # may inform the fit, never scored/traded

# --- generic competition tiers (importance-weight keys) -----------------------
TIER_LEAGUE = "league"
TIER_PLAYOFF = "playoff"
TIER_DOMESTIC_CUP = "domestic_cup"
TIER_CONTINENTAL_CUP = "continental_cup"


class LeagueConfigError(ValueError):
    """Registry misconfiguration. Raised loudly rather than guessed around."""


@dataclass(frozen=True)
class ScheduleShape:
    """How many regular-season matches a season of n_teams should contain."""

    kind: str                        # double_round_robin | fixed_games_per_team
    games_per_team: int | None = None

    def expected_matches(self, n_teams: int) -> int | None:
        if n_teams <= 1:
            return None
        if self.kind == "double_round_robin":
            return n_teams * (n_teams - 1)        # EPL 20 -> 380; BL 18 -> 306
        if self.kind == "fixed_games_per_team":
            if not self.games_per_team:
                return None
            return n_teams * self.games_per_team // 2
        raise LeagueConfigError("unknown schedule kind: " + repr(self.kind))


DOUBLE_ROUND_ROBIN = ScheduleShape("double_round_robin")
MLS_SCHEDULE = ScheduleShape("fixed_games_per_team", games_per_team=34)

SEASON_CALENDAR_YEAR = "calendar_year"    # MLS: Feb-Dec, season == year
SEASON_AUTUMN_SPRING = "autumn_spring"    # Europe: Aug-May, labelled by start


@dataclass(frozen=True)
class CompetitionFeed:
    """One provider feed contributing matches to a league dataset."""

    key: str                 # stable internal key, e.g. league / us_open_cup
    provider_slug: str       # provider identifier, e.g. usa.1
    label: str               # value written to the tournament column
    tier: str                # importance-weight key
    role: str = ROLE_SCORING
    is_primary: bool = False
    playoff_label: str | None = None
    playoff_tier: str | None = None


@dataclass(frozen=True)
class LeagueSpec:
    league_id: str
    display_name: str
    country: str
    season_format: str
    schedule: ScheduleShape
    provider: str
    feeds: tuple
    first_season: int
    model: ModelConfig
    eval_start: str
    eval_end: str
    max_goals_grid: int = 12
    venue_series: dict = field(default_factory=dict)

    @property
    def primary_feed(self) -> CompetitionFeed:
        for f in self.feeds:
            if f.is_primary:
                return f
        raise LeagueConfigError(self.league_id + ": no primary feed")

    @property
    def scoring_tiers(self) -> frozenset:
        tiers = {f.tier for f in self.feeds if f.role == ROLE_SCORING}
        pf = self.primary_feed
        if pf.playoff_tier and pf.role == ROLE_SCORING:
            tiers.add(pf.playoff_tier)
        return frozenset(tiers)

    @property
    def training_only_tiers(self) -> frozenset:
        return frozenset(f.tier for f in self.feeds
                         if f.role == ROLE_TRAINING_ONLY)

    @property
    def tradeable(self) -> bool:
        """True only where venue series coverage has been VERIFIED."""
        return any(bool(v) for v in self.venue_series.values())

    def season_window(self, season: int) -> tuple:
        """Provider date window (start, end) as YYYYMMDD strings."""
        if self.season_format == SEASON_CALENDAR_YEAR:
            return "%d0101" % season, "%d1231" % season
        if self.season_format == SEASON_AUTUMN_SPRING:
            return "%d0701" % season, "%d0630" % (season + 1)
        raise LeagueConfigError("unknown season format: "
                                + repr(self.season_format))

    def season_windows(self, season: int) -> list:
        """Provider date windows covering a season, as (start, end) strings.

        European seasons can overrun their nominal end: the COVID-disrupted
        2019-20 campaigns resumed in June and finished in JULY 2020 (verified:
        EPL 314 matches to 30 Jun + 66 in July = 380). A single Jul-Jun window
        therefore truncates a real season, so a supplementary summer window is
        fetched and events are bucketed by the provider's own season year.
        Each request stays within a 12-month span, which the provider requires.
        """
        if self.season_format == SEASON_CALENDAR_YEAR:
            return [self.season_window(season)]
        start, end = self.season_window(season)
        overrun = ("%d0701" % (season + 1), "%d0831" % (season + 1))
        return [(start, end), overrun]

    def season_label(self, season: int) -> str:
        if self.season_format == SEASON_AUTUMN_SPRING:
            return "%d-%s" % (season, str(season + 1)[2:])
        return str(season)

    # ---- per-league paths (no globals) ----
    @property
    def data_dir(self) -> str:
        return os.path.join(LEAGUES_DIR, self.league_id)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.data_dir, "raw")

    @property
    def matches_csv(self) -> str:
        return os.path.join(self.raw_dir, "matches.csv")

    @property
    def manifest_json(self) -> str:
        return os.path.join(self.raw_dir, "manifest.json")

    @property
    def fitted_params_json(self) -> str:
        return os.path.join(self.data_dir, "fitted_params.json")

    @property
    def calibration_dir(self) -> str:
        return os.path.join(self.data_dir, "calibration")

    @property
    def adjustments_json(self) -> str:
        return os.path.join(self.data_dir, "adjustments.json")

    @property
    def eval_report_json(self) -> str:
        return os.path.join(self.data_dir, "evaluation.json")

    def all_paths(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "raw_dir": self.raw_dir,
            "matches_csv": self.matches_csv,
            "manifest": self.manifest_json,
            "fitted_params": self.fitted_params_json,
            "calibration_dir": self.calibration_dir,
            "adjustments": self.adjustments_json,
            "eval_report": self.eval_report_json,
        }


# Generic tier weights: STARTING POINTS, not tuned values. Each league earns
# its own via walk-forward CV written into its own fitted_params.json.
_BASE_WEIGHTS = {
    TIER_LEAGUE: 1.00,
    TIER_PLAYOFF: 0.90,
    TIER_DOMESTIC_CUP: 0.60,
    TIER_CONTINENTAL_CUP: 0.60,
}

# Untuned European defaults. Deliberately NOT copied from the MLS fit: MLS
# parameters were tuned on MLS data and carry no validity elsewhere.
_EURO_MODEL = dataclasses.replace(
    ModelConfig(),
    half_life_days=180.0,
    rho=-0.05,
    blend_k=12.0,
    elo_k=30.0,
    importance_weights=dict(_BASE_WEIGHTS),
    tournament_calib_k=None,
)

# MLS parameters ARE tuned (walk-forward CV, 2026-07).
_MLS_MODEL = dataclasses.replace(
    ModelConfig(),
    half_life_days=60.0,
    rho=-0.10,
    blend_k=20.0,
    elo_k=40.0,
    importance_weights={**_BASE_WEIGHTS,
                        TIER_DOMESTIC_CUP: 1.0,
                        TIER_CONTINENTAL_CUP: 1.0},
    tournament_calib_k=None,
)


# Kalshi market families that map onto the regulation scoreline grid. Each
# league uses the same six suffixes behind a league-specific prefix. VERIFIED
# live on 2026-08-01: every prefix below returned open markets.
SUPPORTED_KALSHI_SUFFIXES = ("GAME", "TOTAL", "SPREAD", "BTTS",
                             "TEAMTOTAL", "SCORE")


def kalshi_series(prefix: str) -> tuple:
    """The supported series tickers for a Kalshi league prefix."""
    return tuple(prefix + suffix for suffix in SUPPORTED_KALSHI_SUFFIXES)


# Polymarket league tag IDs, read from the venue's own /sports registry and
# confirmed by pulling open per-match events for each (2026-08-24). Discovery
# MUST use these: a generic "soccer" tag returns a different, mostly obscure
# population and badly misrepresents coverage.
POLYMARKET_TAGS = {
    "premier_league": (82, 306),
    "la_liga": (780,),
    "bundesliga": (1494,),
    "ligue_1": (102070,),
    "mls": (100100,),
}


def _euro(league_id, name, country, slug, kalshi_prefix, first_season=2014):
    """A standard European top flight: one primary double-round-robin feed."""
    return LeagueSpec(
        league_id=league_id,
        display_name=name,
        country=country,
        season_format=SEASON_AUTUMN_SPRING,
        schedule=DOUBLE_ROUND_ROBIN,
        provider="espn",
        feeds=(CompetitionFeed(key="league", provider_slug=slug, label=name,
                               tier=TIER_LEAGUE, role=ROLE_SCORING,
                               is_primary=True),),
        first_season=first_season,
        model=_EURO_MODEL,
        eval_start="%d-08-01" % (first_season + 4),
        eval_end="2026-12-31",
        venue_series={"kalshi": kalshi_series(kalshi_prefix),
                      "polymarket": POLYMARKET_TAGS.get(league_id, ())},
    )


LEAGUES = {
    "mls": LeagueSpec(
        league_id="mls",
        display_name="Major League Soccer",
        country="USA",
        season_format=SEASON_CALENDAR_YEAR,
        schedule=MLS_SCHEDULE,
        provider="espn",
        feeds=(
            CompetitionFeed("league", "usa.1", "Major League Soccer",
                            TIER_LEAGUE, ROLE_SCORING, is_primary=True,
                            playoff_label="MLS Cup Playoffs",
                            playoff_tier=TIER_PLAYOFF),
            CompetitionFeed("us_open_cup", "usa.open", "US Open Cup",
                            TIER_DOMESTIC_CUP, ROLE_TRAINING_ONLY),
            CompetitionFeed("leagues_cup", "concacaf.leagues.cup",
                            "Leagues Cup", TIER_CONTINENTAL_CUP,
                            ROLE_TRAINING_ONLY),
        ),
        first_season=2010,
        model=_MLS_MODEL,
        eval_start="2023-01-01",
        eval_end="2026-12-31",
        venue_series={"kalshi": kalshi_series("KXMLS"),
                      "polymarket": POLYMARKET_TAGS["mls"]},
    ),
    "premier_league": _euro("premier_league", "English Premier League",
                            "England", "eng.1", "KXEPL"),
    "la_liga": _euro("la_liga", "Spanish LALIGA", "Spain", "esp.1",
                     "KXLALIGA"),
    "bundesliga": _euro("bundesliga", "German Bundesliga", "Germany", "ger.1",
                        "KXBUNDESLIGA"),
    "ligue_1": _euro("ligue_1", "French Ligue 1", "France", "fra.1",
                     "KXLIGUE1"),
}

# Market families that EXIST on-venue but have NO validated quantitative model
# in this system. Verified present on Kalshi 2026-08-01. They must be recorded
# and routed to UNSUPPORTED -- never valued from qualitative confidence.
UNSUPPORTED_MARKET_FAMILIES = (
    "1H", "1HBTTS", "1HSPREAD", "1HTOTAL", "1HSCORE",   # first half
    "2H", "2HBTTS", "2HSPREAD", "2HTOTAL",              # second half
    "GOAL", "ANYGOAL", "FIRSTGOAL", "SOA", "AST",       # player props
    "CORNERS", "TCORNERS",                              # corners
    "MOV", "FTTS", "ADVANCE",                           # method / first-to-score
    "LEADER", "LAST", "TOP", "TOP2", "TOP4", "TOP6",    # season futures
    "RELEGATION", "POINTMARGIN", "TEAMPOINTS",
    "SEASONSTAT", "POY", "H2H", "H2HFINISH",
)


def get_league(league_id: str) -> LeagueSpec:
    try:
        return LEAGUES[league_id]
    except KeyError:
        raise LeagueConfigError(
            "unknown league %r; known: %s" % (league_id, sorted(LEAGUES))
        ) from None


def league_ids() -> list:
    return sorted(LEAGUES)


def all_leagues() -> list:
    return [LEAGUES[k] for k in league_ids()]


# Season-slug prefixes actually observed across 2014-2026 on the European
# endpoints (surveyed live 2026-08-01), e.g.
#   2019-20-english-premier-league     20162017-english-premier-league
#   2014-2015-barclays-premier-league  201819-german-bundesliga
# Ligue 1 additionally uses a bare "regular-season" for several seasons.
_EURO_SEASON_SLUG = re.compile(r"^(\d{4}-\d{2}|\d{4}-\d{4}|\d{8}|\d{6})-")
_MLS_PLAYOFF_KEYWORDS = ("playoff", "knockout", "semi", "final", "mls-cup",
                         "wild-card", "round", "play-in", "mls-is-back")
# Excluded on purpose:
#   all-star            exhibition, not competitive
#   promotion/relegation  Ligue 1 barrage ties are played against a SECOND
#                       DIVISION club. Including them would inject a team from
#                       another competition into a single-league fit without
#                       modelling the league-strength gap, which the mission
#                       forbids. Recorded as excluded rather than ingested.
_EXCLUDED_KEYWORDS = ("all-star", "promotion", "relegation")


def classify_season_slug(spec: LeagueSpec, slug: str):
    """Map a primary-feed season slug to (tournament_label, tier).

    Returns None for deliberately excluded competitions (e.g. all-star games).
    Raises LeagueConfigError on anything unrecognised -- never guesses, because
    a mis-tiered competition silently corrupts both fitting and scoring.
    """
    text = (slug or "").lower()
    if any(k in text for k in _EXCLUDED_KEYWORDS):
        return None
    feed = spec.primary_feed
    if spec.season_format == SEASON_AUTUMN_SPRING:
        # Either a dated season prefix, or a bare "regular-season" (Ligue 1).
        if _EURO_SEASON_SLUG.match(text) or text.startswith("regular-season"):
            return feed.label, feed.tier
        raise LeagueConfigError(
            "%s: unrecognised season slug %r" % (spec.league_id, slug))
    if text.startswith("regular-season"):
        return feed.label, feed.tier
    if feed.playoff_tier and any(k in text for k in _MLS_PLAYOFF_KEYWORDS):
        return feed.playoff_label or feed.label, feed.playoff_tier
    raise LeagueConfigError(
        "%s: unrecognised season slug %r" % (spec.league_id, slug))
