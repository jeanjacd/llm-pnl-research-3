"""
sources.py
==========
Provenance for every external data source. Keeping this in one auditable place is
deliberate: the whole model is reproducible from public, verifiable inputs, and a
reader can check exactly where each number comes from.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    licence: str
    description: str


# ============================================================================
# ACTIVE ERA: MLS
# ============================================================================
# Primary MLS source, selected 2026-07-20 after a live evaluation of candidates:
#
#   * ESPN public scoreboard API (chosen) -- see criteria below.
#   * football-data.org -- REJECTED: MLS is behind the paid tiers (free tier is
#     12 European/world competitions only; verified on /coverage 2026-07-20).
#   * jfjelstul GitHub datasets -- REJECTED: no MLS dataset exists (only a World
#     Cup database; repo probe 404'd 2026-07-20).
#   * FBref / sports-reference -- REJECTED as a pipeline dependency: HTML
#     scraping subject to anti-bot terms and layout drift; fine for manual
#     cross-checks only.
#   * Kaggle MLS dumps -- REJECTED: stale (no live updates, no fixtures).
#
# Evaluation of ESPN against the selection criteria (all verified live 2026-07-20):
#   accuracy      : 2024 regular season returns exactly 493 events = 29 teams x 34
#                   games / 2; scores spot-checked against public records.
#   latency       : the scoreboard is the live match feed -- results same-day;
#                   upcoming fixtures listed with STATUS_SCHEDULED (47 events
#                   returned for the coming 3 weeks).
#   depth         : full seasons verified back to at least 2010 (16+ seasons).
#   licensing     : unofficial-but-public, unauthenticated JSON endpoint, widely
#                   used by open-source packages (sportsdataverse et al.). No
#                   formal licence; usage here is personal research/betting, low
#                   request volume, cached locally. This is the weakest criterion
#                   of an otherwise dominant option -- documented, not hidden.
#   reliability   : stable for years; no API key; full season in one request.
#   schema        : kickoff timestamp, home/away, regulation score (shootouts
#                   reported separately -- STATUS_FINAL_PEN keeps the 90' score,
#                   exactly what a Dixon-Coles fit needs), venue city/country,
#                   neutral-site flag, season slug (regular / playoff round /
#                   all-star), stable team ids across rebrands.
#   double duty   : history AND forward fixtures from the same endpoint with the
#                   same team ids -- mirrors how results.csv served the WC era.
MLS_ESPN = Source(
    name="ESPN public scoreboard API (MLS)",
    url="https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard",
    licence="Unofficial public endpoint; no formal licence (personal research use, low volume, cached)",
    description=(
        "MLS results and fixtures (league slug usa.1: regular season + MLS Cup "
        "playoffs, distinguished by season slug), plus optional cup slugs "
        "usa.open (US Open Cup) and concacaf.leagues.cup (Leagues Cup). Teams "
        "keyed by stable ESPN team id and canonicalised to the latest display "
        "name, so franchise rebrands merge to one identity."
    ),
)

# ============================================================================
# ARCHIVED ERA: internationals / 2026 World Cup (see archive/wc2026/ARCHIVE.md)
# ============================================================================
# WC-era primary source. One CSV supplied BOTH the ~150 years of historical
# results used to fit team strength AND the live 2026 World Cup fixtures/results.
# The frozen copy lives in archive/wc2026/data/raw/results.csv; nothing in the
# MLS pipeline reads it.
RESULTS = Source(
    name="martj42/international_results",
    url="https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    licence="Public domain (CC0) -- see repository",
    description=(
        "Men's full international match results 1872-present plus scheduled 2026 "
        "World Cup fixtures. Columns: date, home_team, away_team, home_score, "
        "away_score, tournament, city, country, neutral."
    ),
)

# Cross-check only (not ingested at runtime): published national-team Elo. Our own
# Elo is computed transparently from RESULTS so the pipeline has no scraping
# dependency; eloratings.net is listed for manual sanity-checking.
ELO_CROSSCHECK = Source(
    name="World Football Elo Ratings",
    url="https://www.eloratings.net/",
    licence="Permissive reuse (see site)",
    description="Reference national-team Elo ratings for cross-checking our computed Elo.",
)

ALL_SOURCES = (MLS_ESPN, RESULTS, ELO_CROSSCHECK)
