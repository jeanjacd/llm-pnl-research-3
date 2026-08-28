"""League registry: isolation, verified provider facts, and fail-loud rules.

The spec's first hard requirement is that league paths and parameters cannot
bleed into each other. These tests enforce that structurally.
"""
import os

import pytest

from wc2026.leagues import (
    DOUBLE_ROUND_ROBIN,
    MLS_SCHEDULE,
    ROLE_SCORING,
    ROLE_TRAINING_ONLY,
    SUPPORTED_KALSHI_SUFFIXES,
    TIER_LEAGUE,
    LeagueConfigError,
    all_leagues,
    classify_season_slug,
    get_league,
    kalshi_series,
    league_ids,
)

EXPECTED = {"mls", "premier_league", "la_liga", "bundesliga", "ligue_1"}


def test_all_five_leagues_registered():
    assert set(league_ids()) == EXPECTED


# --- isolation ----------------------------------------------------------------
def test_no_two_leagues_share_any_path():
    """Paths must be disjoint: one league's fit can never overwrite another."""
    seen = {}
    for spec in all_leagues():
        for name, path in spec.all_paths().items():
            key = os.path.normcase(os.path.abspath(path))
            assert key not in seen, (
                "path collision between %s and %s at %s"
                % (spec.league_id, seen.get(key), path))
            seen[key] = spec.league_id


def test_every_path_is_under_its_own_league_directory():
    for spec in all_leagues():
        root = os.path.normcase(os.path.abspath(spec.data_dir))
        for name, path in spec.all_paths().items():
            assert os.path.normcase(os.path.abspath(path)).startswith(root), (
                "%s path %s escapes %s" % (spec.league_id, name, root))
        # the directory name is the league id -- no cross-naming
        assert os.path.basename(spec.data_dir.rstrip(os.sep)) == spec.league_id


def test_model_configs_are_independent_objects():
    """Mutating one league's weights must not touch another's."""
    mls = get_league("mls")
    epl = get_league("premier_league")
    assert mls.model.importance_weights is not epl.model.importance_weights
    # MLS is tuned; Europe is not -- they must not share parameter values by
    # accident, which would imply an untuned league inherited a tuned fit.
    assert mls.model.half_life_days != epl.model.half_life_days


def test_registry_specs_are_frozen():
    spec = get_league("mls")
    with pytest.raises(Exception):
        spec.league_id = "hacked"


# --- verified provider facts (checked live 2026-08-01) ------------------------
def test_provider_slugs_match_verified_values():
    assert get_league("mls").primary_feed.provider_slug == "usa.1"
    assert get_league("premier_league").primary_feed.provider_slug == "eng.1"
    assert get_league("la_liga").primary_feed.provider_slug == "esp.1"
    assert get_league("bundesliga").primary_feed.provider_slug == "ger.1"
    assert get_league("ligue_1").primary_feed.provider_slug == "fra.1"


def test_schedule_shapes_reproduce_verified_match_counts():
    # live: EPL/La Liga 380 with 20 teams; Bundesliga/Ligue 1 306 with 18
    assert DOUBLE_ROUND_ROBIN.expected_matches(20) == 380
    assert DOUBLE_ROUND_ROBIN.expected_matches(18) == 306
    # Ligue 1 was 20 teams (380) before 2023-24 -- the rule must follow size,
    # not a hardcoded constant.
    assert DOUBLE_ROUND_ROBIN.expected_matches(20) == 380
    # MLS plays a fixed 34 games per team regardless of league size
    assert MLS_SCHEDULE.expected_matches(30) == 510
    assert MLS_SCHEDULE.expected_matches(29) == 493


def test_season_windows_respect_calendar_vs_autumn_spring():
    assert get_league("mls").season_window(2024) == ("20240101", "20241231")
    assert get_league("premier_league").season_window(2024) == ("20240701",
                                                                "20250630")
    assert get_league("premier_league").season_label(2024) == "2024-25"
    assert get_league("mls").season_label(2024) == "2024"


# --- competition roles --------------------------------------------------------
def test_training_only_competitions_are_never_scored():
    mls = get_league("mls")
    assert mls.training_only_tiers                       # cups exist
    assert not (mls.scoring_tiers & mls.training_only_tiers)
    for feed in mls.feeds:
        if feed.role == ROLE_TRAINING_ONLY:
            assert feed.tier not in mls.scoring_tiers


def test_european_leagues_score_only_the_league_tier():
    for lid in ("premier_league", "la_liga", "bundesliga", "ligue_1"):
        spec = get_league(lid)
        assert spec.scoring_tiers == frozenset({TIER_LEAGUE})
        assert spec.primary_feed.role == ROLE_SCORING


def test_every_tier_has_an_importance_weight():
    for spec in all_leagues():
        for tier in spec.scoring_tiers | spec.training_only_tiers:
            assert tier in spec.model.importance_weights, (
                "%s: tier %s has no weight" % (spec.league_id, tier))


# --- slug classification fails loud ------------------------------------------
def test_european_season_slug_classification():
    epl = get_league("premier_league")
    assert classify_season_slug(epl, "2024-25-english-premier-league") == (
        "English Premier League", TIER_LEAGUE)


def test_all_observed_historical_slug_formats_classify():
    """Formats surveyed live across 2014-2026; a new one must fail loud, but
    every observed one must resolve."""
    cases = [
        ("premier_league", "2019-20-english-premier-league"),
        ("premier_league", "20162017-english-premier-league"),
        ("premier_league", "2014-2015-barclays-premier-league"),
        ("bundesliga", "201819-german-bundesliga"),
        ("la_liga", "20142015-spanish-primera-division"),
        ("ligue_1", "regular-season"),          # Ligue 1 uses the bare form
        ("ligue_1", "2020-21-ligue-1"),
    ]
    for lid, slug in cases:
        assert classify_season_slug(get_league(lid), slug) == (
            get_league(lid).display_name, TIER_LEAGUE), slug


def test_promotion_relegation_playoffs_are_excluded():
    """Ligue 1 barrage ties are played against second-division clubs. They must
    be excluded, not folded into a single-league fit."""
    l1 = get_league("ligue_1")
    for slug in ("promotionrelegation-playoffs",
                 "promotionrelegation-playoff-final",
                 "promotion-playoff-semifinals",
                 "promotionrelegation-playoff-quarterfinals"):
        assert classify_season_slug(l1, slug) is None, slug


def test_mls_slug_classification_covers_all_eras():
    mls = get_league("mls")
    assert classify_season_slug(mls, "regular-season")[1] == TIER_LEAGUE
    assert classify_season_slug(mls, "regular-season-2010")[1] == TIER_LEAGUE
    for slug in ("mls-cup", "conference-finals", "knockout---western-conf",
                 "eastern-conference-playoffs---round-one", "wild-card"):
        assert classify_season_slug(mls, slug)[1] == "playoff", slug
    assert classify_season_slug(mls, "all-star-game") is None


def test_unrecognised_slug_raises_rather_than_guessing():
    for lid, slug in (("premier_league", "mystery-competition"),
                      ("mls", "some-new-format")):
        with pytest.raises(LeagueConfigError):
            classify_season_slug(get_league(lid), slug)


def test_unknown_league_raises():
    with pytest.raises(LeagueConfigError):
        get_league("serie_a")


# --- venue coverage -----------------------------------------------------------
def test_all_five_leagues_have_verified_kalshi_series():
    """Verified live: every league returned open markets for these series."""
    for spec in all_leagues():
        assert spec.tradeable, spec.league_id
        series = spec.venue_series["kalshi"]
        assert len(series) == len(SUPPORTED_KALSHI_SUFFIXES)


def test_no_series_ticker_is_shared_between_leagues():
    seen = {}
    for spec in all_leagues():
        for ticker in spec.venue_series.get("kalshi", ()):
            assert ticker not in seen, (
                "%s and %s both claim %s" % (spec.league_id, seen.get(ticker),
                                             ticker))
            seen[ticker] = spec.league_id


def test_kalshi_series_helper_builds_supported_families_only():
    assert kalshi_series("KXEPL") == ("KXEPLGAME", "KXEPLTOTAL", "KXEPLSPREAD",
                                      "KXEPLBTTS", "KXEPLTEAMTOTAL",
                                      "KXEPLSCORE", "KXEPL1H", "KXEPL1HTOTAL",
                                      "KXEPL1HSCORE")


def test_unsupported_families_are_declared_not_silently_traded():
    from wc2026.leagues import UNSUPPORTED_MARKET_FAMILIES
    # families we know exist on-venue but cannot model
    for fam in ("GOAL", "CORNERS", "MOV", "RELEGATION"):
        assert fam in UNSUPPORTED_MARKET_FAMILIES
    # and none of them leaked into a tradeable series list
    for spec in all_leagues():
        for ticker in spec.venue_series.get("kalshi", ()):
            for suffix in SUPPORTED_KALSHI_SUFFIXES:
                if ticker.endswith(suffix):
                    break
            else:
                raise AssertionError("unsupported family in series: " + ticker)
