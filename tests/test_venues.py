"""Venue layer: normalisation, claim mapping, abstention, book walking,
snapshot hashing and cross-venue equivalence. All offline.

Fixtures mirror wordings enumerated from the live venues on 2026-08-24 (see the
provider module docstrings for the survey).
"""
import pandas as pd
import pytest

from wc2026.leagues import get_league
from wc2026.venues.base import (
    KIND_BINARY,
    KIND_BUNDLE,
    KIND_NATIVE_COMBO,
    Book,
    Leg,
    MarketInstrument,
    UnsupportedInstrument,
    changed_since,
    claim_supported,
    equivalent,
    snapshot_record,
)
from wc2026.venues.kalshi_provider import (
    UNSUPPORTED_SUFFIXES,
    family_of,
    is_supported_family,
    league_prefix,
)
from wc2026.venues.polymarket import (
    PolymarketProvider,
    name_similarity,
    normalise_team,
    parse_ts,
    resolve_fixture,
)

EPL = get_league("premier_league")
H, A = "Fulham FC", "Chelsea FC"


# --------------------------------------------------------------------------- #
# claim support
# --------------------------------------------------------------------------- #
def test_supported_claims_are_exactly_the_grid_derived_family():
    for claim in ("home_win", "away_win", "draw", "total_over_2.5", "btts",
                  "home_over_1.5", "away_wins_by_over_1.5", "score_2-1",
                  "not_btts"):
        assert claim_supported(claim), claim
    for claim in ("player_anytime_goal", "total_corners_over_9.5",
                  "first_half_over_1.5", "method_of_victory", "relegation",
                  "family_CORNERS"):
        assert not claim_supported(claim), claim


# --------------------------------------------------------------------------- #
# Polymarket claim mapping (the wordings that actually exist)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,expected", [
    ("Will Fulham FC win on 2026-08-24?", "home_win"),
    ("Will Chelsea FC win on 2026-08-24?", "away_win"),
    ("Will Fulham FC vs. Chelsea FC end in a draw?", "draw"),
    ("Exact Score: Fulham FC 2 - 1 Chelsea FC?", "score_2-1"),
    ("Exact Score: Chelsea FC 3 - 0 Fulham FC?", "score_0-3"),
    ("Spread: Fulham FC (-1.5)", "home_wins_by_over_1.5"),
    ("Spread: Chelsea FC (-2.5)", "away_wins_by_over_2.5"),
    ("Fulham FC vs. Chelsea FC: O/U 2.5", "total_over_2.5"),
    ("Fulham FC vs. Chelsea FC: Both Teams to Score", "btts"),
    ("Fulham FC vs. Chelsea FC: Fulham FC O/U 1.5", "home_over_1.5"),
    ("Fulham FC vs. Chelsea FC: Chelsea FC O/U 2.5", "away_over_2.5"),
])
def test_polymarket_supported_questions_map(question, expected):
    assert PolymarketProvider.claim_for(question, H, A) == expected


@pytest.mark.parametrize("question", [
    "Fulham FC vs. Chelsea FC: 1st Half O/U 1.5",
    "Fulham FC vs. Chelsea FC: 2nd Half O/U 2.5",
    "Fulham FC vs. Chelsea FC: Fulham FC 1st Half O/U 0.5",
    "Fulham FC vs. Chelsea FC: Both Teams to Score in First Half",
    "Fulham FC vs. Chelsea FC: O/U 3.5 Total Corners",
    "Fulham FC vs. Chelsea FC: 1st Half O/U 3.5 Total Corners",
    "Exact Score: Any Other Score?",
    "Fulham FC to score first vs. Chelsea FC?",
    "Will Erling Haaland score 2+ goals?",
])
def test_polymarket_unmodellable_questions_are_refused(question):
    """Out-of-model wordings must return None, never a nearest-guess claim."""
    assert PolymarketProvider.claim_for(question, H, A) is None


def test_totals_are_recognised_without_the_word_goals():
    """The venue spells full-match totals bare ('O/U 2.5'); searching for the
    word 'goals' finds nothing and wrongly implies the venue has no totals."""
    assert PolymarketProvider.claim_for("Fulham FC vs. Chelsea FC: O/U 0.5",
                                        H, A) == "total_over_0.5"


def test_exact_score_orientation_follows_our_fixture_not_the_venue_order():
    """A question written away-team-first must be flipped to our orientation."""
    assert PolymarketProvider.claim_for("Exact Score: Chelsea FC 3 - 1 Fulham FC?",
                                        H, A) == "score_1-3"


# --------------------------------------------------------------------------- #
# identity resolution
# --------------------------------------------------------------------------- #
def test_name_similarity_handles_affixes_and_accents():
    assert name_similarity("FC Bayern Munchen", "Bayern Munich") >= 0.6
    assert name_similarity("RC Celta de Vigo", "Celta Vigo") >= 0.9
    assert name_similarity("Paris Saint-Germain FC", "Paris Saint-Germain") >= 0.9
    assert normalise_team("FC Bayern Munchen") == {"bayern", "munchen"}


def test_fixture_resolution_refuses_ambiguous_pairs():
    """Manchester City and Manchester United score alike on name; a fixture
    must not be resolved when two candidates are within noise of each other."""
    fixtures = pd.DataFrame({
        "date": [pd.Timestamp("2026-08-24"), pd.Timestamp("2026-08-24")],
        "home_team": ["Manchester City", "Manchester United"],
        "away_team": ["Arsenal", "Arsenal"]})
    assert resolve_fixture("Manchester", "Arsenal",
                           parse_ts("2026-08-24T19:00:00Z"), fixtures) is None


def test_fixture_resolution_matches_a_clear_pair():
    fixtures = pd.DataFrame({
        "date": [pd.Timestamp("2026-08-24")],
        "home_team": ["Fulham"], "away_team": ["Chelsea"]})
    hit = resolve_fixture("Fulham FC", "Chelsea FC",
                          parse_ts("2026-08-24T19:00:00Z"), fixtures)
    assert hit and hit["home"] == "Fulham" and not hit["flipped"]


def test_fixture_resolution_detects_a_flipped_orientation():
    fixtures = pd.DataFrame({
        "date": [pd.Timestamp("2026-08-24")],
        "home_team": ["Chelsea"], "away_team": ["Fulham"]})
    hit = resolve_fixture("Fulham FC", "Chelsea FC",
                          parse_ts("2026-08-24T19:00:00Z"), fixtures)
    assert hit and hit["flipped"] is True


def test_kickoff_parsing_accepts_every_observed_spelling():
    """The venue mixes ISO-Z and '+00' offsets across fields."""
    assert parse_ts("2026-08-24T19:00:00Z") is not None
    assert parse_ts("2026-08-24 19:00:00+00") is not None
    assert parse_ts("") is None and parse_ts(None) is None


# --------------------------------------------------------------------------- #
# Kalshi family handling
# --------------------------------------------------------------------------- #
def test_only_the_six_grid_families_are_supported():
    for fam in ("GAME", "TOTAL", "SPREAD", "BTTS", "TEAMTOTAL", "SCORE"):
        assert is_supported_family(fam)
    for fam in ("1H", "CORNERS", "GOAL", "MOV", "RELEGATION", "TOP4"):
        assert not is_supported_family(fam)
        assert fam in UNSUPPORTED_SUFFIXES


def test_second_division_and_all_star_series_are_never_constructed():
    """KXLALIGA2*/KXBUNDESLIGA2* are the SECOND division and KXMLSAST* is the
    all-star game; a prefix scan would pull both into a top-flight slate."""
    for lid in ("la_liga", "bundesliga", "mls"):
        series = set(get_league(lid).venue_series["kalshi"])
        assert not any(t.startswith("KXLALIGA2") for t in series)
        assert not any(t.startswith("KXBUNDESLIGA2") for t in series)
        assert not any(t.startswith("KXMLSAST") for t in series)


def test_league_prefix_and_family_extraction():
    assert league_prefix(EPL) == "KXEPL"
    assert family_of("KXEPLGAME-26AUG24FULCFC", "KXEPL") == "GAME"
    assert family_of("KXEPL1HSPREAD-26AUG24FULCFC", "KXEPL") == "1HSPREAD"


class _FlakyClient:
    """Fails on one series, succeeds on the rest."""

    def __init__(self, fail_on):
        self.fail_on = fail_on
        self.seen = []

    def get_markets(self, series_ticker=None, status="open", **kw):
        self.seen.append(series_ticker)
        if series_ticker == self.fail_on:
            raise RuntimeError("transient network error")
        return []

    def get_orderbook(self, ticker):
        raise AssertionError("not reached")


def test_a_failed_series_fetch_raises_rather_than_reporting_zero():
    """A swallowed fetch error is indistinguishable from 'this league has no
    such market'. That ambiguity produced a wrong coverage report once, so a
    failure must be loud."""
    from wc2026.venues.kalshi_provider import DiscoveryError, KalshiProvider
    provider = KalshiProvider(client=_FlakyClient("KXEPLGAME"))
    with pytest.raises(DiscoveryError, match="partial coverage"):
        provider.discover(EPL, with_books=False)


def test_partial_discovery_requires_explicit_opt_in_and_records_errors():
    from wc2026.venues.kalshi_provider import KalshiProvider
    provider = KalshiProvider(client=_FlakyClient("KXEPLGAME"))
    provider.discover(EPL, with_books=False, strict=False)
    assert provider.last_errors and provider.last_errors[0][0] == "KXEPLGAME"


def test_unsupported_families_are_still_swept_so_coverage_is_honest():
    """Out-of-model families must be FETCHED and recorded, not skipped, or the
    'unsupported' count silently understates what the venue offers."""
    from wc2026.venues.kalshi_provider import KalshiProvider
    client = _FlakyClient(fail_on=None)
    KalshiProvider(client=client).discover(EPL, with_books=False)
    assert "KXEPLGAME" in client.seen          # supported
    assert "KXEPLCORNERS" in client.seen       # unsupported, still swept
    assert "KXEPL1H" in client.seen


# --------------------------------------------------------------------------- #
# book walking
# --------------------------------------------------------------------------- #
def _book():
    return Book(yes_asks=((44, 100.0), (45, 250.0), (47, 500.0)),
                yes_bids=((42, 80.0),),
                no_asks=((55, 60.0),), no_bids=((53, 40.0),))


def test_walk_prices_size_beyond_the_touch():
    """Size beyond the touch must cost what it actually costs."""
    avg, filled, worst = _book().walk("yes", 300)
    assert filled == 300
    assert avg == pytest.approx((100 * 44 + 200 * 45) / 300)
    assert worst == 45
    assert avg > 44                       # strictly worse than the touch


def test_walk_reports_partial_fill_when_depth_runs_out():
    avg, filled, worst = _book().walk("yes", 10_000)
    assert filled == 850                  # 100 + 250 + 500
    assert worst == 47


def test_max_size_at_or_below_a_limit():
    b = _book()
    assert b.max_size_at_or_below("yes", 44) == 100.0
    assert b.max_size_at_or_below("yes", 45) == 350.0
    assert b.max_size_at_or_below("yes", 43) == 0.0


def test_empty_book_yields_no_price_not_a_midpoint():
    empty = Book()
    assert empty.touch("yes") is None
    assert empty.walk("yes", 10) is None
    assert empty.depth_at_touch("yes") == 0.0


# --------------------------------------------------------------------------- #
# instruments, abstention and combos
# --------------------------------------------------------------------------- #
def _inst(claims, kind=KIND_BINARY, venue="kalshi", regulation=True,
          home="Fulham", away="Chelsea", kickoff="2026-08-24T19:00:00+00:00",
          book=None):
    legs = tuple(Leg.build(c, "ref-%s" % c, home=home, away=away,
                           kickoff_utc=kickoff) for c in claims)
    return MarketInstrument(venue=venue, instrument_id="|".join(claims),
                            kind=kind, title="t", legs=legs,
                            settles_on_regulation=regulation,
                            kickoff_utc=kickoff, book=book or Book())


def test_instrument_with_any_unsupported_leg_abstains():
    inst = _inst(["home_win", "player_anytime_goal"])
    assert not inst.supported
    assert inst.unsupported_reasons
    with pytest.raises(UnsupportedInstrument):
        inst.require_valuable()


def test_same_match_legs_are_flagged_as_dependent():
    """Two legs on one fixture must never be multiplied as independent."""
    combo = _inst(["home_win", "total_over_2.5"], kind=KIND_NATIVE_COMBO)
    assert combo.is_multi_leg and combo.shares_a_match


def test_legs_from_different_matches_are_not_flagged_as_one_fixture():
    legs = (Leg.build("home_win", "a", home="Fulham", away="Chelsea"),
            Leg.build("home_win", "b", home="Arsenal", away="Spurs"))
    combo = MarketInstrument(venue="kalshi", instrument_id="c",
                             kind=KIND_NATIVE_COMBO, title="t", legs=legs)
    assert combo.is_multi_leg and not combo.shares_a_match


def test_native_combo_and_bundle_are_distinct_kinds():
    """A venue parlay and a self-assembled bundle are not the same product."""
    combo = _inst(["home_win", "btts"], kind=KIND_NATIVE_COMBO)
    bundle = _inst(["home_win", "btts"], kind=KIND_BUNDLE)
    assert combo.kind != bundle.kind
    assert combo.decision_hash() != bundle.decision_hash()


# --------------------------------------------------------------------------- #
# snapshot hashing / re-evaluation
# --------------------------------------------------------------------------- #
def test_price_change_changes_the_decision_hash():
    a = _inst(["home_win"], book=Book(yes_asks=((44, 100.0),)))
    b = _inst(["home_win"], book=Book(yes_asks=((45, 100.0),)))
    assert a.decision_hash() != b.decision_hash()


def test_depth_change_changes_the_decision_hash():
    a = _inst(["home_win"], book=Book(yes_asks=((44, 100.0),)))
    b = _inst(["home_win"], book=Book(yes_asks=((44, 900.0),)))
    assert a.decision_hash() != b.decision_hash()


def test_identical_state_is_not_re_reviewed_but_a_moved_price_is():
    """Deduplicating permanently by ticker would suppress a market whose price
    moved; the hash must reopen exactly those."""
    first = [_inst(["home_win"], book=Book(yes_asks=((44, 100.0),)))]
    snap = snapshot_record(first, "premier_league", "kalshi")
    assert changed_since(snap, first) == []
    moved = [_inst(["home_win"], book=Book(yes_asks=((46, 100.0),)))]
    assert len(changed_since(snap, moved)) == 1
    assert len(changed_since(None, moved)) == 1        # no history -> all new


def test_snapshot_counts_supported_and_total_separately():
    insts = [_inst(["home_win"]), _inst(["player_anytime_goal"])]
    snap = snapshot_record(insts, "premier_league", "kalshi")
    assert snap["n_instruments"] == 2 and snap["n_supported"] == 1
    assert snap["snapshot_hash"]


# --------------------------------------------------------------------------- #
# cross-venue equivalence
# --------------------------------------------------------------------------- #
def test_equivalent_requires_a_known_and_matching_settlement_basis():
    kalshi = _inst(["home_win"], venue="kalshi", regulation=True)
    poly_unknown = _inst(["home_win"], venue="polymarket", regulation=None)
    assert not equivalent(kalshi, poly_unknown)       # unknown -> refuse
    poly_known = _inst(["home_win"], venue="polymarket", regulation=True)
    assert equivalent(kalshi, poly_known)
    poly_other = _inst(["home_win"], venue="polymarket", regulation=False)
    assert not equivalent(kalshi, poly_other)


def test_equivalent_requires_the_same_claim_and_fixture():
    a = _inst(["home_win"], venue="kalshi")
    assert not equivalent(a, _inst(["away_win"], venue="polymarket"))
    assert not equivalent(a, _inst(["home_win"], venue="polymarket",
                                   home="Arsenal", away="Spurs"))


def test_same_venue_instruments_are_never_cross_venue_equivalent():
    a = _inst(["home_win"], venue="kalshi")
    assert not equivalent(a, _inst(["home_win"], venue="kalshi"))


def test_far_apart_kickoffs_are_not_equivalent():
    a = _inst(["home_win"], venue="kalshi", kickoff="2026-08-24T19:00:00+00:00")
    b = _inst(["home_win"], venue="polymarket",
              kickoff="2026-08-26T19:00:00+00:00")
    assert not equivalent(a, b)
