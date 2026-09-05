"""Which team a market is about, when both clubs share a word.

Real Betis v Real Madrid, 2026-09-04. Kalshi lists three first-half markets --
`-RBB`, `-RMA`, `-TIE`. Both team markets were recorded as `1h_home_win`,
because the resolver tested the home side first and returned on the first
score over 0.6, and "Real Madrid" scores 0.667 against "Real Betis".

The consequence is not a wrong label. The `-RMA` market was then priced with
P(Betis win the half), so the board compared a Madrid market against a Betis
probability, found edge that did not exist, and bought 18 contracts of it. It
settled a winner only because the half was drawn, which makes both readings
true at once.

`resolve_fixture` has refused ambiguous pairs since it was written -- its own
docstring names Manchester City and Manchester United. This is the same rule
applied one level down, to the team WITHIN a resolved fixture.
"""
import pytest

from wc2026.venues.kalshi_provider import claim_for
from wc2026.venues.naming import name_similarity, team_side

BETIS, MADRID = "Real Betis", "Real Madrid"


def test_the_pair_that_broke_it_really_is_this_similar():
    """Not hypothetical: 0.667 clears a 0.6 bar with room to spare."""
    assert name_similarity(MADRID, BETIS) >= 0.6


@pytest.mark.parametrize("team,home,away,expected", [
    (MADRID, BETIS, MADRID, "away"),
    (BETIS, BETIS, MADRID, "home"),
    ("Manchester United", "Manchester City", "Manchester United", "away"),
    ("Manchester City", "Manchester City", "Manchester United", "home"),
    ("Real Sociedad", MADRID, "Real Sociedad", "away"),
    ("Atletico Madrid", MADRID, "Atlético Madrid", "away"),
])
def test_the_better_match_wins_regardless_of_which_side_is_tested_first(
        team, home, away, expected):
    assert team_side(team, home, away) == expected


def test_a_name_matching_neither_side_resolves_to_nothing():
    assert team_side("Sporting Lisbon", BETIS, MADRID) is None
    assert team_side("", BETIS, MADRID) is None


def test_two_indistinguishable_sides_abstain_rather_than_guess():
    """An unpriced market costs nothing. A market priced on the opposing
    team's probability costs the stake."""
    assert team_side("Real Madrid", "Real Madrid", "Real Madrid CF") is None


def test_a_clear_match_still_resolves_through_the_alias_table():
    assert team_side("PSG", "Paris Saint-Germain", "Lens") == "home"
    assert team_side("FC Köln", "Werder Bremen", "FC Cologne") == "away"
    assert team_side("D.C. United", "DC United", "Inter Miami") == "home"


# --- the whole point: one market per outcome ---------------------------------
def test_each_outcome_market_on_a_real_derby_gets_its_own_claim():
    subs = {"Real Madrid wins 1st Half": "1h_away_win",
            "Real Betis wins 1st Half": "1h_home_win",
            "Tie 1st Half": "1h_draw"}
    got = {s: claim_for("1H", s, BETIS, MADRID) for s in subs}
    assert got == subs
    assert len(set(got.values())) == 3, "three markets, three propositions"


def test_the_full_match_derby_separates_too():
    assert claim_for("GAME", "Real Madrid wins", BETIS, MADRID) == "away_win"
    assert claim_for("GAME", "Real Betis wins", BETIS, MADRID) == "home_win"


def test_a_scoreline_is_written_from_the_named_teams_point_of_view():
    """`-RMA` winning 3-1 is 1-3 in home-away order. Reading the team wrong
    reversed the scoreline as well as the side."""
    assert claim_for("SCORE", "Real Madrid wins 3-1", BETIS, MADRID) == \
        "score_1-3"
    assert claim_for("SCORE", "Real Betis wins 3-1", BETIS, MADRID) == \
        "score_3-1"
