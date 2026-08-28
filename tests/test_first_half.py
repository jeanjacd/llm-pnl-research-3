"""First-half markets, priced by the frozen engine fitted to half-time goals.

No new mathematics -- which is the point. The same grid, the same claim
vocabulary, the same settlement rules, applied to a different target. What
these tests guard is that the two halves of the system agree about WHICH score
a claim settles on, and that missing half-time data is never quietly imputed.
"""
import numpy as np
import pandas as pd
import pytest

from wc2026.model.first_half import (
    NoHalfTimeData,
    build_first_half_strength,
    coverage,
    half_time_frame,
)
from wc2026.paper.cycle import probability_for
from wc2026.paper.outcomes import (
    UnsettleableClaim,
    claim_is_true,
    is_first_half,
    score_for_claim,
)
from wc2026.venues.base import claim_supported


def matches(n=400, with_ht=None):
    """A synthetic history; `with_ht` rows carry a half-time score."""
    with_ht = n if with_ht is None else with_ht
    rows = []
    day = pd.Timestamp("2024-01-01")
    for i in range(n):
        home, away = ("A", "B") if i % 2 else ("B", "A")
        rows.append({
            "date": day + pd.Timedelta(days=i),
            "home_team": home, "away_team": away,
            "home_score": 2, "away_score": 1,
            "home_ht_score": 1.0 if i < with_ht else None,
            "away_ht_score": 0.0 if i < with_ht else None,
            "tournament": "L", "tier": "league", "neutral": False,
            "played": True})
    return pd.DataFrame(rows)


# --- the two halves of the system must agree ----------------------------------
FIRST_HALF_CLAIMS = ["1h_home_win", "1h_away_win", "1h_draw",
                     "1h_total_over_0.5", "1h_total_over_1.5",
                     "1h_total_under_1.5", "1h_score_0-0", "1h_score_1-0"]


class Grid:
    def __init__(self, matrix):
        self.matrix = matrix


@pytest.mark.parametrize("claim", FIRST_HALF_CLAIMS)
def test_a_first_half_claim_is_scored_exactly_as_it_is_forecast(claim):
    """The same cell-for-cell agreement the full-match claims are held to."""
    rng = np.random.default_rng(3)
    m = rng.random((13, 13)) + 0.01
    pred = Grid(m / m.sum())
    expected = sum(pred.matrix[i][j]
                   for i in range(13) for j in range(13)
                   if claim_is_true(claim, i, j))
    assert probability_for(claim, pred) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("claim", FIRST_HALF_CLAIMS)
def test_a_first_half_claim_asks_the_same_question_as_its_full_match_twin(claim):
    """`1h_` is a prefix on the SAME proposition, not a different rule."""
    twin = claim[len("1h_"):]
    for hg, ag in ((0, 0), (1, 0), (0, 2), (2, 2)):
        assert claim_is_true(claim, hg, ag) == claim_is_true(twin, hg, ag)


def test_a_first_half_claim_settles_on_the_INTERVAL_score():
    row = {"played": True, "status": "STATUS_FULL_TIME",
           "home_score": 3.0, "away_score": 1.0, "went_to_extra_time": False,
           "home_ht_score": 1.0, "away_ht_score": 1.0}
    assert score_for_claim("1h_draw", row) == (1, 1)
    assert score_for_claim("draw", row) == (3, 1)
    # 1-1 at the interval, 3-1 at the end: the two claims must disagree.
    assert claim_is_true("1h_draw", *score_for_claim("1h_draw", row))
    assert not claim_is_true("draw", *score_for_claim("draw", row))


def test_a_match_with_no_recorded_half_time_cannot_settle_a_first_half_market():
    """~1-2% of matches carry no usable goal detail. Settling those as 0-0
    would pay out on a number nobody observed."""
    row = {"played": True, "status": "STATUS_FULL_TIME",
           "home_score": 2.0, "away_score": 1.0, "went_to_extra_time": False,
           "home_ht_score": None, "away_ht_score": None}
    assert score_for_claim("1h_draw", row) is None
    assert score_for_claim("draw", row) == (2, 1), "full match still settles"


def test_is_first_half_survives_negation():
    assert is_first_half("1h_draw") and is_first_half("not_1h_draw")
    assert not is_first_half("draw") and not is_first_half("not_draw")


# --- vocabulary ---------------------------------------------------------------
def test_only_the_validated_first_half_families_are_admitted():
    for claim in ("1h_home_win", "1h_draw", "1h_away_win",
                  "1h_total_over_1.5", "1h_score_2-1", "not_1h_draw"):
        assert claim_supported(claim), claim
    # Arithmetically expressible, but nothing has validated them on half-time
    # data, so they stay abstentions.
    for claim in ("1h_btts", "1h_home_over_0.5", "1h_home_wins_by_over_1.5"):
        assert not claim_supported(claim), claim


def test_an_unknown_first_half_claim_still_stalls_settlement():
    with pytest.raises(UnsettleableClaim):
        claim_is_true("1h_corners_over_4.5", 1, 1)


# --- the fit ------------------------------------------------------------------
def test_rows_without_a_half_time_score_are_dropped_not_imputed():
    """Imputing 0-0 would teach the model that goalless first halves are far
    more common than they are -- the direction that makes every over cheap."""
    frame = half_time_frame(matches(n=100, with_ht=60))
    assert len(frame) == 60
    assert (frame["home_score"] == 1).all(), "half-time score becomes THE score"
    assert (frame["away_score"] == 0).all()


def test_coverage_is_reported_rather_than_assumed():
    stats = coverage(matches(n=100, with_ht=60))
    assert stats == {"played": 100, "with_half_time": 60, "fraction": 0.6}


def test_a_league_without_half_time_columns_is_refused_loudly():
    frame = matches(n=100).drop(columns=["home_ht_score", "away_ht_score"])
    with pytest.raises(NoHalfTimeData):
        half_time_frame(frame)
    with pytest.raises(NoHalfTimeData):
        build_first_half_strength(frame)


def test_too_few_matches_is_no_model_rather_than_a_weak_one():
    """A league with no first-half model must not look like one with a poor
    model -- the two lead to different decisions downstream."""
    with pytest.raises(NoHalfTimeData) as exc:
        build_first_half_strength(matches(n=100, with_ht=100))
    assert "required" in str(exc.value)


def test_a_mostly_missing_history_is_refused_even_when_large():
    with pytest.raises(NoHalfTimeData) as exc:
        build_first_half_strength(matches(n=2000, with_ht=400))
    assert "%" in str(exc.value)


def test_a_sufficient_history_fits():
    ratings = build_first_half_strength(matches(n=600, with_ht=600))
    assert set(ratings.teams) == {"A", "B"}


# --- the wiring, which is where the damage would actually be done -------------
# Both of the bugs this system has shipped were wiring, not mathematics: a grid
# answering the wrong question is a well-formed probability, and nothing
# downstream would flag it.
def test_a_first_half_claim_is_answered_from_the_first_half_grid():
    from wc2026.paper.cycle import ratings_for_claim
    full, half = object(), object()
    assert ratings_for_claim("1h_total_over_1.5", full, half) is half
    assert ratings_for_claim("not_1h_draw", full, half) is half


def test_a_full_match_claim_is_answered_from_the_full_match_grid():
    from wc2026.paper.cycle import ratings_for_claim
    full, half = object(), object()
    assert ratings_for_claim("total_over_1.5", full, half) is full
    assert ratings_for_claim("not_draw", full, half) is full


def test_a_league_with_no_first_half_model_abstains_rather_than_falls_back():
    """Falling back to the full-match grid is the expensive failure: every
    1H over would look cheap and the book would buy them all."""
    from wc2026.paper.cycle import ratings_for_claim
    full = object()
    assert ratings_for_claim("1h_total_over_1.5", full, None) is None
    assert ratings_for_claim("total_over_1.5", full, None) is full, \
        "full-match markets must be unaffected"


def test_the_two_grids_would_disagree_enough_for_this_to_matter():
    """Guards the premise. If the first-half and full-match grids gave similar
    answers, mis-routing would be harmless and the abstention over-caution.
    They do not: on a history of 2-1 finals that were 1-0 at the interval, the
    same over is a coin flip on one grid and a long shot on the other."""
    from wc2026.model.ratings import build_team_strength
    from wc2026.sim.match import predict_match

    history = matches(n=600, with_ht=600)
    full = predict_match(build_team_strength(history), "A", "B")
    half = predict_match(build_first_half_strength(history), "A", "B")

    over = probability_for("1h_total_over_1.5", full)
    correct = probability_for("1h_total_over_1.5", half)
    assert over - correct > 0.40, (over, correct)
    # And the error runs the dangerous way: the wrong grid makes the over look
    # far likelier than it is, so a book pricing off it would buy every one.
    assert correct < 0.15 < over
