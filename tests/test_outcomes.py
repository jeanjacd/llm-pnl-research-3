"""Settlement must answer exactly the question the forecast answered."""
import numpy as np
import pytest

from wc2026.paper.cycle import probability_for
from wc2026.paper.outcomes import (
    UnsettleableClaim,
    claim_is_true,
    regulation_score,
    winning_side,
)


class FakePrediction:
    """Any normalised scoreline grid is a valid probability measure."""

    def __init__(self, matrix):
        self.matrix = matrix


def a_grid(size=13, seed=7):
    rng = np.random.default_rng(seed)
    m = rng.random((size, size)) + 0.01
    return FakePrediction(m / m.sum())


# Every claim family the model can price, with lines on both sides of the mass.
CLAIMS = (
    ["home_win", "away_win", "draw", "btts"]
    + ["total_over_%s" % x for x in ("0.5", "1.5", "2.5", "3.5", "4.5")]
    + ["total_under_%s" % x for x in ("1.5", "2.5", "3.5")]
    + ["home_over_%s" % x for x in ("0.5", "1.5", "2.5")]
    + ["away_over_%s" % x for x in ("0.5", "1.5", "2.5")]
    + ["home_wins_by_over_%s" % x for x in ("0.5", "1.5", "2.5")]
    + ["away_wins_by_over_%s" % x for x in ("0.5", "1.5", "2.5")]
    + ["score_0-0", "score_1-0", "score_2-1", "score_3-3"]
)


@pytest.mark.parametrize("claim", CLAIMS)
def test_settlement_agrees_with_the_forecast_cell_for_cell(claim):
    """Sum the model's own grid over the cells settlement calls a win.

    If this drifts, measured P&L is scoring a different question from the one
    the model answered -- which would read as edge (or its absence) with
    nothing in the output to show why.
    """
    pred = a_grid()
    size = pred.matrix.shape[0]
    expected = sum(pred.matrix[i][j]
                   for i in range(size) for j in range(size)
                   if claim_is_true(claim, i, j))
    assert probability_for(claim, pred) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("claim", CLAIMS)
def test_negation_agrees_too(claim):
    pred = a_grid(seed=11)
    got = probability_for("not_" + claim, pred)
    assert got == pytest.approx(1.0 - probability_for(claim, pred), abs=1e-12)
    for hg, ag in ((0, 0), (1, 0), (2, 2), (3, 1)):
        assert claim_is_true("not_" + claim, hg, ag) is not claim_is_true(
            claim, hg, ag)


def test_the_obvious_cases_by_hand():
    assert claim_is_true("home_win", 2, 1)
    assert not claim_is_true("home_win", 1, 1)
    assert claim_is_true("draw", 0, 0)
    assert claim_is_true("btts", 1, 1)
    assert not claim_is_true("btts", 3, 0)
    assert claim_is_true("total_over_2.5", 2, 1)
    assert not claim_is_true("total_over_2.5", 1, 1)
    assert claim_is_true("away_wins_by_over_1.5", 0, 2)
    assert not claim_is_true("away_wins_by_over_1.5", 0, 1)
    assert claim_is_true("score_2-1", 2, 1)
    assert not claim_is_true("score_2-1", 1, 2)


def test_winning_side_is_the_brokers_result_vocabulary():
    assert winning_side("draw", 1, 1) == "yes"
    assert winning_side("draw", 2, 1) == "no"


def test_an_unknown_claim_stalls_settlement_instead_of_losing_silently():
    """A claim we cannot settle must be visible, not resolved to a loss."""
    for claim in ("corners_over_9.5", "first_goalscorer_messi", "", "score_x-y"):
        with pytest.raises(UnsettleableClaim):
            claim_is_true(claim, 1, 1)


def test_a_negative_scoreline_is_refused():
    with pytest.raises(UnsettleableClaim):
        claim_is_true("home_win", -1, 0)


# --- which fixtures may settle at all -----------------------------------------
def row(**kw):
    base = {"played": True, "status": "STATUS_FULL_TIME", "home_score": 2.0,
            "away_score": 1.0, "went_to_extra_time": False}
    base.update(kw)
    return base


def test_a_finished_regulation_match_settles():
    assert regulation_score(row()) == (2, 1)


def test_extra_time_is_refused_because_our_markets_settle_on_regulation():
    """The stored score for an ET match is the post-ET score."""
    assert regulation_score(row(went_to_extra_time=True)) is None


def test_unfinished_abandoned_or_scoreless_rows_are_refused():
    assert regulation_score(row(played=False)) is None
    assert regulation_score(row(status="STATUS_ABANDONED")) is None
    assert regulation_score(row(status="STATUS_POSTPONED")) is None
    assert regulation_score(row(home_score=None)) is None
    assert regulation_score(row(away_score=float("nan"))) is None
