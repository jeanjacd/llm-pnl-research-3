"""Half-time reconstruction from the scoreboard's goal details.

The scoreboard already carries per-goal `details` with a clock and a team, so
this costs no extra request. What it must never do is guess: roughly 10% of
historical matches have no detail list, and reading that as 0-0 would teach a
first-half model that goalless first halves are far more common than they are.
"""
import pytest

from wc2026.data.espn import half_time_score


def comp(*goals, **kw):
    """A competition with goal details at the given (minute, team) pairs."""
    details = [{"type": {"text": "Goal"},
                "clock": {"displayValue": minute},
                "team": {"id": team}} for minute, team in goals]
    details += kw.get("extra", [])
    return {"details": details}


HOME, AWAY = "10", "20"


# --- the ordinary cases -------------------------------------------------------
def test_goals_before_the_interval_count_and_later_ones_do_not():
    got = half_time_score(comp(("23'", HOME), ("31'", AWAY), ("67'", HOME)),
                          HOME, AWAY, (2, 1))
    assert got == (1, 1)


def test_first_half_stoppage_time_belongs_to_the_first_half():
    got = half_time_score(comp(("45'+2'", HOME)), HOME, AWAY, (1, 0))
    assert got == (1, 0)


def test_second_half_stoppage_time_does_not():
    got = half_time_score(comp(("90'+4'", HOME)), HOME, AWAY, (1, 0))
    assert got == (0, 0)


def test_a_goalless_match_is_a_definite_nil_nil_without_any_details():
    """No goals at all means none before the interval either."""
    assert half_time_score({"details": []}, HOME, AWAY, (0, 0)) == (0, 0)
    assert half_time_score({}, HOME, AWAY, (0, 0)) == (0, 0)


# --- everything that must return UNKNOWN rather than a guess -------------------
def test_a_scoring_match_with_no_details_is_unknown_not_nil_nil():
    """The ~10% data gap. Recording it as 0-0 would bias the model hard."""
    assert half_time_score({"details": []}, HOME, AWAY, (2, 1)) == (None, None)
    assert half_time_score({}, HOME, AWAY, (2, 1)) == (None, None)


def test_an_unparseable_clock_is_unknown():
    assert half_time_score(comp(("HT", HOME)), HOME, AWAY, (1, 0)) == (None, None)
    assert half_time_score(comp(("", HOME)), HOME, AWAY, (1, 0)) == (None, None)


def test_a_goal_credited_to_neither_side_is_unknown():
    assert half_time_score(comp(("20'", "999")), HOME, AWAY, (1, 0)) == (None, None)


def test_an_unplayed_match_has_no_half_time_score():
    assert half_time_score(comp(), HOME, AWAY, (None, None)) == (None, None)


def test_a_reconstruction_exceeding_full_time_is_rejected():
    """A half-time score above the final score means the parse is wrong, and a
    wrong number is worse than an absent one."""
    got = half_time_score(comp(("10'", HOME), ("20'", HOME), ("30'", HOME)),
                          HOME, AWAY, (1, 0))
    assert got == (None, None)


def test_details_that_are_not_goals_are_ignored():
    extra = [{"type": {"text": "Yellow Card"}, "clock": {"displayValue": "5'"},
              "team": {"id": HOME}},
             {"type": {"text": "Substitution"}, "clock": {"displayValue": "30'"},
              "team": {"id": AWAY}}]
    got = half_time_score(comp(("23'", HOME), extra=extra), HOME, AWAY, (1, 0))
    assert got == (1, 0)


@pytest.mark.parametrize("minute,expected", [
    ("1'", (1, 0)), ("44'", (1, 0)), ("45'", (1, 0)),
    ("46'", (0, 0)), ("90'", (0, 0)),
])
def test_the_interval_boundary_is_the_45th_minute(minute, expected):
    assert half_time_score(comp((minute, HOME)), HOME, AWAY, (1, 0)) == expected


# --- the invariant that catches a wrong parse downstream ----------------------
def test_half_time_never_exceeds_full_time_for_any_input():
    for hg, ag in ((0, 0), (1, 0), (0, 3), (2, 2), (5, 1)):
        goals = [("%d'" % (10 + i), HOME) for i in range(hg)]
        goals += [("%d'" % (10 + i), AWAY) for i in range(ag)]
        got = half_time_score(comp(*goals), HOME, AWAY, (hg, ag))
        assert got[0] is None or (got[0] <= hg and got[1] <= ag)
