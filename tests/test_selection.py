"""One board decision per fixture, chosen by a rule that is not "highest EV"."""
import datetime as dt

import pytest

from wc2026.paper.selection import (
    base_claim,
    board_window_state,
    claim_rank,
    fixture_key,
    hours_to_kickoff,
    select_one_per_fixture,
)


def in_window(hours=24.0):
    """A kick-off the lead-time rule will accept."""
    return (dt.datetime.now(dt.timezone.utc)
            + dt.timedelta(hours=hours)).isoformat()


class FakeCase:
    def __init__(self, ev):
        self.ev_per_contract = ev


class FakeInstrument:
    def __init__(self, venue="kalshi", instrument_id="I1"):
        self.venue, self.instrument_id = venue, instrument_id


class FakeLeg:
    def __init__(self, kickoff):
        self.kickoff_utc = kickoff


class Cand:
    def __init__(self, claim, ev=1.0, home="A", away="B", league="mls",
                 kickoff=None, venue="kalshi", instrument_id=None):
        kickoff = kickoff or in_window()
        self.claim = claim
        self.case = FakeCase(ev)
        self.instrument = FakeInstrument(venue, instrument_id or claim)
        self.leg = FakeLeg(kickoff)
        self.league_id = league
        self.fixture_key = fixture_key(league, home, away, kickoff)


def test_a_negated_claim_belongs_to_its_own_family():
    assert base_claim("not_home_win") == "home_win"
    assert claim_rank("not_home_win") == claim_rank("home_win")


def test_the_families_are_ordered_by_model_quality_not_alphabetically():
    """1X2 is the family the evaluation scores; exact scorelines are last."""
    assert claim_rank("home_win") < claim_rank("btts")
    assert claim_rank("btts") < claim_rank("total_over_2.5")
    assert claim_rank("total_over_2.5") < claim_rank("home_over_1.5")
    assert claim_rank("home_over_1.5") < claim_rank("home_wins_by_over_1.5")
    assert claim_rank("home_wins_by_over_1.5") < claim_rank("score_2-1")


def test_an_unknown_claim_sorts_last_rather_than_first():
    assert claim_rank("corners_over_9.5") > claim_rank("score_2-1")


def test_one_fixture_yields_exactly_one_selection():
    """The measured shape: 34 markets on a match are 1 observation."""
    cands = [Cand("away_over_%s" % x) for x in ("0.5", "1.5", "2.5")]
    cands += [Cand("away_win"), Cand("btts"), Cand("draw")]
    selected, skipped = select_one_per_fixture(cands)
    assert len(selected) == 1
    assert len(skipped) == len(cands) - 1


def test_the_best_validated_family_wins_over_a_juicier_long_shot():
    """A 40c EV on an exact scoreline must not outrank a 1c EV on 1X2."""
    long_shot = Cand("score_4-3", ev=40.0)
    solid = Cand("home_win", ev=1.0)
    selected, _ = select_one_per_fixture([long_shot, solid])
    assert selected[0].claim == "home_win"


def test_ev_breaks_ties_inside_a_family():
    lo, hi = Cand("home_win", ev=1.0), Cand("away_win", ev=9.0)
    selected, _ = select_one_per_fixture([lo, hi])
    assert selected[0].claim == "away_win"


def test_selection_is_stable_across_runs_over_the_same_data():
    cands = [Cand("home_win", ev=5.0, venue="kalshi", instrument_id="K"),
             Cand("draw", ev=5.0, venue="polymarket", instrument_id="P")]
    first, _ = select_one_per_fixture(cands)
    second, _ = select_one_per_fixture(list(reversed(cands)))
    assert first[0].instrument.instrument_id == second[0].instrument.instrument_id


def test_distinct_fixtures_each_get_a_selection():
    cands = [Cand("home_win", home="A", away="B"),
             Cand("home_win", home="C", away="D"),
             Cand("draw", home="C", away="D")]
    selected, _ = select_one_per_fixture(cands)
    assert len(selected) == 2


def test_the_two_venues_quoting_one_match_are_still_one_fixture():
    """Otherwise the same match counts twice as an 'independent' observation."""
    cands = [Cand("home_win", venue="kalshi", instrument_id="K"),
             Cand("home_win", venue="polymarket", instrument_id="P")]
    selected, _ = select_one_per_fixture(cands)
    assert len(selected) == 1


# --- lead-time targeting ------------------------------------------------------
def test_a_fixture_days_away_is_not_boarded_yet():
    """Boarding four days out wastes the observation on stale team news."""
    assert board_window_state(in_window(96)) == "too_early"


def test_a_fixture_at_the_target_lead_time_is_boarded():
    assert board_window_state(in_window(24)) == "board"


def test_a_fixture_about_to_kick_off_is_let_go():
    """No time left for a resting order to be reached."""
    assert board_window_state(in_window(0.5)) == "too_late"
    assert board_window_state(in_window(-3)) == "too_late"


def test_a_missed_fixture_is_still_boarded_late_rather_than_lost():
    """An outage must cost freshness, not the whole observation."""
    assert board_window_state(in_window(8)) == "board"


def test_an_unknown_kickoff_is_named_not_silently_dropped():
    assert board_window_state(None) == "unknown_kickoff"
    assert board_window_state("not a timestamp") == "unknown_kickoff"


def test_hours_to_kickoff_handles_both_timestamp_spellings():
    now = dt.datetime(2026, 8, 20, tzinfo=dt.timezone.utc)
    assert hours_to_kickoff("2026-08-20T12:00:00Z", now) == pytest.approx(12.0)
    assert hours_to_kickoff("2026-08-20 12:00:00", now) == pytest.approx(12.0)


def test_the_window_is_wider_than_the_schedulers_interval():
    """Six-hourly runs must not let a fixture slip between two of them."""
    from wc2026.paper.selection import BOARD_WINDOW_HOURS
    assert BOARD_WINDOW_HOURS >= 6.0


def test_selection_drops_out_of_window_fixtures_with_a_named_reason():
    early = Cand("home_win", kickoff=in_window(120))
    selected, skipped = select_one_per_fixture([early])
    assert selected == []
    assert "too_early" in skipped[0][1]


def test_lead_time_can_be_turned_off_for_a_backfill():
    early = Cand("home_win", kickoff=in_window(500))
    selected, _ = select_one_per_fixture([early], use_lead_time=False)
    assert len(selected) == 1


def test_kickoff_minutes_apart_do_not_split_one_fixture_in_two():
    """The venues source kick-off differently and can disagree by minutes."""
    a = fixture_key("mls", "A", "B", "2026-09-01T19:00:00+00:00")
    b = fixture_key("mls", "A", "B", "2026-09-01T19:05:00+00:00")
    assert a == b


def test_a_fixture_already_boarded_is_not_boarded_again():
    """Re-boarding produces a correlated duplicate, not a new observation."""
    cand = Cand("home_win")
    selected, skipped = select_one_per_fixture([cand], {cand.fixture_key})
    assert selected == []
    assert skipped and "already boarded" in skipped[0][1]


def test_every_dropped_candidate_carries_a_reason():
    """A silent cap reads as 'we looked at everything'."""
    cands = [Cand("home_win"), Cand("score_1-0"), Cand("btts")]
    selected, skipped = select_one_per_fixture(cands)
    assert len(selected) + len(skipped) == len(cands)
    assert all(isinstance(reason, str) and reason for _, reason in skipped)


def test_nothing_in_yields_nothing_out():
    assert select_one_per_fixture([]) == ([], [])


def test_a_missing_ev_does_not_crash_the_ranking():
    cand = Cand("home_win")
    cand.case.ev_per_contract = None
    selected, _ = select_one_per_fixture([cand])
    assert len(selected) == 1


@pytest.mark.parametrize("claim", ["home_win", "away_win", "draw"])
def test_all_three_1x2_outcomes_share_the_top_rank(claim):
    assert claim_rank(claim) == 0


def test_a_spread_is_not_mistaken_for_a_1x2_market():
    """`home_wins_by_over_1.5`.startswith(`home_win`) is True -- a plain
    prefix test ranks the thinnest family as the best-validated one."""
    assert claim_rank("home_wins_by_over_1.5") > claim_rank("home_win")
    assert claim_rank("away_wins_by_over_2.5") > claim_rank("away_win")
    assert claim_rank("home_wins_by_over_1.5") == claim_rank("away_wins_by_over_1.5")
