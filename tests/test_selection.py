"""One board decision per fixture, chosen by a rule that is not "highest EV"."""
import datetime as dt

import pytest

from wc2026.paper.selection import (
    BOARD_MIN_HOURS,
    BOARD_RUN_INTERVAL_HOURS,
    BOARD_RUN_SLACK_HOURS,
    BOARD_SLATE_CHUNK,
    BOARD_TARGET_HOURS,
    BOARD_WINDOW_HOURS,
    RETRY_MAX_ATTEMPTS,
    actionable_ceiling_hours,
    actionable_fixtures,
    base_claim,
    board_state,
    board_window_state,
    chunk_slate,
    claim_rank,
    fixture_key,
    hours_to_kickoff,
    retry_by_hours,
    select_fixture_slates,
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


def test_a_fixture_yields_ONE_slate_containing_every_market():
    """34 markets on a match are 1 OBSERVATION but 34 tradeable decisions.
    Boarding one and discarding the rest confused the two."""
    cands = [Cand("away_over_%s" % x) for x in ("0.5", "1.5", "2.5")]
    cands += [Cand("away_win"), Cand("btts"), Cand("draw")]
    slates, skipped = select_fixture_slates(cands)
    assert len(slates) == 1, "one fixture"
    assert len(next(iter(slates.values()))) == len(cands), "every market kept"
    assert skipped == []


def test_the_best_validated_family_leads_the_slate():
    """Both are kept, but 1X2 leads the prompt and survives any cap; a 40c EV
    on an exact scoreline must not displace a 1c EV on 1X2."""
    slates, _ = select_fixture_slates([Cand("score_4-3", ev=40.0),
                                       Cand("home_win", ev=1.0)])
    slate = next(iter(slates.values()))
    assert [c.claim for c in slate] == ["home_win", "score_4-3"]


def test_ev_orders_within_a_family():
    slates, _ = select_fixture_slates([Cand("home_win", ev=1.0),
                                       Cand("away_win", ev=9.0)])
    assert [c.claim for c in next(iter(slates.values()))] == ["away_win",
                                                             "home_win"]


def test_nothing_is_discarded_however_many_markets_a_fixture_has():
    """The 40 used to be a CAP and silently dropped 63% of candidates at the
    real distribution -- median 109 per fixture -- cutting mid-family through
    team totals. It is a batch size now."""
    cands = ([Cand("home_win"), Cand("draw"), Cand("away_win")]
             + [Cand("score_%d-%d" % (i, j), instrument_id="s%d%d" % (i, j))
                for i in range(8) for j in range(8)])
    slates, skipped = select_fixture_slates(cands)
    slate = next(iter(slates.values()))
    assert len(slate) == len(cands), "every market survives"
    assert skipped == []


def test_a_fixture_is_boarded_in_batches_that_cover_it_exactly():
    """The response is the bound, not the prompt: the judge returns a decision
    per market, and a truncated reply fails the whole batch closed."""
    slate = list(range(109))
    batches = chunk_slate(slate)
    assert sum(len(b) for b in batches) == 109, "nothing lost, nothing repeated"
    assert [x for b in batches for x in b] == slate, "order preserved"
    assert all(len(b) <= BOARD_SLATE_CHUNK for b in batches)
    assert len(batches) == 3


def test_a_slate_that_fits_is_one_batch():
    assert len(chunk_slate(list(range(BOARD_SLATE_CHUNK)))) == 1
    assert chunk_slate([]) == []


def test_batches_keep_the_family_ordering_so_each_is_coherent():
    """The 1X2 and totals are judged together, not scattered across sittings."""
    cands = ([Cand("home_win"), Cand("draw"), Cand("away_win"), Cand("btts")]
             + [Cand("score_%d-%d" % (i, j), instrument_id="s%d%d" % (i, j))
                for i in range(8) for j in range(8)])
    slate = next(iter(select_fixture_slates(cands)[0].values()))
    first = chunk_slate(slate)[0]
    assert all(claim_rank(c.claim) <= claim_rank(slate[-1].claim) for c in first)
    assert claim_rank(first[0].claim) == 0


def test_slate_order_is_stable_across_runs_over_the_same_data():
    cands = [Cand("home_win", ev=5.0, venue="kalshi", instrument_id="K"),
             Cand("draw", ev=5.0, venue="polymarket", instrument_id="P")]
    a, _ = select_fixture_slates(cands)
    b, _ = select_fixture_slates(list(reversed(cands)))
    ids = lambda d: [c.instrument.instrument_id for c in next(iter(d.values()))]
    assert ids(a) == ids(b)


def test_distinct_fixtures_each_get_a_selection():
    cands = [Cand("home_win", home="A", away="B"),
             Cand("home_win", home="C", away="D"),
             Cand("draw", home="C", away="D")]
    slates, _ = select_fixture_slates(cands)
    assert len(slates) == 2


def test_the_two_venues_quoting_one_match_are_still_one_fixture():
    """One observation, but both venues stay tradeable -- the cheaper book may
    be either of them."""
    cands = [Cand("home_win", venue="kalshi", instrument_id="K"),
             Cand("home_win", venue="polymarket", instrument_id="P")]
    slates, _ = select_fixture_slates(cands)
    assert len(slates) == 1
    assert len(next(iter(slates.values()))) == 2


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
    slates, skipped = select_fixture_slates([early])
    assert slates == {}
    assert "too_early" in skipped[0][1]


def test_lead_time_can_be_turned_off_for_a_backfill():
    early = Cand("home_win", kickoff=in_window(500))
    slates, _ = select_fixture_slates([early], use_lead_time=False)
    assert len(slates) == 1


def test_kickoff_minutes_apart_do_not_split_one_fixture_in_two():
    """The venues source kick-off differently and can disagree by minutes."""
    a = fixture_key("mls", "A", "B", "2026-09-01T19:00:00+00:00")
    b = fixture_key("mls", "A", "B", "2026-09-01T19:05:00+00:00")
    assert a == b


def test_a_fixture_the_board_has_decided_is_not_boarded_again():
    """Re-boarding a DECIDED fixture is a correlated duplicate, not a new
    observation -- and re-asking until it says yes is the multiple-comparisons
    problem."""
    cand = Cand("home_win")
    ledger = {cand.fixture_key: {"action": "PASS", "attempts": 1}}
    slates, skipped = select_fixture_slates([cand], ledger)
    assert slates == {}
    assert "already decided" in skipped[0][1]


# --- deferral retries ---------------------------------------------------------
def deferred(attempts=1):
    return {"action": "DEFER", "attempts": attempts}


def test_a_deferral_is_retried_rather_than_being_permanent():
    """A DEFER is the board asking to be run again once team news lands.
    Recording it as final barred exactly the thing it asked for."""
    cand = Cand("home_win", kickoff=in_window(6))
    slates, _ = select_fixture_slates(
        [cand], {cand.fixture_key: deferred()})
    assert len(slates) == 1


def test_a_pass_or_reject_is_never_retried():
    """The safeguard: only a lack of INFORMATION is retried, never a decision."""
    for action in ("PASS", "PAPER_PLACE_LIMIT", "PAPER_BUY_NOW", "UNSUPPORTED"):
        cand = Cand("home_win", kickoff=in_window(6))
        slates, skipped = select_fixture_slates(
            [cand], {cand.fixture_key: {"action": action, "attempts": 1}})
        assert slates == {}, action
        assert "already decided" in skipped[0][1]


def test_a_deferral_is_retried_only_once():
    cand = Cand("home_win", kickoff=in_window(6))
    slates, skipped = select_fixture_slates(
        [cand], {cand.fixture_key: deferred(attempts=RETRY_MAX_ATTEMPTS)})
    assert slates == {}
    assert "already retried" in skipped[0][1]


def test_the_retry_is_held_back_for_nearer_kickoff():
    """Retrying immediately wastes it on the same missing team news."""
    cand = Cand("home_win", kickoff=in_window(28))
    slates, skipped = select_fixture_slates(
        [cand], {cand.fixture_key: deferred()})
    assert slates == {}
    assert "holding the retry" in skipped[0][1]


def test_the_retry_is_abandoned_if_there_is_no_time_left_to_fill():
    cand = Cand("home_win", kickoff=in_window(0.5))
    slates, skipped = select_fixture_slates(
        [cand], {cand.fixture_key: deferred()})
    assert slates == {}
    assert "too late" in skipped[0][1]


def test_the_retry_window_is_wider_than_the_worst_gap_between_runs():
    """THE guarantee: a retry lands only if a run falls inside the window, so
    the window has to outlast a dropped run plus the scheduler running late.
    Measured on the live schedule: nominal 6.0h, worst observed gap 7.0h."""
    span = retry_by_hours() - BOARD_MIN_HOURS
    worst_gap = 2 * BOARD_RUN_INTERVAL_HOURS + BOARD_RUN_SLACK_HOURS
    assert span >= worst_gap


def test_every_deferred_fixture_gets_its_retry_before_kick_off():
    """Simulated against the real cadence, including a dropped run, for every
    kick-off minute across a day. `paramount: before the game, every time`."""
    interval = BOARD_RUN_INTERVAL_HOURS
    start = dt.datetime(2026, 9, 1, 0, 17, tzinfo=dt.timezone.utc)
    # Enough runs to reach past the latest kick-off tested below, whatever
    # the cadence -- otherwise the simulation runs out of runs and reports a
    # miss that the schedule would not actually produce.
    runs = [start + dt.timedelta(hours=interval * i)
            for i in range(int(80 / interval))]
    # Drop one run outright and run another 42 min late (both observed live).
    degraded = [r for i, r in enumerate(runs) if i != 5]
    degraded = [r + dt.timedelta(minutes=42) if i % 3 == 0 else r
                for i, r in enumerate(degraded)]

    for offset in range(0, 60 * 24, 17):          # kick-offs across a day
        kickoff = start + dt.timedelta(hours=40) + dt.timedelta(minutes=offset)
        record, retried_at = deferred(), None
        for now in degraded:
            if now >= kickoff:
                break
            ok, _ = board_state(record, kickoff.isoformat(), now)
            if ok:
                retried_at = now
                break
        assert retried_at is not None, "no retry for kick-off %s" % kickoff
        lead = (kickoff - retried_at).total_seconds() / 3600
        assert lead >= BOARD_MIN_HOURS, "retry too close to kick-off: %.1fh" % lead


def test_every_dropped_candidate_carries_a_reason():
    """A silent cap reads as 'we looked at everything'."""
    cands = [Cand("home_win"), Cand("score_1-0"), Cand("btts")]
    slates, skipped = select_fixture_slates(cands)
    kept = sum(len(g) for g in slates.values())
    assert kept + len(skipped) == len(cands), "nothing vanishes unaccounted for"
    assert all(isinstance(reason, str) and reason for _, reason in skipped)


def test_nothing_in_yields_nothing_out():
    assert select_fixture_slates([]) == ({}, [])


def test_a_missing_ev_does_not_crash_the_ranking():
    cand = Cand("home_win")
    cand.case.ev_per_contract = None
    slates, _ = select_fixture_slates([cand])
    assert len(slates) == 1


@pytest.mark.parametrize("claim", ["home_win", "away_win", "draw"])
def test_all_three_1x2_outcomes_share_the_top_rank(claim):
    assert claim_rank(claim) == 0


def test_a_spread_is_not_mistaken_for_a_1x2_market():
    """`home_wins_by_over_1.5`.startswith(`home_win`) is True -- a plain
    prefix test ranks the thinnest family as the best-validated one."""
    assert claim_rank("home_wins_by_over_1.5") > claim_rank("home_win")
    assert claim_rank("away_wins_by_over_2.5") > claim_rank("away_win")
    assert claim_rank("home_wins_by_over_1.5") == claim_rank("away_wins_by_over_1.5")


# --- window-filtered discovery ------------------------------------------------
def fixture_table(*hours_out):
    import pandas as pd
    now = dt.datetime.now(dt.timezone.utc)
    return pd.DataFrame([
        {"date": pd.Timestamp(now + dt.timedelta(hours=h)).normalize(),
         "home_team": "H%d" % i, "away_team": "A%d" % i,
         "kickoff_utc": (now + dt.timedelta(hours=h)).isoformat()}
        for i, h in enumerate(hours_out)])


def test_only_fixtures_a_run_could_act_on_reach_discovery():
    """A book is fetched per RESOLVED market, so the season-wide table meant
    thousands of fetches for matches weeks away. Measured on one league:
    2,164 Polymarket requests / 251s before, 4 requests / 0.9s after."""
    frame, skipped = actionable_fixtures(fixture_table(1, 12, 24, 200))
    assert len(frame) == 2          # 12h and 24h
    assert skipped == 2             # 1h too late, 200h too early


def test_the_filter_cannot_hide_a_fixture_still_owed_a_retry():
    """THE safety property. The retry deadline must sit inside the range the
    filter keeps, or a deferred fixture would never be rediscovered and the
    guarantee would be silently void."""
    assert retry_by_hours() <= actionable_ceiling_hours()
    frame, _ = actionable_fixtures(fixture_table(retry_by_hours() - 0.1))
    assert len(frame) == 1


def test_the_filter_keeps_the_whole_first_pass_window():
    assert (BOARD_TARGET_HOURS + BOARD_WINDOW_HOURS) <= actionable_ceiling_hours()
    frame, _ = actionable_fixtures(
        fixture_table(BOARD_TARGET_HOURS + BOARD_WINDOW_HOURS - 0.1))
    assert len(frame) == 1


def test_anything_the_filter_keeps_is_something_selection_would_consider():
    """No fixture should survive the filter only to be dropped immediately --
    that is wasted discovery, which is the thing being fixed."""
    ceiling = actionable_ceiling_hours()
    for hours in (BOARD_MIN_HOURS + 0.1, ceiling / 2, ceiling - 0.1):
        frame, _ = actionable_fixtures(fixture_table(hours))
        assert len(frame) == 1, hours
        state = board_window_state(frame.iloc[0]["kickoff_utc"])
        deferred_ok, _ = board_state(deferred(), frame.iloc[0]["kickoff_utc"])
        assert state == "board" or deferred_ok, (hours, state)


def test_a_fixture_with_no_kickoff_is_left_out_and_counted():
    """It cannot be placed in the window and the board refuses it anyway."""
    import pandas as pd
    frame = fixture_table(12)
    frame.loc[0, "kickoff_utc"] = None
    kept, skipped = actionable_fixtures(frame)
    assert len(kept) == 0 and skipped == 1
    assert isinstance(kept, pd.DataFrame)


def test_a_table_without_kickoffs_is_passed_through_untouched():
    """Older callers pass date/home/away only; filtering must not crash."""
    frame = fixture_table(12).drop(columns=["kickoff_utc"])
    kept, skipped = actionable_fixtures(frame)
    assert len(kept) == 1 and skipped == 0


def test_an_empty_table_is_handled():
    assert actionable_fixtures(None) == (None, 0)
