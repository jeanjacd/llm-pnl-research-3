"""The betting ledger: open, run, book.

The strip this replaces failed in ways a test could not see, because it was
tested for what it PRINTED rather than what it encoded. So these assert the
encoding directly: that magnitude is a length, that a non-event claims no
weight, that a still-running bet is representable at all, and that the closing
line has a channel of its own.
"""
import datetime as dt

import pytest
from paper_fixtures import book, pos, verdict

from wc2026.site import ledger

UTC = dt.timezone.utc
# Kick-offs in `paper_fixtures` are 18:00, so this sits 90 minutes into a
# match played on the 28th -- inside the in-play window, and after one
# played on the 27th has long since finished.
NOW = dt.datetime(2026, 8, 28, 19, 30, tzinfo=UTC)


def held(claim="draw", home="A", away="B", **kw):
    """An unsettled position. `pos` marks a position settled iff it has a P&L."""
    out = pos(claim=claim, home=home, away=away, pnl=None, **kw)
    out["instrument_id"] = "%s-%s-%s" % (claim, home, kw.get("venue", "k"))
    return out


# ── what has to happen ────────────────────────────────────────────────────────
# The one thing the strip could never say. It has to read the claim exactly the
# way settlement does, or the sentence is a lie about the money.
@pytest.mark.parametrize("claim,fragment", [
    ("home_win", "Betis must be ahead at full time"),
    ("not_home_win", "Betis must not be ahead at full time"),
    ("1h_draw", "level at half time"),
    ("not_1h_home_win", "Betis must not be ahead at half time"),
    ("total_over_2.5", "More than 2.5 goals"),
    ("not_total_over_2.5", "Fewer than 2.5 goals"),
    ("not_total_under_2.5", "More than 2.5 goals"),
    ("btts", "Both sides must score"),
    ("not_btts", "at least one side must fail to score".capitalize()),
    ("score_4-0", "must be exactly 4-0"),
    ("not_score_4-0", "anything but 4-0"),
    ("home_over_1.5", "Betis must score more than 1.5"),
    ("not_home_over_1.5", "Betis must score fewer than 1.5"),
    ("home_wins_by_over_1.5", "Betis must win by more than 1.5"),
    ("not_home_wins_by_over_1.5", "Betis must not win by more than 1.5"),
])
def test_the_instruction_is_derived_from_the_claim(claim, fragment):
    assert fragment in ledger.needs(claim, "Betis", "Madrid")


def test_every_sentence_is_a_sentence():
    for claim in ("draw", "not_draw", "away_win", "1h_total_over_1.5",
                  "not_score_0-0", "btts"):
        text = ledger.needs(claim, "A", "B")
        assert text[0].isupper() and text.endswith(".")


def test_an_unknown_claim_says_so_rather_than_guessing():
    assert "settles on the final scoreline" in ledger.needs("mystery", "A", "B")


# ── zone 1: open ──────────────────────────────────────────────────────────────
def test_a_settled_position_is_not_open():
    p = book([pos(pnl=500), held(home="C", away="D")])
    rows = ledger.open_tickets(p, now=NOW)
    assert [r["home"] for r in rows] == ["C"]


def test_the_state_comes_from_the_clock_and_the_kick_off():
    """Pre-game, in play, and waiting on a result are three different answers
    to "what is happening", and the strip had no way to say any of them."""
    p = book([held(home="Soon", kickoff="2026-08-29"),
              held(home="Playing", kickoff="2026-08-28"),
              held(home="Done", kickoff="2026-08-27")])
    got = {r["home"]: r["state"] for r in ledger.open_tickets(p, now=NOW)}
    assert got == {"Soon": "pending", "Playing": "live", "Done": "awaiting"}


def test_the_rail_leads_with_what_is_happening_now():
    p = book([held(home="Soon", kickoff="2026-08-29"),
              held(home="Done", kickoff="2026-08-27"),
              held(home="Playing", kickoff="2026-08-28")])
    assert [r["state"] for r in ledger.open_tickets(p, now=NOW)] == \
        ["live", "awaiting", "pending"]


def test_a_fixture_is_one_card_however_many_markets_it_carries():
    """`paper/clv.py` makes the fixture the unit of independence. Four cards
    for one match would show one opinion four times."""
    p = book([held(claim="draw"), held(claim="btts"),
              held(claim="home_win")])
    rows = ledger.open_tickets(p, now=NOW)
    assert len(rows) == 1 and rows[0]["n_legs"] == 3


def test_exposure_is_what_it_cost_and_what_it_pays():
    """100 contracts at 20c cost $20 and return $100, so $80 is still on the
    table. A binary settles at 100c; nothing here guesses at a probability."""
    p = book([held(size=100.0, cost=20.0)])
    out = ledger.exposure(ledger.open_tickets(p, now=NOW))
    assert out["staked_cents"] == pytest.approx(2000.0)
    assert out["upside_cents"] == pytest.approx(8000.0)
    assert out["n_markets"] == 1 and out["n_fixtures"] == 1


def test_no_open_position_is_an_empty_rail_rather_than_a_zero():
    assert ledger.open_tickets(book([pos(pnl=100)]), now=NOW) == []
    assert ledger.exposure([])["staked_cents"] == 0


# ── zone 3: the book ──────────────────────────────────────────────────────────
def test_the_book_reads_oldest_first():
    p = book(boarded=[verdict(home="Late", day="2026-08-27"),
                      verdict(home="Early", day="2026-08-20")])
    assert [c["home"] for c in ledger.columns(p, now=NOW)] == ["Early", "Late"]


def test_a_board_that_passed_and_a_market_that_never_came_are_different():
    """The strip printed both as `·`. The board approving 130 markets and the
    market never reaching our price are opposite facts about one fixture."""
    order = {"status": "expired", "kind": "limit", "league_id": "mls",
             "home_team": "Leeds", "away_team": "Brentford",
             "kickoff_utc": "2026-08-28T18:00:00"}
    p = book(boarded=[verdict(), verdict(home="Leeds", away="Brentford",
                              action="PAPER_PLACE_LIMIT")],
             orders=[order for _ in range(130)])
    got = {c["home"]: c["outcome"] for c in ledger.columns(p, now=NOW)}
    assert got["A"] == "declined"
    assert got["Leeds"] == "unfilled"


def test_a_running_fixture_is_shown_as_running_rather_than_dropped():
    """The strip only knew about the past: a live fixture was excluded
    entirely, so the record could not represent a bet that had not settled."""
    p = book([held(home="Playing", kickoff="2026-08-28")])
    assert [c["outcome"] for c in ledger.columns(p, now=NOW)] == ["live"]
    p = book([held(home="Later", kickoff="2026-08-29")])
    assert [c["outcome"] for c in ledger.columns(p, now=NOW)] == ["open"]


def test_a_fixture_is_won_or_lost_on_its_net_not_its_hit_rate():
    p = book([pos(claim="a", pnl=900), pos(claim="b", pnl=-100)])
    for row in ledger.columns(p, now=NOW):
        assert row["outcome"] == "won" and row["net_cents"] == 800


def test_a_month_boundary_is_a_flag_on_the_column_not_a_column_of_its_own():
    """A rule between two fixtures must not consume an x-position, or the run
    curve above slides out of step with the bars below it."""
    p = book(boarded=[verdict(home="Aug", day="2026-08-30"),
                      verdict(home="Sep", day="2026-09-02")])
    cols = ledger.columns(p, now=NOW)
    assert len(cols) == 2
    assert [c["new_month"] for c in cols] == [False, True]


def test_the_column_carries_the_closing_line_it_was_measured_at():
    p = book([pos(claim="draw", pnl=-500, clv=2.5)])
    assert ledger.columns(p, now=NOW)[0]["clv_cents"] == pytest.approx(2.5)


def test_a_column_with_no_closing_line_carries_none_rather_than_zero():
    p = book(boarded=[verdict()])
    assert ledger.columns(p, now=NOW)[0]["clv_cents"] is None


# ── the shared scale ──────────────────────────────────────────────────────────
def test_magnitude_is_a_length_and_the_scale_is_shared():
    """The digit had to be parsed one fixture at a time and did not sort
    visually, so no shape emerged across 48 of them. A length is preattentive
    only if every length is measured against the same number."""
    p = book([pos(claim="a", home="A", away="B", pnl=-1000),
              pos(claim="b", home="C", away="D", pnl=-2000)])
    cols = ledger.columns(p, now=NOW)
    info = ledger.scale(cols)
    heights = sorted(ledger.height(c["net_cents"], info)[0] for c in cols)
    assert heights == pytest.approx([0.5, 1.0])


def test_a_graze_and_a_wipeout_are_different_lengths():
    small = ledger.columns(book([pos(pnl=-100, cost=50.0, size=20.0)]),
                           now=NOW)
    big = ledger.columns(book([pos(pnl=-1000, cost=50.0, size=20.0)]),
                         now=NOW)
    info = ledger.scale(small + big)
    a = ledger.height(small[0]["net_cents"], info)[0]
    b = ledger.height(big[0]["net_cents"], info)[0]
    assert 0 < a < b


def test_one_outlier_cannot_flatten_everything_else():
    """Clamped at the 95th percentile, with the overflow flagged rather than
    hidden -- a bar off the scale should say so, not quietly set it."""
    rows = [pos(claim="c%d" % i, home="H%d" % i, pnl=-100) for i in range(20)]
    rows.append(pos(claim="huge", home="Huge", pnl=-100000))
    cols = ledger.columns(book(rows), now=NOW)
    info = ledger.scale(cols)
    assert info["clamped"] == 1
    ordinary = [c for c in cols if c["home"] != "Huge"][0]
    assert ledger.height(ordinary["net_cents"], info)[0] == pytest.approx(1.0)
    frac, over = ledger.height(-100000, info)
    assert frac == 1.0 and over is True


def test_an_empty_book_still_yields_a_usable_scale():
    info = ledger.scale([])
    assert info["top_cents"] > 0
    assert ledger.height(0, info) == (0.0, False)


# ── zone 2: the run ───────────────────────────────────────────────────────────
def test_the_run_accumulates_only_what_has_settled():
    p = book([pos(claim="a", home="A", away="B", pnl=500),
              pos(claim="b", home="C", away="D", pnl=-200),
              held(home="E", away="F")])
    out = ledger.run(ledger.columns(p, now=NOW))
    assert out["final_cents"] == pytest.approx(300.0)
    assert [round(pt["cents"]) for pt in out["points"]][-1] == 300


def test_the_run_shares_the_books_x_positions_exactly():
    p = book(boarded=[verdict(home=str(i), day="2026-08-2%d" % i)
                      for i in range(5)])
    cols = ledger.columns(p, now=NOW)
    out = ledger.run(cols)
    assert [pt["i"] for pt in out["points"]] == list(range(len(cols)))


def test_the_cone_is_the_open_position_stated_as_a_range():
    """Not a forecast. Floor is every open market losing its stake, ceiling is
    every one of them landing, and the record sits between them."""
    p = book([pos(claim="a", home="A", away="B", pnl=500),
              held(home="E", away="F", size=100.0, cost=20.0)])
    out = ledger.run(ledger.columns(p, now=NOW))
    assert out["has_open"]
    assert out["floor_cents"] == pytest.approx(500 - 2000)
    assert out["ceiling_cents"] == pytest.approx(500 + 8000)


def test_a_book_with_nothing_open_draws_no_cone():
    p = book([pos(claim="a", pnl=500)])
    assert ledger.run(ledger.columns(p, now=NOW))["has_open"] is False


# ── the archive ───────────────────────────────────────────────────────────────
def test_a_slug_is_stable_and_safe_to_put_in_a_path():
    a = ledger.slug(("la_liga", "Real Madrid", "Málaga", "2026-08-30"))
    b = ledger.slug(("la_liga", "Real Madrid", "Málaga", "2026-08-30"))
    assert a == b
    assert a == "2026-08-30-la-liga-real-madrid-m-laga"
    assert "/" not in a and "\\\\" not in a and " " not in a


def test_every_settled_market_reaches_the_archive():
    p = book([pos(claim="a", home="A", away="B", pnl=500),
              pos(claim="b", home="A", away="B", pnl=-200),
              held(home="C", away="D")])
    rows = ledger.settled_fixtures(p)
    assert len(rows) == 1, "the open fixture has settled nothing"
    assert rows[0]["n_markets"] == 2 and rows[0]["n_won"] == 1
    assert rows[0]["pnl_cents"] == pytest.approx(300.0)


def test_the_archive_reads_newest_first():
    p = book([pos(claim="a", home="Old", kickoff="2026-08-20", pnl=100),
              pos(claim="b", home="New", kickoff="2026-08-27", pnl=100)])
    assert [r["home"] for r in ledger.settled_fixtures(p)] == ["New", "Old"]
