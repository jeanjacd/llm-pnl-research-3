"""The numbers the published page prints.

Everything here is a claim the site makes in public about a real trading
record, so the failures that matter are the quiet ones: a rate computed over
the wrong denominator, an abstention dropped, a live fixture scored as though
it had finished.
"""
import pytest
from paper_fixtures import book, pos, verdict

from wc2026.site import model


# --- the unit of independence -------------------------------------------------
def test_correlated_bets_on_one_match_are_one_observation():
    """The live book's first three settled bets were all Lille v PSG. Scoring
    that as n=3 would treat one directional view written three ways as three
    independent results, which is how a record flatters itself."""
    p = book([pos("home_over_1.5", clv=4.0, pnl=3700),
              pos("away_over_0.5", clv=-1.0, pnl=4250),
              pos("home_wins_by_over_1.5", clv=0.5, pnl=-640)])
    out = model.clv(p)
    assert out["n_fixtures"] == 1, "one match is one observation"
    assert out["n_bets"] == 3
    # and the headline averages the FIXTURE, not the three bets
    assert out["mean_cents"] == pytest.approx((4.0 - 1.0 + 0.5) / 3)


def test_distinct_matches_are_distinct_observations():
    p = book([pos(home="A", away="B", clv=2.0),
              pos(home="C", away="D", clv=4.0)])
    assert model.clv(p)["n_fixtures"] == 2


def test_a_declined_fixture_merges_with_positions_later_taken_on_it():
    """`boarded` records carry no league and no kickoff -- only their KEY does.
    Reading the record instead of the key files every abstention under an empty
    league, so it never merges and the fixture is counted twice."""
    p = book([pos(league="ligue_1", home="Lille", away="PSG")],
             [verdict("ligue_1", "Lille", "PSG", "2026-08-28",
                      action="PAPER_PLACE_LIMIT")])
    rows = model.fixtures(p)
    assert len(rows) == 1, "the same match must not appear twice"
    assert rows[0]["acted"] and rows[0]["boarded"]
    assert rows[0]["league_id"] == "ligue_1"


# --- the form notation --------------------------------------------------------
def test_a_declined_fixture_reads_as_an_abstention():
    p = book(boarded=[verdict()])
    assert [f["char"] for f in model.form_line(p)] == [model.FORM_DECLINED]


def test_a_fixture_that_finished_up_reads_as_cashed():
    p = book([pos(pnl=4000, clv=1.0)])
    assert [f["char"] for f in model.form_line(p)] == [model.FORM_CASHED]


def test_a_losing_fixture_reads_as_the_count_that_landed():
    """The digit separates losing with nothing landing from losing with several
    landing -- a pricing problem from a sizing problem."""
    p = book([pos("draw", pnl=-1000), pos("btts", pnl=-1000)])
    assert [f["char"] for f in model.form_line(p)] == ["0"]
    q = book([pos("draw", pnl=+500), pos("btts", pnl=-2000)])
    assert [f["char"] for f in model.form_line(q)] == ["1"], "net down, 1 landed"


def test_a_live_fixture_is_absent_from_the_record_rather_than_scored():
    """It has no result. Encoding one would be settling an unplayed match."""
    p = book([pos("draw", pnl=None), pos("btts", pnl=3000)])
    assert model.form_line(p) == []


def test_a_month_boundary_sets_a_rule():
    p = book(boarded=[verdict(home="A", day="2026-08-30"),
                      verdict(home="C", day="2026-09-02")])
    chars = [f["char"] for f in model.form_line(p)]
    assert chars == [model.FORM_DECLINED, model.FORM_MONTH,
                     model.FORM_DECLINED]


def test_the_record_reads_oldest_first():
    p = book(boarded=[verdict(home="Late", day="2026-08-30"),
                      verdict(home="Early", day="2026-08-20")])
    details = [f["detail"] for f in model.form_line(p)]
    assert "Early" in details[0] and "Late" in details[-1]


def test_a_board_reason_is_trimmed_on_a_word_boundary():
    """A readout cut mid-word looks like a bug rather than a measurement."""
    long = "Hoffenheim projects favorably for over 2.5 away goals because " \
           "their expected goals against sits well above the league median"
    p = book(boarded=[verdict(reason=long)])
    detail = model.form_line(p)[0]["detail"]
    assert detail.endswith("…")
    assert not detail.rstrip("…").endswith(" ")
    assert " ".join(detail.rstrip("…").split()[-1:]) in long


# --- families -----------------------------------------------------------------
def test_a_team_total_is_not_filed_as_a_spread():
    """`home_over_1.5` and `home_wins_by_over_1.5` differ by one word and price
    completely differently."""
    p = book([pos("home_over_1.5"), pos("home_wins_by_over_1.5")])
    got = {(r["period"], r["family"]): r["n"] for r in model.claim_families(p)}
    assert got == {("FT", "team total"): 1, ("FT", "spread"): 1}


def test_first_half_markets_are_reported_apart_from_full_match():
    p = book([pos("1h_total_over_1.5"), pos("total_over_1.5")])
    got = {(r["period"], r["family"]) for r in model.claim_families(p)}
    assert got == {("1H", "total"), ("FT", "total")}


def test_a_negated_claim_files_with_the_claim_it_negates():
    p = book([pos("not_1h_draw")])
    assert model.claim_families(p)[0]["period"] == "1H"
    assert model.claim_families(p)[0]["family"] == "result"


# --- headline arithmetic ------------------------------------------------------
def test_a_unit_is_one_percent_of_the_starting_bankroll():
    assert model.unit_cents(book(start=100_000)) == 1000.0
    p = book([pos(pnl=7310)], start=100_000)
    assert model.pnl(p)["realized_units"] == pytest.approx(7.31)


def test_the_decline_rate_counts_every_boarded_fixture():
    p = book(boarded=[verdict(home=str(i)) for i in range(9)]
             + [verdict(home="z", action="PAPER_PLACE_LIMIT")])
    out = model.board(p)
    assert out["n_fixtures"] == 10 and out["n_declined"] == 9
    assert out["decline_rate"] == pytest.approx(0.9)


def test_the_fill_rate_ignores_orders_that_are_still_open():
    """An open order has not failed to fill -- it has not been decided."""
    p = book(orders=[{"status": "filled", "kind": "limit"},
                     {"status": "expired", "kind": "limit"},
                     {"status": "open", "kind": "limit"}])
    out = model.fills(p)
    assert out["n_resolved"] == 2 and out["rate"] == pytest.approx(0.5)
    assert out["n_open"] == 1


def test_signed_values_use_a_real_minus_not_a_hyphen():
    """U+2212 is the same width as + in a tabular face, so signed columns
    align. A hyphen is narrower and the column visibly steps."""
    assert model.signed_units(-1.5).startswith("−")
    assert model.signed_units(1.5).startswith("+")
    assert "-" not in model.signed_units(-1.5)


# --- the thin-sample cases that must not crash or lie --------------------------
def test_an_empty_book_reports_absent_rather_than_zero():
    """0% CLV is a claim about a measurement. No measurement is not."""
    empty = book()
    assert model.clv(empty)["mean_cents"] is None
    assert model.pnl(empty)["roi"] is None
    assert model.pnl(empty)["strike_rate"] is None
    assert model.fills(empty)["rate"] is None
    assert model.board(empty)["decline_rate"] is None
    assert model.form_line(empty) == []
    assert model.summary(empty)["fixtures"] == []


def test_a_position_with_no_closing_line_is_excluded_from_clv():
    p = book([pos(clv=None, pnl=100), pos(home="C", clv=3.0, pnl=100)])
    out = model.clv(p)
    assert out["n_fixtures"] == 1 and out["mean_cents"] == pytest.approx(3.0)


def test_the_summary_carries_every_section_the_page_renders():
    keys = set(model.summary(book()))
    assert {"clv", "pnl", "fills", "board", "fixtures", "form", "leagues",
            "families", "equity", "daily", "unit_cents"} <= keys
