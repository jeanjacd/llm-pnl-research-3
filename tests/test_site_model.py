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


def test_a_losing_fixture_reads_as_the_share_of_stake_it_returned():
    """The digit separates a graze from a wipeout, which is the question a
    reader has about a fixture that finished down."""
    # two markets, $20 staked each; lost $10 of each -> $20 back of $40 -> 5
    p = book([pos("draw", pnl=-1000), pos("btts", pnl=-1000)])
    assert [f["char"] for f in model.form_line(p)] == ["5"]
    # same book, nearly all of it lost -> 0
    q = book([pos("draw", pnl=-1900), pos("btts", pnl=-1900)])
    assert [f["char"] for f in model.form_line(q)] == ["0"]


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
def test_money_is_reported_in_cents_and_never_in_units():
    """Units keep a record readable across a bankroll that MOVES. Every
    position here is sized against the starting balance, so a unit was a flat
    $10 -- a second representation carrying no second fact."""
    p = book([pos(pnl=7310)], start=100_000)
    out = model.pnl(p)
    assert out["realized_cents"] == pytest.approx(7310.0)
    assert not [k for k in out if k.endswith("_units")]
    assert not hasattr(model, "unit_cents")


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
    assert model.signed_money(-150) == "−$1.50"
    assert model.signed_money(150) == "+$1.50"
    assert "-" not in model.signed_money(-150)


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
            "families", "equity", "daily"} <= keys


# ── the form figure has to carry a denominator ────────────────────────────────
def order(status="expired", league="mls", home="A", away="B",
          kickoff="2026-08-28"):
    return {"status": status, "kind": "limit", "league_id": league,
            "home_team": home, "away_team": away,
            "kickoff_utc": kickoff + "T18:00:00"}


def test_the_digit_is_a_share_of_stake_and_not_a_raw_count():
    """A count compared nothing to nothing: the denominator ran from 1 market
    to 76, so 23-of-53 and 29-of-76 both saturated the cap and printed `9`."""
    # 10 markets at 50c: 8 win (8 x 50c profit), 2 lose (2 x 50c). Staked 500c,
    # returned 500 + 400 - 100 = 800c -> 8 tenths... but that is a WIN, so use
    # a losing book: 2 win, 8 lose -> staked 500, pnl = 100 - 400 = -300,
    # returned 200 of 500 = 4 tenths.
    held = [pos(claim="score_%d-0" % i, pnl=(50 if i < 2 else -50), cost=50.0,
                size=1.0) for i in range(10)]
    for i, p in enumerate(held):
        p["instrument_id"] = "i%d" % i
    p = book(held)
    assert [f["char"] for f in model.form_line(p)] == ["4"]


def test_a_near_total_loss_reads_as_zero_however_many_markets_cashed():
    """Celta Vigo returned $0.29 of $42.05 with one market cashing, and the
    old notation printed `1` -- indistinguishable from a graze."""
    held = [pos(claim="a", pnl=20, cost=1.0, size=1.0),
            pos(claim="b", pnl=-4000, cost=40.0, size=100.0)]
    held[0]["instrument_id"], held[1]["instrument_id"] = "a", "b"
    assert [f["char"] for f in model.form_line(book(held))] == ["0"]


def test_a_graze_and_a_wipeout_no_longer_print_the_same_character():
    graze = [pos(claim="x", pnl=-100, cost=50.0, size=20.0)]
    wipe = [pos(claim="x", pnl=-1000, cost=50.0, size=20.0)]
    a = model.form_line(book(graze))[0]["char"]
    b = model.form_line(book(wipe))[0]["char"]
    assert a != b and a > b


# ── ordering and filling nothing is not declining ─────────────────────────────
def test_a_fixture_that_filled_nothing_is_not_shown_as_declined():
    """Leeds v Brentford placed 130 orders and filled none. The record printed
    `·` -- the same character as a board that passed on the fixture."""
    p = book(boarded=[verdict(home="Leeds", away="Brentford",
                              action="PAPER_PLACE_LIMIT")],
             orders=[order(home="Leeds", away="Brentford") for _ in range(130)])
    row = model.fixtures(p)[0]
    assert row["ordered"] and not row["acted"]
    assert not row["declined"], "the market declined, not the board"
    assert model.form_figure(row) == model.FORM_UNFILLED
    assert "filled none" in model.form_line(p)[0]["detail"]


def test_a_board_that_passed_is_still_shown_as_declined():
    p = book(boarded=[verdict()])
    row = model.fixtures(p)[0]
    assert row["declined"] and not row["ordered"]
    assert model.form_figure(row) == model.FORM_DECLINED


def test_a_fixture_counts_the_orders_that_never_filled():
    """39% of resting orders fill. A ticket showing only what filled presents
    a fifth of the attempt as the whole of it."""
    p = book([pos(claim="draw", pnl=100)],
             orders=[order(status="filled"), order(status="expired"),
                     order(status="expired"), order(status="open")])
    row = model.fixtures(p)[0]
    assert row["n_unfilled"] == 2
    assert row["fill_rate"] == pytest.approx(1 / 3), "open orders are undecided"


def test_an_unfilled_order_is_never_counted_as_a_settled_loss():
    """The strike rate is computed over POSITIONS, and a position exists only
    on a fill, so a missed order cannot depress it."""
    p = book([pos(claim="draw", pnl=500)],
             orders=[order(status="expired") for _ in range(50)])
    out = model.pnl(p)
    assert out["n_settled"] == 1 and out["strike_rate"] == pytest.approx(1.0)


# ── the cascade ───────────────────────────────────────────────────────────────
def test_the_ladder_only_ever_narrows():
    """A funnel whose third rung is wider than its second invites the reader to
    distrust the whole thing. `markets_approved` records only each fixture's
    LAST sitting while orders span every attempt, so it read 85 against 1,327
    offered -- it is left out rather than shown with an apology."""
    p = book([pos(claim="draw", pnl=100)],
             [verdict(considered=120)],
             orders=[order(status="filled")] + [order() for _ in range(9)])
    counts = [r["n"] for r in model.funnel(p)]
    assert counts == sorted(counts, reverse=True), counts


def test_every_rung_of_the_ladder_counts_the_same_thing():
    """A fixture is not a subset of a market, so a 45-long bar under a
    5,107-long one compares nothing to nothing."""
    stages = {r["stage"] for r in model.funnel(book())}
    assert stages == {"read", "offered", "filled", "settled"}
    assert not any("fixture" in s for s in stages)


def test_the_ladder_survives_an_empty_book():
    assert [r["n"] for r in model.funnel(book())] == [0, 0, 0, 0]


# ── the closing line reaches the signature element ────────────────────────────
def test_a_form_figure_carries_the_closing_line_it_was_measured_at():
    """The band said what HAPPENED and never what the closing line thought of
    it, so the page's own leading indicator was absent from the one display
    everybody reads first."""
    p = book([pos(claim="draw", pnl=-500, clv=2.5)])
    entry = model.form_line(p)[0]
    assert entry["clv_cents"] == pytest.approx(2.5)
    assert entry["staked_cents"] > 0


def test_a_figure_with_no_closing_line_carries_none_rather_than_zero():
    p = book(boarded=[verdict()])
    assert model.form_line(p)[0]["clv_cents"] is None
