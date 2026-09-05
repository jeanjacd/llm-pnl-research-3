"""Withdrawing a position from the measurement without hiding it.

The case this exists for: on Real Betis v Real Madrid the Kalshi market
`...-RMA` -- "Real Madrid wins 1st Half" -- was recorded as `1h_home_win` and
priced with P(Betis win the half). The trade returned $6.01, and that $6.01 is
not evidence about the model in either direction, because the model never
evaluated the proposition it was bought on.

A book with only "keep it" and "delete it" has to choose between a rate
computed over a non-observation and a record that disagrees with what
happened. Both are worse than saying so.
"""
import datetime as dt

import pytest
from paper_fixtures import book, pos

from wc2026.paper.void import VoidError, apply, plan
from wc2026.site import ledger, model

NOW = dt.datetime(2026, 9, 5, 12, 0, tzinfo=dt.timezone.utc)


def held(pid, pnl=601, size=18.0, cost=65.0, claim="not_1h_home_win",
         case="kalshi:RMA:no", fees=29, clv=-16.5):
    out = pos(claim=claim, home="Real Betis", away="Real Madrid",
              league="la_liga", kickoff="2026-09-04", pnl=pnl, clv=clv,
              size=size, cost=cost)
    out.update({"position_id": pid, "case_id": case, "fees_cents": fees,
                "payout_cents": (size * 100.0) if pnl > 0 else 0.0,
                "instrument_id": "KXLALIGA1H-26SEP04RBBRMA-RMA"})
    return out


def a_book():
    p = book([held("bad"),
              pos(claim="1h_draw", home="Real Betis", away="Real Madrid",
                  league="la_liga", kickoff="2026-09-04", pnl=2592, clv=0.5,
                  size=39.0, cost=32.0)])
    for i, position in enumerate(p["positions"].values()):
        position.setdefault("position_id", "p%d" % i)
    p["orders"] = {"o1": {"order_id": "o1", "case_id": "kalshi:RMA:no",
                          "status": "filled", "kind": "limit",
                          "limit_price_cents": 65, "requested_size": 18.0,
                          "claim": "not_1h_home_win", "league_id": "la_liga",
                          "home_team": "Real Betis", "away_team": "Real Madrid",
                          "kickoff_utc": "2026-09-04T19:00:00"}}
    p["ledger"] = [{"ts": "2026-09-04T22:07:01+00:00", "case_id": "kalshi:RMA:no",
                    "instrument_id": "KXLALIGA1H-26SEP04RBBRMA-RMA",
                    "side": "no", "pnl_cents": 601.0, "won": True}]
    p["cash_cents"] = 100_000
    return p


# ── the operation ─────────────────────────────────────────────────────────────
def test_a_void_needs_a_reason():
    """The reason is the whole point of it: it is what the page prints, and
    what distinguishes this from tidying."""
    with pytest.raises(VoidError):
        apply(a_book(), ["bad"], "")
    with pytest.raises(VoidError):
        apply(a_book(), ["bad"], "   ")


def test_an_unknown_position_raises_rather_than_voiding_the_rest():
    """Never applied partially. A repair that half-lands is worse than one
    that does not run."""
    p = a_book()
    with pytest.raises(VoidError):
        apply(p, ["bad", "nonexistent"], "because")
    assert not any(x.get("void") for x in p["positions"].values())
    assert p["cash_cents"] == 100_000


def test_the_position_stays_on_the_record_and_gains_a_reason():
    p = a_book()
    apply(p, ["bad"], "wrong proposition", now=NOW)
    hit = [x for x in p["positions"].values() if x.get("position_id") == "bad"]
    assert len(hit) == 1, "voiding is not deleting"
    assert hit[0]["void"] is True
    assert hit[0]["void_reason"] == "wrong proposition"
    assert hit[0]["voided_at"] == NOW.isoformat()
    # and every figure it had is still there
    assert hit[0]["avg_cost_cents"] == 65.0 and hit[0]["clv_cents"] == -16.5


def test_the_stake_comes_back():
    """A void means the bet is cancelled and the stake returned. Cash ends
    where it would have been had the order never been placed: 18 at 65c is
    $11.70 out, 29c of fees, $18.00 back -- so $6.01 comes off."""
    p = a_book()
    report = apply(p, ["bad"], "wrong proposition")
    assert report["cash_delta_cents"] == -601
    assert p["cash_cents"] == 100_000 - 601


def test_the_order_and_the_ledger_line_go_with_it():
    p = a_book()
    apply(p, ["bad"], "wrong proposition")
    assert p["orders"]["o1"]["void"] is True
    assert p["ledger"][0]["void"] is True


def test_voiding_twice_is_not_charged_twice():
    p = a_book()
    apply(p, ["bad"], "wrong proposition")
    again = apply(p, ["bad"], "wrong proposition")
    assert again["steps"] == [] and again["already_void"] == ["bad"]
    assert p["cash_cents"] == 100_000 - 601


def test_plan_writes_nothing():
    p = a_book()
    report = plan(p, ["bad"], "wrong proposition")
    assert report["cash_delta_cents"] == -601
    assert p["cash_cents"] == 100_000
    assert not any(x.get("void") for x in p["positions"].values())


# ── what stops counting it ────────────────────────────────────────────────────
def test_nothing_that_measures_counts_a_void():
    p = a_book()
    before = model.pnl(p)
    apply(p, ["bad"], "wrong proposition")
    after = model.pnl(p)
    assert before["n_settled"] - after["n_settled"] == 1
    assert before["n_won"] - after["n_won"] == 1
    assert after["realized_cents"] == pytest.approx(
        before["realized_cents"] - 601)
    # the stake was returned, so it was never at risk
    assert after["staked_cents"] == pytest.approx(
        before["staked_cents"] - 18 * 65)


def test_a_void_is_dropped_from_the_closing_line_too():
    """It cuts both ways, which is the check that this is not a way of
    improving a number: the voided position had the WORST closing line on the
    fixture, so removing it makes CLV look better and P&L look worse."""
    p = a_book()
    clv_before = model.clv(p)["mean_cents"]
    pnl_before = model.pnl(p)["realized_cents"]
    apply(p, ["bad"], "wrong proposition")
    assert model.clv(p)["mean_cents"] > clv_before, "the closing line improves"
    assert model.pnl(p)["realized_cents"] < pnl_before, "and the money worsens"


def test_the_curves_skip_a_voided_ledger_line():
    p = a_book()
    apply(p, ["bad"], "wrong proposition")
    assert model.equity_curve(p) == []
    assert model.daily_ledger(p) == []


def test_the_league_and_family_tables_skip_it():
    p = a_book()
    apply(p, ["bad"], "wrong proposition")
    counted = sum(r["n_markets"] for r in model.by_league(p))
    assert counted == 1
    assert sum(r["n"] for r in model.claim_families(p)) == 1


# ── what still shows it ───────────────────────────────────────────────────────
def test_the_archive_still_carries_it_with_its_reason():
    """A record that quietly drops its own mistakes is worth less than one
    that never had any."""
    p = a_book()
    apply(p, ["bad"], "wrong proposition")
    row = ledger.settled_fixtures(p)[0]
    claims = {m["claim"]: m for m in row["markets"]}
    assert set(claims) == {"1h_draw", "not_1h_home_win"}
    assert claims["not_1h_home_win"]["void"] is True
    assert claims["not_1h_home_win"]["void_reason"] == "wrong proposition"
    assert claims["not_1h_home_win"]["won"] is False, "neither won nor lost"
    # and the fixture's own total is over the markets that count
    assert row["n_markets"] == 1 and row["n_void"] == 1
    assert row["pnl_cents"] == pytest.approx(2592)


def test_a_voided_market_is_printed_on_the_fixture_page():
    from wc2026.site import archive
    p = a_book()
    apply(p, ["bad"], "the market was Real Madrid, the price was Real Betis")
    page = [v for k, v in archive.pages(p).items() if "index" not in k][0]
    assert "1st half Real Betis do not win" in page
    assert "Void" in page
    assert "the market was Real Madrid, the price was Real Betis" in page


def test_a_fixture_whose_only_settled_market_was_voided_reads_as_voided():
    p = book([held("bad")])
    for position in p["positions"].values():
        position["position_id"] = "bad"
    p["cash_cents"] = 100_000
    apply(p, ["bad"], "wrong proposition")
    col = ledger.columns(p, now=NOW)[0]
    assert col["outcome"] == "void"
    assert col["n_void"] == 1 and col["n_settled"] == 0


# ── it survives a rebuild ─────────────────────────────────────────────────────
def test_a_rebuild_does_not_raise_a_voided_order_from_the_dead():
    """`paper.rebuild` replays order INTENT and knows nothing about what was
    later found wrong with it, so the void has to travel with the intent."""
    from wc2026.paper.broker import PaperPortfolio
    from wc2026.paper.rebuild import order_intent, resubmit

    p = a_book()
    apply(p, ["bad"], "wrong proposition")
    intents = [order_intent(o) for o in p["orders"].values()]
    assert intents[0]["void"] is True

    fresh = PaperPortfolio(starting_cash_cents=100_000, cash_cents=100_000,
                           path=None)
    stats = resubmit(fresh, intents)
    assert stats["voided"] == 1 and stats["submitted"] == 0
    assert fresh.orders == {}
