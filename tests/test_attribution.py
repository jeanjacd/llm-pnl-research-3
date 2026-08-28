"""Forecast and spread capture must never be summed into one number.

The board rests both sides of a claim on purpose -- if both fill it is locked
profit. But an offsetting pair carries no view, and its CLV is a constant that
would read as edge. These tests pin the decomposition, including the case
where netting must be REFUSED because the two contracts may not settle alike.
"""
import pytest

from wc2026.paper.attribution import decompose
from wc2026.paper.broker import PaperPortfolio, PaperPosition


def a_portfolio(tmp_path):
    return PaperPortfolio(starting_cash_cents=100_000, cash_cents=100_000,
                          path=str(tmp_path / "p.json"))


def hold(portfolio, claim, side, size, cost, venue="kalshi",
         regulation=True, home="A", away="B", ident=None):
    ident = ident or "%s-%s-%s" % (venue, claim, side)
    pos = PaperPosition(position_id=ident, venue=venue, instrument_id=ident,
                        side=side, size=float(size), avg_cost_cents=float(cost),
                        fees_cents=0, league_id="mls", claim=claim,
                        home_team=home, away_team=away,
                        kickoff_utc="2026-08-28T19:00:00+00:00",
                        settles_on_regulation=regulation)
    portfolio.positions["%s|%s" % (ident, side)] = pos
    return pos


def only(result):
    assert len(result) == 1
    return next(iter(result.values()))


# --- the arithmetic the whole thing rests on ----------------------------------
def test_an_offsetting_pair_is_spread_capture_not_forecast(tmp_path):
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 10, 19)
    hold(p, "draw", "no", 10, 65)
    out = only(decompose(p))
    assert out["matched"] == 10
    assert out["residual"] == 0, "an offsetting pair expresses no view"
    # 10 pairs, each paying 100c for 84c.
    assert out["spread_pnl_cents"] == pytest.approx(160.0)


def test_the_locked_profit_does_not_depend_on_the_outcome(tmp_path):
    """This is why it cannot sit in the forecast number: c cancels."""
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 10, 19)
    hold(p, "draw", "no", 10, 65)
    locked = only(decompose(p))["spread_pnl_cents"]
    for closing in (5, 19, 50, 84, 99):
        clv_yes = closing - 19
        clv_no = (100 - closing) - 65
        assert (clv_yes + clv_no) * 10 == pytest.approx(locked)


def test_an_uneven_pair_splits_into_both_components(tmp_path):
    p = a_portfolio(tmp_path)
    hold(p, "home_win", "yes", 10, 64)
    hold(p, "home_win", "no", 4, 20)
    out = only(decompose(p))
    assert out["matched"] == 4
    assert out["spread_pnl_cents"] == pytest.approx(4 * 100 - 4 * 84)
    assert out["residual"] == 6 and out["residual_side"] == "yes"


def test_a_lone_position_is_entirely_forecast(tmp_path):
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 10, 19)
    out = only(decompose(p))
    assert out["matched"] == 0 and out["residual"] == 10
    assert out["spread_pnl_cents"] == 0.0


# --- the fold that is easy to get wrong ---------------------------------------
def test_no_on_a_claim_and_yes_on_its_negation_are_the_same_position(tmp_path):
    """Holding `no` on draw and `yes` on not_draw is one economic position.
    Folding them apart would make an offsetting pair look like two bets the
    same way, and net nothing."""
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 10, 19, ident="a")
    hold(p, "not_draw", "yes", 10, 65, ident="b")
    out = only(decompose(p))
    assert out["matched"] == 10, "these offset and must net"
    assert out["residual"] == 0


def test_two_bets_the_same_way_never_net(tmp_path):
    """`yes` on draw and `no` on not_draw are the SAME direction."""
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 10, 19, ident="a")
    hold(p, "not_draw", "no", 10, 21, ident="b")
    out = only(decompose(p))
    assert out["matched"] == 0
    assert out["residual"] == 20 and out["residual_side"] == "yes"


# --- the refusal that keeps a fake riskless profit off the books --------------
def test_cross_venue_pairs_net_when_both_settle_on_regulation(tmp_path):
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 10, 19, venue="kalshi", regulation=True)
    hold(p, "draw", "no", 10, 65, venue="polymarket", regulation=True)
    out = only(decompose(p))
    assert out["matched"] == 10 and out["unmatched_reason"] is None


def test_an_unknown_settlement_basis_refuses_the_match(tmp_path):
    """"Probably the same rules" is what produces a losing trade. An
    un-nettable pair is live risk and stays in the forecast bucket."""
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 10, 19, venue="kalshi", regulation=True)
    hold(p, "draw", "no", 10, 65, venue="polymarket", regulation=None)
    out = only(decompose(p))
    assert out["matched"] == 0, "must not book a riskless profit that is not"
    assert out["residual"] == 20
    assert out["unmatched_reason"]


def test_differing_settlement_bases_refuse_the_match(tmp_path):
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 10, 19, venue="kalshi", regulation=True)
    hold(p, "draw", "no", 10, 65, venue="polymarket", regulation=False)
    out = only(decompose(p))
    assert out["matched"] == 0 and out["unmatched_reason"]


def test_a_same_venue_pair_always_nets_regardless_of_recorded_basis(tmp_path):
    """One market, one settlement rule -- there is nothing to disagree about."""
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 10, 19, regulation=None, ident="a")
    hold(p, "draw", "no", 10, 65, regulation=None, ident="b")
    assert only(decompose(p))["matched"] == 10


# --- bookkeeping invariants ---------------------------------------------------
def test_contracts_are_conserved(tmp_path):
    """Nothing may be created or destroyed by the decomposition."""
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 7, 19, ident="a")
    hold(p, "draw", "no", 4, 65, ident="b")
    hold(p, "btts", "yes", 3, 40, ident="c")
    held = sum(pos.size for pos in p.positions.values())
    out = decompose(p)
    accounted = sum(v["matched"] * 2 + v["residual"] for v in out.values())
    assert accounted == pytest.approx(held)


def test_distinct_fixtures_and_claims_never_net_against_each_other(tmp_path):
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 10, 19, home="A", away="B", ident="a")
    hold(p, "draw", "no", 10, 65, home="C", away="D", ident="b")
    out = decompose(p)
    assert len(out) == 2
    assert all(v["matched"] == 0 for v in out.values())


def test_an_empty_book_decomposes_to_nothing(tmp_path):
    assert decompose(a_portfolio(tmp_path)) == {}


def test_a_position_with_no_claim_is_left_out_rather_than_guessed(tmp_path):
    p = a_portfolio(tmp_path)
    pos = hold(p, "draw", "yes", 10, 19)
    pos.claim = None
    assert decompose(p) == {}


# --- attribution of realised money -------------------------------------------
def settle(portfolio, position, pnl_cents):
    position.settled = True
    portfolio.ledger.append({"instrument_id": position.instrument_id,
                             "side": position.side,
                             "pnl_cents": float(pnl_cents),
                             "won": pnl_cents > 0})


def test_forecast_plus_spread_always_equals_the_realised_total(tmp_path):
    """THE invariant. The split may neither lose money nor invent it, and
    nothing downstream would catch a discrepancy."""
    from wc2026.paper.attribution import attribute
    p = a_portfolio(tmp_path)
    a = hold(p, "draw", "yes", 10, 19, ident="a")
    b = hold(p, "draw", "no", 4, 65, ident="b")
    c = hold(p, "btts", "yes", 6, 40, ident="c")
    settle(p, a, 810.0)
    settle(p, b, -260.0)
    settle(p, c, -240.0)
    out = attribute(p)
    for bucket in out.values():
        assert bucket["forecast_pnl_cents"] + bucket["spread_pnl_cents"] == \
            pytest.approx(bucket["total_pnl_cents"])
    total = sum(v["total_pnl_cents"] for v in out.values())
    assert total == pytest.approx(810.0 - 260.0 - 240.0)


def test_a_fully_offsetting_pair_contributes_nothing_to_forecast(tmp_path):
    from wc2026.paper.attribution import attribute
    p = a_portfolio(tmp_path)
    a = hold(p, "draw", "yes", 10, 19, ident="a")
    b = hold(p, "draw", "no", 10, 65, ident="b")
    settle(p, a, 810.0)
    settle(p, b, -650.0)
    bucket = next(iter(attribute(p).values()))
    assert bucket["forecast_pnl_cents"] == pytest.approx(0.0)
    assert bucket["spread_pnl_cents"] == pytest.approx(160.0)
    assert bucket["total_pnl_cents"] == pytest.approx(160.0)


def test_a_lone_position_is_entirely_forecast_pnl(tmp_path):
    from wc2026.paper.attribution import attribute
    p = a_portfolio(tmp_path)
    a = hold(p, "draw", "yes", 10, 19)
    settle(p, a, 810.0)
    bucket = next(iter(attribute(p).values()))
    assert bucket["forecast_pnl_cents"] == pytest.approx(810.0)
    assert bucket["spread_pnl_cents"] == pytest.approx(0.0)


def test_a_partly_netted_position_splits_its_pnl_by_size(tmp_path):
    """10 held, 4 netted -> 60% of that leg's money is forecast."""
    from wc2026.paper.attribution import attribute
    p = a_portfolio(tmp_path)
    a = hold(p, "home_win", "yes", 10, 64, ident="a")
    b = hold(p, "home_win", "no", 4, 20, ident="b")
    settle(p, a, 360.0)
    settle(p, b, -80.0)
    bucket = next(iter(attribute(p).values()))
    assert bucket["forecast_pnl_cents"] == pytest.approx(360.0 * 0.6)
    assert bucket["spread_pnl_cents"] == pytest.approx(360.0 * 0.4 - 80.0)
    assert bucket["forecast_contracts"] == pytest.approx(6.0)


# --- forecast-only CLV --------------------------------------------------------
def test_an_offsetting_pair_is_excluded_from_forecast_clv(tmp_path):
    """Its CLV is a constant whatever the closing price -- reporting it would
    present captured spread as forecasting skill."""
    from wc2026.paper.attribution import forecast_clv
    p = a_portfolio(tmp_path)
    a = hold(p, "draw", "yes", 10, 19, ident="a")
    b = hold(p, "draw", "no", 10, 65, ident="b")
    a.clv_cents, b.clv_cents = 6.0, 10.0
    assert forecast_clv(p) == []


def test_only_the_directional_remainder_is_scored(tmp_path):
    from wc2026.paper.attribution import forecast_clv
    p = a_portfolio(tmp_path)
    a = hold(p, "home_win", "yes", 10, 64, ident="a")
    b = hold(p, "home_win", "no", 4, 20, ident="b")
    a.clv_cents, b.clv_cents = 5.0, -2.0
    rows = forecast_clv(p)
    assert len(rows) == 1
    _, clv, contracts = rows[0]
    assert clv == pytest.approx(5.0) and contracts == pytest.approx(6.0)


def test_a_position_with_no_closing_line_yet_is_absent_not_zero(tmp_path):
    from wc2026.paper.attribution import forecast_clv
    p = a_portfolio(tmp_path)
    hold(p, "draw", "yes", 10, 19)
    assert forecast_clv(p) == []
