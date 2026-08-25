"""Deterministic decision layer: EV, limit ladders, action states and joint
combo valuation. Offline."""
import datetime as dt

import numpy as np
import pytest

from wc2026.decision import (
    BUY_NOW,
    DEFER,
    PASS,
    PLACE_LIMIT,
    UNSUPPORTED,
    WAIT_FOR_QUOTE,
    DependencyError,
    LegClaim,
    build_case,
    bundle_outcomes,
    calculator,
    combo_expected_payout,
    dependence_ratio,
    ev_at_price,
    fee_cents,
    joint_probability,
    naive_independent_probability,
)
from wc2026.decision.calculator import CalcConfig, adverse_selection
from wc2026.sim.match import score_matrix
from wc2026.venues.base import (
    KIND_BINARY,
    Book,
    Leg,
    MarketInstrument,
    utcnow_iso,
)


def soon(hours=6):
    return (dt.datetime.now(dt.timezone.utc)
            + dt.timedelta(hours=hours)).isoformat()


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def inst(claim="home_win", yes_asks=((44, 500.0),), yes_bids=((43, 500.0),),
         venue="kalshi", regulation=True, kickoff=None, tick=1,
         observed=None, supported=True):
    leg = (Leg.build(claim, "ref") if supported else
           Leg(claim=claim, market_ref="ref", supported=False,
               unsupported_reason="no validated model"))
    return MarketInstrument(
        venue=venue, instrument_id="INST", kind=KIND_BINARY, title="t",
        legs=(leg,), settles_on_regulation=regulation,
        kickoff_utc=kickoff or soon(), tick_cents=tick,
        fee_model={"venue": venue},
        book=Book(yes_asks=tuple(yes_asks), yes_bids=tuple(yes_bids),
                  observed_at=observed or now_iso()))


# --------------------------------------------------------------------------- #
# fees
# --------------------------------------------------------------------------- #
def test_kalshi_fee_matches_the_published_schedule():
    assert fee_cents("kalshi", 100, 50) == 175
    assert fee_cents("kalshi", 1, 50) == 2
    assert fee_cents("kalshi", 100, 99) == 7


def test_polymarket_zero_base_fee_is_zero_but_unknown_venue_is_conservative():
    assert fee_cents("polymarket", 100, 50, {"taker_base_fee": 0}) == 0
    # an unknown venue must not be assumed free
    assert fee_cents("mystery_venue", 100, 50) == 175


def test_ev_subtracts_fee_and_reserve():
    ev, roi, fee = ev_at_price(0.60, 50, "kalshi", 100, 1.0)
    assert ev == pytest.approx(60 - 50 - 1.75 - 1.0)
    assert roi == pytest.approx(ev / (50 + 1.75))


def test_adverse_selection_grows_with_spread_and_time_and_is_capped():
    cfg = CalcConfig()
    assert adverse_selection(cfg, 1, 1) < adverse_selection(cfg, 9, 1)
    assert adverse_selection(cfg, 1, 1) < adverse_selection(cfg, 1, 90)
    assert adverse_selection(cfg, 50, 500) <= cfg.adverse_selection_max_cents


# --------------------------------------------------------------------------- #
# action states
# --------------------------------------------------------------------------- #
def test_unsupported_instrument_is_never_priced():
    case = build_case(inst(claim="player_goal", supported=False), 0.9, 0.9)
    assert case.action == UNSUPPORTED and case.ev_per_contract is None


def test_non_regulation_settlement_is_unsupported():
    case = build_case(inst(regulation=False), 0.9, 0.9)
    assert case.action == UNSUPPORTED


def test_missing_model_probability_defers():
    assert build_case(inst(), None, None).action == DEFER


def test_no_executable_quote_waits_rather_than_passing():
    """An attractive valuation with an empty book is WAIT_FOR_QUOTE: the
    candidate must not be thrown away."""
    case = build_case(inst(yes_asks=(), yes_bids=()), 0.80, 0.80)
    assert case.action == WAIT_FOR_QUOTE


def test_stale_book_defers():
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).isoformat()
    assert build_case(inst(observed=old), 0.8, 0.8).action == DEFER


def test_kickoff_passed_defers():
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
    assert build_case(inst(kickoff=past), 0.8, 0.8).action == DEFER


def test_far_future_kickoff_passes():
    far = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=30)).isoformat()
    assert build_case(inst(kickoff=far), 0.8, 0.8).action == PASS


def test_clear_edge_at_the_touch_is_buy_now():
    case = build_case(inst(yes_asks=((40, 500.0),), yes_bids=((39, 500.0),)),
                      0.70, 0.70)
    assert case.action == BUY_NOW
    assert case.ev_per_contract > 0 and case.roi > 0


def test_no_edge_at_any_price_is_pass():
    case = build_case(inst(yes_asks=((95, 500.0),), yes_bids=((94, 500.0),)),
                      0.10, 0.10)
    # a ~4c fair value against a 94c bid is not realistically obtainable
    assert case.action == PASS
    assert "not realistically obtainable" in " ".join(case.reasons)


# --------------------------------------------------------------------------- #
# the limit ladder -- the heart of PLACE_LIMIT
# --------------------------------------------------------------------------- #
def test_unattractive_quote_still_yields_a_conditional_limit():
    """The point of the ladder: a slightly rich ask must become PLACE_LIMIT
    with a usable maximum price, not PASS.

    The disagreement here is a few cents -- the realistic case. A market
    trading 10+ points away from the model is treated as a PASS instead (see
    test_large_disagreement_is_a_pass), because at that distance a fill
    requires the market to be badly wrong rather than merely drifting.
    """
    case = build_case(inst(yes_asks=((55, 500.0),), yes_bids=((54, 500.0),)),
                      0.60, 0.60)
    assert case.action == PLACE_LIMIT
    assert case.max_limit_price_cents is not None
    assert case.max_limit_price_cents < 55
    assert case.ladder and all(r.price_cents <= case.max_limit_price_cents
                               for r in case.ladder)


def test_large_disagreement_is_a_pass():
    """A limit far below the resting bid needs the market to be badly wrong,
    not merely to drift. That is a PASS, not a standing order."""
    case = build_case(inst(yes_asks=((70, 500.0),), yes_bids=((69, 500.0),)),
                      0.60, 0.60)
    assert case.action == PASS
    assert "not realistically obtainable" in " ".join(case.reasons)


def test_absurdly_distant_limit_is_a_pass_not_a_limit_order():
    """A resting buy 90c below the bid is arithmetically +EV and will never
    fill; the realism gate must catch it."""
    case = build_case(inst(yes_asks=((99, 500.0),), yes_bids=((98, 500.0),)),
                      0.05, 0.05)
    assert case.action == PASS


def test_every_rung_is_positive_ev_and_ordered():
    case = build_case(inst(yes_asks=((55, 500.0),), yes_bids=((54, 500.0),)),
                      0.60, 0.60)
    prices = [r.price_cents for r in case.ladder]
    assert prices == sorted(prices)
    assert all(r.ev_per_contract > 0 for r in case.ladder)
    # cheaper rungs earn more
    assert case.ladder[0].ev_per_contract > case.ladder[-1].ev_per_contract


def test_ladder_reports_the_size_available_at_each_price():
    book_asks = ((60, 10.0), (65, 40.0), (70, 900.0))
    case = build_case(inst(yes_asks=book_asks, yes_bids=((59, 50.0),)),
                      0.85, 0.85)
    rung60 = next(r for r in case.ladder if r.price_cents == 60)
    rung65 = next(r for r in case.ladder if r.price_cents == 65)
    assert rung60.size_available == 10.0
    assert rung65.size_available == 50.0        # cumulative through 65c


def test_limit_price_never_exceeds_the_deterministic_ceiling():
    """No rung may price above the maximum acceptable limit."""
    case = build_case(inst(yes_asks=((55, 500.0),), yes_bids=((54, 500.0),)),
                      0.60, 0.60)
    assert max(r.price_cents for r in case.ladder) == case.max_limit_price_cents


def test_wide_spread_refuses_to_cross_and_becomes_a_limit():
    case = build_case(inst(yes_asks=((40, 500.0),), yes_bids=((10, 500.0),)),
                      0.80, 0.80)
    assert case.action == PLACE_LIMIT
    assert "spread" in " ".join(case.reasons)


def test_thin_touch_depth_does_not_buy_now():
    case = build_case(inst(yes_asks=((40, 1.0), (41, 900.0)),
                           yes_bids=((39, 500.0),)), 0.70, 0.70)
    assert case.action == PLACE_LIMIT


# --------------------------------------------------------------------------- #
# book walking feeds EV
# --------------------------------------------------------------------------- #
def test_walking_the_book_raises_the_effective_price():
    """Size beyond the touch must cost more, and that must reduce EV."""
    shallow = build_case(inst(yes_asks=((40, 5.0), (60, 5000.0)),
                              yes_bids=((39, 100.0),)), 0.70, 0.70)
    deep = build_case(inst(yes_asks=((40, 5000.0),), yes_bids=((39, 100.0),)),
                      0.70, 0.70)
    assert shallow.executable_avg_cents > deep.executable_avg_cents
    assert shallow.slippage_cents > 0 and deep.slippage_cents == 0
    assert shallow.ev_per_contract < deep.ev_per_contract


# --------------------------------------------------------------------------- #
# conservative probability
# --------------------------------------------------------------------------- #
def test_value_uses_the_lower_bound_not_the_raw_probability():
    """A model known to be over-confident must not earn phantom edge."""
    optimistic = build_case(inst(yes_asks=((50, 500.0),),
                                 yes_bids=((49, 500.0),)), 0.75, 0.60)
    assert optimistic.p_lower < 0.60
    assert optimistic.ev_per_contract < 100 * 0.75 - 50


# --------------------------------------------------------------------------- #
# joint / combo valuation
# --------------------------------------------------------------------------- #
@pytest.fixture
def grid():
    return score_matrix(1.7, 1.2, -0.1, 12)


def _masks(g):
    k = np.arange(g.shape[0])
    i, j = np.meshgrid(k, k, indexing="ij")
    return i, j


def test_same_match_legs_are_not_multiplied(grid):
    i, j = _masks(grid)
    legs = [LegClaim("m1", i > j), LegClaim("m1", (i + j) > 2.5)]
    joint = joint_probability(legs, {"m1": grid})
    naive = naive_independent_probability(legs, {"m1": grid})
    assert joint != pytest.approx(naive, abs=1e-6)
    assert dependence_ratio(legs, {"m1": grid}) > 1.05   # materially dependent


def test_mutually_exclusive_legs_have_zero_joint(grid):
    i, j = _masks(grid)
    legs = [LegClaim("m1", i > j), LegClaim("m1", i < j)]
    assert joint_probability(legs, {"m1": grid}) == 0.0


def test_different_matches_multiply(grid):
    i, j = _masks(grid)
    legs = [LegClaim("m1", i > j), LegClaim("m2", i > j)]
    grids = {"m1": grid, "m2": grid}
    p1 = float(grid[i > j].sum())
    assert joint_probability(legs, grids) == pytest.approx(p1 * p1)


def test_missing_grid_refuses_to_approximate(grid):
    i, j = _masks(grid)
    legs = [LegClaim("m1", i > j), LegClaim("unknown", i > j)]
    with pytest.raises(DependencyError):
        joint_probability(legs, {"m1": grid})


def test_negated_leg_is_the_complement(grid):
    i, j = _masks(grid)
    yes = joint_probability([LegClaim("m1", i > j)], {"m1": grid})
    no = joint_probability([LegClaim("m1", i > j, negated=True)], {"m1": grid})
    assert yes + no == pytest.approx(1.0)


def test_combo_expected_payout_uses_the_joint(grid):
    i, j = _masks(grid)
    legs = [LegClaim("m1", i > j), LegClaim("m1", (i + j) > 2.5)]
    payout = combo_expected_payout(legs, {"m1": grid})
    assert payout == pytest.approx(100.0 * joint_probability(legs, {"m1": grid}))


def test_bundle_ev_is_additive_not_multiplicative(grid):
    """A separately executed bundle has no parlay payout: each leg settles on
    its own, so EV adds."""
    i, j = _masks(grid)
    legs = [LegClaim("m1", i > j), LegClaim("m2", (i + j) > 2.5)]
    grids = {"m1": grid, "m2": grid}
    out = bundle_outcomes(legs, grids, costs=[40.0, 50.0])
    assert out["kind"] == "bundle"
    assert out["total_ev_per_contract_set"] == pytest.approx(
        sum(l["ev_per_contract"] for l in out["legs"]))
    assert len(out["legs"]) == 2


def test_bundle_requires_a_cost_per_leg(grid):
    i, j = _masks(grid)
    with pytest.raises(DependencyError):
        bundle_outcomes([LegClaim("m1", i > j)], {"m1": grid}, costs=[])


# --- venue fee schedules ------------------------------------------------------
# Regression tests for a fee bug the live board's quant member caught: the
# Polymarket `base_fee` field is in BASIS POINTS and the fee formula is
# symmetric in price. An earlier version read the field as a fraction (10,000x
# too big) and a later one charged a flat rate on notional (5.6x too big at
# 23c), producing a breakeven "probability" of 2.56.
POLY_FEE = {"venue": "polymarket", "units": "basis_points",
            "taker_base_fee": 1000, "maker_base_fee": 0}


def test_polymarket_fee_is_basis_points_not_a_fraction():
    fee = calculator.fee_cents("polymarket", 100, 23, POLY_FEE)
    # 100 * 0.10 * 0.23 * 0.77 = 1.771 USD -> 178c after ceiling.
    assert fee == 178
    # The bug charged the full notional: 1000c on a 2300c position.
    assert fee < 23 * 100, "fee cannot approach the notional of the position"


def test_polymarket_fee_never_exceeds_the_published_ceiling_shape():
    """Max at 50c, symmetric about it, and monotone toward the wings."""
    fees = {p: calculator.fee_cents("polymarket", 100, p, POLY_FEE)
            for p in range(1, 100)}
    assert max(fees.values()) == fees[50] == 250
    for p in range(1, 50):
        assert fees[p] == fees[100 - p]
        assert fees[p] <= fees[p + 1]


def test_polymarket_fee_is_a_small_fraction_of_cost():
    """A sane fee is cents on a dollar, not multiples of the stake."""
    for price in (5, 23, 50, 77, 95):
        fee = calculator.fee_cents("polymarket", 100, price, POLY_FEE)
        assert fee / (price * 100) < 0.11


def test_polymarket_charges_makers_nothing():
    """Polymarket documents 'Makers are never charged fees'."""
    assert calculator.fee_cents("polymarket", 100, 23, POLY_FEE,
                                role="maker") == 0
    assert calculator.fee_cents("polymarket", 100, 23, POLY_FEE,
                                role="taker") > 0


def test_kalshi_ignores_liquidity_role():
    """Only Polymarket's schedule is role-dependent; do not leak the rebate."""
    assert (calculator.fee_cents("kalshi", 100, 50, None, role="maker")
            == calculator.fee_cents("kalshi", 100, 50, None, role="taker")
            == 175)


def test_zero_fee_model_is_free_but_unknown_venue_is_not():
    assert calculator.fee_cents("polymarket", 100, 50,
                                {"taker_base_fee": 0}) == 0
    # An unrecognised venue falls back to the most expensive known schedule
    # rather than being assumed free.
    assert calculator.fee_cents("some_new_venue", 100, 50, None) == 175


def test_breakeven_probability_stays_a_probability():
    """The bug's signature was breakeven_prob = 2.56 on a 23c contract."""
    for price in (5, 23, 50, 77, 95):
        fee = calculator.fee_cents("polymarket", 100, price, POLY_FEE)
        breakeven = (price + fee / 100) / 100.0
        assert 0.0 < breakeven < 1.0


def test_unknown_kickoff_defers_because_in_play_cannot_be_ruled_out():
    """A pre-match model must never price a quote it cannot date."""
    book = Book(observed_at=utcnow_iso(), yes_asks=((10, 500),),
                yes_bids=((9, 500),))
    inst = MarketInstrument(
        venue="kalshi", instrument_id="X", kind=KIND_BINARY, title="t",
        legs=(Leg.build("draw", "X", home="A", away="B", league_id="mls"),),
        league_id="mls", kickoff_utc=None, book=book,
        fee_model={"venue": "kalshi"})
    case = calculator.build_case(inst, 0.40, 0.40)
    assert case.action == "DEFER"
    assert "kickoff" in " ".join(case.reasons).lower()


def test_a_kickoff_already_passed_defers():
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
    book = Book(observed_at=utcnow_iso(), yes_asks=((10, 500),),
                yes_bids=((9, 500),))
    inst = MarketInstrument(
        venue="kalshi", instrument_id="X", kind=KIND_BINARY, title="t",
        legs=(Leg.build("draw", "X", home="A", away="B", league_id="mls"),),
        league_id="mls", kickoff_utc=past, book=book,
        fee_model={"venue": "kalshi"})
    case = calculator.build_case(inst, 0.40, 0.40)
    assert case.action == "DEFER"
    assert "passed" in " ".join(case.reasons).lower()
