"""Fee math against the OFFICIAL Kalshi fee schedule tables (PDF effective
2026-02-05): every published (price, count) -> fee pair is reproduced exactly."""
from fractions import Fraction

import pytest

from wc2026.betting.fees import (
    breakeven_prob,
    ev_per_contract_cents,
    fee_per_contract_cents,
    trading_fee_cents,
)

# (price_cents, fee_cents for 1 contract, fee_cents for 100 contracts)
OFFICIAL_TAKER_TABLE = [
    (1, 1, 7), (5, 1, 34), (10, 1, 63), (15, 1, 90), (20, 2, 112),
    (25, 2, 132), (30, 2, 147), (35, 2, 160), (40, 2, 168), (45, 2, 174),
    (50, 2, 175), (55, 2, 174), (60, 2, 168), (65, 2, 160), (70, 2, 147),
    (75, 2, 132), (80, 2, 112), (85, 1, 90), (90, 1, 63), (95, 1, 34),
    (99, 1, 7),
]


@pytest.mark.parametrize("price,fee1,fee100", OFFICIAL_TAKER_TABLE)
def test_official_taker_fee_table(price, fee1, fee100):
    assert trading_fee_cents(1, price, 0.07) == fee1
    assert trading_fee_cents(100, price, 0.07) == fee100


def test_fee_rounds_up_never_down():
    # 0.07 * 3 * 0.5 * 0.5 = $0.0525 -> $0.06, never $0.05
    assert trading_fee_cents(3, 50, 0.07) == 6


def test_maker_factor():
    # 0.0175 * 100 * 0.25 = $0.4375 -> 44c
    assert trading_fee_cents(100, 50, 0.0175) == 44


def test_zero_contracts_no_fee():
    assert trading_fee_cents(0, 50) == 0


def test_invalid_price_rejected():
    for bad in (0, 100, -5, 101):
        with pytest.raises(ValueError):
            trading_fee_cents(1, bad)


def test_fee_per_contract_is_exact_fraction():
    assert fee_per_contract_cents(100, 50) == Fraction(175, 100)


def test_breakeven_includes_fee_exactly():
    # 100 contracts at 50c: cost 5000c + 175c fee over 10000c payout
    assert breakeven_prob(100, 50) == Fraction(5175, 10000)
    # breakeven always exceeds the naive price probability
    for price in (10, 35, 50, 65, 90):
        assert breakeven_prob(100, price) > Fraction(price, 100)


def test_ev_per_contract_signs():
    # p exactly at breakeven -> EV 0
    be = float(breakeven_prob(100, 50))
    assert abs(ev_per_contract_cents(be, 100, 50)) < 1e-9
    assert ev_per_contract_cents(be + 0.05, 100, 50) > 0
    assert ev_per_contract_cents(be - 0.05, 100, 50) < 0
