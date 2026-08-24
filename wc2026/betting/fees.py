"""
fees.py
=======
Exact Kalshi fee arithmetic, integer-safe (no floating-point drift in money).

Official formula (fee schedule PDF, effective 2026-02-05, verified 2026-07-20):

    fees = roundup_to_cent(factor x C x P x (1-P))      [dollars]

with P the contract price in dollars and C the contract count. In integer
cents with price p (cents), that is

    fee_cents = ceil( factor_num x C x p x (100 - p) / (10000 x factor_den) )

computed with pure integer arithmetic. The published fee table is reproduced in
tests/test_betting_fees.py as ground truth.
"""
from __future__ import annotations

from fractions import Fraction


def _factor_fraction(factor: float) -> Fraction:
    """Exact rational form of a fee factor (0.07 -> 7/100, 0.0175 -> 7/400)."""
    return Fraction(str(factor))


def trading_fee_cents(count: int, price_cents: int, factor: float = 0.07) -> int:
    """Total fee in cents for `count` contracts at `price_cents`, rounded UP to
    the next cent per the official schedule. Exact integer arithmetic."""
    if not (0 < price_cents < 100):
        raise ValueError(f"price_cents must be in 1..99, got {price_cents}")
    if count < 0:
        raise ValueError("count must be >= 0")
    if count == 0:
        return 0
    frac = _factor_fraction(factor)
    # dollars = factor * C * (p/100) * ((100-p)/100); cents = dollars * 100
    fee_cents_exact = frac * count * price_cents * (100 - price_cents) / 100
    return -((-fee_cents_exact.numerator) // fee_cents_exact.denominator)  # ceil


def fee_per_contract_cents(count: int, price_cents: int, factor: float = 0.07) -> Fraction:
    """Exact per-contract fee share (a Fraction -- rounding happens per order,
    so the per-contract share is generally not a whole cent)."""
    return Fraction(trading_fee_cents(count, price_cents, factor), count)


def breakeven_prob(count: int, price_cents: int, factor: float = 0.07) -> Fraction:
    """The win probability at which buying `count` contracts at `price_cents`
    is exactly EV-zero including fees: (cost + fee) / payout. Exact."""
    fee = trading_fee_cents(count, price_cents, factor)
    return Fraction(count * price_cents + fee, count * 100)


def ev_per_contract_cents(p: float, count: int, price_cents: int,
                          factor: float = 0.07) -> float:
    """Expected value per contract in cents at model probability p, including
    the exact (order-level) fee share."""
    fee_share = fee_per_contract_cents(count, price_cents, factor)
    return 100.0 * p - price_cents - float(fee_share)
