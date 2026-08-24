"""
decision/joint.py
=================
Joint valuation of multi-leg instruments.

The rule that matters: legs on the SAME fixture are dependent and must be
valued on the shared scoreline grid. "Home win" and "over 2.5 goals" are
positively related; multiplying their marginals overstates a parlay's
probability, and on a venue whose combos explicitly permit same-event legs
(Kalshi sets `is_single_market_per_event=False`) that error is easy to make and
expensive.

Legs on DIFFERENT fixtures are treated as independent. That is a documented
simplification, not a claim of truth: distinct matches still share weather,
referee policy and league-wide scoring drift. It is stated in the case so a
reviewer can see the assumption rather than discover it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DependencyError(ValueError):
    """A joint probability was requested without the means to compute it."""


@dataclass
class LegClaim:
    """One leg bound to a fixture and an indicator over that fixture's grid."""
    fixture_key: str
    indicator: np.ndarray          # bool (G+1, G+1) over the scoreline grid
    negated: bool = False

    def mask(self) -> np.ndarray:
        return ~self.indicator if self.negated else self.indicator


def joint_probability(legs, grids: dict) -> float:
    """P(all legs settle YES).

    `grids` maps fixture_key -> that fixture's exact scoreline PMF. Legs on one
    fixture are intersected on its grid FIRST (preserving dependence), and only
    then are distinct fixtures combined multiplicatively.
    """
    if not legs:
        raise DependencyError("no legs to value")
    by_fixture: dict = {}
    for leg in legs:
        by_fixture.setdefault(leg.fixture_key, []).append(leg)

    total = 1.0
    for key, group in by_fixture.items():
        grid = grids.get(key)
        if grid is None:
            raise DependencyError(
                "no scoreline grid for fixture %r; refusing to approximate a "
                "joint probability by multiplying marginals" % key)
        mask = np.ones_like(grid, dtype=bool)
        for leg in group:
            mask &= leg.mask()
        total *= float(grid[mask].sum())
    return total


def naive_independent_probability(legs, grids: dict) -> float:
    """Product of marginals -- provided ONLY for comparison/diagnostics.

    Never use this to price a combo whose legs share a fixture. It exists so
    the size of the dependence error can be measured and reported.
    """
    out = 1.0
    for leg in legs:
        grid = grids.get(leg.fixture_key)
        if grid is None:
            raise DependencyError("no grid for %r" % leg.fixture_key)
        out *= float(grid[leg.mask()].sum())
    return out


def dependence_ratio(legs, grids: dict) -> float:
    """joint / naive. 1.0 means independence was harmless; >1 means the naive
    product UNDERSTATES the parlay, <1 means it OVERSTATES it."""
    naive = naive_independent_probability(legs, grids)
    if naive <= 0:
        return float("nan")
    return joint_probability(legs, grids) / naive


def combo_expected_payout(legs, grids: dict, payout_per_contract: float = 100.0
                          ) -> float:
    """Expected settlement value of an all-YES combo, in cents.

    Kalshi's multivariate contracts resolve YES only if every associated market
    resolves YES ("scalar outcomes are multiplied"), so for a binary combo the
    expected payout is simply payout * P(all legs).
    """
    return payout_per_contract * joint_probability(legs, grids)


def bundle_outcomes(legs, grids: dict, costs, payout_per_contract: float = 100.0
                    ) -> dict:
    """Value a SEPARATELY EXECUTED bundle -- not a venue parlay.

    Each leg is its own contract with its own price and its own fill risk, so
    the bundle's expected value is the SUM of per-leg expectations, not the
    product. Partial fills are the normal case and leave a different portfolio
    than intended; that is reported rather than hidden.
    """
    if len(costs) != len(legs):
        raise DependencyError("a cost is required for every leg")
    per_leg, total_ev = [], 0.0
    for leg, cost in zip(legs, costs):
        grid = grids.get(leg.fixture_key)
        if grid is None:
            raise DependencyError("no grid for %r" % leg.fixture_key)
        p = float(grid[leg.mask()].sum())
        ev = payout_per_contract * p - cost
        per_leg.append({"fixture": leg.fixture_key, "p": p, "cost": cost,
                        "ev_per_contract": ev})
        total_ev += ev
    return {
        "kind": "bundle",
        "legs": per_leg,
        "total_ev_per_contract_set": total_ev,
        "all_legs_probability": joint_probability(legs, grids),
        "note": ("a bundle has no venue parlay payout: legs settle "
                 "independently, so EV is additive and partial fills leave a "
                 "different portfolio than intended"),
    }
