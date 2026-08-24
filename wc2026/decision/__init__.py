"""Deterministic decision layer.

All probabilities, fees, EV, limit prices and sizes are computed here, in code,
BEFORE any model is consulted. The board audits these numbers and may only
shrink them.
"""
from .calculator import (
                         ACTIONS,
                         BUY_NOW,
                         CALC,
                         DEFER,
                         PASS,
                         PLACE_LIMIT,
                         PLACEABLE_ACTIONS,
                         UNSUPPORTED,
                         WAIT_FOR_QUOTE,
                         CalcConfig,
                         DeterministicCase,
                         PriceRung,
                         build_case,
                         ev_at_price,
                         fee_cents,
)
from .joint import (
                         DependencyError,
                         LegClaim,
                         bundle_outcomes,
                         combo_expected_payout,
                         dependence_ratio,
                         joint_probability,
                         naive_independent_probability,
)

__all__ = ["ACTIONS", "BUY_NOW", "PLACE_LIMIT", "WAIT_FOR_QUOTE", "PASS",
           "DEFER", "UNSUPPORTED", "PLACEABLE_ACTIONS", "CALC", "CalcConfig",
           "DeterministicCase", "PriceRung", "build_case", "ev_at_price",
           "fee_cents", "LegClaim", "joint_probability", "dependence_ratio",
           "naive_independent_probability", "combo_expected_payout",
           "bundle_outcomes", "DependencyError"]
