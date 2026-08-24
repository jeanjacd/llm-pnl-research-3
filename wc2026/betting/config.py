"""
config.py (betting)
===================
Every betting parameter and safety rail in one documented place, mirroring the
model's config philosophy: if a number affects money, it lives here.

Fee factors are verified against the official Kalshi fee schedule PDF
(https://kalshi.com/docs/kalshi-fee-schedule.pdf, "Last updated and effective:
Feb 5, 2026", retrieved 2026-07-20 via archive.org because kalshi.com
rate-limited the direct fetch):
    taker: fees = roundup_to_cent(0.07   x C x P x (1-P))
    maker: fees = roundup_to_cent(0.0175 x C x P x (1-P))
    settlement fee: none.
A "7.7.26 Update" of the schedule exists; re-verify these factors before any
live session (`python -m wc2026 recommend` prints the assumed factors).
EV always assumes TAKER fees -- the conservative case.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import DATA_DIR

BETTING_DIR = os.path.join(DATA_DIR, "betting")
STATE_JSON = os.path.join(BETTING_DIR, "state.json")
AUDIT_LOG = os.path.join(BETTING_DIR, "audit.jsonl")
RECS_LOG = os.path.join(BETTING_DIR, "recommendations.jsonl")
GATE_LOG = os.path.join(BETTING_DIR, "gate_audit.jsonl")

# Production host (verified 2026-07-23: serves all markets despite "elections").
# Override via env to point at Kalshi's demo/paper environment for safe testing,
# e.g. KALSHI_BASE=https://demo-api.kalshi.co/trade-api/v2
KALSHI_BASE = os.environ.get(
    "KALSHI_BASE", "https://api.elections.kalshi.com/trade-api/v2")

# Credentials come from the environment ONLY. Never from source, ever.
ENV_KEY_ID = "KALSHI_API_KEY_ID"
ENV_PRIVATE_KEY_PATH = "KALSHI_PRIVATE_KEY_PATH"


@dataclass(frozen=True)
class BettingConfig:
    # --- fees (see module docstring for provenance) --------------------------
    taker_fee_factor: float = 0.07
    maker_fee_factor: float = 0.0175

    # --- candidate selection -------------------------------------------------
    # Minimum edge (model probability minus fee-adjusted breakeven) for a bet
    # to be a candidate. 3 points: below that, model-vs-market differences on a
    # 1.7%-better-than-baseline model are noise.
    min_edge: float = 0.03
    # The candidate must ALSO be non-negative-EV under the calibration-corrected
    # probability (the backtest trust curve) -- protects against the measured
    # overconfidence on heavy favourites. House-favour by construction.
    require_calibrated_nonnegative: bool = True
    # Only bet matches starting within this window (news risk grows with time).
    max_hours_to_kickoff: float = 96.0

    # --- execution / liquidity ----------------------------------------------
    # Minimum resting contracts at the touch before a market is biddable.
    min_liquidity_contracts: int = 20
    # Never lift an ask more than this many cents above the best bid midpoint
    # implied fair -- i.e. refuse to cross wide spreads.
    max_spread_cents: int = 10
    # Slippage: fills are simulated by walking the book; average fill price may
    # exceed the touch by at most this many cents or the bet is trimmed.
    max_slippage_cents: int = 2
    # V2 order execution (POST /portfolio/events/orders). IOC = take what rests
    # at our limit now and cancel the remainder -> no surprise resting orders,
    # and never a fill worse than our price. STP cancels our own self-cross.
    order_time_in_force: str = "immediate_or_cancel"
    order_self_trade_prevention: str = "taker_at_cross"

    # --- Kelly sizing --------------------------------------------------------
    # Dynamic fractional Kelly: multiplier on FULL-Kelly joint stakes, scaled by
    # the documented confidence function between these bounds. Half Kelly is a
    # hard ceiling, quarter Kelly the floor for any candidate that passes.
    kelly_fraction_min: float = 0.25
    kelly_fraction_max: float = 0.50
    # Hard cap on the summed full-Kelly fractions per match (before the
    # fractional multiplier), a numerical and sanity guard.
    max_total_fraction_per_match: float = 0.50

    # --- safety rails (execute mode) -- every one enforced in code -----------
    kill_switch: bool = False            # True disables ALL order placement
    max_stake_per_bet_usd: float = 50.0
    max_stake_per_bet_pct: float = 0.05  # of current bankroll
    daily_loss_limit_usd: float = 100.0
    weekly_loss_limit_usd: float = 250.0
    # Runaway/blast-radius backstop only -- how many bets may be open at once.
    # The PRIMARY per-run control is the interactive "how many orders?" prompt
    # in execute mode (you pick, top-ranked first), so this sits high and only
    # bites if the automation ever tries to accumulate an absurd book.
    max_open_positions: int = 50
    default_bankroll_usd: float = 1000.0   # used until a real balance is seen

    # --- news-check gate -----------------------------------------------------
    gate_enabled: bool = True
    gate_timeout_seconds: float = 600.0
    gate_multiplier_min: float = 0.25    # 'reduce' verdict clamp range
    gate_multiplier_max: float = 1.00
    gate_command: tuple = ("claude", "-p")   # headless Claude invocation
    # Model for the gate. Its task -- web research + a bounded JSON verdict -- is
    # well within Haiku's capability at a fraction of Sonnet/Opus cost, and the
    # gate's one-directional authority + fail-closed validation bound the
    # downside of a lighter model. Raise to a stronger model for higher stakes.
    gate_model: str = "claude-haiku-4-5"
    # Web tools the headless gate must be allowed to use -- without these,
    # `claude -p` runs non-interactively with no way to approve them, so it
    # cannot research and (correctly) refuses to produce a verdict.
    gate_allowed_tools: tuple = ("WebSearch", "WebFetch")

    # --- market series in scope ----------------------------------------------
    # Series whose settlement maps exactly onto the regulation scoreline grid.
    # First-half / method-of-victory / first-team-to-score markets are OUT of
    # scope: the model has no first-half or ET/pens layer, and pretending
    # otherwise would be fabricated precision.
    series: tuple = ("KXMLSGAME", "KXMLSTOTAL", "KXMLSSPREAD", "KXMLSBTTS",
                     "KXMLSTEAMTOTAL", "KXMLSSCORE")


BETTING = BettingConfig()
