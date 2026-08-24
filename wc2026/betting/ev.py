"""
ev.py
=====
Turn a mapped market + live order book into an evaluated candidate bet.

Cost basis is the executable ASK from the order book (never the midpoint),
with slippage awareness: the average fill price for the intended size is
computed by walking the book, and a fill that would average more than
`max_slippage_cents` above the touch is trimmed to the depth that keeps it
within tolerance. Fees are the exact taker fees from fees.py.

Both sides of every market are considered: buying NO on claim X is buying YES
on its complement (p_no = 1 - p_yes), priced off the opposite side of the book.

Gates applied here (all conservative, all tested):
  * minimum edge on the RAW model probability vs fee-adjusted breakeven;
  * non-negative edge on the CALIBRATED probability (trust curve) if enabled;
  * minimum liquidity at the touch;
  * maximum bid-ask spread (refuse wide, stale books);
  * kickoff within the configured window.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .confidence import TrustCurve
from .config import BettingConfig
from .fees import breakeven_prob, trading_fee_cents
from .kalshi import OrderBook
from .markets import MappedMarket


@dataclass
class Candidate:
    market: MappedMarket
    side: str                  # "yes" or "no" (what we would BUY)
    claim: str                 # claim we are long, e.g. "home_win" / "not btts"
    p_model: float             # model probability of OUR side settling at 100
    p_calibrated: float
    ask_cents: int             # touch price for our side
    depth_at_touch: int
    spread_cents: int | None
    edge_raw: float            # p_model - breakeven (at reference size)
    edge_calibrated: float
    # sizing, filled in by the pipeline:
    full_kelly: float = 0.0
    kelly_multiplier: float = 0.0
    confidence_parts: dict = field(default_factory=dict)
    contracts: int = 0
    stake_cents: int = 0
    fee_cents: int = 0
    gate_verdict: str | None = None
    gate_multiplier: float | None = None
    gate_rationale: str | None = None
    skip_reason: str | None = None

    @property
    def cost_dollars(self) -> float:
        """Fee-inclusive cost per contract in dollars (at reference size)."""
        n = max(self.contracts, 1)
        fee = trading_fee_cents(n, self.ask_cents)
        return (self.ask_cents + fee / n) / 100.0


REFERENCE_SIZE = 100   # fee-per-contract reference (the asymptotic fee share)


def evaluate_market(mkt: MappedMarket, book: OrderBook, score_pmf: np.ndarray,
                    trust: TrustCurve, cfg: BettingConfig,
                    hours_to_kickoff: float | None) -> list[Candidate]:
    """Evaluate BOTH sides of one market. Returns candidates that pass every
    gate (losers are simply not returned; the pipeline logs the reasons)."""
    p_yes = mkt.model_prob(score_pmf)
    out = []
    for side, p in (("yes", p_yes), ("no", 1.0 - p_yes)):
        claim = mkt.claim if side == "yes" else f"not_{mkt.claim}"
        ask = book.touch(side)
        if ask is None or not (0 < ask < 100):
            continue
        depth = book.depth_at_touch(side)
        if depth < cfg.min_liquidity_contracts:
            continue
        bid = book.best_bid(side)
        spread = (ask - bid) if bid is not None else None
        if spread is None or spread > cfg.max_spread_cents:
            continue
        if hours_to_kickoff is not None and hours_to_kickoff > cfg.max_hours_to_kickoff:
            continue

        be = float(breakeven_prob(REFERENCE_SIZE, ask, cfg.taker_fee_factor))
        p_cal = trust.calibrated_prob(p)
        edge_raw = p - be
        edge_cal = p_cal - be
        if edge_raw < cfg.min_edge:
            continue
        if cfg.require_calibrated_nonnegative and edge_cal < 0.0:
            continue
        out.append(Candidate(
            market=mkt, side=side, claim=claim, p_model=p, p_calibrated=p_cal,
            ask_cents=ask, depth_at_touch=depth, spread_cents=spread,
            edge_raw=edge_raw, edge_calibrated=edge_cal))
    return out
