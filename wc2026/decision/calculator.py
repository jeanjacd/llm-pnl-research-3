"""
decision/calculator.py
======================
The deterministic case calculator: every probability, fee, EV, limit price and
size is computed HERE, in code, before any model is asked for an opinion.

Mission rule 1: deterministic code is the source of truth. The board may audit
and shrink these numbers; it may never replace them with its own arithmetic.

What this produces for a supported instrument
---------------------------------------------
  * raw model probability and a calibrated estimate, plus conservative bounds;
  * the CURRENT EXECUTABLE price obtained by walking the real book (never a
    midpoint, never a displayed "best ask" without depth behind it);
  * venue-specific fees, spread, depth, modelled slippage and an
    adverse-selection reserve;
  * break-even probability, EV per contract, ROI, uncertainty-adjusted EV;
  * the maximum size executable right now;
  * a LIMIT LADDER: every viable price tick up to the maximum acceptable
    limit, with the size available at each and the EV/ROI it would earn.

Why the ladder matters: a candidate must not disappear merely because the
current ask is unattractive. An unattractive quote with an attractive
conditional price is PLACE_LIMIT, not PASS.

Action states
-------------
  BUY_NOW          the executable book clears every EV and risk hurdle now
  PLACE_LIMIT      the quote is unattractive but a lower price is sufficiently +EV
  WAIT_FOR_QUOTE   valuation may be attractive but nothing is executable
  PASS             no realistically obtainable price clears the hurdle
  DEFER            inputs, rules, model support or freshness are inadequate
  UNSUPPORTED      no validated quantitative model exists for the contract
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field

from ..betting.fees import trading_fee_cents

# --- action states ------------------------------------------------------------
BUY_NOW = "BUY_NOW"
PLACE_LIMIT = "PLACE_LIMIT"
WAIT_FOR_QUOTE = "WAIT_FOR_QUOTE"
PASS = "PASS"
DEFER = "DEFER"
UNSUPPORTED = "UNSUPPORTED"

ACTIONS = (BUY_NOW, PLACE_LIMIT, WAIT_FOR_QUOTE, PASS, DEFER, UNSUPPORTED)
# Only these two may ever reach the paper broker.
PLACEABLE_ACTIONS = (BUY_NOW, PLACE_LIMIT)


def deterministic_size(case, cfg) -> int:
    """Contracts to buy at the computed limit, from the edge and nothing else.

    Fractional Kelly on `p_lower` -- the CONSERVATIVE end of the probability
    bound, not the point estimate -- at the price the ladder cleared. Floored
    to whole contracts, so a market whose edge does not justify one contract
    gets none rather than a token position.
    """
    from ..betting.kelly import contracts_for_stake, single_kelly

    price = case.max_limit_price_cents
    if not price or case.p_lower is None:
        return 0
    cost = price / 100.0
    if not (0.0 < cost < 1.0):
        return 0
    edge_fraction = single_kelly(float(case.p_lower), cost)
    if edge_fraction <= 0:
        return 0
    stake = min(cfg.kelly_fraction * edge_fraction,
                cfg.max_stake_fraction_per_market)
    return max(0, contracts_for_stake(stake, cfg.bankroll_cents / 100.0, cost))


@dataclass(frozen=True)
class CalcConfig:
    min_edge: float = 0.03
    min_roi: float = 0.02
    min_depth_contracts: float = 5.0
    max_spread_cents: int = 10
    # Reserve against being picked off by better-informed flow. Widens with the
    # spread (a wide book is an uninformed book) and with time to kickoff.
    adverse_selection_cents: float = 1.0
    adverse_selection_per_hour: float = 0.02
    adverse_selection_max_cents: float = 4.0
    # --- deterministic sizing -------------------------------------------------
    # The board's quant prompt says a member may "CONFIRM or REDUCE the computed
    # maximum price and size". No size was ever computed: `run_board` passed
    # `ceiling_size = None`, so there was nothing to reduce and the member
    # invented a number. On the live book that produced 333 positions of one
    # contract and a handful of 20, 50, 60 and 100 -- six positions holding 40%
    # of all money staked, with no rule behind the split.
    #
    # Quarter Kelly on the conservative probability bound. Quarter because it is
    # the documented floor in `betting.confidence`, and a floor is the right
    # choice while the forecasts are still unproven: half Kelly on a mis-stated
    # edge is ruinous, quarter Kelly on a real one is merely slow.
    kelly_fraction: float = 0.25
    # Sized against the STARTING bankroll, matching the exposure cap, so that a
    # run of luck cannot compound the stake before the edge is established.
    bankroll_cents: float = 100_000.0
    # No single market may take more than this share of the bankroll, so one
    # high-probability contract cannot swallow a fixture's whole budget.
    max_stake_fraction_per_market: float = 0.0125
    # Probability uncertainty: how much of the calibration gap to treat as
    # error when computing the conservative bound.
    uncertainty_floor: float = 0.02
    max_hours_to_kickoff: float = 96.0
    stale_book_seconds: float = 900.0
    # A resting buy far below the current bid is arithmetically +EV and
    # practically unfillable. The allowance scales with time to kickoff --
    # a price hours away has room to drift onto our limit, one minute away
    # does not -- and is capped so an absurd limit is still a PASS.
    limit_gap_base_cents: float = 6.0
    limit_gap_per_hour_cents: float = 1.5
    limit_gap_max_cents: float = 30.0


CALC = CalcConfig()


def _utcnow():
    return dt.datetime.now(dt.timezone.utc)


def _parse(ts):
    if not ts:
        return None
    text = str(ts).strip().replace(" ", "T").replace("Z", "+00:00")
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)


# --- fees ---------------------------------------------------------------------
def fee_cents(venue: str, count: int, price_cents: int, fee_model=None,
              role: str = "taker") -> int:
    """Exact venue fee for `count` contracts at `price_cents`, ceiled to a cent.

    Both venues use the SAME formula shape -- fee = C * rate * p * (1-p) --
    which is symmetric in p and therefore largest at 50c and near zero at the
    wings. Charging a flat rate on notional instead (an earlier version of this
    function did) overstates the fee by ~5.6x at 23c.

    Kalshi: rate 0.07, taker only, verified against the published fee table.

    Polymarket: rate comes from the venue's own `base_fee` field, in BASIS
    POINTS. Measured 2026-08-24 across all five leagues: Gamma `takerBaseFee`,
    the CLOB market record `taker_base_fee`, and the authoritative
    `GET /fee-rate?token_id=..` endpoint ALL return 1000 for every soccer
    market sampled (88/88 markets, 73/73 tokens). 1000bps = 0.10 is exactly
    twice the published sports rate of 0.05, and Polymarket's own client has an
    open, unanswered issue about this contradiction (py-clob-client#326). We
    charge the reported field rather than the published rate: over-charging
    costs a missed trade, under-charging books a bad one.

      documented   : 0.05 -> max $1.25 / 100 shares at 50c
      charged here : 0.10 -> max $2.50 / 100 shares at 50c

    `role`: Polymarket documents "Makers are never charged fees", so a rung
    that RESTS on the book pays nothing while a marketable order pays the taker
    rate. Kalshi is charged the taker schedule either way. The default is
    "taker" so any caller that does not reason about liquidity role gets the
    conservative number.

    An unknown venue is charged the Kalshi schedule rather than assumed free.
    """
    if count <= 0:
        return 0
    if venue == "polymarket":
        if role == "maker":
            return 0
        model = fee_model or {}
        try:
            basis_points = float(model.get("taker_base_fee") or 0)
        except (TypeError, ValueError):
            basis_points = 0.0
        if basis_points <= 0:
            return 0
        rate = basis_points / 10_000.0
        # p*(1-p) is formed from INTEGER cents. Computing it in floats makes
        # the fee asymmetric about 50c -- 0.7*(1-0.7) != 0.3*(1-0.3) in binary
        # floating point -- which would charge a different fee for the two
        # sides of the same contract.
        price = max(0, min(100, int(price_cents)))
        exact = count * rate * price * (100 - price) / 100.0   # cents
        return int(-(-exact // 1))                             # ceil
    return trading_fee_cents(count, price_cents, 0.07)


# --- rungs and cases ----------------------------------------------------------
@dataclass
class PriceRung:
    """One viable limit price and what it would earn."""
    price_cents: int
    size_available: float
    fee_cents: int
    cost_cents: float
    ev_per_contract: float
    roi: float
    immediately_executable: bool
    # "taker" if the rung crosses the spread, "maker" if it rests. Polymarket
    # charges makers nothing; Kalshi charges the same schedule either way.
    liquidity_role: str = "taker"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeterministicCase:
    case_id: str
    venue: str
    instrument_id: str
    league_id: str | None
    claim: str
    side: str
    kind: str

    # probabilities
    p_raw: float | None = None
    p_calibrated: float | None = None
    p_lower: float | None = None
    p_upper: float | None = None

    # execution
    touch_cents: int | None = None
    executable_avg_cents: float | None = None
    executable_size: float = 0.0
    worst_cents: int | None = None
    spread_cents: int | None = None
    depth_at_touch: float = 0.0
    slippage_cents: float = 0.0
    adverse_selection_cents: float = 0.0

    # value
    breakeven_prob: float | None = None
    ev_per_contract: float | None = None
    roi: float | None = None
    uncertainty_adjusted_ev: float | None = None
    max_limit_price_cents: int | None = None
    max_contracts: int = 0
    ladder: list = field(default_factory=list)

    # decision
    action: str = DEFER
    reasons: list = field(default_factory=list)
    actionable_until: str | None = None
    hours_to_kickoff: float | None = None
    model_version: str | None = None
    observed_at: str | None = None

    def as_dict(self) -> dict:
        out = asdict(self)
        out["ladder"] = [r.as_dict() if isinstance(r, PriceRung) else r
                         for r in self.ladder]
        return out

    def case_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def placeable(self) -> bool:
        return self.action in PLACEABLE_ACTIONS


# --- the calculator -----------------------------------------------------------
def conservative_bounds(p_raw: float, p_cal: float, cfg: CalcConfig) -> tuple:
    """A defensible interval around the model probability.

    The calibration correction is itself uncertain, so the half-width is at
    least `uncertainty_floor` and at least the size of the correction. The
    LOWER bound is what every value test uses -- being wrong about our own
    accuracy must cost size, not produce phantom edge.
    """
    gap = abs(p_cal - p_raw)
    half = max(cfg.uncertainty_floor, gap)
    return max(0.0, p_cal - half), min(1.0, p_cal + half)


def adverse_selection(cfg: CalcConfig, spread_cents, hours_to_kickoff) -> float:
    """Cents per contract reserved against informed flow."""
    reserve = cfg.adverse_selection_cents
    if spread_cents is not None:
        reserve += 0.25 * max(0, spread_cents - 1)
    if hours_to_kickoff is not None and hours_to_kickoff > 0:
        reserve += cfg.adverse_selection_per_hour * min(hours_to_kickoff, 96.0)
    return min(reserve, cfg.adverse_selection_max_cents)


def ev_at_price(p: float, price_cents: int, venue: str, count: int,
                reserve_cents: float, fee_model=None,
                role: str = "taker") -> tuple:
    """(ev_per_contract_cents, roi, fee_cents) for buying at `price_cents`.

    EV = conservative expected payout - price - fee share - adverse-selection
    reserve. A binary contract pays 100c.
    """
    fee = fee_cents(venue, count, price_cents, fee_model, role)
    fee_share = fee / count if count else 0.0
    ev = 100.0 * p - price_cents - fee_share - reserve_cents
    cost = price_cents + fee_share
    roi = ev / cost if cost > 0 else 0.0
    return ev, roi, fee


def build_case(instrument, p_raw: float | None, p_calibrated: float | None,
               side: str = "yes", cfg: CalcConfig = CALC,
               model_version: str | None = None,
               reference_size: int = 100) -> DeterministicCase:
    """Deterministic evaluation of one instrument on one side.

    Every branch that cannot produce a trustworthy number returns DEFER or
    UNSUPPORTED rather than a guess.
    """
    leg = instrument.legs[0] if instrument.legs else None
    case = DeterministicCase(
        case_id="%s:%s:%s" % (instrument.venue, instrument.instrument_id, side),
        venue=instrument.venue, instrument_id=instrument.instrument_id,
        league_id=instrument.league_id,
        claim=(leg.claim if leg else "unknown"), side=side,
        kind=instrument.kind, model_version=model_version,
        observed_at=(instrument.book.observed_at if instrument.book else None))

    # --- support and rules -------------------------------------------------
    if not instrument.supported:
        case.action = UNSUPPORTED
        case.reasons = instrument.unsupported_reasons or ["unsupported"]
        return case
    if instrument.settles_on_regulation is False:
        case.action = UNSUPPORTED
        case.reasons = ["settles on a basis the model does not represent"]
        return case
    if p_raw is None:
        case.action = DEFER
        case.reasons = ["no model probability available"]
        return case

    p_cal = p_raw if p_calibrated is None else p_calibrated
    case.p_raw, case.p_calibrated = p_raw, p_cal
    case.p_lower, case.p_upper = conservative_bounds(p_raw, p_cal, cfg)

    # --- freshness ---------------------------------------------------------
    kickoff = _parse(instrument.kickoff_utc)
    now = _utcnow()
    if kickoff is None:
        # Without a kick-off there is no way to tell a pre-match quote from an
        # in-play one, and this model has no in-play validity whatsoever.
        case.action = DEFER
        case.reasons = ["kickoff time unknown; cannot confirm the match has "
                        "not started"]
        return case
    if kickoff is not None:
        hours = (kickoff - now).total_seconds() / 3600.0
        case.hours_to_kickoff = hours
        case.actionable_until = kickoff.isoformat()
        if hours <= 0:
            case.action = DEFER
            case.reasons = ["kickoff has passed"]
            return case
        if hours > cfg.max_hours_to_kickoff:
            case.action = PASS
            case.reasons = ["kickoff is %.0fh away (limit %.0fh)"
                            % (hours, cfg.max_hours_to_kickoff)]
            return case
    observed = _parse(case.observed_at)
    if observed is not None:
        age = (now - observed).total_seconds()
        if age > cfg.stale_book_seconds:
            case.action = DEFER
            case.reasons = ["book snapshot is %.0fs old" % age]
            return case

    book = instrument.book
    case.adverse_selection_cents = adverse_selection(
        cfg, None if book is None else book.spread_cents(side),
        case.hours_to_kickoff)

    # --- executable price --------------------------------------------------
    if book is None or book.touch(side) is None:
        case.action = WAIT_FOR_QUOTE
        case.reasons = ["no executable quote on this side"]
        return case
    case.touch_cents = book.touch(side)
    case.depth_at_touch = book.depth_at_touch(side)
    case.spread_cents = book.spread_cents(side)

    walked = book.walk(side, max(reference_size, 1))
    if walked is None:
        case.action = WAIT_FOR_QUOTE
        case.reasons = ["book has no fillable depth"]
        return case
    avg, filled, worst = walked
    case.executable_avg_cents, case.executable_size, case.worst_cents = (
        avg, filled, worst)
    case.slippage_cents = avg - case.touch_cents

    # --- the limit ladder --------------------------------------------------
    # Every tick from 1c up to the highest price that still clears the hurdles
    # at the CONSERVATIVE probability. This is what turns an unattractive quote
    # into a conditional order instead of a discarded candidate.
    tick = max(1, int(instrument.tick_cents or 1))
    max_limit = None
    ladder = []
    touch_now = case.touch_cents or 0
    for price in range(tick, 100, tick):
        # At or above the touch the order crosses and pays the taker rate;
        # below it the order rests, so a fill makes us the maker.
        role = "taker" if (touch_now and price >= touch_now) else "maker"
        ev, roi, fee = ev_at_price(case.p_lower, price, instrument.venue,
                                   reference_size,
                                   case.adverse_selection_cents,
                                   instrument.fee_model, role)
        if ev <= 0 or roi < cfg.min_roi:
            continue
        breakeven = (price + fee / reference_size
                     + case.adverse_selection_cents) / 100.0
        if case.p_lower - breakeven < cfg.min_edge:
            continue
        max_limit = price
        size_here = book.max_size_at_or_below(side, price)
        ladder.append(PriceRung(
            price_cents=price, size_available=size_here, fee_cents=fee,
            cost_cents=price + fee / reference_size,
            ev_per_contract=ev, roi=roi, liquidity_role=role,
            immediately_executable=size_here >= cfg.min_depth_contracts))
    case.ladder = ladder
    case.max_limit_price_cents = max_limit
    case.max_contracts = deterministic_size(case, cfg)

    # --- headline value at the current executable price --------------------
    price_now = int(round(avg))
    ev_now, roi_now, fee_now = ev_at_price(
        case.p_lower, max(1, min(99, price_now)), instrument.venue,
        reference_size, case.adverse_selection_cents, instrument.fee_model)
    case.ev_per_contract = ev_now
    case.roi = roi_now
    case.breakeven_prob = (price_now + fee_now / reference_size
                           + case.adverse_selection_cents) / 100.0
    case.uncertainty_adjusted_ev = ev_now      # already uses the lower bound

    # --- decide ------------------------------------------------------------
    if not ladder:
        case.action = PASS
        case.reasons = ["no price up to 99c clears the EV/edge hurdles"]
        return case
    # Realism: a limit far below the resting bid will never be hit.
    best_bid = book.best_bid(side)
    allowed_gap = min(
        cfg.limit_gap_max_cents,
        cfg.limit_gap_base_cents
        + cfg.limit_gap_per_hour_cents * max(0.0, case.hours_to_kickoff or 0.0))
    if (best_bid is not None and max_limit is not None
            and max_limit < best_bid - allowed_gap):
        case.action = PASS
        case.reasons = ["best acceptable limit %dc is %dc below the %dc bid "
                        "(allowance %.0fc); not realistically obtainable"
                        % (max_limit, best_bid - max_limit, best_bid,
                           allowed_gap)]
        return case
    if case.spread_cents is not None and case.spread_cents > cfg.max_spread_cents:
        case.action = PLACE_LIMIT
        case.reasons = ["spread %dc exceeds %dc; will not cross"
                        % (case.spread_cents, cfg.max_spread_cents)]
        return case
    if (max_limit is not None and price_now <= max_limit
            and case.depth_at_touch >= cfg.min_depth_contracts
            and ev_now > 0 and roi_now >= cfg.min_roi
            and case.p_lower - case.breakeven_prob >= cfg.min_edge):
        case.action = BUY_NOW
        case.reasons = ["executable at %dc, EV %.2fc/contract, ROI %.1f%%"
                        % (price_now, ev_now, 100 * roi_now)]
        return case
    case.action = PLACE_LIMIT
    case.reasons = ["current %dc is unattractive; conditional at <=%dc"
                    % (price_now, max_limit)]
    return case
