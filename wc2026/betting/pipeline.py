"""
pipeline.py
===========
Orchestration for the two modes:

  recommend (default) -- pull one league's live markets, compute fee-aware EV and
      dynamic-Kelly stakes, run the news-check gate, emit a ranked timestamped
      report and a persistent log. PLACES NOTHING, ever.

  execute -- everything recommend does, then walks the safety gauntlet and
      places LIMIT orders. Dry-run is the default even here: '--live' plus an
      interactive typed confirmation is required for real orders, and every
      rail below is enforced in code (and covered by tests):

        1. kill_switch off          5. per-bet stake caps ($ and % bankroll)
        2. credentials present      6. max open positions
        3. --live + typed confirm   7. min liquidity / max spread (in ev.py)
        4. daily/weekly loss halts  8. limit orders only; abort on any error

      Any API error or unexpected market state aborts the ENTIRE run -- no
      blind retries into a position.
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

from ..data import loader
from ..model.ratings import build_team_strength
from ..sim.match import predict_match
from .bankroll import BettingState, audit, reconcile_positions
from .confidence import TrustCurve, kelly_multiplier, w_data, w_liq, w_time
from .config import BETTING, BettingConfig
from .ev import Candidate, evaluate_market
from .fees import trading_fee_cents
from .kalshi import KalshiClient, OrderError
from .kelly import JointBet, contracts_for_stake, joint_kelly
from .markets import MarketMapper, match_event_to_fixture
from .tracking import log_placement, log_recommendations


class UntradeableLeague(RuntimeError):
    """A league without verified venue coverage or validated parameters."""


def _utcnow():
    return dt.datetime.now(dt.timezone.utc)


# --------------------------------------------------------------------------- #
# bankroll of record
# --------------------------------------------------------------------------- #
def resolve_bankroll(state: BettingState, client: KalshiClient | None,
                     verbose: bool = True) -> int:
    """The bankroll used for sizing, in cents. With credentials, the LIVE
    Kalshi balance is authoritative (fetched fresh and persisted to state);
    without them, the last persisted balance (or the configured default) is
    used with a clear notice. A failed live fetch in this advisory path warns
    loudly and falls back -- execute mode separately fetches the balance
    fail-HARD before any order (run_execute)."""
    if client is not None and getattr(client, "authenticated", False):
        try:
            state.bankroll_cents = client.get_balance_cents()
            state.save()
            if verbose:
                print(f"Bankroll: ${state.bankroll_cents/100:,.2f} "
                      f"(live Kalshi balance)")
            return state.bankroll_cents
        except Exception as e:                                # noqa: BLE001
            audit("balance_fetch_failed", {"error": str(e)})
            if verbose:
                print(f"WARNING: live balance fetch failed ({e}); using last "
                      f"known bankroll ${state.bankroll_cents/100:,.2f}")
            return state.bankroll_cents
    if verbose:
        print(f"Bankroll: ${state.bankroll_cents/100:,.2f} (no Kalshi "
              f"credentials -- last known/default; sizes are indicative)")
    return state.bankroll_cents


# --------------------------------------------------------------------------- #
# slate construction (shared by both modes)
# --------------------------------------------------------------------------- #
def require_tradeable(spec) -> None:
    """Refuse to build a slate for a league that is not validated for trading.

    Two independent gates, both fail-closed:
      * no VERIFIED venue coverage -> research-only;
      * no tuned parameters -> the forecast has never been validated for this
        league, so it must not price a market at any size.
    """
    from ..eval.tune import effective_model
    if not spec.tradeable:
        raise UntradeableLeague(
            "%s has no verified venue coverage; it is research-only."
            % spec.league_id)
    _, tuned = effective_model(spec)
    if not tuned:
        raise UntradeableLeague(
            "%s has no tuned parameters (data/leagues/%s/fitted_params.json "
            "missing). Run `python -m wc2026 tune --league %s` and validate "
            "before trading it."
            % (spec.league_id, spec.league_id, spec.league_id))


def build_slate(cfg: BettingConfig = BETTING, client: KalshiClient | None = None,
                min_edge: float | None = None, verbose: bool = True,
                _model_cfg=None, spec=None) -> list[Candidate]:
    """Fetch markets, map fixtures, evaluate EV, and size stakes jointly.
    Returns the fully-sized candidate slate (before the news-check gate)."""
    from ..cli import FIT_MAX_AGE_DAYS
    from ..eval.tune import effective_model
    from ..leagues import get_league
    if min_edge is not None:
        import dataclasses
        cfg = dataclasses.replace(cfg, min_edge=min_edge)

    spec = spec or get_league("mls")
    require_tradeable(spec)
    client = client or KalshiClient()
    model_cfg = _model_cfg or effective_model(spec)[0]
    trust = TrustCurve.load(TrustCurve.path_for(spec, "1x2"))

    # This league's own series only -- a league can never scan another's book.
    import dataclasses as _dc
    cfg = _dc.replace(cfg, series=tuple(spec.venue_series.get("kalshi", ())))

    # Fresh data + ratings WITH that league's human-curated soft-factor layer.
    df = loader.load_league(spec, tiers="training")
    train = loader.training_matches(df)
    ratings = build_team_strength(train, as_of=train["date"].max(), cfg=model_cfg,
                                  max_age_days=FIT_MAX_AGE_DAYS, verbose=verbose,
                                  adjustments_path=spec.adjustments_json)
    fix_cols = ["date", "home_team", "away_team"]
    if "kickoff_utc" in df.columns:
        fix_cols.append("kickoff_utc")
    fixtures = df[~df["played"]][fix_cols].copy()

    # --- discover markets, grouped by event ---------------------------------
    by_event: dict[str, list[dict]] = {}
    for series in cfg.series:
        for m in client.get_markets(series_ticker=series, status="open"):
            by_event.setdefault(m.get("event_ticker", ""), []).append(m)
    if verbose:
        print(f"Fetched {sum(len(v) for v in by_event.values())} open markets "
              f"across {len(by_event)} events in {len(cfg.series)} series")

    mapper = MarketMapper(grid_size=model_cfg.max_goals_grid)
    now = _utcnow()
    state = BettingState.load(cfg)
    bankroll_usd = resolve_bankroll(state, client, verbose=verbose) / 100.0

    # The league's ...GAME series anchors fixture identification; the other
    # families share the same event date/team suffix, so resolve fixtures
    # from the game events first.
    game_series = next((t for t in cfg.series if t.endswith("GAME")), None)
    fixture_by_key: dict[str, tuple[str, str]] = {}
    for ev, mkts in by_event.items():
        if game_series and ev.startswith(game_series + "-"):
            hit = match_event_to_fixture(mkts, fixtures)
            if hit:
                fixture_by_key[ev.split("-", 1)[1]] = hit

    candidates: list[Candidate] = []
    per_match: dict[tuple[str, str], list[Candidate]] = {}
    n_unmapped = 0
    for ev, mkts in sorted(by_event.items()):
        key = ev.split("-", 1)[1] if "-" in ev else ev
        pair = fixture_by_key.get(key) or match_event_to_fixture(mkts, fixtures)
        if not pair:
            n_unmapped += len(mkts)
            continue
        home, away = pair
        pred = predict_match(ratings, home, away, neutral=False, cfg=model_cfg)
        pmf = pred.matrix
        espn_kickoff = _fixture_kickoff(fixtures, home, away)
        hours = None
        exp = mkts[0].get("expected_expiration_time")
        if exp:
            hours = (pd.Timestamp(exp).tz_convert(None) - pd.Timestamp(now).tz_convert(None)
                     ).total_seconds() / 3600.0
        for m in mkts:
            mapped = mapper.map_market(m, home, away)
            if mapped is None:
                n_unmapped += 1
                continue
            book = client.get_orderbook(m["ticker"])
            for cand in evaluate_market(mapped, book, pmf, trust, cfg, hours):
                cand.confidence_parts["hours_to_kickoff"] = hours
                cand.confidence_parts["n_eff"] = (
                    ratings.n_eff.get(home, 0.0), ratings.n_eff.get(away, 0.0))
                if espn_kickoff is not None:
                    cand.confidence_parts["espn_kickoff_utc"] = espn_kickoff
                per_match.setdefault((home, away), []).append(cand)
    if verbose and n_unmapped:
        print(f"  ({n_unmapped} markets skipped: no exact fixture/settlement map)")

    # --- joint Kelly sizing per match ---------------------------------------
    for (home, away), cands in per_match.items():
        pred = predict_match(ratings, home, away, neutral=False, cfg=model_cfg)
        bets = [JointBet(key=c.market.ticker + ":" + c.side,
                         indicator=(c.market.indicator if c.side == "yes"
                                    else ~c.market.indicator),
                         cost=c.cost_dollars) for c in cands]
        fractions = joint_kelly(pred.matrix, bets,
                                total_cap=cfg.max_total_fraction_per_match)
        for c, fx in zip(cands, fractions):
            c.full_kelly = float(fx)
            if fx <= 0:
                c.skip_reason = "joint Kelly sized to zero"
                continue
            n_h, n_a = c.confidence_parts["n_eff"]
            hours = c.confidence_parts.get("hours_to_kickoff")
            wanted = contracts_for_stake(fx * cfg.kelly_fraction_max,
                                         bankroll_usd, c.cost_dollars)
            parts = {
                "w_cal": trust.w_cal(c.p_model),
                "w_data": w_data(n_h, n_a),
                "w_liq": w_liq(c.depth_at_touch, max(wanted, 1)),
                "w_time": w_time(hours if hours is not None else 1e9),
            }
            conf = float(np.prod(list(parts.values())))
            lam = kelly_multiplier(conf, cfg.kelly_fraction_min,
                                   cfg.kelly_fraction_max)
            c.confidence_parts.update(parts, confidence=conf)
            c.kelly_multiplier = lam
            stake_fraction = lam * fx

            # hard stake caps (house-favour: min of all)
            stake_usd = min(stake_fraction * bankroll_usd,
                            cfg.max_stake_per_bet_usd,
                            cfg.max_stake_per_bet_pct * bankroll_usd)
            contracts = min(contracts_for_stake(1.0, stake_usd, c.cost_dollars),
                            c.depth_at_touch)
            if contracts < 1:
                c.skip_reason = "stake below one contract after caps"
                continue
            c.contracts = contracts
            c.fee_cents = trading_fee_cents(contracts, c.ask_cents,
                                            cfg.taker_fee_factor)
            c.stake_cents = contracts * c.ask_cents + c.fee_cents
            # final exact re-check at the true size (fee ceiling effects)
            from .fees import breakeven_prob
            be = float(breakeven_prob(contracts, c.ask_cents, cfg.taker_fee_factor))
            if c.p_model - be < cfg.min_edge:
                c.skip_reason = "edge below minimum at final size"
                c.contracts = 0
        candidates.extend(cands)

    slate = [c for c in candidates if c.contracts >= 1 and not c.skip_reason]
    slate.sort(key=lambda c: -(c.edge_raw * c.contracts * c.ask_cents))
    return slate


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def render_report(slate: list[Candidate], cfg: BettingConfig,
                  gated: bool) -> str:
    ts = _utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"Kalshi MLS EV report -- {ts}",
             f"  fee model: taker {cfg.taker_fee_factor} (verify vs current "
             f"schedule); min edge {cfg.min_edge:.0%}",
             f"  news-check gate: {'ON' if gated else 'OFF -- UNSCREENED'}",
             ""]
    if not slate:
        lines.append("No positive-EV candidates passed the gates.")
        return "\n".join(lines)
    hdr = (f"{'match':34s} {'claim':22s} {'side':4s} {'ask':>4s} {'model':>6s} "
           f"{'cal':>6s} {'edge':>6s} {'conf':>5s} {'kelly':>6s} {'n':>4s} "
           f"{'stake':>8s} {'gate':>7s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for c in slate:
        gate = c.gate_verdict or "-"
        lines.append(
            f"{c.market.home[:16]:16s} v {c.market.away[:15]:15s} "
            f"{c.claim[:22]:22s} {c.side:4s} {c.ask_cents:3d}c "
            f"{c.p_model:6.1%} {c.p_calibrated:6.1%} {c.edge_raw:6.1%} "
            f"{c.confidence_parts.get('confidence', 0):5.2f} "
            f"{c.kelly_multiplier:5.2f}x {c.contracts:4d} "
            f"${c.stake_cents/100:7.2f} {gate:>7s}")
    total = sum(c.stake_cents for c in slate) / 100
    lines.append(f"\n  total proposed stake: ${total:.2f}")
    return "\n".join(lines)


def parse_order_selection(answer: str, n_available: int) -> int:
    """How many of the proposed orders to place. Returns 0 to place NOTHING.

    Accepts a count (1..n, clamped to n) or 'all'/'a'. Everything else --
    empty, 0, negative, non-numeric -- returns 0 and aborts the run: with real
    money the least-effort reply must be the least-risky one, so pressing Enter
    never places anything.
    """
    a = (answer or "").strip().lower()
    if a in ("all", "a"):
        return n_available
    try:
        n = int(a)
    except ValueError:
        return 0
    return min(n, n_available) if n > 0 else 0


def _fixture_kickoff(fixtures: pd.DataFrame, home: str, away: str):
    """The exact ESPN kickoff timestamp for a fixture, or None. Used to read the
    true closing line from candlesticks later (tracking.py)."""
    if "kickoff_utc" not in fixtures.columns:
        return None
    m = fixtures[(fixtures["home_team"] == home) & (fixtures["away_team"] == away)]
    if len(m) and pd.notna(m.iloc[0]["kickoff_utc"]):
        return m.iloc[0]["kickoff_utc"]
    return None


def _slate_records(slate: list[Candidate]) -> list[dict]:
    return [{
        "ticker": c.market.ticker, "side": c.side, "claim": c.claim,
        "match": f"{c.market.home} v {c.market.away}",
        # The exact ESPN kickoff -- the closing-line reference. Deliberately NOT
        # falling back to the Kalshi expiration (~game end): a wrong kickoff
        # would read an in-game price as the "close". None -> CLV unavailable.
        "kickoff_utc": c.confidence_parts.get("espn_kickoff_utc"),
        "p_model": round(c.p_model, 4), "p_calibrated": round(c.p_calibrated, 4),
        "price_cents": c.ask_cents, "edge": round(c.edge_raw, 4),
        "confidence": c.confidence_parts.get("confidence"),
        "kelly_multiplier": c.kelly_multiplier, "full_kelly": round(c.full_kelly, 4),
        "contracts": c.contracts, "stake_cents": c.stake_cents,
        "fee_cents": c.fee_cents,
        "gate_verdict": c.gate_verdict, "gate_multiplier": c.gate_multiplier,
        "gate_rationale": c.gate_rationale,
        # for counterfactual tracking of vetoed/reduced bets:
        "counterfactual_contracts": c.confidence_parts.get("pre_gate_contracts",
                                                           c.contracts),
    } for c in slate]


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def run_recommend(args, cfg: BettingConfig = BETTING,
                  client: KalshiClient | None = None,
                  spec=None) -> list[Candidate]:
    client = client or KalshiClient()
    slate = build_slate(cfg, client, min_edge=getattr(args, "min_edge", None),
                        spec=spec)
    gated = False
    if cfg.gate_enabled and not getattr(args, "skip_gate", False):
        from .gate import screen_slate
        slate = screen_slate(slate, cfg)
        gated = True
    else:
        audit("gate_skipped", {"mode": "recommend",
                               "reason": "disabled by flag/config"})
    report = render_report(slate, cfg, gated)
    print()
    print(report)
    n_logged, n_dup = log_recommendations(_slate_records(slate), mode="recommend")
    if n_dup:
        print(f"  ({n_logged} newly logged; {n_dup} already logged this "
              f"market/side -- not duplicated)")
    audit("recommend_run", {"n_candidates": len(slate), "n_logged": n_logged,
                            "n_duplicates_skipped": n_dup, "gated": gated})
    return slate


def run_execute(args, cfg: BettingConfig = BETTING,
                client: KalshiClient | None = None,
                state: BettingState | None = None,
                confirm_input=input, select_input=input,
                spec=None) -> list[dict]:
    """Execute mode. Returns the list of orders actually placed (possibly
    empty). Every rail is checked in order; the first failure stops the run."""
    client = client or KalshiClient()
    state = state or BettingState.load(cfg)
    placed: list[dict] = []

    # Rail 1: kill switch.
    if cfg.kill_switch:
        audit("execute_blocked", {"rail": "kill_switch"})
        print("KILL SWITCH is on (betting config). No orders will be placed.")
        return placed
    # Rail 2: credentials.
    client.require_auth("Execute mode")
    # Reconcile FIRST: settle positions Kalshi no longer holds and book their
    # P&L, so the open-position count and the loss-limit ledger reflect reality
    # before any rail below is evaluated (otherwise stale settled positions
    # would block trading forever and hide realized losses).
    reconcile_positions(state, client)
    # Rail 4 (before doing any work): loss-limit halts.
    breach = state.loss_limits_breached(cfg)
    if breach:
        audit("execute_blocked", {"rail": "loss_limit", "detail": breach})
        print(f"TRADING HALTED: {breach}")
        return placed

    # Live balance becomes the bankroll of record.
    balance = client.get_balance_cents()
    state.bankroll_cents = balance
    state.save()

    slate = build_slate(cfg, client, min_edge=getattr(args, "min_edge", None),
                        spec=spec)

    # The news-check gate is mandatory in execute mode unless explicitly
    # disabled in config (loudly logged). Unscreened or vetoed bets are never
    # placed -- gate.placeable() is the fail-closed filter.
    if cfg.gate_enabled:
        from .gate import placeable, screen_slate
        slate = screen_slate(slate, cfg)
        print()
        print(render_report(slate, cfg, gated=True))
        n_logged, n_dup = log_recommendations(_slate_records(slate), mode="execute")
        slate = placeable(slate)
    else:
        audit("gate_disabled_in_execute", {"warning": "explicit config override"})
        print("WARNING: news-check gate explicitly disabled in config.")
        print()
        print(render_report(slate, cfg, gated=False))
        n_logged, n_dup = log_recommendations(_slate_records(slate), mode="execute")
    if n_dup:
        print(f"  ({n_logged} newly logged; {n_dup} already logged this "
              f"market/side -- not duplicated)")
    if not slate:
        return placed

    # Rail 6: max open positions (runaway backstop; you pick the count below).
    room = cfg.max_open_positions - state.n_open_positions
    if room <= 0:
        audit("execute_blocked", {"rail": "max_open_positions"})
        print(f"Open positions at limit ({cfg.max_open_positions}); not placing.")
        return placed
    if len(slate) > room:
        print(f"  (capping {len(slate)} candidates to {room} -- "
              f"{state.n_open_positions} of {cfg.max_open_positions} slots in use)")
        slate = slate[:room]

    # Rail 3: --live flag + interactive selection + typed confirmation.
    # DRY-RUN DEFAULT.
    if not getattr(args, "live", False):
        print("\nDRY RUN (default). Re-run with --live to place these orders.")
        audit("execute_dry_run", {"n_orders": len(slate)})
        return placed
    if not sys.stdin.isatty():
        audit("execute_blocked", {"rail": "no_tty_for_confirmation"})
        print("Refusing --live without an interactive terminal.")
        return placed

    # You choose how many to place. The slate is ranked by expected edge, so
    # "3" means the three best bets. Enter/0/anything unparseable places NOTHING.
    reply = select_input(
        f"\n{len(slate)} order(s) proposed, ranked best-first. How many do you "
        f"want to place? [1-{len(slate)}, 'all', or 0 to cancel]: ")
    n_selected = parse_order_selection(reply, len(slate))
    if n_selected <= 0:
        audit("execute_aborted", {"rail": "selection_none", "reply": str(reply)})
        print("Nothing selected. No orders placed.")
        return placed
    slate = slate[:n_selected]
    audit("orders_selected", {"n_selected": n_selected, "reply": str(reply)})

    print(f"\nSelected {len(slate)} order(s):")
    for c in slate:
        print(f"  {c.market.home} v {c.market.away} | {c.claim} {c.side} "
              f"{c.contracts} @ {c.ask_cents}c = ${c.stake_cents/100:.2f}")
    total = sum(c.stake_cents for c in slate) / 100
    answer = confirm_input(
        f"\nAbout to place {len(slate)} REAL order(s) totalling ${total:.2f}. "
        f"Type 'PLACE ORDERS' to proceed: ")
    if answer.strip() != "PLACE ORDERS":
        audit("execute_aborted", {"rail": "confirmation_declined"})
        print("Not confirmed. No orders placed.")
        return placed

    # Order placement: IOC limit orders at the evaluated ask, one by one; any
    # error aborts the remainder of the run. Records the ACTUAL fill the
    # exchange reports (IOC can fill partially or not at all), never an assumed
    # one -- so bankroll, P&L, and placement logs reflect reality.
    for c in slate:
        try:
            order = client.place_limit_order(
                c.market.ticker, c.side, c.contracts, c.ask_cents,
                time_in_force=cfg.order_time_in_force,
                self_trade_prevention=cfg.order_self_trade_prevention)
        except OrderError as e:
            audit("order_error_abort", {"ticker": c.market.ticker,
                                        "error": str(e)})
            print(f"ORDER ERROR on {c.market.ticker}: {e}\nAborting the run.")
            break
        filled = order.get("filled", 0)
        fill_px = order.get("fill_price_cents", c.ask_cents)
        fee = order.get("fee_cents", 0)
        audit("order_placed", {"ticker": c.market.ticker, "side": c.side,
                               "requested": c.contracts, "filled": filled,
                               "fill_price_cents": fill_px, "fee_cents": fee,
                               "order_id": order.get("order_id")})
        if filled <= 0:
            print(f"  0 filled for {c.market.ticker} (IOC, no liquidity at "
                  f"{c.ask_cents}c); nothing recorded.")
            continue
        state.record_fill(c.market.ticker, c.side, filled, fill_px, fee)
        state.save()
        # Log the ACTUAL execution so realized P&L can count it.
        log_placement({"ticker": c.market.ticker, "side": c.side,
                       "contracts": filled, "price_cents": fill_px,
                       "fee_cents": fee, "order_id": order.get("order_id")})
        placed.append(order)
        print(f"  filled {filled}/{c.contracts} x {c.market.ticker} {c.side} "
              f"@ {fill_px}c (order {order.get('order_id')})")
    return placed
