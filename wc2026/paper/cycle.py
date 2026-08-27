"""
paper/cycle.py
==============
One scheduled paper-trading cycle, in the order the mission specifies:

  1. restore prior paper state          8. skip unchanged case hashes
  2. refresh near-term fixtures         9. pre-screen unsupported / nonviable
  3. discover markets (both venues)    10. sealed quant + coach
  4. save timestamped snapshots        11. judge, only after both validate
  5. synchronise settlements           12. submit approved decisions
  6. revalue open paper orders         13. update positions and P&L
  7. build deterministic cases         14. sanitised summary + persist

Idempotence is a requirement, not a nicety: the same cycle run twice must not
double-submit an order, double-fill one, or settle a position twice. Orders are
keyed by (case_id, instrument, side, price); settlement refuses a second call.

PAPER ONLY. Nothing here can reach a venue.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
from collections import Counter

from ..board.orchestrator import run_board
from ..data import loader
from ..decision.calculator import BUY_NOW, PLACE_LIMIT, UNSUPPORTED, build_case
from ..eval.tune import effective_model
from ..leagues import all_leagues, get_league
from ..venues.base import changed_since, snapshot_record
from .broker import PaperPortfolio
from .clv import capture_closing_lines, clv_summary
from .fills import KalshiFillProbe, PolymarketFillProbe, replay_fills
from .selection import (
    fixture_key,
    hours_to_kickoff,
    select_one_per_fixture,
)
from .settlement import settle_portfolio

SNAPSHOT_DIR = os.path.join("data", "snapshots")
# The board's own audit trail for PAPER runs. Deliberately not the default
# `data/betting/board_audit.jsonl`, which is the real-money directory: paper
# reasoning has no business there, and on a runner that path is destroyed with
# the container, so every explanation of every DEFER was being discarded.
# `data/paper/` is uploaded as a run artifact (30-day retention) but is NOT
# pushed to the public state branch -- the ledger is published, the transcripts
# are not.
PAPER_BOARD_LOG = os.path.join("data", "paper", "board_audit.jsonl")


def board_reason(verdict: dict) -> tuple:
    """(who decided, one short sentence) -- never the full transcript.

    The run summary is written to a public Action log, so this returns the
    member's own one-line rationale rather than its findings, sources or
    prompt. The complete record stays in PAPER_BOARD_LOG.
    """
    failure = str(verdict.get("failure") or "")
    lowered = failure.lower()
    quant = verdict.get("quant") or {}
    coach = verdict.get("coach") or {}
    judge = verdict.get("judge") or {}
    if "quant" in lowered:
        return "quant", str(quant.get("rationale") or failure)
    if "coach" in lowered:
        text = str(coach.get("rationale") or "")
        if not text:
            reruns = coach.get("required_reruns") or []
            text = "; ".join(str(r) for r in reruns[:2])
        return "coach", (text or failure)
    if judge:
        return "judge", str(judge.get("decisive_reason") or failure)
    return "board", (failure or "no reason recorded")


@dataclasses.dataclass
class Candidate:
    """One placeable case, held until the whole slate can be ranked."""
    case: object
    instrument: object
    leg: object
    side: str
    claim: str
    league_id: str
    fixture_key: str


def _utcnow():
    return dt.datetime.now(dt.timezone.utc)


def _load_previous_snapshot(league_id: str, venue: str):
    path = os.path.join(SNAPSHOT_DIR, "%s_%s_latest.json" % (league_id, venue))
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_snapshot(record: dict, league_id: str, venue: str) -> str:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    stamp = _utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive = os.path.join(SNAPSHOT_DIR,
                           "%s_%s_%s.json" % (league_id, venue, stamp))
    with open(archive, "w", encoding="utf-8") as fh:
        json.dump(record, fh, default=str)
    latest = os.path.join(SNAPSHOT_DIR, "%s_%s_latest.json" % (league_id, venue))
    with open(latest, "w", encoding="utf-8") as fh:
        json.dump(record, fh, default=str)
    return archive


def probability_for(claim: str, prediction) -> float | None:
    """Model probability for a supported claim, from the exact scoreline grid."""
    import numpy as np
    matrix = prediction.matrix
    size = matrix.shape[0]
    k = np.arange(size)
    i, j = np.meshgrid(k, k, indexing="ij")
    negate = claim.startswith("not_")
    base = claim[4:] if negate else claim

    mask = None
    if base == "home_win":
        mask = i > j
    elif base == "away_win":
        mask = i < j
    elif base == "draw":
        mask = i == j
    elif base == "btts":
        mask = (i >= 1) & (j >= 1)
    elif base.startswith("total_over_"):
        mask = (i + j) > float(base.rsplit("_", 1)[1])
    elif base.startswith("total_under_"):
        mask = (i + j) < float(base.rsplit("_", 1)[1])
    elif base.startswith("home_over_"):
        mask = i > float(base.rsplit("_", 1)[1])
    elif base.startswith("away_over_"):
        mask = j > float(base.rsplit("_", 1)[1])
    elif base.startswith("home_wins_by_over_"):
        mask = (i - j) > float(base.rsplit("_", 1)[1])
    elif base.startswith("away_wins_by_over_"):
        mask = (j - i) > float(base.rsplit("_", 1)[1])
    elif base.startswith("score_"):
        try:
            hg, ag = base.split("_", 1)[1].split("-")
            hg, ag = int(hg), int(ag)
        except ValueError:
            return None
        if hg >= size or ag >= size:
            return None
        mask = (i == hg) & (j == ag)
    if mask is None:
        return None
    p = float(matrix[mask].sum())
    return 1.0 - p if negate else p


def run_maintenance(state_path: str | None = None,
                    summary_path: str | None = None, providers=None,
                    league_ids=None, fill_probes=None,
                    verbose: bool = True) -> dict:
    """The cheap half of the loop: no model calls, no market discovery.

    Replays resting fills from venue history, expires what has run out,
    captures closing lines and settles finished matches. It is separated from
    `run_cycle` because the two halves have completely different cost
    profiles: this one costs a handful of HTTP calls and can therefore run
    EVERY day, while boarding costs minutes of model time per fixture and only
    needs to run when a fixture is near its lead time.

    Running it daily is what makes the numbers real. Midweek fixtures settle
    on the day they finish rather than waiting for the next matchday, and a
    resting order that traded through on a Tuesday is recorded as a fill
    instead of expiring unexamined.
    """
    portfolio = PaperPortfolio.load(state_path)
    stats = {"started_at": _utcnow().isoformat(), "mode": "maintenance",
             "leagues": {}, "cases_built": 0, "board_run": 0,
             "orders_submitted": 0, "fills": 0, "resting_fills": 0,
             "expired": 0, "settled": 0, "settled_pnl_usd": 0.0}

    probes = fill_probes if fill_probes is not None else _default_probes(providers)
    if probes:
        replay = replay_fills(portfolio, probes)
        stats["resting_fills"] = replay["filled"]
        stats["fills"] = replay["filled"]
        stats["fill_replay"] = replay
        stats["clv_capture"] = capture_closing_lines(portfolio, probes)

    stats["expired"] = len(portfolio.expire_due())

    # Only the leagues we actually hold something in need loading.
    wanted = set(league_ids or [])
    if not wanted:
        wanted = {p.league_id for p in portfolio.positions.values()
                  if p.league_id and not p.settled}
    frames = {}
    for league in sorted(wanted):
        try:
            frames[league] = loader.load_league(get_league(league),
                                                tiers="training")
        except Exception as exc:                              # noqa: BLE001
            stats["leagues"][league] = {"error": str(exc)[:120]}
    if frames:
        settled = settle_portfolio(portfolio, frames)
        stats["settled"] = settled["settled"]
        stats["settled_pnl_usd"] = round(settled["pnl_cents"] / 100.0, 2)
        stats["settlement"] = settled

    stats["clv"] = clv_summary(portfolio)
    portfolio.save(state_path)
    stats["portfolio"] = portfolio.summary()
    stats["finished_at"] = _utcnow().isoformat()
    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(render_summary(stats))
    if verbose:
        print(render_summary(stats))
    return stats


def _default_probes(providers) -> dict:
    """A fill probe per venue actually in play.

    Built from the providers so a cycle run against one venue does not make
    network calls to the other, and so a caller that passes no providers (a
    test, or a settle-only run) gets no probes and no traffic.
    """
    probes = {}
    for provider in (providers or []):
        venue = getattr(provider, "venue", None)
        if venue == "kalshi" and venue not in probes:
            # Reuse the provider's authenticated client; candlesticks are a
            # public GET but the client carries the rate limiting.
            probes[venue] = KalshiFillProbe(getattr(provider, "client", None))
        elif venue == "polymarket" and venue not in probes:
            probes[venue] = PolymarketFillProbe(
                getattr(provider, "session", None))
    return probes


def run_cycle(league_ids=None, state_path: str | None = None,
              summary_path: str | None = None, providers=None,
              invoke=None, board_enabled: bool = True,
              verbose: bool = True, fill_probes=None,
              settle_enabled: bool = True) -> dict:
    """Run one full paper cycle. Returns a sanitised summary dict."""
    from ..model.ratings import build_team_strength
    from ..sim.match import predict_match

    portfolio = PaperPortfolio.load(state_path)
    stats = {"started_at": _utcnow().isoformat(), "leagues": {},
             "cases_built": 0, "cases_skipped_unchanged": 0,
             "unsupported": 0, "board_run": 0, "orders_submitted": 0,
             "fills": 0, "expired": 0, "resting_fills": 0, "settled": 0,
             "settled_pnl_usd": 0.0}

    # Replay the tape FIRST. An order whose market traded through at 10:40 and
    # whose deadline passed at 11:00 was filled, not expired -- running
    # `expire_due` first would silently discard the fill and understate both
    # the fill rate and the P&L.
    probes = fill_probes if fill_probes is not None else _default_probes(providers)
    if probes:
        replay = replay_fills(portfolio, probes)
        stats["resting_fills"] = replay["filled"]
        stats["fills"] += replay["filled"]
        stats["fill_replay"] = replay
        # Capture the closing line before settling: it exists at kick-off,
        # whereas the result may not be ingested for hours, and CLV is the
        # faster-converging of the two measurements.
        stats["clv_capture"] = capture_closing_lines(portfolio, probes)

    # 6. revalue: expire anything past its deadline before anything new.
    stats["expired"] = len(portfolio.expire_due())

    # Settlement needs PLAYED matches, which is the same table the model is
    # fitted from, so it is collected as each league is loaded rather than
    # re-read afterwards.
    settle_frames = {}
    candidates = []

    specs = ([get_league(l) for l in league_ids] if league_ids
             else all_leagues())
    for spec in specs:
        league_stats = {"instruments": 0, "supported": 0, "changed": 0,
                        "cases": 0, "placeable": 0, "submitted": 0}
        model_cfg, tuned = effective_model(spec)
        if not tuned:
            league_stats["skipped"] = "untuned -- forecasts not validated"
            stats["leagues"][spec.league_id] = league_stats
            continue

        df = loader.load_league(spec, tiers="training")
        settle_frames[spec.league_id] = df
        train = loader.training_matches(df)
        if train.empty:
            league_stats["skipped"] = "no training data"
            stats["leagues"][spec.league_id] = league_stats
            continue
        ratings = build_team_strength(train, as_of=train["date"].max(),
                                      cfg=model_cfg, verbose=False,
                                      adjustments_path=spec.adjustments_json)
        # kickoff_utc travels with the fixtures: it is the only reliable
        # kick-off for venues that publish a settlement time instead.
        fixture_cols = [c for c in ("date", "home_team", "away_team",
                                    "kickoff_utc") if c in df.columns]
        fixtures = df[~df["played"]][fixture_cols]

        for provider in (providers or []):
            try:
                # BOTH providers need the fixture table: without it a leg
                # carries no home/away, so no claim resolves and the venue
                # silently contributes zero cases. Kalshi was excluded
                # here and produced exactly that -- 223 "supported"
                # instruments, not one of them priceable.
                instruments = provider.discover(spec, fixtures=fixtures)
            except Exception as exc:                          # noqa: BLE001
                league_stats.setdefault("errors", []).append(
                    "%s: %s" % (provider.venue, str(exc)[:120]))
                continue

            league_stats["instruments"] += len(instruments)
            league_stats["supported"] += sum(1 for i in instruments
                                             if i.supported)
            stats["unsupported"] += sum(1 for i in instruments
                                        if not i.supported)

            previous = _load_previous_snapshot(spec.league_id, provider.venue)
            fresh = changed_since(previous, instruments)
            league_stats["changed"] += len(fresh)
            stats["cases_skipped_unchanged"] += len(instruments) - len(fresh)
            _save_snapshot(snapshot_record(instruments, spec.league_id,
                                           provider.venue),
                           spec.league_id, provider.venue)

            for instrument in fresh:
                if not instrument.supported:
                    continue
                leg = instrument.legs[0]
                if not (leg.home and leg.away):
                    continue
                try:
                    prediction = predict_match(ratings, leg.home, leg.away,
                                               neutral=False, cfg=model_cfg)
                except Exception:                             # noqa: BLE001
                    continue      # unrated team -> fail closed, no forecast
                for side in ("yes", "no"):
                    claim = leg.claim if side == "yes" else "not_" + leg.claim
                    p = probability_for(claim, prediction)
                    if p is None:
                        continue
                    case = build_case(instrument, p, p, side=side)
                    stats["cases_built"] += 1
                    league_stats["cases"] += 1
                    if case.action == UNSUPPORTED:
                        continue
                    if case.action not in (BUY_NOW, PLACE_LIMIT):
                        continue
                    league_stats["placeable"] += 1
                    # COLLECTED, not boarded here. Selection is a decision
                    # about the whole slate -- one market per fixture -- and
                    # cannot be made while still walking one venue's markets.
                    candidates.append(Candidate(
                        case=case, instrument=instrument, leg=leg, side=side,
                        claim=claim, league_id=spec.league_id,
                        fixture_key=fixture_key(spec.league_id, leg.home,
                                                leg.away, leg.kickoff_utc)))

        stats["leagues"][spec.league_id] = league_stats

    if settle_enabled and settle_frames:
        settled = settle_portfolio(portfolio, settle_frames)
        stats["settled"] = settled["settled"]
        stats["settled_pnl_usd"] = round(settled["pnl_cents"] / 100.0, 2)
        stats["settlement"] = settled

    # --- selection, then the board ----------------------------------------
    selected, dropped = select_one_per_fixture(candidates, portfolio.boarded)
    stats["placeable_cases"] = len(candidates)
    stats["fixtures_selected"] = len(selected)
    # Never a silent cap: a dropped candidate is reported with its reason,
    # because "we boarded 9 of 307" and "we looked at everything" must not
    # read the same in the summary.
    stats["candidates_dropped"] = len(dropped)
    stats["drop_reasons"] = dict(Counter(reason for _, reason in dropped))

    coach_cache: dict = {}
    stats["board_decisions"] = []
    for cand in selected:
        case, instrument, leg = cand.case, cand.instrument, cand.leg
        fixture = {"home": leg.home, "away": leg.away,
                   "league_id": cand.league_id,
                   "kickoff_utc": leg.kickoff_utc}
        decision = {"action": "PAPER_" + case.action,
                    "limit_price_cents": case.max_limit_price_cents,
                    "contracts": 1.0}
        if board_enabled:
            stats["board_run"] += 1
            kwargs = {"coach_cache": coach_cache, "log_path": PAPER_BOARD_LOG}
            if invoke:
                kwargs["invoke"] = invoke
            verdict = run_board(case, fixture, **kwargs)
            decided_by, why = board_reason(verdict)
            # Attempts accumulate so a deferral is retried ONCE, never
            # indefinitely -- see selection.RETRY_MAX_ATTEMPTS.
            prior = portfolio.boarded.get(cand.fixture_key) or {}
            attempts = int(prior.get("attempts") or 0) + 1
            portfolio.boarded[cand.fixture_key] = {
                "attempts": attempts,
                "first_boarded_at": prior.get("first_boarded_at")
                                    or _utcnow().isoformat(),
                "hours_to_kickoff": round(
                    hours_to_kickoff(leg.kickoff_utc) or 0.0, 1),
                "ts": _utcnow().isoformat(), "action": verdict["action"],
                "case_id": case.case_id, "claim": cand.claim,
                "home": leg.home, "away": leg.away,
                # Kept so a DEFER is answerable months later without the
                # transcript, which does not survive the runner.
                "decided_by": decided_by, "reason": why[:400]}
            stats["board_decisions"].append(
                dict(portfolio.boarded[cand.fixture_key]))
            if verdict["action"] not in ("PAPER_BUY_NOW", "PAPER_PLACE_LIMIT"):
                continue
            judged = verdict.get("judge") or {}
            decision = {"action": verdict["action"],
                        "limit_price_cents": judged.get("limit_price_cents"),
                        "contracts": judged.get("contracts", 1.0)}
        else:
            prior = portfolio.boarded.get(cand.fixture_key) or {}
            portfolio.boarded[cand.fixture_key] = {
                "attempts": int(prior.get("attempts") or 0) + 1,
                "first_boarded_at": prior.get("first_boarded_at")
                                    or _utcnow().isoformat(),
                "ts": _utcnow().isoformat(), "action": decision["action"],
                "case_id": case.case_id, "claim": cand.claim,
                "home": leg.home, "away": leg.away,
                "decided_by": "deterministic", "reason": "board disabled"}

        price = decision.get("limit_price_cents")
        size = decision.get("contracts") or 0
        if not price or size <= 0:
            continue
        league_stats = stats["leagues"].setdefault(cand.league_id, {})
        try:
            order = portfolio.submit(
                case.case_id, instrument.venue, instrument.instrument_id,
                cand.side, int(price), float(size), league_id=cand.league_id,
                expires_at=case.actionable_until,
                # Carried so the position can settle itself later without
                # re-querying a market that has closed, or re-running venue
                # name matching weeks on.
                claim=cand.claim, home_team=leg.home, away_team=leg.away,
                kickoff_utc=leg.kickoff_utc)
        except Exception as exc:                              # noqa: BLE001
            league_stats.setdefault("errors", []).append(
                "submit: %s" % str(exc)[:80])
            continue
        stats["orders_submitted"] += 1
        league_stats["submitted"] = league_stats.get("submitted", 0) + 1
        if decision["action"] == "PAPER_BUY_NOW":
            portfolio.fill_marketable(order.order_id, instrument.book)
            if order.filled_size > 0:
                stats["fills"] += 1

    stats["clv"] = clv_summary(portfolio)
    portfolio.save(state_path)
    stats["portfolio"] = portfolio.summary()
    stats["finished_at"] = _utcnow().isoformat()

    if summary_path:
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(render_summary(stats))
    if verbose:
        print(render_summary(stats))
    return stats


def render_summary(stats: dict) -> str:
    """A SANITISED summary: counts and P&L only.

    No prompts, no model transcripts, no raw authenticated payloads, no
    per-order identifiers -- this is written to a public Action log.
    """
    lines = ["## Paper cycle %s" % stats.get("started_at", ""),
             "",
             "PAPER TRADING ONLY -- no real orders exist.",
             ""]

    clv = stats.get("clv") or {}
    if clv.get("n_bets"):
        # CLV leads because it converges roughly 4x faster than P&L, and the
        # fixture count sits beside it because that -- not the bet count -- is
        # the sample size any significance claim rests on.
        lines += ["### Closing line value (the faster measurement)", "",
                  "| metric | value |", "|---|---|",
                  "| CLV per fixture | %s c |" % clv.get("clv_cents"),
                  "| t-stat (on fixtures) | %s |" % clv.get("t_stat"),
                  "| independent fixtures | %s |" % clv.get("n_fixtures"),
                  "| bets behind them | %s |" % clv.get("n_bets"),
                  "| bets per fixture | %s |" % clv.get("bets_per_fixture"),
                  ""]
        if (clv.get("n_fixtures") or 0) < 246:
            lines += ["_%d of ~246 fixtures needed to resolve a 0.5c edge at "
                      "80%% power; below that this figure is noise._"
                      % clv["n_fixtures"], ""]

    lines += ["### This cycle", "", "| metric | value |", "|---|---|"]
    for key in ("cases_built", "cases_skipped_unchanged", "unsupported",
                "placeable_cases", "fixtures_selected", "candidates_dropped",
                "board_run", "orders_submitted", "fills", "resting_fills",
                "expired", "settled", "settled_pnl_usd"):
        if key in stats:
            lines.append("| %s | %s |" % (key, stats.get(key, 0)))

    # A cap that is not reported reads as "we looked at everything".
    for reason, count in sorted((stats.get("drop_reasons") or {}).items()):
        lines.append("| dropped: %s | %s |" % (reason, count))

    # WHY each fixture went the way it did. Without this a DEFER is only ever
    # a count, and the explanation lives in a file that dies with the runner.
    decisions = stats.get("board_decisions") or []
    if decisions:
        lines += ["", "### Board decisions this cycle", "",
                  "| fixture | claim | h to k/o | attempt | action | "
                  "decided by | why |",
                  "|---|---|---|---|---|---|---|"]
        for d in decisions:
            why = " ".join(str(d.get("reason") or "").split())
            if len(why) > 180:
                why = why[:177] + "..."
            attempt = int(d.get("attempts") or 1)
            lines.append("| %s v %s | %s | %s | %s | %s | %s | %s |"
                         % (d.get("home") or "?", d.get("away") or "?",
                            d.get("claim") or "?", d.get("hours_to_kickoff", "?"),
                            "retry" if attempt > 1 else "1st",
                            d.get("action") or "?", d.get("decided_by") or "?",
                            why.replace("|", "/") or "-"))
        lines += ["", "_Full board transcripts are in the run's artifacts "
                      "(`data/paper/board_audit.jsonl`, 30-day retention); "
                      "they are deliberately not published to the state "
                      "branch._"]

    port = stats.get("portfolio") or {}
    lines += ["", "### Portfolio", "", "| metric | value |", "|---|---|"]
    for key in ("cash_usd", "reserved_usd", "n_open_orders",
                "n_positions_open", "n_settled", "n_fixtures_boarded",
                "realized_pnl_usd", "fees_paid_usd", "fill_rate"):
        if key in port:
            lines.append("| %s | %s |" % (key, port[key]))

    settlement = stats.get("settlement") or {}
    blocked = {k: v for k, v in settlement.items()
               if k not in ("settled", "pnl_cents", "problems") and v}
    if blocked:
        lines += ["", "_Positions left open: %s_"
                  % ", ".join("%s=%s" % kv for kv in sorted(blocked.items()))]

    lines += ["", "| league | instruments | supported | changed | cases | submitted |",
              "|---|---|---|---|---|---|"]
    for league, st in sorted((stats.get("leagues") or {}).items()):
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (league, st.get("instruments", 0), st.get("supported", 0),
                        st.get("changed", 0), st.get("cases", 0),
                        st.get("submitted", 0)))
    return "\n".join(lines)
