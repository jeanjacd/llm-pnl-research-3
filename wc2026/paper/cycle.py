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

import datetime as dt
import json
import os

from ..board.orchestrator import run_board
from ..data import loader
from ..decision.calculator import BUY_NOW, PLACE_LIMIT, UNSUPPORTED, build_case
from ..eval.tune import effective_model
from ..leagues import all_leagues, get_league
from ..venues.base import changed_since, snapshot_record
from .broker import PaperPortfolio

SNAPSHOT_DIR = os.path.join("data", "snapshots")


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


def run_cycle(league_ids=None, state_path: str | None = None,
              summary_path: str | None = None, providers=None,
              invoke=None, board_enabled: bool = True,
              verbose: bool = True) -> dict:
    """Run one full paper cycle. Returns a sanitised summary dict."""
    from ..model.ratings import build_team_strength
    from ..sim.match import predict_match

    portfolio = PaperPortfolio.load(state_path)
    stats = {"started_at": _utcnow().isoformat(), "leagues": {},
             "cases_built": 0, "cases_skipped_unchanged": 0,
             "unsupported": 0, "board_run": 0, "orders_submitted": 0,
             "fills": 0, "expired": 0}

    # 6. revalue: expire anything past its deadline before anything new.
    stats["expired"] = len(portfolio.expire_due())

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
        train = loader.training_matches(df)
        if train.empty:
            league_stats["skipped"] = "no training data"
            stats["leagues"][spec.league_id] = league_stats
            continue
        ratings = build_team_strength(train, as_of=train["date"].max(),
                                      cfg=model_cfg, verbose=False,
                                      adjustments_path=spec.adjustments_json)
        fixtures = df[~df["played"]][["date", "home_team", "away_team"]]

        for provider in (providers or []):
            try:
                instruments = provider.discover(spec, fixtures=fixtures) \
                    if provider.venue == "polymarket" \
                    else provider.discover(spec)
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

                    decision = {"action": "PAPER_" + case.action,
                                "limit_price_cents":
                                    case.max_limit_price_cents,
                                "contracts": 1.0}
                    if board_enabled:
                        stats["board_run"] += 1
                        verdict = run_board(
                            case, {"home": leg.home, "away": leg.away,
                                   "league_id": spec.league_id,
                                   "kickoff_utc": leg.kickoff_utc},
                            invoke=invoke) if invoke else \
                            run_board(case,
                                      {"home": leg.home, "away": leg.away,
                                       "league_id": spec.league_id,
                                       "kickoff_utc": leg.kickoff_utc})
                        if verdict["action"] not in ("PAPER_BUY_NOW",
                                                     "PAPER_PLACE_LIMIT"):
                            continue
                        judged = verdict.get("judge") or {}
                        decision = {"action": verdict["action"],
                                    "limit_price_cents":
                                        judged.get("limit_price_cents"),
                                    "contracts": judged.get("contracts", 1.0)}

                    price = decision.get("limit_price_cents")
                    size = decision.get("contracts") or 0
                    if not price or size <= 0:
                        continue
                    try:
                        order = portfolio.submit(
                            case.case_id, instrument.venue,
                            instrument.instrument_id, side, int(price),
                            float(size), league_id=spec.league_id,
                            expires_at=case.actionable_until)
                    except Exception as exc:                  # noqa: BLE001
                        league_stats.setdefault("errors", []).append(
                            "submit: %s" % str(exc)[:80])
                        continue
                    stats["orders_submitted"] += 1
                    league_stats["submitted"] += 1
                    if decision["action"] == "PAPER_BUY_NOW":
                        portfolio.fill_marketable(order.order_id,
                                                  instrument.book)
                        if order.filled_size > 0:
                            stats["fills"] += 1

        stats["leagues"][spec.league_id] = league_stats

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
             "",
             "| metric | value |", "|---|---|"]
    for key in ("cases_built", "cases_skipped_unchanged", "unsupported",
                "board_run", "orders_submitted", "fills", "expired"):
        lines.append("| %s | %s |" % (key, stats.get(key, 0)))
    port = stats.get("portfolio") or {}
    for key in ("cash_usd", "reserved_usd", "n_open_orders",
                "n_positions_open", "n_settled", "realized_pnl_usd",
                "fill_rate"):
        if key in port:
            lines.append("| %s | %s |" % (key, port[key]))
    lines += ["", "| league | instruments | supported | changed | cases | submitted |",
              "|---|---|---|---|---|---|"]
    for league, s in sorted((stats.get("leagues") or {}).items()):
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (league, s.get("instruments", 0), s.get("supported", 0),
                        s.get("changed", 0), s.get("cases", 0),
                        s.get("submitted", 0)))
    return "\n".join(lines)
