"""
cli.py
======
Command-line entry point:  python -m wc2026 <command> --league <id> [options]

Five leagues are modelled independently: mls, premier_league, la_liga,
bundesliga, ligue_1. There is no implicit "active" league -- every command that
touches data takes `--league`.

  leagues                     list registered leagues and their status
  update    --league|--all    ingest results + fixtures for a league
  fit       --league          fit ratings; print the strength ranking
  match     --league H A      probabilistic match report (home side first)
  backtest  --league          walk-forward calibration report
  evaluate  --league|--all    nested DEV/CAL/HOLDOUT evaluation (headline)
  tune      --league|--all    select hyperparameters on DEV only
  recommend --league          Kalshi EV report (places nothing)
  execute   --league          gated execution (DRY-RUN default)
  track                       resolve recommendations: CLV + P&L
  tournament                  legacy: simulate the 2026 WC from the archive
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

from .config import CONFIG, WC_FITTED_PARAMS_JSON, WC_RESULTS_CSV
from .data import loader
from .eval.backtest import run_backtest
from .eval.nested import run_nested, write_report
from .eval.tune import effective_model, tune_league
from .leagues import all_leagues, get_league, league_ids
from .model.ratings import build_team_strength, calibrate_to_tournament, strength_table
from .sim.match import predict_match

FIT_MAX_AGE_DAYS = 8 * 365


def _spec(args):
    return get_league(args.league)


def _targets(args):
    """The league specs a command should act on (--all or --league)."""
    if getattr(args, "all", False):
        return all_leagues()
    return [get_league(args.league)]


def _model_for(spec, quiet: bool = False):
    cfg, tuned = effective_model(spec)
    if not quiet:
        if tuned:
            print("(%s: CV-tuned params half_life=%s rho=%s blend_k=%s)"
                  % (spec.league_id, cfg.half_life_days, cfg.rho, cfg.blend_k))
        else:
            print("(%s: UNTUNED defaults -- forecasts are not validated)"
                  % spec.league_id)
    return cfg


def _load_ratings(spec, cfg, use_adjustments: bool = False, verbose=True):
    df = loader.load_league(spec, tiers="training")
    train = loader.training_matches(df)
    if train.empty:
        sys.exit("%s: no played matches. Run: python -m wc2026 update --league %s"
                 % (spec.league_id, spec.league_id))
    ratings = build_team_strength(
        train, as_of=train["date"].max(), cfg=cfg,
        max_age_days=FIT_MAX_AGE_DAYS, verbose=verbose,
        adjustments_path=spec.adjustments_json if use_adjustments else None)
    return df, train, ratings


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_leagues(args):
    print("%-16s %-26s %-9s %-9s %-8s %s"
          % ("league", "name", "provider", "tradeable", "tuned", "matches"))
    for spec in all_leagues():
        _, tuned = effective_model(spec)
        n = "-"
        if os.path.exists(spec.matches_csv):
            with open(spec.matches_csv, encoding="utf-8") as fh:
                n = str(sum(1 for _ in fh) - 1)
        print("%-16s %-26s %-9s %-9s %-8s %s"
              % (spec.league_id, spec.display_name,
                 spec.primary_feed.provider_slug, spec.tradeable, tuned, n))


def cmd_update(args):
    from .data.espn import ingest_league
    for spec in _targets(args):
        manifest = ingest_league(spec, verbose=True)
        print("  %s: %d rows, %d teams, sha %s"
              % (spec.league_id, manifest["n_rows"], manifest["n_teams"],
                 manifest["matches_csv_sha256"][:12]))


def cmd_fit(args):
    spec = _spec(args)
    cfg = _model_for(spec)
    _, train, ratings = _load_ratings(spec, cfg)
    print("\nFitted on %s matches through %s"
          % (format(len(train), ","), ratings.as_of.date()))
    print("  intercept=%.3f  home_adv=%.3f  rho=%.3f"
          % (ratings.intercept, ratings.home_adv, ratings.rho))
    table = strength_table(ratings, cfg)
    print("\nTop %d by net strength:" % args.top)
    print(table.head(args.top).to_string(index=False,
                                         float_format=lambda v: "%.3f" % v))


def cmd_match(args):
    from .model.dixon_coles import UnknownTeamError
    spec = _spec(args)
    cfg = _model_for(spec)
    _, _, ratings = _load_ratings(spec, cfg, use_adjustments=True)
    try:
        pred = predict_match(ratings, args.home, args.away,
                             neutral=args.neutral, cfg=cfg)
    except UnknownTeamError as exc:
        sys.exit("%s\nKnown teams include: %s"
                 % (exc, ", ".join(sorted(ratings.attack)[:8])))
    print()
    print(pred.report())


def cmd_backtest(args):
    spec = _spec(args)
    cfg = _model_for(spec)
    df = loader.load_league(spec, tiers="training")
    res = run_backtest(df, test_start=args.start or spec.eval_start,
                       test_end=args.end or spec.eval_end,
                       max_age_days=FIT_MAX_AGE_DAYS, cfg=cfg,
                       min_train=args.min_train, verbose=args.verbose,
                       test_tiers=spec.scoring_tiers, league_id=spec.league_id,
                       with_uncertainty=True)
    print()
    print(res.report())


def cmd_evaluate(args):
    for spec in _targets(args):
        cfg = _model_for(spec, quiet=True)
        df = loader.load_league(spec, tiers="training")
        print("\n=== %s ===" % spec.league_id)
        report = run_nested(df, dataclasses.replace(spec, model=cfg),
                            min_train=args.min_train, verbose=args.verbose)
        path = write_report(report, spec)
        # Persist the DEV-only selection as this league's fitted parameters.
        # Same selection, same region -- no additional information is used.
        with open(spec.fitted_params_json, "w", encoding="utf-8") as fh:
            json.dump({"league_id": spec.league_id,
                       "best": report["selection"]["best"],
                       "selected_on": "DEV region only",
                       "dev_window": report["split"]["dev"],
                       "inner_log_loss": report["selection"]["inner_log_loss"],
                       "stability": report["selection"]["stability"],
                       "caveat": "selection score, not a performance estimate"},
                      fh, indent=2)
        # Calibration for the 1X2 family, fitted on the CAL window only.
        os.makedirs(spec.calibration_dir, exist_ok=True)
        with open(os.path.join(spec.calibration_dir, "1x2.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"league_id": spec.league_id, "family": "1x2",
                       "fitted_on": "calibration window only",
                       "window": report["split"]["calibration"],
                       "n_matches": report["calibration_window"]["n"],
                       "bins": report["calibration_window"]["bins"]},
                      fh, indent=2)
        hold = report["holdout"]
        print("  HOLDOUT n=%d  log-loss %.4f (base %.4f)  RPS %.4f"
              % (hold["n"], hold["metrics"]["log_loss"],
                 hold["baseline"]["log_loss"], hold["metrics"]["rps"]))
        if "calibration_slope" in hold.get("extras", {}):
            print("  calibration slope %.3f  ECE %.4f"
                  % (hold["extras"]["calibration_slope"], hold["extras"]["ece"]))
        print("  wrote %s" % path)


def cmd_tune(args):
    for spec in _targets(args):
        df = loader.load_league(spec, tiers="training")
        print("\n=== tuning %s ===" % spec.league_id)
        tune_league(df, spec, min_train=args.min_train, verbose=args.verbose)


def cmd_recommend(args):
    from .betting.pipeline import run_recommend
    run_recommend(args, spec=_spec(args))


def cmd_execute(args):
    from .betting.kalshi import ConfigError
    from .betting.pipeline import run_execute
    try:
        run_execute(args, spec=_spec(args))
    except ConfigError as exc:
        sys.exit(str(exc))


def cmd_track(args):
    from .betting import tracking
    from .betting.bankroll import BettingState, reconcile_positions
    from .betting.config import BETTING
    from .betting.kalshi import KalshiClient
    client = KalshiClient()
    lookup = None
    if getattr(args, "recompute", False):
        frames = []
        for spec in all_leagues():
            if os.path.exists(spec.matches_csv):
                frames.append(loader.load_league(spec))
        if frames:
            import pandas as pd
            lookup = _make_kickoff_lookup(pd.concat(frames, ignore_index=True))
    tracking.resolve(client=client, recompute=getattr(args, "recompute", False),
                     kickoff_lookup=lookup)
    if client.authenticated:
        reconcile_positions(BettingState.load(BETTING), client)


def _make_kickoff_lookup(df):
    """fn(rec) -> ISO kickoff | None, mapping a logged bet to the exact kickoff."""
    import re

    import pandas as pd
    months = {m: i for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
         "NOV", "DEC"], start=1)}
    fixtures = df[["date", "home_team", "away_team", "kickoff_utc"]].dropna(
        subset=["kickoff_utc"])

    def lookup(rec):
        match, ticker = rec.get("match", ""), rec.get("ticker", "")
        if " v " not in match or "-" not in ticker:
            return None
        home, away = match.split(" v ", 1)
        mo = re.match(r"(\d{2})([A-Z]{3})(\d{2})", ticker.split("-")[1])
        if not mo:
            return None
        yy, mon, dd = mo.groups()
        try:
            when = pd.Timestamp(2000 + int(yy), months[mon], int(dd))
        except (ValueError, KeyError):
            return None
        cand = fixtures[(fixtures["home_team"] == home)
                        & (fixtures["away_team"] == away)]
        cand = cand[(cand["date"] >= when - pd.Timedelta(days=1))
                    & (cand["date"] <= when + pd.Timedelta(days=1))]
        if len(cand):
            return pd.Timestamp(cand.iloc[0]["kickoff_utc"]).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
        return None

    return lookup


def cmd_paper_cycle(args):
    """One scheduled paper-trading cycle. PAPER ONLY -- places no real order."""
    from .paper.cycle import run_cycle
    from .venues.kalshi_provider import KalshiProvider
    from .venues.polymarket import PolymarketProvider
    leagues = [x.strip() for x in (args.leagues or "").split(",") if x.strip()]
    run_cycle(league_ids=leagues or None,
              state_path=args.state, summary_path=args.summary,
              providers=[KalshiProvider(), PolymarketProvider()],
              board_enabled=not args.no_board)


def cmd_paper_maintain(args):
    """Portfolio upkeep only. PAPER ONLY -- places no order of any kind."""
    from .paper.cycle import run_maintenance
    from .venues.kalshi_provider import KalshiProvider
    from .venues.polymarket import PolymarketProvider
    leagues = [x.strip() for x in (args.leagues or "").split(",") if x.strip()]
    run_maintenance(state_path=args.state, summary_path=args.summary,
                    providers=[KalshiProvider(), PolymarketProvider()],
                    league_ids=leagues or None)


def cmd_paper_rebuild(args):
    """Recompute the paper book from ORDER INTENT. PAPER ONLY, places nothing.

    Two defects made the recorded book unusable rather than merely inaccurate:
    fills were taken from tape running past each order's expiry, and every
    no-side position was settled backwards. Both are fixed, and neither can be
    patched in place -- which fills happened decides which positions exist.

    Nothing here is guessed. Orders carry their own intent, the venues still
    serve the tape for the window each order was live for, and the results are
    in the fixture table.
    """
    import json as _json

    from .data import loader
    from .leagues import all_leagues, get_league
    from .paper.broker import PaperPortfolio
    from .paper.rebuild import archive, compare, rebuild
    from .venues.kalshi_provider import KalshiProvider
    from .venues.polymarket import PolymarketProvider

    old = PaperPortfolio.load(args.state)
    if not old.orders:
        sys.exit("no orders on record at %s -- nothing to rebuild" % args.state)

    leagues = [x.strip() for x in (args.leagues or "").split(",") if x.strip()]
    specs = [get_league(x) for x in leagues] if leagues else all_leagues()
    frames = {}
    for spec in specs:
        try:
            frames[spec.league_id] = loader.load_league(spec, tiers="training")
        except Exception as exc:                              # noqa: BLE001
            print("  %s: no results available (%s)" % (spec.league_id, exc))

    from .paper.cycle import _default_probes
    providers = [KalshiProvider(), PolymarketProvider()]
    print("replaying %d orders against the venue tape..." % len(old.orders))
    report = rebuild(old, _default_probes(providers), frames)

    book = report.pop("book")
    diff = compare(old, book)
    b, a = diff["before"], diff["after"]
    print()
    print("  positions   %4d -> %4d" % (b["positions"], a["positions"]))
    print("  settled     %4d -> %4d" % (b["settled"], a["settled"]))
    print("  win rate    %s -> %s"
          % ("  n/a" if b["win_rate"] is None else "%5.1f%%" % (100*b["win_rate"]),
             "  n/a" if a["win_rate"] is None else "%5.1f%%" % (100*a["win_rate"])))
    print("  realised    $%8.2f -> $%8.2f   (swing $%.2f)"
          % (b["realized_cents"]/100, a["realized_cents"]/100,
             diff["pnl_swing_cents"]/100))
    unresolved = report["summary"]["unresolved"]
    if unresolved:
        print("  %d order(s) could not be resolved from the tape and are left "
              "neither filled nor expired." % unresolved)

    if args.dry_run:
        print("")
        print("dry run: nothing written. Re-run without --dry-run to keep it.")
        return
    with open(args.archive, "w", encoding="utf-8") as fh:
        _json.dump(archive(old.to_dict() if hasattr(old, "to_dict")
                           else _json.loads(_json.dumps(old, default=str))),
                   fh, default=str)
    book.path = args.state
    book.save()
    print("")
    print("  old book archived to %s" % args.archive)
    print("  rebuilt book written to %s" % args.state)


def cmd_tournament(args):
    """Legacy WC-era workflow, reading exclusively from the archive."""
    cfg = CONFIG
    if os.path.exists(WC_FITTED_PARAMS_JSON):
        with open(WC_FITTED_PARAMS_JSON, encoding="utf-8") as fh:
            best = json.load(fh).get("best", {})
        cfg = dataclasses.replace(
            cfg, half_life_days=best.get("half_life_days", cfg.half_life_days),
            rho=best.get("rho", cfg.rho), blend_k=best.get("blend_k", cfg.blend_k))
    if not os.path.exists(WC_RESULTS_CSV):
        sys.exit("Archived WC dataset not found at %s" % WC_RESULTS_CSV)
    from .sim.tournament import simulate_tournament
    df = loader.load_matches(WC_RESULTS_CSV)
    train = loader.training_matches(df)
    state = loader.tournament_state(df)
    ratings = build_team_strength(train, as_of=state.as_of, cfg=cfg,
                                  max_age_days=FIT_MAX_AGE_DAYS, verbose=True)
    ratings = calibrate_to_tournament(ratings, state.played, cfg=cfg)
    res = simulate_tournament(ratings, state, n_sims=args.sims, seed=args.seed,
                              cfg=cfg)
    print()
    print(res.report(top_n=args.top))


# --------------------------------------------------------------------------- #
def _add_league(parser, allow_all=False):
    parser.add_argument("--league", choices=league_ids(),
                        default=None if allow_all else "mls",
                        help="league to act on")
    if allow_all:
        parser.add_argument("--all", action="store_true",
                            help="act on every registered league")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wc2026", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("leagues", help="list registered leagues").set_defaults(
        func=cmd_leagues)

    pu = sub.add_parser("update", help="ingest results + fixtures")
    _add_league(pu, allow_all=True)
    pu.set_defaults(func=cmd_update)

    pf = sub.add_parser("fit", help="fit ratings and show the ranking")
    _add_league(pf)
    pf.add_argument("--top", type=int, default=30)
    pf.set_defaults(func=cmd_fit)

    pm = sub.add_parser("match", help="match report (home side first)")
    _add_league(pm)
    pm.add_argument("home")
    pm.add_argument("away")
    pm.add_argument("--neutral", action="store_true")
    pm.set_defaults(func=cmd_match)

    pb = sub.add_parser("backtest", help="walk-forward calibration report")
    _add_league(pb)
    pb.add_argument("--start", default=None)
    pb.add_argument("--end", default=None)
    pb.add_argument("--min-train", type=int, default=400, dest="min_train")
    pb.add_argument("--verbose", action="store_true")
    pb.set_defaults(func=cmd_backtest)

    pe = sub.add_parser("evaluate", help="nested DEV/CAL/HOLDOUT evaluation")
    _add_league(pe, allow_all=True)
    pe.add_argument("--min-train", type=int, default=400, dest="min_train")
    pe.add_argument("--verbose", action="store_true")
    pe.set_defaults(func=cmd_evaluate)

    pt = sub.add_parser("tune", help="select hyperparameters on DEV only")
    _add_league(pt, allow_all=True)
    pt.add_argument("--min-train", type=int, default=400, dest="min_train")
    pt.add_argument("--verbose", action="store_true")
    pt.set_defaults(func=cmd_tune)

    pr = sub.add_parser("recommend", help="Kalshi EV report (places nothing)")
    _add_league(pr)
    pr.add_argument("--min-edge", type=float, default=None)
    pr.add_argument("--skip-gate", action="store_true")
    pr.set_defaults(func=cmd_recommend)

    px = sub.add_parser("execute", help="place gated orders (dry-run default)")
    _add_league(px)
    px.add_argument("--live", action="store_true")
    px.add_argument("--min-edge", type=float, default=None)
    px.set_defaults(func=cmd_execute)

    ptr = sub.add_parser("track", help="resolve recommendations: CLV + P&L")
    ptr.add_argument("--recompute-clv", action="store_true", dest="recompute")
    ptr.set_defaults(func=cmd_track)

    pc = sub.add_parser("paper-cycle",
                        help="one paper-trading cycle (places no real order)")
    pc.add_argument("--leagues", default="",
                    help="comma-separated league ids (blank = all)")
    pc.add_argument("--state", default=None, help="portfolio state path")
    pc.add_argument("--summary", default=None, help="sanitised summary path")
    pc.add_argument("--no-board", action="store_true",
                    help="skip the board (deterministic decisions only)")
    pc.set_defaults(func=cmd_paper_cycle)

    prb = sub.add_parser("paper-rebuild",
                         help="recompute the paper book from order intent")
    prb.add_argument("--state", default=os.path.join("data", "paper",
                                                     "portfolio.json"))
    prb.add_argument("--archive", default=os.path.join("data", "paper",
                                                       "portfolio_before_rebuild.json"))
    prb.add_argument("--leagues", default="")
    prb.add_argument("--dry-run", action="store_true",
                     help="report the change without writing anything")
    prb.set_defaults(func=cmd_paper_rebuild)

    pm = sub.add_parser("paper-maintain",
                        help="settle, fill and price-check the paper book "
                             "(no model calls, no market discovery)")
    pm.add_argument("--leagues", default="",
                    help="comma-separated league ids (blank = whatever is held)")
    pm.add_argument("--state", default=None, help="portfolio state path")
    pm.add_argument("--summary", default=None, help="sanitised summary path")
    pm.set_defaults(func=cmd_paper_maintain)

    pto = sub.add_parser("tournament", help="legacy: simulate the 2026 WC")
    pto.add_argument("--sims", type=int, default=30000)
    pto.add_argument("--top", type=int, default=16)
    pto.add_argument("--seed", type=int, default=12345)
    pto.set_defaults(func=cmd_tournament)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "league", None) is None and not getattr(args, "all", False):
        args.league = "mls"
    args.func(args)


if __name__ == "__main__":
    main()
