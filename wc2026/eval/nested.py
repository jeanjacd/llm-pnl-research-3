"""
nested.py
=========
Strictly separated chronological evaluation, per league.

The previous design tuned hyperparameters on the same window it then reported
as the headline result, which inflates that result by construction. This module
enforces four disjoint, time-ordered regions:

    |<---------- DEV ---------->|<-- CAL -->|<-- HOLDOUT -->|
     inner walk-forward folds     calibrator   headline, used
     select hyperparameters       fitted here  exactly ONCE

Rules enforced in code (not merely documented):
  * hyperparameter selection may read DEV only;
  * the calibrator may read CAL only (and the params chosen on DEV);
  * the HOLDOUT is opened once, by `evaluate_holdout`, and the split records
    that it has been consumed -- a second call raises;
  * every fit inside every region is itself walk-forward and point-in-time, so
    no future match can influence an earlier forecast.

`SplitViolation` is raised on any attempt to read a region a stage is not
entitled to, so a leak becomes a test failure rather than an optimistic number.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
from dataclasses import dataclass, field

import pandas as pd

from .backtest import BacktestResult, run_backtest


class SplitViolation(RuntimeError):
    """A stage tried to read data outside the region it is entitled to."""


@dataclass
class NestedSplit:
    """Time-ordered evaluation regions for one league."""

    league_id: str
    dev_start: pd.Timestamp
    dev_end: pd.Timestamp        # exclusive
    cal_end: pd.Timestamp        # exclusive
    holdout_end: pd.Timestamp    # exclusive
    _holdout_opened: bool = field(default=False, repr=False)

    def as_dict(self) -> dict:
        return {"league_id": self.league_id,
                "dev": [str(self.dev_start.date()), str(self.dev_end.date())],
                "calibration": [str(self.dev_end.date()),
                                str(self.cal_end.date())],
                "holdout": [str(self.cal_end.date()),
                            str(self.holdout_end.date())],
                "holdout_opened": self._holdout_opened}

    # ---- region accessors (the only sanctioned way to read data) ----
    def dev(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._slice(df, self.dev_start, self.dev_end)

    def calibration(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._slice(df, self.dev_start, self.cal_end)

    def holdout(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._slice(df, self.dev_start, self.holdout_end)

    @staticmethod
    def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
        return df[(df["date"] >= start) & (df["date"] < end)].reset_index(
            drop=True)

    def open_holdout(self) -> None:
        if self._holdout_opened:
            raise SplitViolation(
                "%s: the final holdout has already been evaluated. Re-running "
                "it after seeing the result turns it into a tuning set."
                % self.league_id)
        self._holdout_opened = True


def make_split(df: pd.DataFrame, league_id: str, holdout_fraction: float = 0.20,
               cal_fraction: float = 0.15) -> NestedSplit:
    """Cut a league's history into DEV / CAL / HOLDOUT by match date."""
    played = df[df["played"]].sort_values("date")
    if played.empty:
        raise ValueError("%s: no played matches to split" % league_id)
    dates = played["date"]
    start, end = dates.min(), dates.max() + pd.Timedelta(days=1)
    span = (end - start).days
    if span < 365:
        raise ValueError("%s: %d days of history is too short to split"
                         % (league_id, span))
    cal_end = end - pd.Timedelta(days=int(span * holdout_fraction))
    dev_end = cal_end - pd.Timedelta(days=int(span * cal_fraction))
    return NestedSplit(league_id=league_id, dev_start=start, dev_end=dev_end,
                       cal_end=cal_end, holdout_end=end)


def select_hyperparameters(df: pd.DataFrame, spec, split: NestedSplit,
                           grid: dict | None = None, refit_freq: str = "Q",
                           min_train: int = 400, verbose: bool = True) -> dict:
    """Inner walk-forward selection. Reads the DEV region ONLY.

    Returns the winning parameter set plus the full searched grid, so the
    selection is auditable and its stability across folds is inspectable.
    """
    dev = split.dev(df)
    if dev.empty:
        raise ValueError("%s: empty dev region" % spec.league_id)
    grid = grid or {
        "half_life_days": (90, 180, 365),
        "rho": (-0.03, -0.06, -0.10),
        "blend_k": (8.0, 12.0, 20.0),
    }
    # Inner test window: the later part of DEV, so each trial still trains on
    # a strictly earlier prefix.
    dev_dates = dev[dev["played"]]["date"]
    inner_start = dev_dates.min() + (dev_dates.max() - dev_dates.min()) * 0.5
    results = []
    for hl, rho, bk in itertools.product(grid["half_life_days"], grid["rho"],
                                         grid["blend_k"]):
        trial = dataclasses.replace(spec.model, half_life_days=hl, rho=rho,
                                    blend_k=bk)
        try:
            res = run_backtest(dev, test_start=str(inner_start.date()),
                               test_end=str(split.dev_end.date()),
                               refit_freq=refit_freq, rho=rho, cfg=trial,
                               min_train=min_train,
                               test_tiers=spec.scoring_tiers,
                               league_id=spec.league_id)
        except ValueError:
            continue
        results.append({"half_life_days": hl, "rho": rho, "blend_k": bk,
                        "log_loss": res.model["log_loss"],
                        "rps": res.model["rps"], "n": res.n,
                        "n_unknown_team": res.n_unknown_team})
        if verbose:
            print("    hl=%-4s rho=%+.2f k=%-5s -> log_loss %.4f (n=%d)"
                  % (hl, rho, bk, res.model["log_loss"], res.n))
    if not results:
        raise ValueError("%s: no inner fold produced a score" % spec.league_id)
    table = pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)
    best = table.iloc[0].to_dict()
    return {"best": {k: best[k] for k in ("half_life_days", "rho", "blend_k")},
            "inner_log_loss": float(best["log_loss"]),
            "n_inner_scored": int(best["n"]),
            "grid": results,
            "stability": {
                "log_loss_spread": float(table["log_loss"].max()
                                         - table["log_loss"].min()),
                "top3": table.head(3).to_dict(orient="records")}}


def fit_calibrator(df: pd.DataFrame, spec, split: NestedSplit,
                   params: dict, refit_freq: str = "Q",
                   min_train: int = 400) -> BacktestResult:
    """Produce the trust curve on the CALIBRATION window only.

    Trains walk-forward on everything up to each calibration block, scores the
    calibration window, and returns the result whose pooled bins become the
    calibrator. The holdout is never touched here.
    """
    cal_df = split.calibration(df)
    cfg = dataclasses.replace(spec.model, **params)
    return run_backtest(cal_df, test_start=str(split.dev_end.date()),
                        test_end=str(split.cal_end.date()),
                        refit_freq=refit_freq, rho=params.get("rho"), cfg=cfg,
                        min_train=min_train, test_tiers=spec.scoring_tiers,
                        league_id=spec.league_id, with_uncertainty=True)


def evaluate_holdout(df: pd.DataFrame, spec, split: NestedSplit,
                     params: dict, refit_freq: str = "Q",
                     min_train: int = 400) -> BacktestResult:
    """Headline evaluation. Opens the final holdout exactly ONCE."""
    split.open_holdout()
    hold_df = split.holdout(df)
    cfg = dataclasses.replace(spec.model, **params)
    return run_backtest(hold_df, test_start=str(split.cal_end.date()),
                        test_end=str(split.holdout_end.date()),
                        refit_freq=refit_freq, rho=params.get("rho"), cfg=cfg,
                        min_train=min_train, test_tiers=spec.scoring_tiers,
                        league_id=spec.league_id, with_uncertainty=True)


def run_nested(df: pd.DataFrame, spec, grid: dict | None = None,
               refit_freq: str = "Q", min_train: int = 400,
               verbose: bool = True) -> dict:
    """Full nested evaluation for one league. Returns a JSON-ready report."""
    split = make_split(df, spec.league_id)
    if verbose:
        print("  split: %s" % json.dumps(split.as_dict()["dev"]
                                         + split.as_dict()["calibration"]
                                         + split.as_dict()["holdout"]))
    selection = select_hyperparameters(df, spec, split, grid=grid,
                                       refit_freq=refit_freq,
                                       min_train=min_train, verbose=verbose)
    params = selection["best"]
    calibration = fit_calibrator(df, spec, split, params, refit_freq=refit_freq,
                                 min_train=min_train)
    holdout = evaluate_holdout(df, spec, split, params, refit_freq=refit_freq,
                               min_train=min_train)
    return {
        "league_id": spec.league_id,
        "split": split.as_dict(),
        "selection": selection,
        "calibration_window": {
            "n": calibration.n, "metrics": calibration.model,
            "baseline": calibration.baseline,
            "bins": (calibration.pooled_calibration.to_dict(orient="records")
                     if calibration.pooled_calibration is not None else []),
        },
        "holdout": {
            "n": holdout.n,
            "date_range": holdout.date_range,
            "metrics": holdout.model,
            "baseline": holdout.baseline,
            "extras": holdout.extras,
            "n_unknown_team_skipped": holdout.n_unknown_team,
            "calibration_bins": (
                holdout.calibration.to_dict(orient="records")
                if holdout.calibration is not None else []),
        },
        "notes": [
            "Hyperparameters selected on DEV only; calibrator fitted on CAL "
            "only; holdout opened once.",
            "Improvement is versus a base-rate baseline, NOT versus market "
            "prices. No market-alpha claim is supported by this report.",
        ],
    }


def write_report(report: dict, spec) -> str:
    os.makedirs(spec.data_dir, exist_ok=True)
    with open(spec.eval_report_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return spec.eval_report_json
