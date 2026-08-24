"""
backtest.py
===========
Walk-forward (leakage-free) evaluation of the match model, per league.

Scoring rules
-------------
  * log-loss   -- penalises confident wrong calls; the headline metric.
  * Brier      -- multiclass squared error.
  * RPS        -- ranked probability score, respects home > draw > away order.
  * calibration-- predicted vs. observed frequency by probability bin.

Point-in-time discipline
------------------------
For each refit the training set is restricted with `loader.known_at`, i.e. to
matches whose RESULT was knowable at the cutoff (final whistle), not merely
whose kickoff had passed. A match that kicks off before the cutoff and finishes
after it therefore cannot leak its own result into the fit that predicts it.

Unknown teams
-------------
A promoted club has no fitted rating in its first appearance. The model no
longer silently substitutes the league average (see model/dixon_coles.py):
such fixtures are SKIPPED and counted in `n_unknown_team`, so their absence is
visible in the report instead of being quietly scored against an invented
forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CONFIG
from ..data.loader import known_at
from ..model.dixon_coles import UnknownTeamError
from ..model.ratings import build_team_strength
from ..sim.match import predict_match


def _outcome_index(hs: int, as_: int) -> int:
    return 0 if hs > as_ else (1 if hs == as_ else 2)   # home / draw / away


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def rps(probs: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score for ordered categories (home, draw, away)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    cp = np.cumsum(probs, axis=1)
    cy = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / (probs.shape[1] - 1)))


def calibration_slope_intercept(probs: np.ndarray, y: np.ndarray) -> tuple:
    """Logistic recalibration of the pooled forecasts: (slope, intercept).

    A perfectly calibrated model gives slope 1, intercept 0. Slope < 1 means
    over-confidence (probabilities too extreme).
    """
    p = np.clip(probs.ravel(), 1e-6, 1 - 1e-6)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    hit = onehot.ravel()
    x = np.log(p / (1 - p))
    design = np.column_stack([np.ones_like(x), x])

    # Two-parameter logistic recalibration fitted by bounded L-BFGS-B rather
    # than raw Newton-Raphson: saturated probabilities drive the Hessian
    # singular, and an undamped Newton step then diverges to a meaningless
    # slope. Bounds keep the answer interpretable.
    from scipy.optimize import minimize

    def neg_ll(beta):
        eta = np.clip(design @ beta, -30.0, 30.0)
        # log(1+exp(eta)) computed stably
        softplus = np.logaddexp(0.0, eta)
        value = -np.sum(hit * eta - softplus)
        mu = 1.0 / (1.0 + np.exp(-eta))
        grad = -(design.T @ (hit - mu))
        return value, grad

    res = minimize(neg_ll, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B",
                   bounds=[(-10.0, 10.0), (0.0, 10.0)],
                   options={"maxiter": 200})
    intercept, slope = float(res.x[0]), float(res.x[1])
    return slope, intercept


def expected_calibration_error(probs: np.ndarray, y: np.ndarray,
                               n_bins: int = 10) -> float:
    p = probs.ravel()
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    hit = onehot.ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    total, ece = len(p), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.sum():
            ece += (m.sum() / total) * abs(p[m].mean() - hit[m].mean())
    return float(ece)


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, statistic,
                 n_boot: int = 400, alpha: float = 0.05,
                 seed: int = 12345) -> tuple:
    """Cluster bootstrap CI, resampling GROUPS (matches/matchdays) not rows.

    Several forecasts can come from one match; treating them as independent
    would understate the interval.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    if len(unique) < 2:
        return (float("nan"), float("nan"))
    index = {g: np.flatnonzero(groups == g) for g in unique}
    stats = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[g] for g in picked])
        stats.append(statistic(rows))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass
class BacktestResult:
    n: int
    model: dict
    baseline: dict
    calibration: pd.DataFrame
    pooled_calibration: pd.DataFrame | None = None
    n_unknown_team: int = 0
    skipped_examples: list = field(default_factory=list)
    league_id: str | None = None
    date_range: tuple | None = None
    extras: dict = field(default_factory=dict)

    def write_calibration(self, path: str) -> None:
        """Persist the pooled trust curve for the betting layer."""
        import json
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "produced_by": "wc2026 backtest (walk-forward, leakage-free)",
            "league_id": self.league_id,
            "n_matches": int(self.n),
            "n_unknown_team_skipped": int(self.n_unknown_team),
            "date_range": [str(d) for d in (self.date_range or ())],
            "model_metrics": self.model,
            "baseline_metrics": self.baseline,
            "bins": (self.pooled_calibration.to_dict(orient="records")
                     if self.pooled_calibration is not None else []),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def report(self) -> str:
        def fmt(d):
            return ("log-loss %.4f  brier %.4f  rps %.4f"
                    % (d["log_loss"], d["brier"], d["rps"]))
        head = "Walk-forward backtest"
        if self.league_id:
            head += " [%s]" % self.league_id
        lines = ["%s on %s matches" % (head, format(self.n, ","))]
        if self.date_range:
            lines.append("  window    : %s .. %s" % self.date_range)
        lines.append("  model     : " + fmt(self.model))
        lines.append("  base-rate : " + fmt(self.baseline))
        gap = self.baseline["log_loss"] - self.model["log_loss"]
        pct = 100 * (1 - self.model["log_loss"] / self.baseline["log_loss"])
        lines.append("  improvement (log-loss): %.4f (%.1f%% better)" % (gap, pct))
        if "log_loss_ci" in self.extras:
            lo, hi = self.extras["log_loss_ci"]
            lines.append("  model log-loss 95%% CI (cluster bootstrap): "
                         "[%.4f, %.4f]" % (lo, hi))
        if "calibration_slope" in self.extras:
            lines.append("  calibration slope %.3f  intercept %+.3f  ECE %.4f"
                         % (self.extras["calibration_slope"],
                            self.extras["calibration_intercept"],
                            self.extras["ece"]))
        if self.n_unknown_team:
            lines.append("  SKIPPED %d fixture(s) with an unrated team "
                         "(promoted/renamed) -- not silently averaged"
                         % self.n_unknown_team)
        lines.append("  calibration (home-win prob bins):")
        for _, r in self.calibration.iterrows():
            lines.append("    pred %.1f-%.1f: predicted %5.1f%%  observed %5.1f%%"
                         "  (n=%d)" % (r["bin_lo"], r["bin_hi"],
                                       r["mean_pred"] * 100,
                                       r["observed"] * 100, int(r["n"])))
        return "\n".join(lines)


def run_backtest(matches: pd.DataFrame, test_start: str = "2022-08-01",
                 test_end: str = "2026-06-01", refit_freq: str = "MS",
                 max_age_days: float | None = 8 * 365, rho: float | None = None,
                 min_train: int = 2000, cfg=CONFIG, verbose: bool = False,
                 test_tiers=None, league_id: str | None = None,
                 with_uncertainty: bool = False) -> BacktestResult:
    """Refit periodically on the past, predict the next block, score it.

    `test_tiers` restricts SCORING to the competitions actually evaluated or
    traded. Training always sees every supplied match; whether a competition
    helps is governed by importance weights, not by this filter.
    """
    played = matches[matches["played"]].sort_values("date").reset_index(drop=True)
    test = played[(played["date"] >= pd.Timestamp(test_start)) &
                  (played["date"] < pd.Timestamp(test_end))].copy()
    if test_tiers is not None:
        in_tiers = test["tier"].isin(set(test_tiers))
        if in_tiers.any():
            test = test[in_tiers].copy()

    pre = played[played["date"] < pd.Timestamp(test_start)]
    if test_tiers is not None and pre["tier"].isin(set(test_tiers)).any():
        pre = pre[pre["tier"].isin(set(test_tiers))]
    if pre.empty:
        raise ValueError("no pre-test matches to build a base rate from")
    yb = np.array([_outcome_index(h, a)
                   for h, a in zip(pre["home_score"], pre["away_score"])])
    base_rates = np.array([(yb == k).mean() for k in range(3)])

    test["period"] = test["date"].dt.to_period(
        "M" if refit_freq == "MS" else "Q")
    probs, ys, groups = [], [], []
    n_unknown, skipped_examples = 0, []
    for period, grp in test.groupby("period"):
        cutoff = grp["date"].min()
        # Point-in-time: only results KNOWABLE before this block began.
        train = known_at(played, cutoff)
        train = train[train["date"] < cutoff]
        if len(train) < min_train:
            continue
        ratings = build_team_strength(train, as_of=cutoff, rho=rho, cfg=cfg,
                                      max_age_days=max_age_days)
        for _, m in grp.iterrows():
            try:
                pred = predict_match(ratings, m["home_team"], m["away_team"],
                                     neutral=bool(m["neutral"]), cfg=cfg)
            except UnknownTeamError as exc:
                n_unknown += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(str(exc)[:120])
                continue
            probs.append([pred.p_home_win, pred.p_draw, pred.p_away_win])
            ys.append(_outcome_index(m["home_score"], m["away_score"]))
            groups.append("%s|%s|%s" % (m["date"].date(), m["home_team"],
                                        m["away_team"]))
        if verbose:
            print("  %s: scored %d (train %d)" % (period, len(grp), len(train)))

    if not probs:
        raise ValueError("backtest scored no matches (check windows/min_train)")
    probs = np.array(probs)
    ys = np.array(ys)
    groups = np.array(groups)
    base = np.tile(base_rates, (len(ys), 1))

    model_metrics = {"log_loss": log_loss(probs, ys), "brier": brier(probs, ys),
                     "rps": rps(probs, ys)}
    base_metrics = {"log_loss": log_loss(base, ys), "brier": brier(base, ys),
                    "rps": rps(base, ys)}

    ph = probs[:, 0]
    home_won = (ys == 0).astype(float)
    bins = np.linspace(0, 1, 11)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (ph >= lo) & (ph < hi if hi < 1.0 else ph <= hi)
        if m.sum() > 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "mean_pred": ph[m].mean(),
                         "observed": home_won[m].mean(), "n": m.sum()})
    cal = pd.DataFrame(rows)

    pooled_p = probs.ravel()
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(ys)), ys] = 1.0
    pooled_hit = onehot.ravel()
    prows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pooled_p >= lo) & (pooled_p < hi if hi < 1.0 else pooled_p <= hi)
        if m.sum() > 0:
            prows.append({"bin_lo": float(lo), "bin_hi": float(hi),
                          "mean_pred": float(pooled_p[m].mean()),
                          "observed": float(pooled_hit[m].mean()),
                          "n": int(m.sum())})
    pooled = pd.DataFrame(prows)

    extras = {}
    if with_uncertainty:
        slope, intercept = calibration_slope_intercept(probs, ys)
        extras["calibration_slope"] = slope
        extras["calibration_intercept"] = intercept
        extras["ece"] = expected_calibration_error(probs, ys)
        extras["log_loss_ci"] = bootstrap_ci(
            probs, groups, lambda rows: log_loss(probs[rows], ys[rows]))
        extras["rps_ci"] = bootstrap_ci(
            probs, groups, lambda rows: rps(probs[rows], ys[rows]))
        extras["n_clusters"] = int(len(np.unique(groups)))

    return BacktestResult(
        n=len(ys), model=model_metrics, baseline=base_metrics, calibration=cal,
        pooled_calibration=pooled, n_unknown_team=n_unknown,
        skipped_examples=skipped_examples, league_id=league_id,
        date_range=(str(test["date"].min().date()),
                    str(test["date"].max().date())) if len(test) else None,
        extras=extras)
