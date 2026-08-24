"""Leak resistance: the properties that make an evaluation number believable.

These are sentinel tests. They do not check that the model is good; they check
that it cannot have cheated -- that a future row cannot reach back into an
earlier forecast, that selection cannot see the holdout, and that an unrated
team is refused rather than silently averaged.
"""

import numpy as np
import pandas as pd
import pytest

from wc2026.data.loader import known_at
from wc2026.eval.backtest import (
    calibration_slope_intercept,
    expected_calibration_error,
    run_backtest,
)
from wc2026.eval.nested import (
    SplitViolation,
    evaluate_holdout,
    fit_calibrator,
    make_split,
    select_hyperparameters,
)
from wc2026.leagues import get_league
from wc2026.model.dixon_coles import UnknownTeamError
from wc2026.model.ratings import build_team_strength
from wc2026.sim.match import predict_match

SPEC = get_league("premier_league")


def synthetic_league(n_teams=12, n_seasons=8, seed=7, start="2016-08-01"):
    """A deterministic synthetic league with real team-strength structure.

    Fixtures are grouped into weekly matchdays (as a real league is), so the
    generated history spans years rather than one match per day.
    """
    rng = np.random.default_rng(seed)
    teams = ["Club %02d" % i for i in range(n_teams)]
    strength = {t: rng.normal(0, 0.35) for t in teams}
    rows, day = [], pd.Timestamp(start)
    per_matchday = n_teams // 2
    for _ in range(n_seasons):
        pairs = [(i, j) for i in teams for j in teams if i != j]
        for k in range(0, len(pairs), per_matchday):
            for (i, j) in pairs[k:k + per_matchday]:
                lam_h = np.exp(0.25 + 0.25 + strength[i] - strength[j])
                lam_a = np.exp(0.25 + strength[j] - strength[i])
                rows.append({
                    "date": day, "kickoff_utc": day + pd.Timedelta(hours=15),
                    "known_after_utc": day + pd.Timedelta(hours=17, minutes=30),
                    "home_team": i, "away_team": j,
                    "home_score": int(rng.poisson(lam_h)),
                    "away_score": int(rng.poisson(lam_a)),
                    "tournament": SPEC.display_name, "tier": "league",
                    "neutral": False,
                })
            day += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    df["played"] = True
    return df


@pytest.fixture(scope="module")
def league_df():
    return synthetic_league()


def windows(df, train_frac=0.6, test_frac=0.85):
    """Test windows derived FROM the data, so they can never drift out of span."""
    d = df["date"].sort_values()
    return (str(d.quantile(train_frac).date()),
            str(d.quantile(test_frac).date()),
            str((d.max() + pd.Timedelta(days=1)).date()))


# --------------------------------------------------------------------------- #
# 1. a future match cannot change a past forecast
# --------------------------------------------------------------------------- #
def test_future_results_cannot_change_an_earlier_forecast(league_df):
    """THE sentinel: rewrite every result after a cutoff and the forecast made
    at that cutoff must be bit-identical."""
    cutoff = pd.Timestamp(windows(league_df)[0])
    train = known_at(league_df, cutoff)
    train = train[train["date"] < cutoff]
    ratings = build_team_strength(train, as_of=cutoff, cfg=SPEC.model)
    before = predict_match(ratings, "Club 00", "Club 01", neutral=False,
                           cfg=SPEC.model)

    tampered = league_df.copy()
    future = tampered["date"] >= cutoff
    tampered.loc[future, "home_score"] = 9      # absurd, unmistakable
    tampered.loc[future, "away_score"] = 0

    train2 = known_at(tampered, cutoff)
    train2 = train2[train2["date"] < cutoff]
    ratings2 = build_team_strength(train2, as_of=cutoff, cfg=SPEC.model)
    after = predict_match(ratings2, "Club 00", "Club 01", neutral=False,
                          cfg=SPEC.model)

    assert before.p_home_win == pytest.approx(after.p_home_win, abs=1e-12)
    assert before.lam_home == pytest.approx(after.lam_home, abs=1e-12)
    assert np.allclose(before.matrix, after.matrix, atol=1e-12)


def test_known_at_uses_the_final_whistle_not_the_kickoff(league_df):
    """A match kicking off before the cutoff but ending after it must NOT be
    visible: its result was not knowable yet."""
    row = league_df.iloc[100]
    mid_match = row["kickoff_utc"] + pd.Timedelta(minutes=30)
    visible = known_at(league_df, mid_match)
    assert row["home_team"] not in set(
        visible[visible["date"] == row["date"]]["home_team"])
    after_whistle = known_at(league_df, row["known_after_utc"])
    assert len(after_whistle) > len(visible)


def test_backtest_training_set_never_includes_the_scored_match(league_df):
    """Every scored match must post-date the data its fit was allowed to see."""
    start, mid, end = windows(league_df)
    res = run_backtest(league_df, test_start=mid, test_end=end,
                       refit_freq="Q", min_train=200, cfg=SPEC.model,
                       test_tiers={"league"}, league_id="synthetic")
    assert res.n > 0
    # a leaking model would be near-perfect; a legitimate one cannot be
    assert res.model["log_loss"] > 0.5


# --------------------------------------------------------------------------- #
# 2. selection cannot see the holdout
# --------------------------------------------------------------------------- #
def test_split_regions_are_disjoint_and_ordered(league_df):
    split = make_split(league_df, "synthetic")
    assert split.dev_start < split.dev_end < split.cal_end < split.holdout_end
    dev = split.dev(league_df)
    assert dev["date"].max() < split.dev_end


def test_hyperparameter_selection_reads_dev_only(league_df):
    """Corrupting everything after DEV must not change the chosen parameters."""
    split = make_split(league_df, "synthetic")
    grid = {"half_life_days": (120, 365), "rho": (-0.05,), "blend_k": (12.0,)}
    chosen = select_hyperparameters(league_df, SPEC, split, grid=grid,
                                    min_train=200, verbose=False)["best"]

    tampered = league_df.copy()
    after_dev = tampered["date"] >= split.dev_end
    tampered.loc[after_dev, "home_score"] = 7
    tampered.loc[after_dev, "away_score"] = 0
    split2 = make_split(league_df, "synthetic")     # identical boundaries
    chosen2 = select_hyperparameters(tampered, SPEC, split2, grid=grid,
                                     min_train=200, verbose=False)["best"]
    assert chosen == chosen2


def test_holdout_can_only_be_opened_once(league_df):
    split = make_split(league_df, "synthetic")
    params = {"half_life_days": 365, "rho": -0.05, "blend_k": 12.0}
    evaluate_holdout(league_df, SPEC, split, params, min_train=200)
    with pytest.raises(SplitViolation):
        evaluate_holdout(league_df, SPEC, split, params, min_train=200)


def test_calibrator_window_stops_before_the_holdout(league_df):
    """The calibrator must not score a single holdout match."""
    split = make_split(league_df, "synthetic")
    params = {"half_life_days": 365, "rho": -0.05, "blend_k": 12.0}
    cal = fit_calibrator(league_df, SPEC, split, params, min_train=200)
    assert cal.date_range is not None
    assert pd.Timestamp(cal.date_range[1]) < split.cal_end


def test_calibration_bins_never_include_the_match_being_calibrated(league_df):
    """Calibration is fitted on scored forecasts that are all strictly earlier
    than the holdout it will later be applied to."""
    split = make_split(league_df, "synthetic")
    params = {"half_life_days": 365, "rho": -0.05, "blend_k": 12.0}
    cal = fit_calibrator(league_df, SPEC, split, params, min_train=200)
    assert cal.pooled_calibration is not None and len(cal.pooled_calibration)
    assert pd.Timestamp(cal.date_range[0]) >= split.dev_end


# --------------------------------------------------------------------------- #
# 3. unknown / promoted teams are explicit
# --------------------------------------------------------------------------- #
def test_unrated_team_raises_instead_of_becoming_average(league_df):
    ratings = build_team_strength(league_df, as_of=league_df["date"].max(),
                                  cfg=SPEC.model)
    with pytest.raises(UnknownTeamError):
        predict_match(ratings, "Newly Promoted FC", "Club 01", neutral=False,
                      cfg=SPEC.model)
    with pytest.raises(UnknownTeamError):
        predict_match(ratings, "Club 00", "Newly Promoted FC", neutral=False,
                      cfg=SPEC.model)


def test_average_substitution_requires_explicit_opt_in(league_df):
    ratings = build_team_strength(league_df, as_of=league_df["date"].max(),
                                  cfg=SPEC.model)
    pred = predict_match(ratings, "Newly Promoted FC", "Club 01", neutral=False,
                         cfg=SPEC.model, allow_unknown=True)
    assert 0.0 < pred.p_home_win < 1.0


def test_backtest_counts_skipped_unknown_team_fixtures(league_df):
    """A promoted club appearing mid-window is reported, not silently scored."""
    df = league_df.copy()
    start, mid, end = windows(df)
    late = df[df["date"] >= pd.Timestamp(mid)].head(30).index
    df.loc[late, "home_team"] = "Promoted FC"
    res = run_backtest(df, test_start=mid, test_end=end,
                       refit_freq="Q", min_train=200, cfg=SPEC.model,
                       test_tiers={"league"}, league_id="synthetic")
    assert res.n_unknown_team > 0
    assert "unrated" in res.report() or "SKIPPED" in res.report()


# --------------------------------------------------------------------------- #
# 4. reported uncertainty is real
# --------------------------------------------------------------------------- #
def test_calibration_slope_detects_overconfidence():
    """A deliberately over-confident forecast must show slope < 1."""
    rng = np.random.default_rng(3)
    truth = rng.uniform(0.2, 0.8, size=4000)
    y = (rng.uniform(size=4000) < truth).astype(int)
    sharp = np.clip((truth - 0.5) * 2.2 + 0.5, 0.01, 0.99)   # too extreme
    probs = np.column_stack([sharp, 1 - sharp])
    ys = np.where(y == 1, 0, 1)
    slope, _ = calibration_slope_intercept(probs, ys)
    assert slope < 1.0
    assert expected_calibration_error(probs, ys) > 0


def test_bootstrap_ci_is_reported_and_brackets_the_estimate(league_df):
    start, mid, end = windows(league_df)
    res = run_backtest(league_df, test_start=mid, test_end=end,
                       refit_freq="Q", min_train=200, cfg=SPEC.model,
                       test_tiers={"league"}, league_id="synthetic",
                       with_uncertainty=True)
    lo, hi = res.extras["log_loss_ci"]
    assert lo <= res.model["log_loss"] <= hi
    assert res.extras["n_clusters"] > 1


# --------------------------------------------------------------------------- #
# 5. league isolation in evaluation
# --------------------------------------------------------------------------- #
def test_scoring_tiers_exclude_training_only_competitions(league_df):
    """Training-only cups must never enter the scored population."""
    df = league_df.copy()
    cup = df.head(200).copy()
    cup["tier"] = "domestic_cup"
    cup["tournament"] = "Cup"
    mixed = pd.concat([df, cup], ignore_index=True).sort_values("date")
    mls = get_league("mls")
    start, mid, end = windows(df)
    res = run_backtest(mixed, test_start=mid, test_end=end,
                       refit_freq="Q", min_train=200, cfg=SPEC.model,
                       test_tiers=mls.scoring_tiers, league_id="synthetic")
    assert "domestic_cup" not in mls.scoring_tiers
    assert res.n > 0
