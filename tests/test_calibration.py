"""In-tournament goal calibration: direction, shrinkage, and disable switch."""
import dataclasses

import numpy as np
import pandas as pd

from wc2026.config import CONFIG
from wc2026.model.dixon_coles import DCRatings
from wc2026.model.ratings import calibrate_to_tournament


def _ratings():
    return DCRatings(attack={"A": 0.0, "B": 0.0}, defense={"A": 0.0, "B": 0.0},
                     intercept=np.log(1.25), home_adv=0.25, rho=-0.05,
                     n_eff={"A": 20.0, "B": 20.0}, teams=["A", "B"],
                     as_of=pd.Timestamp("2026-07-01"))


def _games(n, goals_each):
    return pd.DataFrame({
        "date": [pd.Timestamp("2026-06-15")] * n,
        "home_team": ["A"] * n, "away_team": ["B"] * n,
        "home_score": [goals_each] * n, "away_score": [goals_each] * n,
        "neutral": [True] * n,
    })


def test_hot_tournament_raises_lambda():
    R = _ratings()  # model expects 2.5 total; tournament delivering 4.0
    cal = calibrate_to_tournament(R, _games(20, 2.0))
    assert cal.lambda_mult > 1.0
    lh0, la0 = R.expected_goals("A", "B")
    lh1, la1 = cal.expected_goals("A", "B")
    assert lh1 > lh0 and la1 > la0


def test_shrinkage_grows_with_sample():
    R = _ratings()
    small = calibrate_to_tournament(R, _games(3, 2.0)).lambda_mult
    large = calibrate_to_tournament(R, _games(60, 2.0)).lambda_mult
    assert 1.0 < small < large   # more evidence -> stronger (but still partial) pull


def test_disable_switch_and_no_games():
    R = _ratings()
    off = dataclasses.replace(CONFIG, tournament_calib_k=None)
    assert calibrate_to_tournament(R, _games(20, 2.0), cfg=off).lambda_mult == 1.0
    empty = _games(0, 2.0)
    assert calibrate_to_tournament(R, empty).lambda_mult == 1.0


def test_future_games_excluded():
    R = _ratings()
    g = _games(20, 2.0)
    g["date"] = pd.Timestamp("2026-07-02")   # after as_of -> must not count
    assert calibrate_to_tournament(R, g).lambda_mult == 1.0
