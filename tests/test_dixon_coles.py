"""Dixon-Coles model: correct rho sign, low-score correction, and parameter recovery."""
import numpy as np
import pandas as pd

from wc2026.config import CONFIG
from wc2026.model.dixon_coles import dc_tau, fit
from wc2026.sim.match import score_matrix


def test_rho_is_negative_default():
    # The whole point of the rebuild: rho must be negative (legacy bug was positive).
    assert CONFIG.rho < 0


def test_tau_sign_boosts_low_draws():
    """With rho<0, tau boosts 0-0 and 1-1 and trims 1-0/0-1 (the empirical pattern)."""
    lam_h = np.array([1.4]); lam_a = np.array([1.2]); rho = -0.05
    assert dc_tau(np.array([0]), np.array([0]), lam_h, lam_a, rho)[0] > 1.0  # 0-0 up
    assert dc_tau(np.array([1]), np.array([1]), lam_h, lam_a, rho)[0] > 1.0  # 1-1 up
    assert dc_tau(np.array([1]), np.array([0]), lam_h, lam_a, rho)[0] < 1.0  # 1-0 down
    assert dc_tau(np.array([0]), np.array([1]), lam_h, lam_a, rho)[0] < 1.0  # 0-1 down


def test_score_matrix_correction_direction():
    """A negative-rho grid has MORE draw mass than independent Poisson (rho=0)."""
    indep = score_matrix(1.4, 1.2, 0.0, 10)
    dc = score_matrix(1.4, 1.2, -0.06, 10)
    assert dc[0, 0] > indep[0, 0]
    assert dc[1, 1] > indep[1, 1]
    assert np.trace(dc) > np.trace(indep)         # total draw prob up
    assert dc[1, 0] < indep[1, 0]                 # 1-0 down


def _synthetic_matches(seed=0, n_teams=12, n_matches=4000):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    att = {t: rng.normal(0, 0.4) for t in teams}
    dfn = {t: rng.normal(0, 0.4) for t in teams}
    att = {t: v - np.mean(list(att.values())) for t, v in att.items()}
    dfn = {t: v - np.mean(list(dfn.values())) for t, v in dfn.items()}
    c, home = np.log(1.35), 0.25
    rows = []
    base = pd.Timestamp("2020-01-01")
    for k in range(n_matches):
        h, a = rng.choice(teams, 2, replace=False)
        lam_h = np.exp(c + home + att[h] - dfn[a])
        lam_a = np.exp(c + att[a] - dfn[h])
        rows.append({"date": base + pd.Timedelta(days=k // 5),
                     "home_team": h, "away_team": a,
                     "home_score": rng.poisson(lam_h), "away_score": rng.poisson(lam_a),
                     "neutral": False, "tier": "friendly", "played": True})
    return pd.DataFrame(rows), att, dfn


def test_parameter_recovery():
    """Fitting synthetic data recovers the true attack/defense ordering well."""
    df, att, dfn = _synthetic_matches()
    # long half-life + no decay cutoff so all synthetic games count equally
    from dataclasses import replace
    cfg = replace(CONFIG, half_life_days=1e6)
    R = fit(df, rho=-0.0001, ridge=0.5, cfg=cfg)
    teams = list(att)
    true_a = np.array([att[t] for t in teams])
    fit_a = np.array([R.attack[t] for t in teams])
    corr = np.corrcoef(true_a, fit_a)[0, 1]
    assert corr > 0.9, f"attack recovery correlation too low: {corr:.2f}"
    # home advantage recovered to the right ballpark
    assert 0.15 < R.home_adv < 0.35
