"""Tournament simulation: knockout win-prob sanity and bracket coherence."""
import numpy as np
import pandas as pd

from wc2026.data.loader import TournamentState
from wc2026.model.dixon_coles import DCRatings
from wc2026.sim.tournament import knockout_win_prob, simulate_tournament


def _ratings_for(teams):
    rng = np.random.default_rng(1)
    att = {t: rng.normal(0, 0.3) for t in teams}
    dfn = {t: rng.normal(0, 0.3) for t in teams}
    return DCRatings(attack=att, defense=dfn, intercept=np.log(1.35),
                     home_adv=0.25, rho=-0.05, n_eff={t: 50.0 for t in teams},
                     teams=list(teams), as_of=pd.Timestamp("2026-06-27"))


def test_knockout_win_prob_complementary():
    R = _ratings_for(["A", "B"])
    pab = knockout_win_prob(R, "A", "B", neutral=True)
    pba = knockout_win_prob(R, "B", "A", neutral=True)
    # someone always advances -> the two win probs sum to 1 (no draws in knockout)
    assert abs(pab + pba - 1.0) < 1e-9
    assert 0.0 < pab < 1.0


def _r32_state():
    teams = [f"T{i:02d}" for i in range(32)]
    rows = []
    for k in range(16):
        rows.append({"date": pd.Timestamp("2026-06-28") + pd.Timedelta(days=k // 4),
                     "home_team": teams[2 * k], "away_team": teams[2 * k + 1],
                     "home_score": None, "away_score": None,
                     "tournament": "FIFA World Cup", "neutral": True, "played": False})
    up = pd.DataFrame(rows)
    return TournamentState(participants=teams, played=pd.DataFrame(),
                           upcoming=up, as_of=pd.Timestamp("2026-06-27")), teams


def test_bracket_probabilities_coherent():
    state, teams = _r32_state()
    R = _ratings_for(teams)
    res = simulate_tournament(R, state, n_sims=5000, seed=3)
    t = res.table
    # Champion probs sum to 1; finalists to 2; R16 qualifiers to 16.
    assert abs(t["Champion"].sum() - 1.0) < 1e-9
    assert abs(t["Reach Final"].sum() - 2.0) < 1e-9
    assert abs(t["Reach R16"].sum() - 16.0) < 1e-9
    # Monotonic nesting: a team can't reach the final more often than the SF.
    assert (t["Reach Final"] <= t["Reach SF"] + 1e-9).all()
    assert (t["Reach SF"] <= t["Reach QF"] + 1e-9).all()
    assert res.frontier_round == "round_of_32"
