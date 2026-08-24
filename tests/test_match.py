"""Match prediction: grid is a valid PMF, markets are coherent, grid == Monte Carlo."""
import numpy as np

from wc2026.model.dixon_coles import DCRatings
from wc2026.sim.match import predict_match, sample_scorelines, score_matrix


def _toy_ratings():
    return DCRatings(
        attack={"A": 0.30, "B": -0.10}, defense={"A": 0.20, "B": -0.05},
        intercept=np.log(1.35), home_adv=0.25, rho=-0.05,
        n_eff={"A": 50.0, "B": 50.0}, teams=["A", "B"], as_of=None,
    )


def test_grid_is_valid_pmf():
    mat = score_matrix(1.5, 1.1, -0.05, 12)
    assert abs(mat.sum() - 1.0) < 1e-12
    assert (mat >= 0).all()


def test_result_probabilities_sum_to_one():
    R = _toy_ratings()
    p = predict_match(R, "A", "B", neutral=True)
    assert abs(p.p_home_win + p.p_draw + p.p_away_win - 1.0) < 1e-9


def test_totals_distribution_sums_to_one():
    R = _toy_ratings()
    p = predict_match(R, "A", "B", neutral=True)
    assert abs(p.total_goals_distribution().sum() - 1.0) < 1e-9
    # over + under around a half-line are complementary
    assert abs(p.prob_over(2.5) + p.prob_under(2.5) - 1.0) < 1e-9


def test_grid_matches_monte_carlo():
    R = _toy_ratings()
    p = predict_match(R, "A", "B", neutral=True)
    rng = np.random.default_rng(0)
    gh, ga = sample_scorelines(p.lam_home, p.lam_away, p.rho, 300_000, rng)
    assert abs((gh > ga).mean() - p.p_home_win) < 0.005
    assert abs((gh == ga).mean() - p.p_draw) < 0.005
    assert abs((gh + ga > 2.5).mean() - p.prob_over(2.5)) < 0.005


def test_home_advantage_increases_home_win():
    R = _toy_ratings()
    neutral = predict_match(R, "A", "B", neutral=True).p_home_win
    at_home = predict_match(R, "A", "B", neutral=False).p_home_win
    assert at_home > neutral
