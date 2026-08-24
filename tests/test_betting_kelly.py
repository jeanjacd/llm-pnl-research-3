"""Kelly sizing: closed form, joint-correlated solver, dynamic multiplier
bounds, and integer conversion."""
import numpy as np
import pytest

from wc2026.betting.confidence import kelly_multiplier, w_data, w_liq, w_time
from wc2026.betting.kelly import (
    JointBet,
    contracts_for_stake,
    joint_kelly,
    single_kelly,
)
from wc2026.sim.match import score_matrix


def _grid(lh=1.6, la=1.2, rho=-0.1):
    return score_matrix(lh, la, rho, 12)


def _ind(g, kind):
    k = np.arange(g.shape[0])
    i, j = np.meshgrid(k, k, indexing="ij")
    return {"home": i > j, "draw": i == j, "away": i < j,
            "over25": (i + j) > 2.5}[kind]


# --- single closed form -------------------------------------------------------
def test_single_kelly_closed_form():
    # p=0.6, cost=0.5 -> f* = (0.6-0.5)/0.5 = 0.2
    assert single_kelly(0.6, 0.5) == pytest.approx(0.2)
    # no edge -> zero
    assert single_kelly(0.5, 0.5) == 0.0
    assert single_kelly(0.4, 0.5) == 0.0


def test_joint_recovers_single_closed_form():
    g = _grid()
    ind = _ind(g, "home")
    p = float(g[ind].sum())
    cost = p - 0.06                       # a 6-point edge
    x = joint_kelly(g, [JointBet("hw", ind, cost)])
    assert x[0] == pytest.approx(single_kelly(p, cost), abs=1e-4)


# --- correlated sizing --------------------------------------------------------
def _elog(g, bets, x):
    pi = g.ravel()
    R = np.stack([(b.indicator.ravel().astype(float) / b.cost) - 1.0
                  for b in bets], axis=1)
    w = 1.0 + R @ np.asarray(x)
    return float(np.sum(pi * np.log(w)))


def test_joint_solution_beats_independent_stacking():
    """Correlation changes optimal sizing, and the joint solver must find it:
    its expected log wealth beats naive independent Kelly stakes and is locally
    optimal. (For MUTUALLY EXCLUSIVE bets like home+draw the joint optimum is
    MORE aggressive than independent stacking -- they hedge each other, the
    classic horse-race result -- so the direction of the correction is not a
    universal inequality; optimality of E[log W] is the invariant.)"""
    g = _grid()
    ih, id_ = _ind(g, "home"), _ind(g, "draw")
    ph, pd_ = float(g[ih].sum()), float(g[id_].sum())
    bets = [JointBet("h", ih, ph - 0.05), JointBet("d", id_, pd_ - 0.05)]
    x = joint_kelly(g, bets)
    assert (x >= 0).all()
    x_indep = np.array([single_kelly(ph, ph - 0.05), single_kelly(pd_, pd_ - 0.05)])
    assert _elog(g, bets, x) >= _elog(g, bets, x_indep) - 1e-10
    # local optimality: small perturbations only hurt
    for d in ([0.01, 0], [-0.01, 0], [0, 0.01], [0, -0.01]):
        xp = np.clip(x + d, 0, None)
        assert _elog(g, bets, x) >= _elog(g, bets, xp) - 1e-9


def test_positively_correlated_bets_are_shrunk_vs_independent():
    """Home-win and home-wins-by-2+ mostly pay together: doubling up on the
    same risk. Joint sizing must total LESS than independent Kelly stakes."""
    g = _grid()
    k = np.arange(g.shape[0])
    i, j = np.meshgrid(k, k, indexing="ij")
    ih = i > j
    ib = (i - j) > 1.5
    ph, pb = float(g[ih].sum()), float(g[ib].sum())
    bets = [JointBet("h", ih, ph - 0.05), JointBet("big", ib, pb - 0.05)]
    x = joint_kelly(g, bets)
    indep = single_kelly(ph, ph - 0.05) + single_kelly(pb, pb - 0.05)
    assert x.sum() < indep
    assert (x >= 0).all()


def test_joint_kelly_respects_total_cap():
    g = _grid()
    bets = [JointBet("h", _ind(g, "home"), 0.3),      # huge edges
            JointBet("o", _ind(g, "over25"), 0.3)]
    x = joint_kelly(g, bets, total_cap=0.4)
    assert x.sum() <= 0.4 + 1e-8


def test_joint_kelly_zero_edge_bets_get_nothing():
    g = _grid()
    ind = _ind(g, "home")
    p = float(g[ind].sum())
    x = joint_kelly(g, [JointBet("h", ind, p + 0.05)])   # negative edge
    assert x[0] == pytest.approx(0.0, abs=1e-4)


# --- dynamic multiplier bounds ------------------------------------------------
def test_kelly_multiplier_bounds_quarter_to_half():
    assert kelly_multiplier(0.0) == pytest.approx(0.25)   # floor: quarter
    assert kelly_multiplier(1.0) == pytest.approx(0.50)   # ceiling: half
    assert kelly_multiplier(0.5) == pytest.approx(0.375)  # continuous between
    # out-of-range confidence is clamped, never exceeds half Kelly
    assert kelly_multiplier(7.3) == pytest.approx(0.50)
    assert kelly_multiplier(-2.0) == pytest.approx(0.25)


def test_confidence_components_in_unit_range():
    assert 0 <= w_data(0, 0) <= 1
    assert w_data(100, 100) > w_data(5, 100)      # thin history shrinks
    assert w_liq(0, 50) == 0.0
    assert w_liq(500, 50) == 1.0
    assert w_time(2.0) == 1.0
    assert 0.5 <= w_time(500.0) < 1.0             # far out decays, floored


# --- integer conversion -------------------------------------------------------
def test_contracts_floor_never_round_up():
    # $99.9 of stake at 50c contracts -> 199 contracts, not 200
    assert contracts_for_stake(0.0999, 1000.0, 0.50) == 199
    assert contracts_for_stake(0.0, 1000.0, 0.5) == 0
    assert contracts_for_stake(0.5, 0.0, 0.5) == 0
