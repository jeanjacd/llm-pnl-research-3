"""
kelly.py
========
Kelly sizing on the EXACT joint outcome distribution.

Single binary bet (cost c dollars per $1 payout, win prob p) has the closed
form  f* = (p - c) / (1 - c).  Multiple markets on the same match are highly
correlated (they settle on the same scoreline), so they are sized JOINTLY:
maximise expected log wealth over the scoreline grid

    max_x  sum_ij  pi_ij * log( 1 + sum_k x_k * (I_k(i,j)/c_k - 1) )

where pi_ij is the model's scoreline PMF, I_k the settlement indicator of bet
k, c_k its cost (ask + exact fee share, in dollars), and x_k >= 0 the fraction
of bankroll staked. No closed form exists for the joint problem; the exact
objective is optimised numerically (SLSQP with analytic gradient), warm-started
from the single-bet closed forms. A single bet passed through the joint solver
recovers its closed form (tested).

The optimiser returns FULL-Kelly fractions. The caller scales each stake by its
dynamic multiplier (confidence.py, in [1/4, 1/2]) and converts to integer
contracts by flooring -- both steps strictly reduce stakes (house-favour).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def single_kelly(p: float, cost: float) -> float:
    """Closed-form full-Kelly fraction for one binary contract."""
    if not (0.0 < cost < 1.0):
        raise ValueError(f"cost must be in (0,1), got {cost}")
    return max(0.0, (p - cost) / (1.0 - cost))


@dataclass
class JointBet:
    """One candidate bet expressed on the shared scoreline grid."""
    key: str
    indicator: np.ndarray      # bool (G+1,G+1)
    cost: float                # dollars per $1 payout, fee-inclusive


def joint_kelly(score_pmf: np.ndarray, bets: list[JointBet],
                total_cap: float = 0.5) -> np.ndarray:
    """Full-Kelly fractions x_k for correlated bets on one match.

    Exact expected-log-wealth objective over the scoreline grid; constraints
    x_k >= 0 and sum x_k <= total_cap (a sanity/feasibility guard that also
    keeps wealth strictly positive in every outcome).
    """
    if not bets:
        return np.zeros(0)
    pi = score_pmf.ravel()
    keep = pi > 0
    pi = pi[keep]
    # payout ratio per dollar staked: I/c - 1  (win: (1-c)/c, lose: -1)
    R = np.stack([(b.indicator.ravel()[keep].astype(float) / b.cost) - 1.0
                  for b in bets], axis=1)                    # (cells, K)
    K = R.shape[1]

    def neg_elog(x):
        w = 1.0 + R @ x
        if np.any(w <= 1e-9):
            return 1e9, np.zeros(K)
        val = -np.sum(pi * np.log(w))
        grad = -(pi / w) @ R
        return val, grad

    x0 = np.array([min(single_kelly(float((score_pmf * b.indicator).sum()), b.cost),
                       total_cap / K) for b in bets])
    res = minimize(neg_elog, x0, jac=True, method="SLSQP",
                   bounds=[(0.0, total_cap)] * K,
                   constraints=[{"type": "ineq",
                                 "fun": lambda x: total_cap - np.sum(x),
                                 "jac": lambda x: -np.ones(K)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    x = np.clip(res.x, 0.0, total_cap)
    x[x < 1e-6] = 0.0
    return x


def contracts_for_stake(stake_fraction: float, bankroll_usd: float,
                        cost_dollars: float) -> int:
    """Integer contracts bought with `stake_fraction` of bankroll at
    `cost_dollars` per contract. Floors -- never rounds up."""
    if stake_fraction <= 0 or bankroll_usd <= 0:
        return 0
    return int(stake_fraction * bankroll_usd / cost_dollars)
