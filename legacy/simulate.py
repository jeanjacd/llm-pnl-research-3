"""
simulate.py
===========
Monte Carlo match simulator using a Dixon-Coles adjusted bivariate Poisson model.

Why Dixon-Coles instead of plain independent Poisson:
  Independent Poisson under-predicts low-scoring draws (0-0, 1-1). Dixon-Coles adds
  a correction term `tau` on the four lowest scorelines, which materially improves
  calibration on real football data.

The simulator returns a MarketBook holding, for EVERY market, a boolean numpy array
of length N (one entry per simulated match). This is the key to correct parlays:
the optimizer computes the joint probability of several legs by AND-ing their boolean
arrays and taking the mean -- so correlation between legs in the same match is
captured exactly, instead of naively multiplying marginal probabilities.

Confidence: each probability is reported with a Wilson score interval, and the raw
Monte Carlo standard error is sqrt(p(1-p)/N) (~0.0015 at N=100k).
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass

from ratings import Team, Player, expected_goals, scorer_distribution

MAX_GOALS = 12          # scoreline grid cap; P(>12 goals) is negligible
DC_RHO = 0.06           # Dixon-Coles low-score correlation parameter (tunable)


def _dixon_coles_tau(i: int, j: int, lam: float, mu: float, rho: float) -> float:
    if i == 0 and j == 0:
        return 1.0 - lam * mu * rho
    if i == 0 and j == 1:
        return 1.0 + lam * rho
    if i == 1 and j == 0:
        return 1.0 + mu * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def _joint_scoreline_pmf(lam: float, mu: float, rho: float = DC_RHO) -> np.ndarray:
    """Build the (MAX_GOALS+1 x MAX_GOALS+1) joint probability matrix of scorelines."""
    k = np.arange(MAX_GOALS + 1)
    # Poisson pmf via logs for stability.
    log_h = -lam + k * math.log(lam) - np.array([math.lgamma(x + 1) for x in k])
    log_a = -mu + k * math.log(mu) - np.array([math.lgamma(x + 1) for x in k])
    pmf_h = np.exp(log_h)
    pmf_a = np.exp(log_a)
    joint = np.outer(pmf_h, pmf_a)

    # Apply Dixon-Coles correction on the 2x2 low-score corner.
    for i in (0, 1):
        for j in (0, 1):
            joint[i, j] *= _dixon_coles_tau(i, j, lam, mu, rho)

    joint = np.clip(joint, 0.0, None)
    joint /= joint.sum()
    return joint


@dataclass
class Market:
    name: str
    prob: float
    ci_low: float
    ci_high: float
    n: int

    @property
    def std_error(self) -> float:
        return math.sqrt(max(self.prob * (1 - self.prob), 0) / self.n)


class MarketBook:
    """Holds boolean outcome arrays for every market in a single simulated matchup."""

    def __init__(self, home: str, away: str, n: int):
        self.home = home
        self.away = away
        self.n = n
        self.outcomes: dict[str, np.ndarray] = {}

    def add(self, name: str, boolean_array: np.ndarray) -> None:
        self.outcomes[name] = boolean_array

    def names(self) -> list[str]:
        return list(self.outcomes)

    def market(self, name: str) -> Market:
        arr = self.outcomes[name]
        p = float(arr.mean())
        lo, hi = wilson_interval(arr.sum(), self.n)
        return Market(name=name, prob=p, ci_low=lo, ci_high=hi, n=self.n)

    def all_markets(self, min_prob: float = 0.0, max_prob: float = 1.0) -> list[Market]:
        out = [self.market(nm) for nm in self.outcomes]
        return [m for m in out if min_prob <= m.prob <= max_prob]

    def joint(self, names: list[str]) -> Market:
        """True joint probability that ALL named markets hit (correlation-aware)."""
        mask = np.ones(self.n, dtype=bool)
        for nm in names:
            mask &= self.outcomes[nm]
        name = " & ".join(names)
        p = float(mask.mean())
        lo, hi = wilson_interval(mask.sum(), self.n)
        return Market(name=name, prob=p, ci_low=lo, ci_high=hi, n=self.n)


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def simulate_match(
    home: Team,
    away: Team,
    home_players: list[Player],
    away_players: list[Player],
    n_iter: int = 100_000,
    neutral: bool = True,
    seed: int | None = None,
) -> MarketBook:
    rng = np.random.default_rng(seed)
    lam_h, lam_a = expected_goals(home, away, neutral=neutral)

    # Sample scorelines from the Dixon-Coles joint pmf.
    joint = _joint_scoreline_pmf(lam_h, lam_a)
    flat = joint.ravel()
    idx = rng.choice(flat.size, size=n_iter, p=flat)
    gh = (idx // (MAX_GOALS + 1)).astype(np.int16)   # home goals
    ga = (idx % (MAX_GOALS + 1)).astype(np.int16)    # away goals

    book = MarketBook(home.name, away.name, n_iter)
    total = gh + ga

    # --- Result markets ---
    book.add(f"{home.name} win", gh > ga)
    book.add(f"{away.name} win", ga > gh)
    book.add("Draw", gh == ga)
    book.add(f"{home.name} win or draw", gh >= ga)
    book.add(f"{away.name} win or draw", ga >= gh)
    book.add("Either team win (no draw)", gh != ga)

    # --- Totals (over/under) --- (5.5 included to match Kalshi KXWCTOTAL)
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        book.add(f"Over {line} goals", total > line)
        book.add(f"Under {line} goals", total < line)

    # --- Both teams to score ---
    book.add("Both teams to score", (gh >= 1) & (ga >= 1))
    book.add("Both teams to score - No", (gh == 0) | (ga == 0))

    # --- Team totals (per team, multiple lines) -> Kalshi KXWCTEAMTOTAL ---
    for line in (0.5, 1.5, 2.5, 3.5):
        book.add(f"{home.name} over {line} goals", gh > line)
        book.add(f"{away.name} over {line} goals", ga > line)
    book.add(f"{home.name} clean sheet", ga == 0)
    book.add(f"{away.name} clean sheet", gh == 0)

    # --- Spread / handicap -> Kalshi KXWCSPREAD ("wins by more than X.5") ---
    margin = gh - ga
    for n in (2, 3):  # "more than 1.5" => 2+, "more than 2.5" => 3+
        book.add(f"{home.name} wins by {n}+ goals", margin >= n)
        book.add(f"{away.name} wins by {n}+ goals", -margin >= n)

    # --- Correct score -> Kalshi KXWCSCORE ("Score {home}-{away}") ---
    for i in range(0, 7):
        for j in range(0, 7):
            book.add(f"Score {i}-{j}", (gh == i) & (ga == j))

    # --- Anytime goalscorer (per listed player) ---
    _add_scorers(book, home, home_players, gh, rng)
    _add_scorers(book, away, away_players, ga, rng)

    return book


def _add_scorers(book: MarketBook, team: Team, players: list[Player],
                 team_goals: np.ndarray, rng: np.random.Generator) -> None:
    """
    For each listed player, sample 'scored at least once' per iteration.
    P(scored | team scored g goals) = 1 - (1 - w)^g, where w is the player's
    goal share. Driven by the same team_goals array, so scorer markets stay
    correlated with result/total markets (correct for parlays).
    """
    dist = scorer_distribution(team, players)  # [(name, weight), ...]
    g = team_goals.astype(np.float64)
    u = rng.random((len(dist), len(team_goals)))
    for k, (name, w) in enumerate(dist):
        prob_scored = 1.0 - np.power(1.0 - w, g)
        book.add(f"{name} to score", u[k] < prob_scored)


if __name__ == "__main__":
    from ratings import load_database
    db = load_database()
    h, a = db.team("Brazil"), db.team("Germany")
    book = simulate_match(h, a, db.team_players("Brazil"), db.team_players("Germany"),
                          n_iter=100_000, seed=42)
    print(f"{h.name} vs {a.name}  (100k sims)\n")
    for m in sorted(book.all_markets(), key=lambda x: -x.prob):
        print(f"  {m.name:38s} {m.prob*100:5.1f}%  "
              f"[{m.ci_low*100:4.1f}, {m.ci_high*100:4.1f}]")
    print("\nExample correlation-aware parlay (same match):")
    j = book.joint([f"{h.name} win", "Over 2.5 goals", "Vinicius Junior to score"])
    print(f"  {j.name}\n    joint = {j.prob*100:.2f}%  "
          f"(naive multiply would differ)")
