"""
match.py
========
Turn fitted ratings into a full probabilistic picture of a single match.

The scoreline distribution is computed EXACTLY on a grid (no Monte-Carlo noise):
build the independent-Poisson outer product from (lambda_home, lambda_away), apply
the Dixon-Coles low-score correction, renormalise. Every match-level market is then
a deterministic sum over that grid -- so P(home win), the total-goals distribution,
most-likely scorelines, BTTS, clean sheets, etc. are all internally consistent and
free of sampling error. A matched Monte-Carlo sampler is provided for the tournament
simulator and as an independent cross-check (tests assert grid == MC within noise).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from ..config import CONFIG
from ..model.dixon_coles import DCRatings, dc_tau


def score_matrix(lam_h: float, lam_a: float, rho: float,
                 max_goals: int | None = None) -> np.ndarray:
    """Exact Dixon-Coles-adjusted joint scoreline PMF.

    Returns a (G+1) x (G+1) array M where M[i, j] = P(home i, away j).
    """
    g = CONFIG.max_goals_grid if max_goals is None else max_goals
    k = np.arange(g + 1)
    ph = poisson.pmf(k, lam_h)
    pa = poisson.pmf(k, lam_a)
    mat = np.outer(ph, pa)

    # Dixon-Coles correction on the four corner cells.
    ii, jj = np.meshgrid(k, k, indexing="ij")
    lam_h_arr = np.full_like(mat, lam_h)
    lam_a_arr = np.full_like(mat, lam_a)
    mat = mat * dc_tau(ii, jj, lam_h_arr, lam_a_arr, rho)

    mat = np.clip(mat, 0.0, None)
    mat /= mat.sum()
    return mat


@dataclass
class MatchPrediction:
    home: str
    away: str
    neutral: bool
    lam_home: float
    lam_away: float
    rho: float
    matrix: np.ndarray            # exact scoreline PMF

    # --- result ---
    @property
    def p_home_win(self) -> float:
        return float(np.tril(self.matrix, -1).sum())   # home goals > away goals

    @property
    def p_away_win(self) -> float:
        return float(np.triu(self.matrix, 1).sum())

    @property
    def p_draw(self) -> float:
        return float(np.trace(self.matrix))

    # --- goals ---
    @property
    def exp_goals_home(self) -> float:
        i = np.arange(self.matrix.shape[0])
        return float((i[:, None] * self.matrix).sum())

    @property
    def exp_goals_away(self) -> float:
        j = np.arange(self.matrix.shape[1])
        return float((j[None, :] * self.matrix).sum())

    @property
    def exp_total(self) -> float:
        return self.exp_goals_home + self.exp_goals_away

    def total_goals_distribution(self) -> np.ndarray:
        """P(total = s) for s = 0 .. 2*G (sum of anti-diagonals)."""
        g = self.matrix.shape[0] - 1
        dist = np.zeros(2 * g + 1)
        for s in range(2 * g + 1):
            dist[s] = np.trace(self.matrix[:, ::-1], offset=(self.matrix.shape[1] - 1) - s)
        return dist

    def prob_over(self, line: float) -> float:
        dist = self.total_goals_distribution()
        totals = np.arange(dist.size)
        return float(dist[totals > line].sum())

    def prob_under(self, line: float) -> float:
        return 1.0 - self.prob_over(line)

    @property
    def p_btts(self) -> float:
        return float(self.matrix[1:, 1:].sum())

    @property
    def p_clean_sheet_home(self) -> float:
        return float(self.matrix[:, 0].sum())     # away scores 0

    @property
    def p_clean_sheet_away(self) -> float:
        return float(self.matrix[0, :].sum())     # home scores 0

    def top_scorelines(self, n: int = 6) -> list[tuple[int, int, float]]:
        flat = np.argsort(self.matrix, axis=None)[::-1][:n]
        out = []
        for idx in flat:
            i, j = np.unravel_index(idx, self.matrix.shape)
            out.append((int(i), int(j), float(self.matrix[i, j])))
        return out

    def report(self) -> str:
        venue = "neutral" if self.neutral else f"{self.home} at home"
        lines = [
            f"{self.home} vs {self.away}  ({venue})",
            f"  expected goals : {self.home} {self.lam_home:.2f} - {self.lam_away:.2f} {self.away}"
            f"   (total {self.exp_total:.2f})",
            f"  result         : {self.home} win {self.p_home_win*100:5.1f}%  |  "
            f"draw {self.p_draw*100:5.1f}%  |  {self.away} win {self.p_away_win*100:5.1f}%",
            f"  both teams score {self.p_btts*100:5.1f}%   "
            f"clean sheet: {self.home} {self.p_clean_sheet_home*100:.0f}% / "
            f"{self.away} {self.p_clean_sheet_away*100:.0f}%",
        ]
        lines.append("  totals         : " + "  ".join(
            f"O{ln} {self.prob_over(ln)*100:4.1f}%" for ln in (0.5, 1.5, 2.5, 3.5, 4.5)))
        tops = self.top_scorelines(6)
        lines.append("  likely scores  : " + "  ".join(
            f"{i}-{j} {p*100:.1f}%" for i, j, p in tops))
        return "\n".join(lines)


def predict_match(ratings: DCRatings, home: str, away: str,
                  neutral: bool = True, cfg=CONFIG,
                  allow_unknown: bool = False) -> MatchPrediction:
    """Full match prediction. Raises UnknownTeamError when either side has no
    fitted rating (see DCRatings.expected_goals) unless `allow_unknown`."""
    lam_h, lam_a = ratings.expected_goals(home, away, neutral=neutral, cfg=cfg,
                                          allow_unknown=allow_unknown)
    mat = score_matrix(lam_h, lam_a, ratings.rho, cfg.max_goals_grid)
    return MatchPrediction(home=home, away=away, neutral=neutral,
                           lam_home=lam_h, lam_away=lam_a, rho=ratings.rho, matrix=mat)


# --- Monte-Carlo sampler (for the tournament sim and as a grid cross-check) ----
def sample_scorelines(lam_h: float, lam_a: float, rho: float, n: int,
                      rng: np.random.Generator, cfg=CONFIG) -> tuple[np.ndarray, np.ndarray]:
    """Sample n (home_goals, away_goals) pairs from the exact DC scoreline PMF."""
    mat = score_matrix(lam_h, lam_a, rho, cfg.max_goals_grid)
    flat = mat.ravel()
    cols = mat.shape[1]
    idx = rng.choice(flat.size, size=n, p=flat)
    return idx // cols, idx % cols
