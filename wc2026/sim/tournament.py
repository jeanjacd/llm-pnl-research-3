"""
tournament.py
=============
Monte-Carlo simulation of the remaining 2026 World Cup knockout bracket from the
live state in the data, producing each team's probability of reaching every round
and of winning the title.

How it works
------------
1. Read the current knockout frontier (the upcoming knockout fixtures) from the data.
   Right now that is the 16 Round-of-32 ties; as results come in it becomes the R16,
   then QF, etc. The frontier ties give the actual pairings; subsequent rounds are
   generated from the official bracket adjacency in data/format_2026.json.
2. Precompute, ONCE, a knockout win-probability matrix W[a, b] = P(a eliminates b)
   accounting for regulation + extra time + penalty shootout. A knockout tie then
   reduces to a single Bernoulli draw, which lets us vectorise the WHOLE bracket
   across all N simulations with numpy -> tens of thousands of sims in a blink.
3. Aggregate how often each team reaches each round / wins the final.

Knockout win probability (a as nominal home, neutral by default):
    P(a wins) = P(a wins reg) + P(draw reg) * [ P(a wins ET) + P(draw ET)*P(a wins SO) ]
Extra time scales the goal rates by 30/90; the shootout is a near-coin-flip nudged
slightly by team strength.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CONFIG, FORMAT_JSON
from ..model.dixon_coles import DCRatings
from .match import predict_match, score_matrix

ROUND_ORDER = ["round_of_32", "round_of_16", "quarter_finals",
               "semi_finals", "final", "champion"]
ROUND_LABEL = {
    "round_of_16": "Reach R16", "quarter_finals": "Reach QF",
    "semi_finals": "Reach SF", "final": "Reach Final", "champion": "Champion",
}


def load_format(path: str | None = None) -> dict:
    with open(path or FORMAT_JSON, encoding="utf-8") as f:
        return json.load(f)


def _shootout_prob(ratings: DCRatings, a: str, b: str, cfg=CONFIG) -> float:
    """Near-coin-flip nudged by overall strength (attack+defense)."""
    sa = ratings.attack.get(a, 0.0) + ratings.defense.get(a, 0.0)
    sb = ratings.attack.get(b, 0.0) + ratings.defense.get(b, 0.0)
    edge = cfg.shootout_max_skill_edge * np.tanh(sa - sb)
    return float(np.clip(0.5 + edge, 0.0, 1.0))


def knockout_win_prob(ratings: DCRatings, a: str, b: str,
                      neutral: bool = True, cfg=CONFIG) -> float:
    """P(a eliminates b) over a single knockout tie (reg + ET + shootout)."""
    pred = predict_match(ratings, a, b, neutral=neutral, cfg=cfg)
    p_a_reg, p_draw = pred.p_home_win, pred.p_draw

    lh, la = ratings.expected_goals(a, b, neutral=neutral, cfg=cfg)
    et = score_matrix(lh * cfg.extra_time_fraction, la * cfg.extra_time_fraction,
                      ratings.rho, cfg.max_goals_grid)
    p_a_et = float(np.tril(et, -1).sum())
    p_draw_et = float(np.trace(et))

    p_a_so = _shootout_prob(ratings, a, b, cfg)
    return p_a_reg + p_draw * (p_a_et + p_draw_et * p_a_so)


@dataclass
class TournamentResult:
    table: pd.DataFrame          # per-team probabilities by round, sorted by title odds
    n_sims: int
    frontier_round: str
    champion_se: float           # MC standard error on the top title probability

    def report(self, top_n: int = 16) -> str:
        cols = [c for c in ["Reach R16", "Reach QF", "Reach SF", "Reach Final", "Champion"]
                if c in self.table.columns]
        head = (f"2026 World Cup simulation  ({self.n_sims:,} sims, "
                f"frontier = {self.frontier_round.replace('_', ' ')}, "
                f"champion SE +/-{self.champion_se*100:.2f}pts)\n")
        lines = [head, f"  {'team':22s}" + "".join(f"{c:>12s}" for c in cols)]
        for _, r in self.table.head(top_n).iterrows():
            lines.append(f"  {r['team']:22s}" +
                         "".join(f"{r[c]*100:>11.1f}%" for c in cols))
        return "\n".join(lines)


def _frontier_from_state(state, fmt: dict) -> tuple[list[tuple[str, str, bool]], str]:
    """Return the current knockout frontier ties [(home, away, neutral)] and round name.

    The upcoming 2026 knockout fixtures define the frontier. Round is inferred from
    the number of ties (16 -> R32, 8 -> R16, 4 -> QF, 2 -> SF, 1 -> Final).
    """
    # Knockout = 2026 WC matches beyond the group stage; the frontier is the
    # earliest still-upcoming knockout round.
    up = state.upcoming
    # All currently-upcoming WC matches are the frontier round (future rounds are TBD).
    ties = [(r.home_team, r.away_team, bool(r.neutral)) for _, r in up.iterrows()]
    size_to_round = {16: "round_of_32", 8: "round_of_16", 4: "quarter_finals",
                     2: "semi_finals", 1: "final"}
    rnd = size_to_round.get(len(ties))
    if rnd is None:
        raise ValueError(
            f"Expected 1/2/4/8/16 upcoming knockout ties, got {len(ties)}. "
            "The group stage may still be in progress, or the schedule is partial."
        )
    return ties, rnd


def simulate_tournament(ratings: DCRatings, state, n_sims: int = 20000,
                        fmt: dict | None = None, seed: int | None = 12345,
                        cfg=CONFIG) -> TournamentResult:
    fmt = fmt or load_format()
    ties, frontier_round = _frontier_from_state(state, fmt)
    rng = np.random.default_rng(seed)

    # Index the knockout teams.
    teams = sorted({t for tie in ties for t in (tie[0], tie[1])})
    tidx = {t: i for i, t in enumerate(teams)}
    nT = len(teams)

    # Precompute neutral-venue knockout win matrix W[a, b] = P(a beats b).
    W = np.full((nT, nT), 0.5)
    for a in teams:
        for b in teams:
            if a != b:
                W[tidx[a], tidx[b]] = knockout_win_prob(ratings, a, b, neutral=True, cfg=cfg)

    # Frontier ties: respect each fixture's actual neutral flag (host games aren't neutral).
    front_home = np.array([tidx[h] for h, a, n in ties])
    front_away = np.array([tidx[a] for h, a, n in ties])
    front_p = np.array([knockout_win_prob(ratings, h, a, neutral=n, cfg=cfg)
                        for h, a, n in ties])

    # --- simulate frontier round (vectorised across sims) ---
    u = rng.random((len(ties), n_sims))
    winners = np.where(u < front_p[:, None], front_home[:, None], front_away[:, None])

    # Which bracket-tree rounds still need to be played after the frontier.
    frontier_pos = ROUND_ORDER.index(frontier_round)
    tree = fmt["knockout_bracket"]
    tree_rounds = [r for r in ["round_of_16", "quarter_finals", "semi_finals", "final"]
                   if ROUND_ORDER.index(r) > frontier_pos]

    # Stage tallies: team -> count reaching that stage.
    counts: dict[str, np.ndarray] = {}
    # Frontier winners advance to the next round.
    counts[ROUND_ORDER[frontier_pos + 1]] = np.bincount(winners.ravel(), minlength=nT)

    cur = winners  # shape (n_current_ties, n_sims)
    for rname in tree_rounds:
        pairs = tree[rname]
        nxt = np.empty((len(pairs), n_sims), dtype=int)
        for t, (i, j) in enumerate(pairs):
            ai, bj = cur[i], cur[j]
            p = W[ai, bj]
            uu = rng.random(n_sims)
            nxt[t] = np.where(uu < p, ai, bj)
        cur = nxt
        advanced_to = ROUND_ORDER[ROUND_ORDER.index(rname) + 1]
        counts[advanced_to] = np.bincount(cur.ravel(), minlength=nT)

    # --- assemble table ---
    rows = []
    for t in teams:
        i = tidx[t]
        row = {"team": t}
        for stage, label in ROUND_LABEL.items():
            if stage in counts:
                row[label] = counts[stage][i] / n_sims
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("Champion", ascending=False).reset_index(drop=True)

    top_p = float(table["Champion"].iloc[0]) if len(table) else 0.0
    se = float(np.sqrt(max(top_p * (1 - top_p), 0) / n_sims))
    return TournamentResult(table=table, n_sims=n_sims,
                            frontier_round=frontier_round, champion_se=se)
