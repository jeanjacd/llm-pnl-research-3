"""
ratings.py
==========
Turns the raw team/player database into the numbers the match simulator needs:
the expected goals (lambda) for each side, plus a per-player scoring distribution.

Design choice
-------------
Real per-team attacking/defensive xG for *national* teams is hard to source cleanly,
so by default we derive expected goals from World Football Elo. If a team in
teams.json carries explicit `attack_rating` / `defense_rating` (xG per match), those
are preferred automatically -- so when you later ingest real numbers via
build_database.py, the model upgrades itself with zero code changes.

Elo -> goals mapping (documented assumptions, tune in CALIBRATION):
  win_prob_A   = 1 / (1 + 10**(-(eloA - eloB + HFA) / 400))      # standard Elo
  supremacy    = (eloA - eloB + HFA) / ELO_PER_GOAL              # expected goal diff
  total_goals  = BASE_TOTAL_GOALS                                # avg goals in match
  lambda_A     = max((total + supremacy) / 2, LAMBDA_FLOOR)
  lambda_B     = max((total - supremacy) / 2, LAMBDA_FLOOR)
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---- CALIBRATION CONSTANTS (tune against historical results) -----------------
ELO_PER_GOAL = 190.0       # backtest-tuned on 49k internationals (was 150; higher = less extreme)
BASE_TOTAL_GOALS = 2.80    # backtest-tuned (was 2.70)
LAMBDA_FLOOR = 0.18        # no team's expected goals drops below this
HOME_FIELD_ADVANTAGE = 60.0  # Elo bonus applied only when a match is NOT neutral
                             # (backtest: ~0.03 log-loss gain from modeling home edge)

# League average goals per team — the divisor in the attack/defense model. Defaults to
# half the base total, but load_database() overrides it from teams.json _meta when
# fit_team_strength has written a fitted value (so totals stay calibrated to real data).
_LEAGUE_AVG_GF = BASE_TOTAL_GOALS / 2.0

# How fast the in-tournament attack/defense fit takes over from the Elo prior.
# weight_on_data = games / (games + BLEND_K). Set to 40 after a full walk-forward over all
# 54 WC games (backtest_blend.py) showed PURE ELO beats the 2-3 game attack/defense fit on
# both results AND totals -- the fit is still noise at this sample. At 3 games K=40 gives
# only ~7% data weight (~pure Elo). RE-RUN backtest_blend.py as games accumulate and lower
# BLEND_K once the data earns its weight.
BLEND_K = 15.0   # leakage-free walk-forward (backtest_v2.py) with a 540-day attack/defense
                 # window: K=15 gave the best totals Brier and ~best 1X2 -> the blend now
                 # beats pure Elo on BOTH once attack/defense is fit on enough games.


@dataclass
class Team:
    name: str
    elo: float
    group: str = ""
    fifa_rank: int = 999
    gk_quality: float = 0.50
    set_piece_xg_share: float = 0.25
    attack_rating: float | None = None   # optional real xG-for per match
    defense_rating: float | None = None  # optional real xG-against per match
    wc_games: int = 0                    # games behind attack/defense fit (blend weight)
    elo_source: str = "unknown"


@dataclass
class Player:
    name: str
    team: str
    position: str
    xG90: float
    team_shot_share: float
    penalty_taker: bool = False
    expected_minutes: float = 80.0


@dataclass
class Database:
    teams: dict[str, Team] = field(default_factory=dict)
    players: dict[str, list[Player]] = field(default_factory=dict)  # keyed by team name

    def team(self, name: str) -> Team:
        if name not in self.teams:
            raise KeyError(f"Unknown team '{name}'. Known: {sorted(self.teams)}")
        return self.teams[name]

    def team_players(self, name: str) -> list[Player]:
        return self.players.get(name, [])


def load_database(data_dir: str = DATA_DIR) -> Database:
    with open(os.path.join(data_dir, "teams.json"), encoding="utf-8") as f:
        traw = json.load(f)
    with open(os.path.join(data_dir, "players.json"), encoding="utf-8") as f:
        praw = json.load(f)

    # Pick up the fitted league average (written by fit_team_strength) so the
    # attack/defense model's totals stay calibrated to the current scoring environment.
    global _LEAGUE_AVG_GF
    _LEAGUE_AVG_GF = float(traw.get("_meta", {}).get("league_avg_gf", BASE_TOTAL_GOALS / 2.0))

    db = Database()
    for t in traw["teams"]:
        db.teams[t["name"]] = Team(
            name=t["name"],
            elo=float(t["elo"]),
            group=t.get("group", ""),
            fifa_rank=int(t.get("fifa_rank", 999)),
            gk_quality=float(t.get("gk_quality", 0.50)),
            set_piece_xg_share=float(t.get("set_piece_xg_share", 0.25)),
            attack_rating=t.get("attack_rating"),
            defense_rating=t.get("defense_rating"),
            wc_games=int(t.get("wc_games", 0)),
            elo_source=t.get("elo_source", "unknown"),
        )
    for p in praw["players"]:
        db.players.setdefault(p["team"], []).append(Player(
            name=p["name"],
            team=p["team"],
            position=p.get("position", "FW"),
            xG90=float(p["xG90"]),
            team_shot_share=float(p["team_shot_share"]),
            penalty_taker=bool(p.get("penalty_taker", False)),
            expected_minutes=float(p.get("expected_minutes", 80.0)),
        ))
    return db


def expected_goals(home: Team, away: Team, neutral: bool = True) -> tuple[float, float]:
    """
    Return (lambda_home, lambda_away): expected goals for each side.

    The Elo-derived lambdas are always computed (well-calibrated, full-history prior).
    If both teams have an in-tournament attack/defense fit, we BLEND the two by games
    played -- weight_on_data = games/(games + BLEND_K) -- so early in a tournament the
    strong Elo prior dominates and the noisy 2-game fit only nudges it, rather than
    replacing it outright.
    """
    hfa = 0.0 if neutral else HOME_FIELD_ADVANTAGE

    # --- Elo-derived supremacy model (always available) ---
    diff = (home.elo - away.elo) + hfa
    supremacy = diff / ELO_PER_GOAL
    lam_h = (BASE_TOTAL_GOALS + supremacy) / 2.0 * (1.0 - (away.gk_quality - 0.5) * 0.30)
    lam_a = (BASE_TOTAL_GOALS - supremacy) / 2.0 * (1.0 - (home.gk_quality - 0.5) * 0.30)

    # --- Attack/defense model, blended in by sample size if available ---
    if home.attack_rating is not None and away.defense_rating is not None \
       and away.attack_rating is not None and home.defense_rating is not None:
        # Dixon-Coles multiplicative form: lam = attack_home * defense_away / avg.
        # No gk_quality term here -- defense_rating already reflects goals conceded
        # (keeper included), so applying gk again would double-count.
        ad_h = home.attack_rating * (away.defense_rating / _LEAGUE_AVG_GF)
        ad_a = away.attack_rating * (home.defense_rating / _LEAGUE_AVG_GF)
        g = (home.wc_games + away.wc_games) / 2.0
        w = g / (g + BLEND_K)             # weight on in-tournament data
        lam_h = w * ad_h + (1.0 - w) * lam_h
        lam_a = w * ad_a + (1.0 - w) * lam_a

    return max(lam_h, LAMBDA_FLOOR), max(lam_a, LAMBDA_FLOOR)


def scorer_distribution(team: Team, players: list[Player]) -> list[tuple[str, float]]:
    """
    Weight each known player by their share of the team's goal expectation.
    Returns list of (player_name, weight) where weights sum to <= 1; the
    remainder is implicitly 'other / own goal / unlisted player'.
    """
    weights = []
    for p in players:
        minute_factor = p.expected_minutes / 90.0
        w = p.xG90 * p.team_shot_share * minute_factor
        if p.penalty_taker:
            w *= 1.12  # modest bump for penalty duty
        weights.append((p.name, w))

    total = sum(w for _, w in weights)
    if total <= 0:
        return []
    # Listed players typically account for ~70-80% of goals; reserve the rest.
    listed_share = 0.78
    return [(name, listed_share * w / total) for name, w in weights]


if __name__ == "__main__":
    db = load_database()
    print(f"Loaded {len(db.teams)} teams, "
          f"{sum(len(v) for v in db.players.values())} players.\n")
    a, b = db.team("Brazil"), db.team("Germany")
    lh, la = expected_goals(a, b)
    print(f"{a.name} (Elo {a.elo}) vs {b.name} (Elo {b.elo})")
    print(f"  expected goals: {a.name} {lh:.2f}  -  {la:.2f} {b.name}")
    print(f"  {a.name} scorer weights: {scorer_distribution(a, db.team_players(a.name))}")
