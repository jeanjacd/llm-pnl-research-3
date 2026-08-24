"""
config.py
=========
Single source of truth for every tunable parameter in the model. Nothing
statistical is hard-coded elsewhere -- if a number affects a prediction, it lives
here with a comment explaining what it is and how it was chosen.

Parameters marked "(CV-tuned)" are not guesses: `python -m wc2026 tune` searches
them by walk-forward cross-validation against out-of-sample log-loss and writes the
winners back here (or into data/fitted_params.json). The defaults below are sane
starting points grounded in the Dixon-Coles literature.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# --- Paths -------------------------------------------------------------------
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PKG_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# WC / international era (ARCHIVED 2026-07-20 -- see archive/wc2026/ARCHIVE.md).
# These paths keep the frozen tournament code exercisable; no MLS code path reads
# them, and the archived fitted params must never be applied to MLS fits.
ARCHIVE_WC_DIR = os.path.join(PROJECT_DIR, "archive", "wc2026")
WC_RESULTS_CSV = os.path.join(ARCHIVE_WC_DIR, "data", "raw", "results.csv")
WC_FITTED_PARAMS_JSON = os.path.join(ARCHIVE_WC_DIR, "data", "fitted_params.json")
WC_ADJUSTMENTS_JSON = os.path.join(ARCHIVE_WC_DIR, "data", "adjustments.json")
FORMAT_JSON = os.path.join(ARCHIVE_WC_DIR, "data", "format_2026.json")

# NOTE: there is deliberately NO active-league path here.
#
# A single RESULTS_CSV / FITTED_PARAMS_JSON / ADJUSTMENTS_JSON / CALIBRATION
# global made one league implicitly "the" league, which is exactly how a fit,
# a calibration or an adjustments file leaks from one competition into another.
# Every per-league path now lives on its LeagueSpec (wc2026/leagues.py) and must
# be passed explicitly:
#
#     from wc2026.leagues import get_league
#     spec = get_league("premier_league")
#     spec.matches_csv, spec.fitted_params_json, spec.calibration_dir, ...


@dataclass(frozen=True)
class ModelConfig:
    # --- Time weighting ------------------------------------------------------
    # Exponential half-life of a match's influence, in days. Recent matches carry
    # more weight, so "recent form" falls out of the fit naturally rather than
    # being bolted on. Dixon-Coles used ~0.0065/day; we parameterise by half-life
    # for interpretability. 540 days ~= two seasons of international football.
    # (CV-tuned)
    half_life_days: float = 540.0

    # --- Dixon-Coles low-score dependence -----------------------------------
    # Correlation/dependence parameter. MUST be slightly NEGATIVE: it boosts
    # 0-0 and 1-1 and trims 1-0/0-1, which is the empirically observed deviation
    # from independent Poisson. (The legacy code had this POSITIVE, which inverted
    # the correction.) Bounded to a sane negative range during fitting. (CV-tuned)
    rho: float = -0.05
    rho_bounds: tuple[float, float] = (-0.20, -0.001)

    # --- Match-importance weights -------------------------------------------
    # Competitive matches are more predictive of tournament performance than
    # friendlies, so they get more weight in the likelihood. Keyed by a coarse
    # competition tier derived from the `tournament` column (see data/loader.py).
    importance_weights: dict = field(default_factory=lambda: {
        "world_cup": 1.00,        # FIFA World Cup (finals)
        "continental": 0.90,      # Euro, Copa America, AFCON, Asian Cup, Gold Cup
        "nations_league": 0.85,   # UEFA/CONCACAF Nations League
        "qualifier": 0.80,        # World Cup / continental qualification
        "confederations": 0.75,   # Confederations Cup, Finalissima, etc.
        "friendly": 0.50,         # Friendlies / minor invitationals
    })

    # --- Elo prior (shrinkage) ----------------------------------------------
    # Standard World Football Elo constants (used by our transparent Elo in
    # model/elo.py). K scales update size; goal-difference multiplier follows the
    # eloratings.net convention. HFA in Elo points for non-neutral matches.
    elo_k: float = 40.0
    elo_hfa: float = 65.0
    elo_initial: float = 1500.0

    # Shrinkage of the Dixon-Coles attack/defense toward the Elo-implied strength:
    # weight_on_fit = n_eff / (n_eff + blend_k), where n_eff is a team's
    # time-weighted match count. Larger blend_k => trust the Elo prior more for
    # teams with little recent data (e.g. World Cup minnows). (CV-tuned)
    blend_k: float = 12.0

    # --- Goal model bounds ---------------------------------------------------
    lambda_floor: float = 0.15     # expected goals never drop below this
    lambda_cap: float = 6.0        # nor exceed this (numerical guard)
    max_goals_grid: int = 12       # scoreline grid is 0..max_goals_grid each side

    # --- In-tournament goal calibration --------------------------------------
    # The WC 2026 runs hotter than the all-internationals scoring level the DC
    # intercept encodes, and no recency weighting can fix that (validated: the
    # under-3.5 bias is flat across the half-life grid). Fix: scale both lambdas
    # by r = exp(w * log(actual/predicted total goals in played WC games)),
    # w = n/(n+k) -- an empirical-Bayes intercept nudge, walk-forward safe.
    # k=15 won a walk-forward sweep over all 100 played games on totals Brier
    # AND 1X2 log-loss (2026-07-14). Set to None to disable.
    tournament_calib_k: float | None = 15.0

    # --- Knockout / extra-time ----------------------------------------------
    # Extra time is 30 min vs 90 regulation => goal rates scale by 30/90.
    extra_time_fraction: float = 30.0 / 90.0
    # Probability the higher-Elo side wins a penalty shootout (mild skill edge;
    # 0.5 would be a pure coin flip). Interpolated by Elo gap in sim/tournament.py.
    shootout_max_skill_edge: float = 0.10   # caps deviation from 50/50

    # --- Host nations (2026): partial home edge even though most games are
    # "neutral". The `neutral` column in results.csv already flags host home games
    # as non-neutral, so this is informational; the home term is applied whenever
    # neutral is False. Listed for transparency / tournament venue logic.
    host_nations: tuple = ("United States", "Mexico", "Canada")


CONFIG = ModelConfig()


# --- Per-league configuration lives in wc2026/leagues.py ----------------------
# The old MLS_CONFIG / MLS_IMPORTANCE_WEIGHTS singletons are gone: an
# importance-weight table keyed to one league's competition names cannot be
# reused by another league, and keeping it here invited exactly that. Each
# LeagueSpec now owns its own ModelConfig.
