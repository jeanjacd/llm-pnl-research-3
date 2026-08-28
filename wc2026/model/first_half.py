"""
model/first_half.py
===================
First-half forecasts from the SAME Dixon-Coles engine, fitted to a different
target.

No new mathematics. `dixon_coles` and `ratings` read the goal columns by name,
so a frame whose `home_score`/`away_score` hold the HALF-TIME score produces
first-half attack and defence ratings through exactly the frozen path the
full-match model uses. The scoreline grid, the joint valuation, the calibration
hooks and the claim vocabulary all carry over unchanged.

WHY IT IS WORTH FITTING SEPARATELY RATHER THAN SCALING THE FULL-MATCH MODEL.
Measured over every completed match in the five leagues after backfilling
half-time scores:

    league          H1 goals   full match   share
    bundesliga         1.26       3.06       41%
    la_liga            1.07       2.66       40%
    ligue_1            1.09       2.72       40%
    mls                1.19       2.94       40%
    premier_league     1.18       2.82       42%

Goals are not uniform across the halves -- roughly 40% arrive before the
interval -- so halving a full-match rate would misprice every first-half line.
The share is stable across leagues, which is also the strongest available
evidence that the minute attribution in `data.espn.half_time_score` is right:
a mis-parse would not reproduce the known pattern five times independently.

COVERAGE IS NOT COMPLETE, AND THE GAP IS DROPPED RATHER THAN FILLED. About 1-2%
of completed matches have no usable goal detail, so their half-time score is
unknown. Those rows are EXCLUDED from the fit. Treating an unknown as 0-0
would teach the model that goalless first halves are far more common than they
are, which is precisely the direction that makes every `1h_total_over` line
look cheap.
"""
from __future__ import annotations

import pandas as pd

HT_COLUMNS = ("home_ht_score", "away_ht_score")


class NoHalfTimeData(RuntimeError):
    """Not enough half-time history to fit. Never silently downgraded."""


def half_time_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """The same matches, with the half-time score standing in as THE score.

    Rows whose half-time score is unknown are dropped, not imputed. The
    returned frame is otherwise identical, so every dating, weighting and
    tier rule in the fit continues to apply.
    """
    missing = [c for c in HT_COLUMNS if c not in matches.columns]
    if missing:
        raise NoHalfTimeData(
            "no half-time columns (%s); re-ingest with `wc2026 update`"
            % ", ".join(missing))
    frame = matches.dropna(subset=list(HT_COLUMNS)).copy()
    if frame.empty:
        raise NoHalfTimeData("no match has a known half-time score")
    frame["home_score"] = frame["home_ht_score"].astype(int)
    frame["away_score"] = frame["away_ht_score"].astype(int)
    return frame


def coverage(matches: pd.DataFrame) -> dict:
    """How much of the history can actually be fitted, stated rather than assumed."""
    played = matches[matches["played"]] if "played" in matches else matches
    if not all(c in matches.columns for c in HT_COLUMNS):
        return {"played": len(played), "with_half_time": 0, "fraction": 0.0}
    known = played.dropna(subset=list(HT_COLUMNS))
    return {"played": len(played), "with_half_time": len(known),
            "fraction": (len(known) / len(played)) if len(played) else 0.0}


MIN_MATCHES = 200
MIN_FRACTION = 0.60


def build_first_half_strength(matches: pd.DataFrame, min_matches: int = MIN_MATCHES,
                              min_fraction: float = MIN_FRACTION, **kwargs):
    """First-half `DCRatings`, or raise.

    Refuses a thin or unrepresentative sample rather than returning ratings
    that would price markets badly. A league whose half-time history is mostly
    missing is not a league with a weak first-half model -- it is a league with
    no first-half model, and the two must not look alike downstream.
    """
    from .ratings import build_team_strength
    stats = coverage(matches)
    if stats["with_half_time"] < min_matches:
        raise NoHalfTimeData(
            "only %d matches with a half-time score; %d required"
            % (stats["with_half_time"], min_matches))
    if stats["fraction"] < min_fraction:
        raise NoHalfTimeData(
            "half-time score known for only %.0f%% of matches; %.0f%% required"
            % (100 * stats["fraction"], 100 * min_fraction))
    return build_team_strength(half_time_frame(matches), **kwargs)
