"""
loader.py
=========
Parse the cached results.csv into a tidy, typed match table and expose the live
2026 World Cup state (participants, played results, upcoming fixtures).

The single dataset is used two ways:
  * `training_matches()` -- every completed full international, for fitting strength.
  * `tournament_state()` -- the 2026 World Cup rows, split into played vs. upcoming,
    which the tournament simulator advances forward.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# --- Competition tier classification (drives match-importance weighting) ------
def classify_tier(tournament: str) -> str:
    """Map a raw `tournament` string to a coarse INTERNATIONAL competition tier.

    This is the archived World-Cup-era rule set, kept so the archived dataset
    (which has no `tier` column) still loads. League datasets do NOT use it:
    `data/espn.py` writes an explicit `tier` per row from the league registry,
    so a competition can never be re-derived -- and mis-derived -- from a label.

    Order matters: 'qualification' is checked before the continental names so
    that 'UEFA Euro qualification' is a qualifier, not a continental finals.
    """
    t = (tournament or "").lower()
    if "world cup" in t and "qualif" not in t:
        return "world_cup"
    if "qualif" in t:
        return "qualifier"
    if "nations league" in t:
        return "nations_league"
    if "confederations" in t or "finalissima" in t:
        return "confederations"
    continental_keys = (
        "uefa euro", "copa am", "african cup of nations", "afc asian cup",
        "gold cup", "concacaf championship", "oceania nations", "gulf cup",
        "afcon",
    )
    if any(k in t for k in continental_keys):
        return "continental"
    return "friendly"


def load_matches(path: str) -> pd.DataFrame:
    """Load and type a match table from an EXPLICIT path.

    The path is required: there is no implicit "active league" dataset. Adds
    parsed `date`, numeric goals, a `played` flag, and `tier` (taken from the
    file when present -- league datasets carry it -- otherwise derived with the
    archived international rules)."""
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Exact kickoff time (MLS era only; the archived WC dataset has no such
    # column). Kept as naive UTC for the betting layer's closing-line capture.
    if "kickoff_utc" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce",
                                           utc=True).dt.tz_convert(None)
    # Upcoming fixtures have empty scores -> NaN.
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    if "tier" not in df.columns:
        # Archived international dataset: derive the tier from the label.
        df["tier"] = df["tournament"].map(classify_tier)
    if "known_after_utc" in df.columns:
        df["known_after_utc"] = pd.to_datetime(df["known_after_utc"],
                                               errors="coerce", utc=True
                                               ).dt.tz_convert(None)
    # Drop rows without a date or without both team names (the upstream file adds
    # placeholder fixtures for undecided ties, e.g. the final before the SFs finish).
    df = (df.dropna(subset=["date", "home_team", "away_team"])
            .sort_values("date").reset_index(drop=True))
    return df


def load_league(spec, tiers: str = "all") -> pd.DataFrame:
    """Load one league's dataset from its own path.

    `tiers`: "all" (default), "scoring" (competitions eligible for evaluation
    and trading) or "training" (scoring + training-only). Restricting here is
    how a training-only cup is kept out of a scored or traded population.
    """
    df = load_matches(spec.matches_csv)
    if tiers == "scoring":
        df = df[df["tier"].isin(spec.scoring_tiers)].reset_index(drop=True)
    elif tiers == "training":
        allowed = set(spec.scoring_tiers) | set(spec.training_only_tiers)
        df = df[df["tier"].isin(allowed)].reset_index(drop=True)
    elif tiers != "all":
        raise ValueError("tiers must be all|scoring|training, got %r" % tiers)
    return df


def known_at(df: pd.DataFrame, as_of) -> pd.DataFrame:
    """Point-in-time filter: only rows whose result was KNOWABLE by `as_of`.

    Uses `known_after_utc` (final whistle) rather than the match date, so a
    match kicking off before the decision time but finishing after it cannot
    leak its result backwards. Rows without the column (archived data) fall
    back to the match date, which is the older, weaker guarantee.
    """
    cutoff = pd.Timestamp(as_of)
    if "known_after_utc" in df.columns:
        effective = df["known_after_utc"].fillna(
            df["date"] + pd.Timedelta(days=1))
    else:
        effective = df["date"]
    return df[effective <= cutoff].reset_index(drop=True)


def training_matches(df: pd.DataFrame, since: str | None = None) -> pd.DataFrame:
    """Completed matches only, optionally restricted to date >= `since`."""
    out = df[df["played"]].copy()
    if since is not None:
        out = out[out["date"] >= pd.Timestamp(since)]
    return out.reset_index(drop=True)


# --- 2026 World Cup live state ------------------------------------------------
@dataclass
class TournamentState:
    participants: list[str]          # all 48 teams appearing in 2026 WC rows
    played: pd.DataFrame             # 2026 WC matches already decided
    upcoming: pd.DataFrame           # 2026 WC fixtures not yet played
    as_of: pd.Timestamp              # latest played-match date (the "now")

    @property
    def n_played(self) -> int:
        return len(self.played)

    @property
    def n_upcoming(self) -> int:
        return len(self.upcoming)


def wc2026_matches(df: pd.DataFrame) -> pd.DataFrame:
    """All 2026 FIFA World Cup rows (played and scheduled)."""
    mask = (df["tournament"] == "FIFA World Cup") & (df["date"] >= pd.Timestamp("2026-01-01"))
    return df[mask].copy().reset_index(drop=True)


def tournament_state(df: pd.DataFrame) -> TournamentState:
    """Split the 2026 World Cup into played vs. upcoming and list participants."""
    wc = wc2026_matches(df)
    teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))
    played = wc[wc["played"]].reset_index(drop=True)
    upcoming = wc[~wc["played"]].reset_index(drop=True)
    as_of = played["date"].max() if len(played) else pd.Timestamp("2026-06-11")
    return TournamentState(participants=teams, played=played,
                           upcoming=upcoming, as_of=as_of)


def all_teams(df: pd.DataFrame) -> list[str]:
    """Every team that has ever appeared (home or away)."""
    return sorted(set(df["home_team"]) | set(df["away_team"]))
