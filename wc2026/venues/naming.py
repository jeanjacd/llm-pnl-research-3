"""
venues/naming.py
================
Cross-venue club-name matching and fixture resolution.

Venues, our data source (ESPN), and the leagues themselves disagree about what
a club is called. Three distinct problems show up, and only the first is
solvable by string similarity:

  1. AFFIXES AND ACCENTS -- "FC Chelsea" / "Chelsea FC" / "Chelsea",
     "Alaves" / "Alavés". Handled by `normalise_team`.
  2. ENDONYM VS EXONYM -- Kalshi writes "FC Köln", ESPN writes "FC Cologne".
     No amount of edit distance bridges Köln -> Cologne; it needs a table.
  3. VENUE ABBREVIATION AND TRUNCATION -- "PSG", "M´gladbach", and Kalshi's
     genuinely truncated "Los Angeles F" / "Los Angeles G" (its own rules text
     cuts LAFC and LA Galaxy short). Also needs a table.

`ALIASES` therefore exists, but every entry in it was added because a MEASURED
resolution failure demanded it -- not pre-emptively. Adding a speculative alias
is worse than leaving a fixture unresolved: an unresolved fixture is skipped
and counted, a wrong alias silently prices the wrong match.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

# Club-name affixes carrying no identifying information across venues.
_AFFIXES = {"fc", "cf", "afc", "sc", "ac", "cd", "rc", "ud", "ca", "sv", "vfb",
            "vfl", "bsc", "tsg", "sd", "as", "ss", "us", "rcd", "club", "de",
            "the", "1", "04", "05", "96"}

# Venue spelling -> the name our fixture table uses. Keys are matched on the
# NORMALISED token set, so "FC Köln" and "Köln" both hit the "koln" entry.
# Each entry cites the observed failure that justifies it (2026-08-24 sweep).
ALIASES = {
    # endonym vs exonym
    "koln": "FC Cologne",                     # Kalshi "FC Köln" -> 0.27
    # "M´gladbach" folds to the tokens {m, gladbach}: the acute is not a
    # combining accent, so it survives NFKD and splits the word.
    "m gladbach": "Borussia Mönchengladbach",
    # venue abbreviation
    "psg": "Paris Saint-Germain",             # Kalshi "PSG" -> 0.25
    "bilbao": "Athletic Club",                # Kalshi "Bilbao" -> 0.14
    "atletico": "Atlético Madrid",            # Kalshi "Atletico" -> 0.19
    # Kalshi's rules text truncates these two clubs mid-word
    "los angeles f": "LAFC",
    "los angeles g": "LA Galaxy",
}


def normalise_team(name: str) -> set:
    """Accent-folded, affix-stripped token set for cross-venue comparison."""
    folded = unicodedata.normalize("NFKD", name or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = re.sub(r"[^a-z0-9 ]", " ", folded.lower())
    return {t for t in folded.split() if t and t not in _AFFIXES}


def apply_alias(name: str) -> str:
    """Rewrite a venue spelling to our canonical name, or return it unchanged.

    Matched on the normalised form so punctuation and affixes cannot cause a
    miss -- "M´gladbach", "M'gladbach" and "Mgladbach" all resolve.
    """
    tokens = normalise_team(name)
    if not tokens:
        return name
    joined = " ".join(sorted(tokens))
    for key, canonical in ALIASES.items():
        key_tokens = normalise_team(key)
        if key_tokens and (key_tokens == tokens
                           or " ".join(sorted(key_tokens)) == joined):
            return canonical
    return name


def _raw_similarity(a: str, b: str) -> float:
    ta, tb = normalise_team(a), normalise_team(b)
    if not ta or not tb:
        return 0.0
    if ta & tb:
        return 0.5 + 0.5 * len(ta & tb) / len(ta | tb)
    best = 0.0
    for x in ta:
        for y in tb:
            best = max(best, difflib.SequenceMatcher(None, x, y).ratio())
    return 0.5 * best


# An alias asserts a SPECIFIC club, so the expansion only counts against a name
# that really is that club. Without this, "PSG" -> "Paris Saint-Germain" scores
# 0.67 against "Paris FC" -- a different Ligue 1 club -- purely on the shared
# token "paris", and on a matchday when PSG is not playing there is no runner-up
# for the margin rule to catch. Measured, not hypothetical.
ALIAS_CONFIRM = 0.9


def name_similarity(a: str, b: str) -> float:
    """0..1 similarity of two club names, robust to affixes and accents.

    Raising the acceptance threshold is NOT an alternative to the alias check
    below: genuine matches score as low as 0.62 ("DC United" -> "D.C. United",
    "Saint Louis" -> "St. Louis CITY SC"), so a higher bar would discard real
    fixtures while still admitting the Paris FC confusion.
    """
    ca, cb = apply_alias(a), apply_alias(b)
    aliased = (ca != a) or (cb != b)
    score = _raw_similarity(ca, cb)
    if aliased and score < ALIAS_CONFIRM:
        # The alias did not land on the club it names; fall back to comparing
        # what the venue actually wrote, which for an abbreviation is weak.
        return _raw_similarity(a, b)
    return score


def resolve_fixture(home_raw, away_raw, kickoff, fixtures, min_score=0.6,
                    margin=0.05):
    """Resolve a venue event to exactly one of OUR fixtures, or None.

    Matching uses the kickoff date (+/- 1 day) AND both club names, and demands
    a unique winner clearly ahead of the runner-up. Name similarity alone is
    never sufficient: "Manchester City" and "Manchester United" score highly
    against each other, so an ambiguous pair is refused rather than guessed.

    Returns {"home", "away", "date", "kickoff_utc", "flipped", "score"} or
    None. `flipped` is True when the venue listed the away side first, so a
    caller that derived a claim from the venue's ordering can correct it.

    `kickoff_utc` is carried through because OUR fixture table is the only
    trustworthy source for it on some venues: Kalshi publishes an expiry
    (kickoff + 3h), not a kick-off, and using it made a match that was already
    72 minutes old look like it was 1.9 hours from starting.
    """
    if kickoff is None or fixtures is None or len(fixtures) == 0:
        return None
    import pandas as pd
    day = pd.Timestamp(kickoff)
    day = day.tz_localize(None) if day.tzinfo else day
    day = day.normalize()
    window = fixtures[(fixtures["date"] >= day - pd.Timedelta(days=1))
                      & (fixtures["date"] <= day + pd.Timedelta(days=1))]
    scored = []
    for _, row in window.iterrows():
        straight = min(name_similarity(home_raw, row["home_team"]),
                       name_similarity(away_raw, row["away_team"]))
        flipped = min(name_similarity(home_raw, row["away_team"]),
                      name_similarity(away_raw, row["home_team"]))
        if flipped > straight:
            scored.append((flipped, True, row))
        else:
            scored.append((straight, False, row))
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    best_score, best_flipped, best_row = scored[0]
    if best_score < min_score:
        return None
    if len(scored) > 1 and best_score - scored[1][0] < margin:
        return None                      # ambiguous: refuse rather than guess
    kickoff = best_row["kickoff_utc"] if "kickoff_utc" in best_row else None
    if kickoff is not None and pd.isna(kickoff):
        kickoff = None
    return {"home": best_row["home_team"], "away": best_row["away_team"],
            "date": best_row["date"], "kickoff_utc": kickoff,
            "flipped": best_flipped, "score": best_score}
