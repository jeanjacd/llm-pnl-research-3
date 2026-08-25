"""
paper/outcomes.py
=================
What a claim actually PAID, given a final scoreline.

This is the settlement twin of `paper.cycle.probability_for`, which says what
the model thought a claim was WORTH. The two must agree exactly, or measured
P&L is scoring a different question from the one the model answered -- a bias
that would look like edge (or like its absence) and be invisible in the output.

`test_outcomes.py` enforces that agreement directly rather than by inspection:
for every supported claim it sums the model's own scoreline grid over the cells
this module calls a win, and asserts the total equals `probability_for`. A
divergence in either file fails the suite.

SETTLEMENT BASIS. Every venue market we price settles on regulation time --
90 minutes plus stoppage, excluding extra time and penalties (Kalshi states it
in `rules_primary`; the provider refuses any market whose rules do not).
A final score that INCLUDES extra time therefore cannot settle these claims,
so `regulation_score` refuses it rather than paying out on the wrong number.
"""
from __future__ import annotations


class UnsettleableClaim(ValueError):
    """The claim has no defined outcome. Never guessed, never defaulted."""


def _line(base: str) -> float:
    return float(base.rsplit("_", 1)[1])


def claim_is_true(claim: str, home_goals: int, away_goals: int) -> bool:
    """Did `claim` happen, given a regulation-time scoreline?

    Mirrors `probability_for` mask-for-mask. Raises rather than returning a
    default for anything it does not recognise: an unknown claim must stall
    settlement visibly, not silently resolve to a loss.
    """
    if home_goals < 0 or away_goals < 0:
        raise UnsettleableClaim("negative scoreline %d-%d"
                                % (home_goals, away_goals))
    negate = claim.startswith("not_")
    base = claim[4:] if negate else claim
    i, j = int(home_goals), int(away_goals)

    if base == "home_win":
        out = i > j
    elif base == "away_win":
        out = i < j
    elif base == "draw":
        out = i == j
    elif base == "btts":
        out = i >= 1 and j >= 1
    elif base.startswith("total_over_"):
        out = (i + j) > _line(base)
    elif base.startswith("total_under_"):
        out = (i + j) < _line(base)
    elif base.startswith("home_over_"):
        out = i > _line(base)
    elif base.startswith("away_over_"):
        out = j > _line(base)
    elif base.startswith("home_wins_by_over_"):
        out = (i - j) > _line(base)
    elif base.startswith("away_wins_by_over_"):
        out = (j - i) > _line(base)
    elif base.startswith("score_"):
        try:
            hg, ag = base.split("_", 1)[1].split("-")
            out = (i == int(hg)) and (j == int(ag))
        except ValueError as exc:
            raise UnsettleableClaim("malformed scoreline claim %r"
                                    % claim) from exc
    else:
        raise UnsettleableClaim("no settlement rule for claim %r" % claim)
    return (not out) if negate else out


def winning_side(claim: str, home_goals: int, away_goals: int) -> str:
    """"yes" if the claim happened, else "no" -- the broker's `result`."""
    return "yes" if claim_is_true(claim, home_goals, away_goals) else "no"


# --- what a fixture row can settle --------------------------------------------
FINAL_STATUSES = ("STATUS_FULL_TIME", "STATUS_FINAL")


def regulation_score(row) -> tuple:
    """(home_goals, away_goals) for a finished match, or None.

    None -- never a guess -- when the match is unfinished, abandoned, missing a
    score, or went to extra time. The last case matters: our markets settle on
    regulation, but the stored score for an extra-time match is the score AFTER
    extra time, so paying out on it would settle the wrong question. League
    fixtures never go to extra time; cup and playoff ties do.
    """
    def get(key):
        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            return None

    if not bool(get("played")):
        return None
    status = str(get("status") or "")
    if status and status not in FINAL_STATUSES:
        return None
    if bool(get("went_to_extra_time")):
        return None
    home, away = get("home_score"), get("away_score")
    if home is None or away is None:
        return None
    try:
        home, away = float(home), float(away)
    except (TypeError, ValueError):
        return None
    # pandas stores a missing score as NaN, and NaN fails every comparison --
    # including the integrality check below, which would raise on it.
    if home != home or away != away:
        return None
    if home != int(home) or away != int(away):
        return None
    return int(home), int(away)
