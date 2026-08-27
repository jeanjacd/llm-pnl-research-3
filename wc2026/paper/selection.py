"""
paper/selection.py
==================
Which candidates actually reach the board.

Measured on the live venues 2026-08-24: **307 placeable cases across 9 distinct
fixtures** -- 34 bets per match. And not merely correlated ones. A single
fixture produced `away_win`, `away_over_0.5`, `away_over_1.5`, `away_over_2.5`,
`away_wins_by_over_1.5` and `away_wins_by_over_2.5`: one directional view --
*the away side scores freely and wins comfortably* -- written six ways off one
scoreline grid. If the model is wrong about that team's attack they all lose
together, and a naive tally reads six independent failures.

Boarding all 307 costs about 28 hours of model time per run (measured: quant
87s, coach 180s) against a 45-minute job, and buys **9 independent
observations**. Boarding one market per fixture costs about 50 minutes and buys
the same 9.

WHY NOT TOP-N BY EV. It is the obvious cap and the wrong one for measurement.
The candidates with the highest apparent EV are disproportionately those where
the model's error happens to be positive -- the winner's curse -- so ranking by
EV samples the model precisely where it is most over-confident, then
generalises from there. The fixture set has to stay unbiased.

WHAT IS USED INSTEAD. A fixed preference over claim FAMILIES, ordered by how
well-validated the underlying model is and how deep the books are, with EV only
as a tie-break inside a family. The choice is therefore stable across runs and
across fixtures, which also makes the resulting observations comparable to each
other -- a mixture of 1X2 reads on some matches and exact-scoreline reads on
others would not be.
"""
from __future__ import annotations

import datetime as dt

# When to board a fixture, in hours before kick-off. A CONSISTENT lead time
# matters for measurement: a decision taken four days out and one taken two
# hours out are not comparable observations, and mixing them confounds any
# reading of the result. 24h is late enough for team news to have formed and
# early enough that a resting limit still has time to fill.
BOARD_TARGET_HOURS = 24.0
# The window is wider than the scheduler's interval so no fixture can slip
# between two runs; `PaperPortfolio.boarded` is what keeps it to once.
BOARD_WINDOW_HOURS = 6.0
# Below this there is no time for a resting order to be reached, so a fixture
# missed entirely is let go rather than boarded at the whistle.
BOARD_MIN_HOURS = 2.0

# --- deferral retries ---------------------------------------------------------
# A DEFER is the board saying it LACKED INFORMATION -- the coach returns
# `required_reruns`, literally asking to be run again once the team news lands.
# Recording that as "boarded, done" was incoherent: the fixture asked to be
# revisited and was then permanently barred from it. It also quietly cost the
# sample, because a fixture that never gets an order never produces a fill, a
# closing line, or a P&L.
#
# PASS and REJECT are NOT retried. That distinction is the whole safeguard:
# those are the board actively declining, and re-asking until it says yes is
# just re-rolling the dice -- the multiple-comparisons problem that biases
# everything toward false positives. Only a DEFER is retried, and only once.
RETRY_MAX_ATTEMPTS = 2

# The schedule the retry has to fit inside. `test_workflows.py` pins this
# against the actual cron so the two cannot drift.
BOARD_RUN_INTERVAL_HOURS = 6.0
# Measured over the live schedule: nominal 6.0h, largest observed gap 7.0h,
# lateness up to +42 min, and on the previous hourly cron an entire run was
# dropped. The retry deadline allows for one whole run being lost plus that
# lateness, because "before the game" has to mean every time, not usually.
BOARD_RUN_SLACK_HOURS = 1.0


def retry_by_hours(interval_hours: float = BOARD_RUN_INTERVAL_HOURS,
                   slack_hours: float = BOARD_RUN_SLACK_HOURS,
                   min_hours: float = BOARD_MIN_HOURS) -> float:
    """Latest lead time at which a retry can still be GUARANTEED to happen.

    A retry lands if any run falls between `min_hours` and this value, so the
    span has to be wider than the worst gap between runs. Two intervals covers
    one dropped run; the slack covers the scheduler running late.
    """
    return min_hours + 2 * interval_hours + slack_hours

# Lower rank wins. Ordered by validated model quality and book depth: 1X2 is
# the family the evaluation actually scores and the deepest book on both
# venues; exact scorelines sit last because the 13x13 grid is truncated and
# their books are thinnest.
CLAIM_PRIORITY = (
    ("home_win", "away_win", "draw"),
    ("btts",),
    ("total_over_", "total_under_"),
    ("home_over_", "away_over_"),
    ("home_wins_by_over_", "away_wins_by_over_"),
    ("score_",),
)


def base_claim(claim: str) -> str:
    """A negated claim belongs to the same family as the claim it negates."""
    text = str(claim or "")
    return text[4:] if text.startswith("not_") else text


def claim_rank(claim: str) -> int:
    """Family rank, lower first.

    A matcher ending in "_" is a PREFIX (the claim carries a line); anything
    else must match exactly. The distinction is not cosmetic: a plain
    startswith test classifies `home_wins_by_over_1.5` -- a spread -- as
    `home_win`, ranking the thinnest family as the best-validated one.
    """
    base = base_claim(claim)
    for rank, family in enumerate(CLAIM_PRIORITY):
        for matcher in family:
            hit = (base.startswith(matcher) if matcher.endswith("_")
                   else base == matcher)
            if hit:
                return rank
    return len(CLAIM_PRIORITY)


def fixture_key(league_id, home, away, kickoff_utc) -> str:
    """One match, however many markets or venues quote it.

    Keyed on the DATE rather than the exact timestamp: the two venues publish
    kick-off from different sources and can disagree by minutes, and a fixture
    that is the same match must not become two observations because of it.
    """
    day = str(kickoff_utc or "")[:10]
    return "%s|%s|%s|%s" % (league_id, home, away, day)


def board_state(record, kickoff_utc, now=None) -> tuple:
    """(should_board, reason) for one fixture, given what happened before.

    `record` is the boarded-ledger entry, or None if this fixture has never
    been seen.
    """
    window = board_window_state(kickoff_utc, now)
    if record is None:
        return (window == "board"), ("first pass" if window == "board"
                                     else "outside the board window (%s)" % window)

    action = str(record.get("action") or "")
    attempts = int(record.get("attempts") or 1)
    if action != "DEFER":
        return False, "fixture already decided (%s)" % (action or "unknown")
    if attempts >= RETRY_MAX_ATTEMPTS:
        return False, "deferral already retried"

    hours = hours_to_kickoff(kickoff_utc, now)
    if hours is None:
        return False, "deferred, but kick-off is unknown"
    if hours < BOARD_MIN_HOURS:
        return False, "deferred, too late to retry"
    if hours > retry_by_hours():
        # Waiting deliberately: a retry is worth more the closer it is to
        # kick-off, when the team news the coach asked for exists.
        return False, "deferred, holding the retry for nearer kick-off"
    return True, "retrying a deferral"


def select_one_per_fixture(candidates, already_boarded=None, now=None,
                           use_lead_time: bool = True):
    """One candidate per fixture, plus the ones deliberately left out.

    Returns (selected, skipped) where `skipped` carries a reason per dropped
    candidate, because a silent cap reads as "we looked at everything".
    """
    # Accepts the boarded LEDGER (key -> record). A bare set of keys is still
    # honoured -- an entry with no recorded action reads as "already decided",
    # which is the old behaviour.
    raw = already_boarded or {}
    ledger = dict(raw) if isinstance(raw, dict) else {key: {} for key in raw}
    best, skipped = {}, []
    for cand in candidates:
        key = cand.fixture_key
        if use_lead_time:
            ok, reason = board_state(ledger.get(key), cand.leg.kickoff_utc, now)
            if not ok:
                skipped.append((cand, reason))
                continue
        elif key in ledger:
            skipped.append((cand, "fixture already boarded"))
            continue
        rank = claim_rank(cand.claim)
        ev = cand.case.ev_per_contract or 0.0
        # Family first, then EV inside the family, then a stable tie-break so
        # two runs over identical data pick the same market.
        score = (rank, -ev, cand.instrument.venue, cand.instrument.instrument_id)
        current = best.get(key)
        if current is None or score < current[0]:
            if current is not None:
                skipped.append((current[1], "another market on this fixture "
                                            "ranked higher"))
            best[key] = (score, cand)
        else:
            skipped.append((cand, "another market on this fixture ranked higher"))
    return [pair[1] for pair in best.values()], skipped


def hours_to_kickoff(kickoff_utc, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    text = str(kickoff_utc or "").strip().replace(" ", "T").replace("Z", "+00:00")
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return (stamp - now).total_seconds() / 3600.0


def board_window_state(kickoff_utc, now=None,
                       target_hours: float = BOARD_TARGET_HOURS,
                       window_hours: float = BOARD_WINDOW_HOURS,
                       min_hours: float = BOARD_MIN_HOURS) -> str:
    """"board", "too_early", "too_late", or "unknown_kickoff".

    Deliberately generous at the near end: a fixture the scheduler missed --
    an outage, a late-added market -- is still boarded on the first run that
    sees it, rather than being dropped and costing an observation. The
    once-only guarantee comes from the boarded ledger, not from this window.
    """
    hours = hours_to_kickoff(kickoff_utc, now)
    if hours is None:
        return "unknown_kickoff"
    if hours > target_hours + window_hours:
        return "too_early"
    if hours < min_hours:
        return "too_late"
    return "board"
