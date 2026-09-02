"""Three defects that each made the book report money it never made.

Every one of them was invisible to the suite and visible only in the record:
a fill rate that was too good, a `no` book that lost more often than it paid
for, and a stake nobody computed. They are grouped here because they share a
property -- each produced a perfectly well-formed number that was wrong, and
nothing downstream could have caught it.
"""
import datetime as dt

import pytest

from wc2026.paper.broker import PaperPortfolio
from wc2026.paper.fills import replay_fills
from wc2026.paper.outcomes import market_claim, winning_side

UTC = dt.timezone.utc
KICKOFF = dt.datetime(2026, 8, 30, 15, 0, tzinfo=UTC)


# ── 1. the fill window ────────────────────────────────────────────────────────
class Tape:
    """A venue whose tape is cheap on a stated interval and dear before it."""

    def __init__(self, cheap_from, cheap_cents=5.0, dear_cents=80.0):
        self.cheap_from, self.cheap, self.dear = cheap_from, cheap_cents, dear_cents
        self.windows = []

    def best_executable_cents(self, order, since, until):
        self.windows.append((since, until))
        end = until if isinstance(until, dt.datetime) else None
        # The cheap price only ever existed from `cheap_from` onwards.
        if end is not None and end > self.cheap_from:
            return self.cheap
        return self.dear

    def closing_price_cents(self, venue_id, side, kickoff):
        return None


def a_book(limit=20, expires=KICKOFF, placed_hours_before=14):
    book = PaperPortfolio(starting_cash_cents=100_000, cash_cents=100_000,
                          path=None)
    order = book.submit("c1", "kalshi", "INST", "yes", limit, 10.0,
                        expires_at=expires.isoformat(), claim="draw",
                        home_team="A", away_team="B",
                        kickoff_utc=expires.isoformat())
    # Placed while it was live. `submit` stamps the wall clock, which in a test
    # sits after the fixture and would make every window empty.
    order.created_at = (expires
                        - dt.timedelta(hours=placed_hours_before)).isoformat()
    return book


def test_a_dead_order_cannot_fill_on_a_price_from_after_the_whistle():
    """A resting bid sits BELOW the pre-match price, so after kick-off the
    market only reaches it once the match has turned against that outcome. The
    fill would be conditioned on the result: filled precisely when wrong."""
    book = a_book()
    # Cheap only after kick-off, and the cycle runs two hours late.
    probe = Tape(cheap_from=KICKOFF)
    out = replay_fills(book, {"kalshi": probe},
                       now=KICKOFF + dt.timedelta(hours=2))
    assert out["filled"] == 0, "the order was dead at kick-off"
    _, until = probe.windows[0]
    assert until == KICKOFF, "the window ends when the ORDER does"


def test_an_order_still_fills_on_a_price_from_while_it_was_live():
    book = a_book()
    probe = Tape(cheap_from=KICKOFF - dt.timedelta(hours=6))
    out = replay_fills(book, {"kalshi": probe},
                       now=KICKOFF + dt.timedelta(hours=2))
    assert out["filled"] == 1


def test_the_window_does_not_grow_with_a_late_cron():
    """Replaying the tape exists so the answer does not depend on the
    schedule. A window ending at `now` gave a later run more tape."""
    seen = []
    for late in (1, 5, 24):
        book = a_book()
        probe = Tape(cheap_from=KICKOFF)
        replay_fills(book, {"kalshi": probe},
                     now=KICKOFF + dt.timedelta(hours=late))
        seen.append(probe.windows[0][1])
    assert len(set(seen)) == 1 and seen[0] == KICKOFF


def test_an_order_with_no_expiry_is_still_bounded_by_now():
    book = PaperPortfolio(starting_cash_cents=100_000, cash_cents=100_000,
                          path=None)
    order = book.submit("c1", "kalshi", "INST", "yes", 20, 10.0, claim="draw",
                        home_team="A", away_team="B")
    order.created_at = (KICKOFF - dt.timedelta(days=1)).isoformat()
    probe = Tape(cheap_from=KICKOFF - dt.timedelta(days=9))
    replay_fills(book, {"kalshi": probe}, now=KICKOFF)
    assert probe.windows[0][1] == KICKOFF


# ── 2. the settlement inversion ───────────────────────────────────────────────
# Ground truth, verified against the fixture table: Real Madrid 4-0 Málaga.
RM, MAL = 4, 0


@pytest.mark.parametrize("claim,side,should_win", [
    # We paid ~92c for "the score is not 0-1". It finished 4-0, so it is not
    # 0-1, and the position won. It was booked as a loss.
    ("not_score_0-1", "no", True),
    ("not_score_2-2", "no", True),
    ("not_score_4-0", "no", False),      # it WAS 4-0, so this one loses
    ("score_4-0", "yes", True),
    ("score_0-1", "yes", False),
    ("not_draw", "no", True),            # 4-0 is not a draw
    ("draw", "yes", False),
])
def test_a_position_settles_on_the_proposition_it_pays_on(claim, side,
                                                          should_win):
    """`cycle` builds the no side as `not_ + leg.claim` and prices it with
    `probability_for(claim)`, so the claim ALREADY carries the side. Comparing
    the result to the side then negated it a second time."""
    assert (winning_side(claim, RM, MAL) == side) is should_win


def test_the_market_question_is_recovered_from_the_positions_claim():
    assert market_claim("not_score_0-1") == "score_0-1"
    assert market_claim("score_0-1") == "score_0-1"
    assert market_claim("not_1h_draw") == "1h_draw"


def test_exactly_one_side_of_a_market_wins():
    """Under the bug both sides of a market could lose, because each was
    scored against a differently-negated proposition. On the live book the
    `no` side paid an average 64c and won 47% of the time, while the `yes`
    side paid 25.5c and won 26.5% -- the same engine, one side calibrated."""
    for claim in ("score_1-0", "draw", "total_over_2.5", "home_win"):
        yes_wins = winning_side(claim, RM, MAL) == "yes"
        no_wins = winning_side("not_" + claim, RM, MAL) == "no"
        assert yes_wins != no_wins, claim


# ── 3. deterministic sizing ───────────────────────────────────────────────────
from wc2026.decision.calculator import (  # noqa: E402
    CalcConfig,
    deterministic_size,
)


class Case:
    def __init__(self, p, price):
        self.p_lower, self.max_limit_price_cents = p, price


def test_no_edge_buys_nothing():
    """A stake nobody computed is a stake nobody can check. At the price the
    model agrees with, the correct size is zero."""
    assert deterministic_size(Case(0.50, 50), CalcConfig()) == 0
    assert deterministic_size(Case(0.40, 50), CalcConfig()) == 0


def test_more_edge_buys_more_until_the_cap():
    cfg = CalcConfig()
    small = deterministic_size(Case(0.52, 50), cfg)
    big = deterministic_size(Case(0.75, 50), cfg)
    assert 0 < small < big


def test_no_single_market_can_swallow_the_fixture_budget():
    """One 95%-at-90c contract would otherwise size to a quarter of the book."""
    cfg = CalcConfig()
    n = deterministic_size(Case(0.99, 90), cfg)
    staked = n * 90 / 100.0
    assert staked <= cfg.max_stake_fraction_per_market * cfg.bankroll_cents / 100.0


def test_size_is_the_same_every_time_for_the_same_inputs():
    """The whole point: 333 positions of one contract and a handful of 20, 50,
    60 and 100, with no rule behind the split, is not a measurement."""
    cfg = CalcConfig()
    got = {deterministic_size(Case(0.60, 40), cfg) for _ in range(5)}
    assert len(got) == 1 and got != {0}


def test_a_degenerate_price_sizes_to_nothing_rather_than_dividing_by_zero():
    cfg = CalcConfig()
    for price in (0, 100, None):
        assert deterministic_size(Case(0.9, price), cfg) == 0
    assert deterministic_size(Case(None, 50), cfg) == 0
