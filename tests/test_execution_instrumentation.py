"""What the book records about HOW a trade happened, not just that it did.

Closing line value is `close - fill`, and that single number is the sum of
three different things: the model's forecast error, what resting earned or
cost, and how the market drifted after the decision. They have different
fixes, and until these fields existed the book could not tell them apart.

The live evidence that made this urgent: 1,702 orders placed, every one of
them resting, a 10.6% fill rate, and 74% of those fills closing BELOW what
they paid. A resting order fills because the market came to it, and the market
comes to a bid when the news is bad -- so the filled set is not a sample of
our opinions, it is a sample of the times the market disagreed with us.
"""
import datetime as dt

import pytest

from wc2026.paper.broker import PaperPortfolio
from wc2026.paper.clv import (
    capture_expired_closing_lines,
    counterfactual_summary,
)
from wc2026.paper.fills import replay_fills, synthetic_book

UTC = dt.timezone.utc
KICKOFF = dt.datetime(2026, 9, 14, 15, 0, tzinfo=UTC)
AFTER = KICKOFF + dt.timedelta(hours=3)


def a_book(cash=100_000):
    return PaperPortfolio(starting_cash_cents=cash, cash_cents=cash, path=None)


def an_order(book, price=30, size=10.0, **kw):
    order = book.submit("c1", "kalshi", "INST", "yes", price, size,
                        expires_at=KICKOFF.isoformat(), claim="draw",
                        home_team="A", away_team="B",
                        kickoff_utc=KICKOFF.isoformat(), **kw)
    order.created_at = (KICKOFF - dt.timedelta(hours=10)).isoformat()
    return order


class Tape:
    """A venue whose tape prints one price, at one stated moment."""

    def __init__(self, price, when=None, close=None):
        self.price, self.when, self.close = price, when, close

    def best_executable(self, order, since, until):
        return self.price, self.when

    def best_executable_cents(self, order, since, until):
        return self.price

    def closing_price_cents(self, venue_id, side, kickoff):
        return self.close


class OldProbe:
    """A probe that predates `best_executable` -- or any test double."""

    def __init__(self, price):
        self.price = price

    def best_executable_cents(self, order, since, until):
        return self.price

    def closing_price_cents(self, venue_id, side, kickoff):
        return None


# ── the decision is recorded, not just the order ──────────────────────────────
def test_the_model_probability_and_the_screen_price_reach_the_order():
    book = a_book()
    order = an_order(book, p_model=0.62, decision_price_cents=41)
    assert order.p_model == pytest.approx(0.62)
    assert order.decision_price_cents == 41


def test_they_carry_through_to_the_position_that_the_fill_creates():
    """The position is what settles and what CLV is computed on, so the
    decomposition has to survive the fill."""
    book = a_book()
    order = an_order(book, p_model=0.62, decision_price_cents=41)
    book.try_fill_resting(order.order_id,
                          synthetic_book("yes", order.limit_price_cents, 10.0),
                          require_trade_through=False)
    pos = next(iter(book.positions.values()))
    assert pos.p_model == pytest.approx(0.62)
    assert pos.decision_price_cents == 41


def test_an_order_placed_without_them_is_not_invented_for():
    book = a_book()
    order = an_order(book)
    assert order.p_model is None and order.decision_price_cents is None


# ── when the fill happened ────────────────────────────────────────────────────
def test_the_fill_is_stamped_with_the_tapes_moment_not_the_cron_s():
    """`opened_at` is written when the replay runs, so a three-hourly schedule
    rounds every fill to itself and a rebuild rewrites them all at once."""
    book = a_book()
    an_order(book, price=30)
    printed = (KICKOFF - dt.timedelta(hours=4)).isoformat()
    out = replay_fills(book, {"kalshi": Tape(29.0, when=printed)}, now=AFTER)
    assert out["filled"] == 1
    order = next(iter(book.orders.values()))
    assert order.filled_at == printed
    assert next(iter(book.positions.values())).filled_at == printed


def test_a_probe_with_no_moment_falls_back_to_the_wall_clock():
    """Honest rather than absent: the fill did happen, we just cannot say
    when more precisely than now."""
    book = a_book()
    an_order(book, price=30)
    replay_fills(book, {"kalshi": Tape(29.0, when=None)}, now=AFTER)
    order = next(iter(book.orders.values()))
    assert order.filled_at is not None


def test_a_probe_that_predates_the_method_still_fills():
    """The probes are duck-typed and several test doubles are plain objects."""
    book = a_book()
    an_order(book, price=30)
    out = replay_fills(book, {"kalshi": OldProbe(29.0)}, now=AFTER)
    assert out["filled"] == 1


def test_the_first_fill_wins_the_stamp():
    """A partial fill followed by another must not move the moment we first
    traded to the moment we last did."""
    book = a_book()
    order = an_order(book, price=30, size=10.0)
    first = (KICKOFF - dt.timedelta(hours=5)).isoformat()
    book.try_fill_resting(order.order_id,
                          synthetic_book("yes", 30, 4.0),
                          require_trade_through=False, filled_at=first)
    book.try_fill_resting(order.order_id,
                          synthetic_book("yes", 30, 6.0),
                          require_trade_through=False,
                          filled_at=KICKOFF.isoformat())
    assert order.filled_at == first


# ── the orders that never filled ──────────────────────────────────────────────
def test_an_expired_order_is_scored_against_the_close():
    """1,521 of 1,701 resolved orders expired carrying no information. They
    cannot be adversely selected, because nothing selected them."""
    book = a_book()
    an_order(book, price=30)
    book.expire_due(now=AFTER)
    out = capture_expired_closing_lines(book, {"kalshi": Tape(None, close=34.0)},
                                        now=AFTER)
    assert out["captured"] == 1
    order = next(iter(book.orders.values()))
    assert order.closing_price_cents == 34.0
    # the close finished above the limit, so the opinion was on the right side
    assert order.counterfactual_clv_cents == pytest.approx(4.0)


def test_a_limit_on_the_wrong_side_of_the_close_scores_negative():
    book = a_book()
    an_order(book, price=30)
    book.expire_due(now=AFTER)
    capture_expired_closing_lines(book, {"kalshi": Tape(None, close=26.5)},
                                  now=AFTER)
    order = next(iter(book.orders.values()))
    assert order.counterfactual_clv_cents == pytest.approx(-3.5)


def test_a_filled_order_is_not_scored_here():
    """It has a position, and the position carries the real reading. Scoring
    it twice would double-count it."""
    book = a_book()
    order = an_order(book, price=30)
    book.try_fill_resting(order.order_id,
                          synthetic_book("yes", 30, 10.0),
                          require_trade_through=False)
    out = capture_expired_closing_lines(book, {"kalshi": Tape(None, close=34.0)},
                                        now=AFTER)
    assert out["captured"] == 0
    assert order.counterfactual_clv_cents is None


def test_a_match_that_has_not_kicked_off_is_left_alone():
    book = a_book()
    an_order(book, price=30)
    book.expire_due(now=AFTER)
    out = capture_expired_closing_lines(
        book, {"kalshi": Tape(None, close=34.0)},
        now=KICKOFF - dt.timedelta(hours=1))
    assert out["captured"] == 0 and out["not_kicked_off"] == 1


def test_no_closing_history_is_counted_rather_than_guessed():
    book = a_book()
    an_order(book, price=30)
    book.expire_due(now=AFTER)
    out = capture_expired_closing_lines(book, {"kalshi": Tape(None, close=None)},
                                        now=AFTER)
    assert out["captured"] == 0 and out["no_history"] == 1
    assert next(iter(book.orders.values())).counterfactual_clv_cents is None


def test_capture_is_idempotent_across_cycles():
    book = a_book()
    an_order(book, price=30)
    book.expire_due(now=AFTER)
    probes = {"kalshi": Tape(None, close=34.0)}
    capture_expired_closing_lines(book, probes, now=AFTER)
    again = capture_expired_closing_lines(book, probes, now=AFTER)
    assert again["captured"] == 0 and again["already_had"] == 1


# ── the summary ───────────────────────────────────────────────────────────────
def test_the_counterfactual_is_averaged_over_fixtures_not_orders():
    """A hundred limits on one match are one opinion written a hundred ways --
    the same rule every other rate in this package follows."""
    book = a_book()
    for i in range(4):
        order = book.submit("c%d" % i, "kalshi", "INST%d" % i, "yes", 30, 5.0,
                            expires_at=KICKOFF.isoformat(), claim="draw",
                            home_team="A", away_team="B", league_id="mls",
                            kickoff_utc=KICKOFF.isoformat())
        order.created_at = (KICKOFF - dt.timedelta(hours=10)).isoformat()
    other = book.submit("z", "kalshi", "OTHER", "yes", 30, 5.0,
                        expires_at=KICKOFF.isoformat(), claim="draw",
                        home_team="C", away_team="D", league_id="mls",
                        kickoff_utc=KICKOFF.isoformat())
    other.created_at = (KICKOFF - dt.timedelta(hours=10)).isoformat()
    book.expire_due(now=AFTER)
    capture_expired_closing_lines(book, {"kalshi": Tape(None, close=34.0)},
                                  now=AFTER)
    out = counterfactual_summary(book)
    assert out["n_orders"] == 5
    assert out["n_fixtures"] == 2, "four markets on one match are one fixture"
    assert out["mean_cents"] == pytest.approx(4.0)
    assert out["beat"] == 2 and out["beat_rate"] == pytest.approx(1.0)


def test_an_empty_book_reports_absent_rather_than_zero():
    out = counterfactual_summary(a_book())
    assert out["mean_cents"] is None and out["n_fixtures"] == 0


def test_a_voided_order_is_not_counted():
    book = a_book()
    an_order(book, price=30)
    book.expire_due(now=AFTER)
    capture_expired_closing_lines(book, {"kalshi": Tape(None, close=34.0)},
                                  now=AFTER)
    next(iter(book.orders.values())).void = True
    assert counterfactual_summary(book)["mean_cents"] is None
