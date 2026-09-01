"""The paper loop must be able to reach a non-zero P&L.

Before this wiring it could not, at any cadence: `try_fill_resting` and
`settle` had no production caller, so a limit order rested until it expired and
`realized_pnl_usd` was structurally 0.00. The suite was green throughout,
because it exercised the broker's methods directly instead of through the
cycle. These tests go through the cycle.
"""
import datetime as dt

import pandas as pd
import pytest

from wc2026.paper.broker import PaperPortfolio
from wc2026.paper.fills import (
    POLY_MID_TO_ASK_CENTS,
    KalshiFillProbe,
    PolymarketFillProbe,
    replay_fills,
    synthetic_book,
)
from wc2026.paper.settlement import find_fixture, settle_portfolio


def a_portfolio(tmp_path, cash=100_000):
    return PaperPortfolio(starting_cash_cents=cash, cash_cents=cash,
                          path=str(tmp_path / "portfolio.json"))


def an_order(portfolio, price=30, size=10.0, claim="home_win", venue="kalshi",
             kickoff="2026-08-20T19:00:00+00:00", case="c1",
             created="2026-08-19T18:00:00+00:00"):
    order = portfolio.submit(case, venue, "INST-1", "yes", price, size,
                             league_id="mls", claim=claim, home_team="A",
                             away_team="B", kickoff_utc=kickoff,
                             expires_at="2026-08-20T18:00:00+00:00")
    # Placed while it was LIVE. `submit` stamps the wall clock, which in a test
    # sits long after the fixture, so the replay window would be empty --
    # the tape is now read from creation to expiry, not to whenever the cron
    # happens to run.
    order.created_at = created
    return order


class StubProbe:
    """Stands in for the venue tape."""

    def __init__(self, best):
        self.best = best
        self.windows = []

    def best_executable_cents(self, order, since, until):
        self.windows.append((since, until))
        return self.best


def played(home=2, away=1, **kw):
    row = {"date": pd.Timestamp("2026-08-20"), "home_team": "A",
           "away_team": "B", "home_score": float(home), "away_score": float(away),
           "played": True, "status": "STATUS_FULL_TIME",
           "went_to_extra_time": False}
    row.update(kw)
    return pd.DataFrame([row])


# --- resting fills ------------------------------------------------------------
def test_a_resting_order_fills_when_the_tape_traded_through(tmp_path):
    p = a_portfolio(tmp_path)
    order = an_order(p, price=30)
    stats = replay_fills(p, {"kalshi": StubProbe(29.0)})
    assert stats["filled"] == 1
    assert order.filled_size == 10.0


def test_a_passive_fill_pays_our_limit_not_the_price_the_tape_printed(tmp_path):
    """We are the resting side; the aggressor takes the price improvement."""
    p = a_portfolio(tmp_path)
    order = an_order(p, price=30)
    replay_fills(p, {"kalshi": StubProbe(22.0)})     # tape went far through
    assert order.avg_fill_price_cents == 30


def test_touching_the_limit_is_not_a_fill(tmp_path):
    """At our own price we cannot know our place in the queue."""
    p = a_portfolio(tmp_path)
    order = an_order(p, price=30)
    stats = replay_fills(p, {"kalshi": StubProbe(30.0)})
    assert stats["not_through"] == 1 and stats["filled"] == 0
    assert order.filled_size == 0.0


def test_a_history_fill_is_labelled_so_it_can_be_told_from_a_real_book(tmp_path):
    p = a_portfolio(tmp_path)
    order = an_order(p, price=30)
    replay_fills(p, {"kalshi": StubProbe(28.0)})
    bases = [e for e in order.events if e["event"] == "fill_basis"]
    assert bases and bases[0]["basis"] == "history"


def test_the_replay_window_starts_where_the_last_one_ended(tmp_path):
    """No gap between runs, and no window scanned twice.

    The clock is pinned rather than read. An earlier version of this test
    compared watermarks taken from two real `now()` calls and passed only
    because Windows returns an IDENTICAL timestamp for consecutive calls;
    on CI's finer clock the two differed by microseconds and it failed.
    """
    first = dt.datetime(2026, 8, 20, 6, 0, tzinfo=dt.timezone.utc)
    second = dt.datetime(2026, 8, 20, 12, 0, tzinfo=dt.timezone.utc)
    p = a_portfolio(tmp_path)
    order = an_order(p, price=30)
    probe = StubProbe(None)

    replay_fills(p, {"kalshi": probe}, now=first)
    assert probe.windows[0][0] == order.created_at    # from the order's birth
    assert probe.windows[0][1] == first
    assert order.last_checked_at == first.isoformat()

    replay_fills(p, {"kalshi": probe}, now=second)
    # Starts exactly where the last one stopped: no gap, no overlap.
    assert probe.windows[1][0] == first.isoformat()
    assert probe.windows[1][1] == second
    assert order.last_checked_at == second.isoformat()



def test_no_probe_for_a_venue_is_counted_not_silently_skipped(tmp_path):
    p = a_portfolio(tmp_path)
    an_order(p, venue="polymarket")
    assert replay_fills(p, {"kalshi": StubProbe(1.0)})["no_probe"] == 1


def test_polymarket_is_held_to_a_stricter_bar_than_kalshi():
    """Its history publishes the mid, so the ask is bounded, not observed."""
    assert POLY_MID_TO_ASK_CENTS > 0


def test_synthetic_book_puts_depth_on_the_side_being_bought():
    yes = synthetic_book("yes", 28, 5)
    assert yes.touch("yes") == 28 and yes.touch("no") is None
    no = synthetic_book("no", 28, 5)
    assert no.touch("no") == 28 and no.touch("yes") is None


# --- settlement ---------------------------------------------------------------
def filled_position(tmp_path, claim="home_win", price=30, size=10.0):
    p = a_portfolio(tmp_path)
    order = an_order(p, price=price, size=size, claim=claim)
    replay_fills(p, {"kalshi": StubProbe(price - 2)})
    assert order.filled_size == size
    return p


def test_a_won_bet_pays_out_and_lands_in_the_ledger(tmp_path):
    p = filled_position(tmp_path, claim="home_win")
    stats = settle_portfolio(p, {"mls": played(2, 1)})
    assert stats["settled"] == 1
    assert stats["pnl_cents"] > 0
    assert p.summary()["realized_pnl_usd"] > 0
    assert len(p.ledger) == 1 and p.ledger[0]["won"] is True


def test_a_lost_bet_costs_exactly_what_was_paid(tmp_path):
    p = filled_position(tmp_path, claim="away_win")
    stats = settle_portfolio(p, {"mls": played(2, 1)})
    assert stats["settled"] == 1 and stats["pnl_cents"] < 0
    assert p.summary()["realized_pnl_usd"] < 0


def test_settling_twice_does_not_double_count(tmp_path):
    p = filled_position(tmp_path)
    first = settle_portfolio(p, {"mls": played(2, 1)})
    second = settle_portfolio(p, {"mls": played(2, 1)})
    assert first["settled"] == 1 and second["settled"] == 0
    assert len(p.ledger) == 1


def test_an_unfinished_match_leaves_the_position_open(tmp_path):
    p = filled_position(tmp_path)
    frame = played()
    frame.loc[0, "played"] = False
    stats = settle_portfolio(p, {"mls": frame})
    assert stats["settled"] == 0 and stats["no_result_yet"] == 1
    assert stats["still_open"] == 1


def test_extra_time_blocks_settlement_because_markets_settle_on_regulation(tmp_path):
    p = filled_position(tmp_path)
    stats = settle_portfolio(p, {"mls": played(went_to_extra_time=True)})
    assert stats["settled"] == 0 and stats["no_result_yet"] == 1


def test_an_unknown_fixture_is_counted_rather_than_resolved_to_a_loss(tmp_path):
    p = filled_position(tmp_path)
    stats = settle_portfolio(p, {"mls": played(home_team="Z")})
    assert stats["settled"] == 0 and stats["fixture_not_found"] == 1


def test_a_position_with_no_settlement_identity_is_reported(tmp_path):
    """State written before positions carried claim/teams must not vanish."""
    p = a_portfolio(tmp_path)
    order = p.submit("c9", "kalshi", "INST-9", "yes", 30, 5.0, league_id="mls")
    replay_fills(p, {"kalshi": StubProbe(28.0)})
    assert order.filled_size == 5.0
    stats = settle_portfolio(p, {"mls": played()})
    assert stats["missing_identity"] == 1 and stats["settled"] == 0


def test_an_unsettleable_claim_is_surfaced_with_the_reason(tmp_path):
    p = filled_position(tmp_path, claim="total_corners_over_9.5")
    stats = settle_portfolio(p, {"mls": played()})
    assert stats["unsettleable_claim"] == 1 and stats["problems"]


def test_a_postponed_match_still_settles_within_the_lookup_window(tmp_path):
    p = filled_position(tmp_path)
    late = played()
    late.loc[0, "date"] = pd.Timestamp("2026-08-22")
    assert settle_portfolio(p, {"mls": late})["settled"] == 1


def test_two_meetings_inside_the_window_are_refused_rather_than_guessed():
    frame = pd.concat([played(), played()], ignore_index=True)
    frame.loc[1, "date"] = pd.Timestamp("2026-08-21")
    assert find_fixture(frame, "A", "B", "2026-08-20T19:00:00+00:00") is None


# --- the number the whole exercise exists to produce --------------------------
def test_submit_then_fill_then_settle_produces_a_real_pnl(tmp_path):
    p = a_portfolio(tmp_path, cash=100_000)
    an_order(p, price=30, size=10.0, claim="home_win")
    assert p.summary()["realized_pnl_usd"] == 0.0        # nothing resolved yet

    replay_fills(p, {"kalshi": StubProbe(29.0)})
    assert p.summary()["n_positions_open"] == 1

    settle_portfolio(p, {"mls": played(2, 1)})           # home_win pays
    out = p.summary()
    assert out["n_settled"] == 1
    assert out["realized_pnl_usd"] == pytest.approx(
        (100.0 * 10 - 30 * 10 - p.ledger[0]["fees_cents"]) / 100.0, abs=0.01),         "a passive fill pays our limit, so cost is exactly 30c x 10 + fee"
    assert out["realized_pnl_usd"] > 0
    assert out["fill_rate"] == 1.0


# --- the probes' own parsing --------------------------------------------------
class FakeKalshiClient:
    def __init__(self, candles):
        self.candles = candles

    def get_candlesticks(self, ticker, start, end, period_interval=1):
        return self.candles


class Order:
    def __init__(self, side="yes", instrument_id="T"):
        self.side, self.instrument_id = side, instrument_id


def candle(ask_low, bid_high):
    return {"yes_ask": {"low_dollars": "%.4f" % ask_low},
            "yes_bid": {"high_dollars": "%.4f" % bid_high}}


def test_kalshi_probe_reads_the_ask_low_for_a_yes_order():
    probe = KalshiFillProbe(FakeKalshiClient([candle(0.28, 0.27),
                                              candle(0.31, 0.30)]))
    got = probe.best_executable_cents(Order("yes"), "2026-08-20T00:00:00Z",
                                      "2026-08-20T06:00:00Z")
    assert got == pytest.approx(28.0)


def test_kalshi_probe_reads_the_bid_high_for_a_no_order():
    """Buying NO is selling YES: a NO buy is reached when the YES bid RISES."""
    probe = KalshiFillProbe(FakeKalshiClient([candle(0.28, 0.27),
                                              candle(0.80, 0.75)]))
    got = probe.best_executable_cents(Order("no"), "2026-08-20T00:00:00Z",
                                      "2026-08-20T06:00:00Z")
    assert got == pytest.approx(25.0)          # 100 - 75


def test_a_backwards_or_empty_window_probes_nothing():
    probe = KalshiFillProbe(FakeKalshiClient([candle(0.10, 0.09)]))
    assert probe.best_executable_cents(
        Order(), "2026-08-20T06:00:00Z", "2026-08-20T00:00:00Z") is None
    assert probe.best_executable_cents(Order(), None, "2026-08-20T00:00:00Z") is None


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, history):
        self.history = history

    def get(self, url, params=None, timeout=None):
        if "/markets/" in url:
            return FakeResponse({"tokens": [{"token_id": "YES"},
                                            {"token_id": "NO"}]})
        return FakeResponse({"history": self.history})


def test_polymarket_probe_converts_the_published_mid_to_a_bounded_ask():
    probe = PolymarketFillProbe(FakeSession([{"t": 1, "p": 0.28},
                                             {"t": 2, "p": 0.31}]))
    got = probe.best_executable_cents(Order(), "2026-08-20T00:00:00Z",
                                      "2026-08-20T06:00:00Z")
    assert got == pytest.approx(28.0 + POLY_MID_TO_ASK_CENTS)


def test_polymarket_probe_picks_the_token_matching_the_side():
    probe = PolymarketFillProbe(FakeSession([{"t": 1, "p": 0.5}]))
    assert probe.token_for("cond", "yes") == "YES"
    assert probe.token_for("cond", "no") == "NO"


def test_settlement_reports_zero_cleanly_on_an_empty_portfolio(tmp_path):
    stats = settle_portfolio(a_portfolio(tmp_path), {})
    assert stats["settled"] == 0 and stats["pnl_cents"] == 0.0


def test_replay_reports_zero_cleanly_with_no_orders(tmp_path):
    assert replay_fills(a_portfolio(tmp_path), {})["checked"] == 0


def test_a_terminal_order_is_never_refilled(tmp_path):
    p = a_portfolio(tmp_path)
    order = an_order(p, price=30)
    p.cancel(order.order_id)
    assert replay_fills(p, {"kalshi": StubProbe(1.0)})["checked"] == 0
    assert order.filled_size == 0.0


def test_settlement_identity_survives_a_save_and_reload(tmp_path):
    p = filled_position(tmp_path)
    path = p.save()
    reloaded = PaperPortfolio.load(path)
    pos = next(iter(reloaded.positions.values()))
    assert (pos.claim, pos.home_team, pos.away_team) == ("home_win", "A", "B")
    assert settle_portfolio(reloaded, {"mls": played(2, 1)})["settled"] == 1


def test_expiry_after_a_fill_does_not_erase_the_position(tmp_path):
    """An order that traded through then expired was filled, not expired."""
    p = a_portfolio(tmp_path)
    order = an_order(p, price=30, size=10.0)
    replay_fills(p, {"kalshi": StubProbe(29.0)})
    p.expire_due(now=dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc))
    assert order.filled_size == 10.0
    assert p.summary()["n_positions_open"] == 1
