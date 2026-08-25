"""Closing line value, and the sample-size honesty around it."""
import datetime as dt

import pytest

from wc2026.paper.broker import PaperPortfolio
from wc2026.paper.clv import capture_closing_lines, clv_summary
from wc2026.paper.fills import KalshiFillProbe, PolymarketFillProbe


def a_portfolio(tmp_path):
    return PaperPortfolio(starting_cash_cents=100_000, cash_cents=100_000,
                          path=str(tmp_path / "p.json"))


PAST = "2026-08-20T19:00:00+00:00"
FUTURE = (dt.datetime.now(dt.timezone.utc)
          + dt.timedelta(days=5)).isoformat()


def a_position(portfolio, instrument="I1", paid=30.0, kickoff=PAST,
               home="A", away="B", venue="kalshi", side="yes"):
    from wc2026.paper.broker import PaperPosition
    pos = PaperPosition(position_id=instrument, venue=venue,
                        instrument_id=instrument, side=side, size=10.0,
                        avg_cost_cents=paid, fees_cents=0, league_id="mls",
                        home_team=home, away_team=away, kickoff_utc=kickoff)
    portfolio.positions["%s|%s" % (instrument, side)] = pos
    return pos


class StubProbe:
    venue = "kalshi"

    def __init__(self, price):
        self.price = price
        self.asked = []

    def closing_price_cents(self, venue_id, side, kickoff):
        self.asked.append((venue_id, side, kickoff))
        return self.price


def test_clv_is_positive_when_the_market_moved_toward_us(tmp_path):
    p = a_portfolio(tmp_path)
    pos = a_position(p, paid=30.0)
    capture_closing_lines(p, {"kalshi": StubProbe(35.0)})
    assert pos.closing_price_cents == 35.0
    assert pos.clv_cents == pytest.approx(5.0)


def test_clv_is_negative_when_it_moved_away(tmp_path):
    p = a_portfolio(tmp_path)
    pos = a_position(p, paid=30.0)
    capture_closing_lines(p, {"kalshi": StubProbe(26.0)})
    assert pos.clv_cents == pytest.approx(-4.0)


def test_a_match_that_has_not_kicked_off_has_no_closing_line_yet(tmp_path):
    p = a_portfolio(tmp_path)
    pos = a_position(p, kickoff=FUTURE)
    stats = capture_closing_lines(p, {"kalshi": StubProbe(50.0)})
    assert stats["not_kicked_off"] == 1 and stats["captured"] == 0
    assert pos.clv_cents is None


def test_capture_is_idempotent(tmp_path):
    p = a_portfolio(tmp_path)
    a_position(p)
    first = capture_closing_lines(p, {"kalshi": StubProbe(35.0)})
    second = capture_closing_lines(p, {"kalshi": StubProbe(99.0)})
    assert first["captured"] == 1
    assert second["captured"] == 0 and second["already_had"] == 1
    assert next(iter(p.positions.values())).closing_price_cents == 35.0


def test_a_missing_tape_is_counted_not_recorded_as_zero_clv(tmp_path):
    p = a_portfolio(tmp_path)
    pos = a_position(p)
    stats = capture_closing_lines(p, {"kalshi": StubProbe(None)})
    assert stats["no_history"] == 1
    assert pos.clv_cents is None, "no reading must stay absent, not become 0"


def test_the_probe_is_asked_about_our_side(tmp_path):
    p = a_portfolio(tmp_path)
    a_position(p, side="no")
    probe = StubProbe(40.0)
    capture_closing_lines(p, {"kalshi": probe})
    assert probe.asked[0][1] == "no"


# --- the sample-size honesty --------------------------------------------------
def test_thirty_four_bets_on_one_match_are_not_thirty_four_observations(tmp_path):
    """Measured on the live venues: 307 placeable cases, 9 fixtures."""
    p = a_portfolio(tmp_path)
    for i in range(34):
        a_position(p, instrument="I%d" % i, paid=30.0)
    capture_closing_lines(p, {"kalshi": StubProbe(35.0)})
    out = clv_summary(p)
    assert out["n_bets"] == 34
    assert out["n_fixtures"] == 1
    assert out["bets_per_fixture"] == 34.0


def test_the_headline_averages_over_fixtures_not_bets(tmp_path):
    """One match with many bets must not outvote a match with one."""
    p = a_portfolio(tmp_path)
    for i in range(9):                       # fixture A, CLV +10 each
        a_position(p, instrument="A%d" % i, paid=30.0, home="A", away="B")
    a_position(p, instrument="C1", paid=30.0, home="C", away="D")
    probe = StubProbe(40.0)
    capture_closing_lines(p, {"kalshi": probe})
    out = clv_summary(p)
    assert out["n_bets"] == 10 and out["n_fixtures"] == 2
    # Both fixtures average +10, so the headline is +10 either way -- what
    # matters is that it was computed over 2 points, not 10.
    assert out["clv_cents"] == pytest.approx(10.0)


def test_a_lopsided_fixture_cannot_dominate_the_headline(tmp_path):
    p = a_portfolio(tmp_path)
    for i in range(20):
        a_position(p, instrument="A%d" % i, paid=30.0, home="A", away="B")
    a_position(p, instrument="C1", paid=10.0, home="C", away="D")
    capture_closing_lines(p, {"kalshi": StubProbe(40.0)})
    out = clv_summary(p)
    # fixture A: +10 each; fixture C: +30. Per-fixture mean = 20.
    # Per-bet mean would be (20*10 + 30)/21 = 10.95 -- dominated by fixture A.
    assert out["clv_cents"] == pytest.approx(20.0)
    assert out["clv_cents_per_bet"] == pytest.approx(10.95, abs=0.01)
    assert out["n_fixtures"] == 2


def test_an_empty_portfolio_reports_no_reading_rather_than_zero(tmp_path):
    out = clv_summary(a_portfolio(tmp_path))
    assert out["n_bets"] == 0 and out["clv_cents"] is None


def test_a_single_fixture_has_no_t_statistic(tmp_path):
    """One observation cannot support a significance claim."""
    p = a_portfolio(tmp_path)
    a_position(p)
    capture_closing_lines(p, {"kalshi": StubProbe(35.0)})
    assert clv_summary(p)["t_stat"] is None


def test_t_statistic_appears_once_there_are_several_fixtures(tmp_path):
    p = a_portfolio(tmp_path)
    for i, paid in enumerate((28.0, 29.0, 31.0, 27.0, 30.0)):
        a_position(p, instrument="I%d" % i, paid=paid,
                   home="H%d" % i, away="A%d" % i)
    capture_closing_lines(p, {"kalshi": StubProbe(35.0)})
    out = clv_summary(p)
    assert out["n_fixtures"] == 5
    assert out["t_stat"] is not None
    assert out["clv_std_cents"] > 0


# --- the venue readers --------------------------------------------------------
class FakeKalshi:
    def __init__(self, candles):
        self.candles = candles

    def get_candlesticks(self, ticker, start, end, period_interval=1):
        return self.candles


def candle(ts, ask, bid):
    return {"end_period_ts": ts,
            "yes_ask": {"close_dollars": "%.4f" % ask},
            "yes_bid": {"close_dollars": "%.4f" % bid}}


def test_kalshi_closing_line_ignores_candles_after_kickoff():
    """Kalshi trades through the match; an in-play price is not the close."""
    kickoff = dt.datetime(2026, 8, 20, 19, 0, tzinfo=dt.timezone.utc)
    ts = int(kickoff.timestamp())
    probe = KalshiFillProbe(FakeKalshi([
        candle(ts - 120, 0.30, 0.28),        # pre-game
        candle(ts - 60, 0.32, 0.30),         # the close
        candle(ts + 600, 0.90, 0.88),        # in-play: must be ignored
    ]))
    got = probe.closing_price_cents("T", "yes", kickoff)
    assert got == pytest.approx(31.0)        # mid of 32 / 30


def test_kalshi_closing_line_flips_for_a_no_position():
    kickoff = dt.datetime(2026, 8, 20, 19, 0, tzinfo=dt.timezone.utc)
    ts = int(kickoff.timestamp())
    probe = KalshiFillProbe(FakeKalshi([candle(ts - 60, 0.32, 0.30)]))
    assert probe.closing_price_cents("T", "no", kickoff) == pytest.approx(69.0)


class FakeResponse:
    def __init__(self, payload):
        self.payload, self.status_code = payload, 200

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


def test_polymarket_closing_line_takes_the_last_point_before_kickoff():
    kickoff = dt.datetime(2026, 8, 20, 19, 0, tzinfo=dt.timezone.utc)
    ts = int(kickoff.timestamp())
    probe = PolymarketFillProbe(FakeSession([
        {"t": ts - 300, "p": 0.40},
        {"t": ts - 60, "p": 0.42},
        {"t": ts + 300, "p": 0.95},          # in-play
    ]))
    assert probe.closing_price_cents("cond", "yes", kickoff) == pytest.approx(42.0)
