"""Market mapping: settlement indicators vs the exact grid, rules gating,
team aliasing, order-book arithmetic, and candidate gates."""
import numpy as np
import pandas as pd
import pytest

from wc2026.betting.confidence import TrustCurve
from wc2026.betting.config import BettingConfig
from wc2026.betting.ev import evaluate_market
from wc2026.betting.kalshi import OrderBook, OrderBookSide
from wc2026.betting.markets import (
    MarketMapper,
    canonical_from_text,
    match_event_to_fixture,
    rules_confirm_regulation,
)
from wc2026.sim.match import score_matrix

RULES_OK = ("If Tie is the result of the game after 90 minutes plus stoppage "
            "time (does not include extra time or penalties), then Yes.")
RULES_BAD = "Settles on the final result including extra time and penalties."


def _mkt(ticker="KXMLSGAME-26JUL25SJLAG-SJ", sub="San Jose",
         title="San Jose vs Los Angeles G Winner?", rules=RULES_OK,
         strike=None, exp="2026-07-26T05:30:00Z"):
    ev = ticker.rsplit("-", 1)[0]
    return {"ticker": ticker, "event_ticker": ev, "title": title,
            "yes_sub_title": sub, "rules_primary": rules,
            "floor_strike": strike, "expected_expiration_time": exp}


HOME, AWAY = "San Jose Earthquakes", "LA Galaxy"


@pytest.fixture
def mapper():
    return MarketMapper(grid_size=12)


@pytest.fixture
def pmf():
    return score_matrix(1.7, 1.2, -0.1, 12)


# --- aliasing -----------------------------------------------------------------
def test_canonical_aliases():
    assert canonical_from_text("San Jose") == "San Jose Earthquakes"
    assert canonical_from_text("Los Angeles G") == "LA Galaxy"
    assert canonical_from_text("New York City FC") == "New York City FC"
    assert canonical_from_text("New York RB") == "Red Bull New York"
    assert canonical_from_text("St. Louis") == "St. Louis CITY SC"
    assert canonical_from_text("Real Madrid") is None


# --- rules gating (fail-closed on settlement basis) ---------------------------
def test_rules_regulation_check():
    assert rules_confirm_regulation(RULES_OK)
    assert not rules_confirm_regulation(RULES_BAD)
    assert not rules_confirm_regulation("")


def test_market_without_regulation_rules_is_skipped(mapper):
    assert mapper.map_market(_mkt(rules=RULES_BAD), HOME, AWAY) is None


# --- settlement indicators == exact grid sums ---------------------------------
def test_game_markets_map_to_1x2(mapper, pmf):
    mh = mapper.map_market(_mkt(sub="San Jose"), HOME, AWAY)
    ma = mapper.map_market(_mkt(ticker="KXMLSGAME-26JUL25SJLAG-LAG",
                                sub="Los Angeles G"), HOME, AWAY)
    md = mapper.map_market(_mkt(ticker="KXMLSGAME-26JUL25SJLAG-TIE",
                                sub="Tie"), HOME, AWAY)
    assert mh.claim == "home_win" and ma.claim == "away_win" and md.claim == "draw"
    tot = mh.model_prob(pmf) + ma.model_prob(pmf) + md.model_prob(pmf)
    assert tot == pytest.approx(1.0, abs=1e-12)
    assert mh.model_prob(pmf) == pytest.approx(float(np.tril(pmf, -1).sum()))


def test_total_market_uses_strike(mapper, pmf):
    m = mapper.map_market(_mkt(ticker="KXMLSTOTAL-26JUL25SJLAG-3",
                               sub="Over 2.5 goals scored", strike=2.5),
                          HOME, AWAY)
    assert m.claim == "total_over_2.5"
    k = np.arange(13)
    i, j = np.meshgrid(k, k, indexing="ij")
    assert m.model_prob(pmf) == pytest.approx(float(pmf[(i + j) > 2.5].sum()))


def test_spread_and_team_total_orientation(mapper, pmf):
    sp = mapper.map_market(_mkt(ticker="KXMLSSPREAD-26JUL25SJLAG-LAG2",
                                sub="Los Angeles G wins by more than 1.5 goals",
                                strike=1.5), HOME, AWAY)
    assert sp.claim == "away_wins_by_over_1.5"
    k = np.arange(13)
    i, j = np.meshgrid(k, k, indexing="ij")
    assert sp.model_prob(pmf) == pytest.approx(float(pmf[(j - i) > 1.5].sum()))

    tt = mapper.map_market(_mkt(ticker="KXMLSTEAMTOTAL-26JUL25SJLAG-SJ2",
                                sub="San Jose over 1.5 goals", strike=1.5),
                           HOME, AWAY)
    assert tt.claim == "home_over_1.5"
    assert tt.model_prob(pmf) == pytest.approx(float(pmf[i > 1.5].sum()))


def test_exact_score_orientation(mapper, pmf):
    m = mapper.map_market(_mkt(ticker="KXMLSSCORE-26JUL25SJLAG-LAG2SJ1",
                               sub="Los Angeles G wins 2-1"), HOME, AWAY)
    assert m.claim == "score_1-2"          # home 1, away 2
    assert m.model_prob(pmf) == pytest.approx(float(pmf[1, 2]))


def test_unmappable_subtitle_returns_none(mapper):
    assert mapper.map_market(_mkt(sub="Real Madrid"), HOME, AWAY) is None


# --- fixture matching ---------------------------------------------------------
def test_event_matched_to_fixture_within_day():
    fixtures = pd.DataFrame({
        "date": [pd.Timestamp("2026-07-26")],
        "home_team": [HOME], "away_team": [AWAY]})
    mkts = [_mkt(sub="San Jose"), _mkt(ticker="KXMLSGAME-26JUL25SJLAG-LAG",
                                       sub="Los Angeles G")]
    assert match_event_to_fixture(mkts, fixtures) == (HOME, AWAY)


def test_event_with_unknown_teams_is_unmatched():
    fixtures = pd.DataFrame({"date": [pd.Timestamp("2026-07-26")],
                             "home_team": [HOME], "away_team": [AWAY]})
    assert match_event_to_fixture([_mkt(sub="Real Madrid",
                                        title="Real Madrid vs Barcelona")],
                                  fixtures) is None


# --- order book arithmetic ----------------------------------------------------
def _book():
    # resting YES bids 40@200, 41@100; resting NO bids 55@150, 56@50
    return OrderBook(ticker="T",
                     yes_bids=OrderBookSide([(40, 200), (41, 100)]),
                     no_bids=OrderBookSide([(55, 150), (56, 50)]))


def test_executable_ask_is_complement_of_opposite_bids():
    b = _book()
    assert b.touch("yes") == 44            # 100 - best NO bid 56
    assert b.depth_at_touch("yes") == 50
    assert b.touch("no") == 59             # 100 - best YES bid 41
    assert b.best_bid("yes") == 41


def test_avg_fill_walks_the_book():
    b = _book()
    price, filled = b.avg_fill_price("yes", 100)
    # 50 @ 44c then 50 @ 45c -> 44.5 average
    assert filled == 100 and price == pytest.approx(44.5)


# --- candidate gates ----------------------------------------------------------
def _trust_flat():
    bins = [{"bin_lo": lo / 10, "bin_hi": lo / 10 + 0.1, "mean_pred": lo / 10 + 0.05,
             "observed": lo / 10 + 0.05, "n": 1000} for lo in range(10)]
    return TrustCurve(bins=bins)


def test_evaluate_market_gates(mapper, pmf):
    cfg = BettingConfig()
    mkt = mapper.map_market(_mkt(sub="San Jose"), HOME, AWAY)
    p = mkt.model_prob(pmf)
    price = int(round(100 * (p - 0.08)))   # 8 points cheap -> passes min_edge
    book = OrderBook("T", yes_bids=OrderBookSide([(price - 2, 500)]),
                     no_bids=OrderBookSide([(100 - price, 500)]))
    cands = evaluate_market(mkt, book, pmf, _trust_flat(), cfg, hours_to_kickoff=24)
    assert len(cands) == 1 and cands[0].side == "yes"
    assert cands[0].edge_raw >= cfg.min_edge

    # thin book -> rejected
    thin = OrderBook("T", yes_bids=OrderBookSide([(price - 2, 500)]),
                     no_bids=OrderBookSide([(100 - price, 3)]))
    assert evaluate_market(mkt, thin, pmf, _trust_flat(), cfg, 24) == []

    # wide spread -> rejected
    wide = OrderBook("T", yes_bids=OrderBookSide([(price - 30, 500)]),
                     no_bids=OrderBookSide([(100 - price, 500)]))
    assert evaluate_market(mkt, wide, pmf, _trust_flat(), cfg, 24) == []

    # too far from kickoff -> rejected
    assert evaluate_market(mkt, book, pmf, _trust_flat(), cfg, 500.0) == []


def test_calibration_gate_blocks_overconfident_bins(mapper, pmf):
    """A bin where the model has been badly overconfident must veto the bet
    even when the raw edge clears the minimum."""
    cfg = BettingConfig()
    mkt = mapper.map_market(_mkt(sub="San Jose"), HOME, AWAY)
    p = mkt.model_prob(pmf)
    price = int(round(100 * (p - 0.05)))
    book = OrderBook("T", yes_bids=OrderBookSide([(price - 2, 500)]),
                     no_bids=OrderBookSide([(100 - price, 500)]))
    # trust curve says this bin is ~12 points overconfident
    bins = [{"bin_lo": 0.0, "bin_hi": 1.01, "mean_pred": p,
             "observed": max(p - 0.12, 0.01), "n": 1000}]
    cands = evaluate_market(mkt, book, pmf, TrustCurve(bins=bins), cfg, 24)
    assert cands == []
