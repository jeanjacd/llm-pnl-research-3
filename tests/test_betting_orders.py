"""V2 order construction & response parsing (POST /portfolio/events/orders).

SAFETY CRITICAL: the yes/no -> bid/ask mapping decides which side of the bet
is actually bought. Kalshi's single-book model is quoted from the YES leg:
bid = buy YES, ask = sell YES (= buy NO at 1 - price)."""
import pytest

from wc2026.betting.kalshi import OrderError, build_order_body, parse_order_response

TIF, STP = "immediate_or_cancel", "taker_at_cross"


# --- side / price mapping -----------------------------------------------------
def test_buy_yes_is_a_bid_at_the_yes_price():
    b = build_order_body("KXMLSGAME-X-CLB", "yes", 5, 44, TIF, STP, "cid")
    assert b["side"] == "bid"
    assert b["price"] == "0.4400"          # YES leg = our price
    assert b["count"] == "5"
    assert b["ticker"] == "KXMLSGAME-X-CLB"
    assert b["client_order_id"] == "cid"
    assert b["time_in_force"] == TIF
    assert b["self_trade_prevention_type"] == STP


def test_buy_no_is_an_ask_at_the_complement_price():
    # Buying NO at 60c == selling YES at 40c; the wire price is the YES leg.
    b = build_order_body("KXMLSGAME-X-CLB", "no", 5, 60, TIF, STP, "cid")
    assert b["side"] == "ask"
    assert b["price"] == "0.4000"          # 100 - 60 = 40c -> 0.4000
    assert b["count"] == "5"


def test_no_side_price_is_always_the_yes_complement():
    for no_cost in (1, 25, 50, 75, 99):
        b = build_order_body("T", "no", 1, no_cost, TIF, STP, "c")
        assert b["price"] == f"{(100 - no_cost) / 100:.4f}"
        assert b["side"] == "ask"


def test_order_body_rejects_bad_input():
    for bad_price in (0, 100, -1, 150):
        with pytest.raises(OrderError):
            build_order_body("T", "yes", 5, bad_price, TIF, STP, "c")
    with pytest.raises(OrderError):
        build_order_body("T", "maybe", 5, 40, TIF, STP, "c")   # bad side
    with pytest.raises(OrderError):
        build_order_body("T", "yes", 0, 40, TIF, STP, "c")     # zero count


# --- response parsing (ACTUAL fills) -----------------------------------------
def test_parse_full_fill_yes():
    r = parse_order_response(
        {"order_id": "o1", "fill_count": "5", "average_fill_price": "0.4400",
         "average_fee_paid": "0.0100"}, "yes", 44)
    assert r["order_id"] == "o1"
    assert r["filled"] == 5
    assert r["fill_price_cents"] == 44          # YES leg -> our (yes) side
    assert r["fee_cents"] == 5                  # 1c/contract * 5


def test_parse_fill_no_side_converts_back_from_yes_leg():
    # YES-leg avg 0.40 -> our NO cost = 60c
    r = parse_order_response(
        {"order_id": "o1", "fill_count": "5", "average_fill_price": "0.4000"},
        "no", 60)
    assert r["fill_price_cents"] == 60


def test_parse_zero_fill_falls_back_to_requested_price():
    r = parse_order_response(
        {"order_id": "o1", "fill_count": "0", "remaining_count": "5"}, "yes", 40)
    assert r["filled"] == 0 and r["fill_price_cents"] == 40 and r["fee_cents"] == 0


def test_parse_missing_order_id_raises():
    with pytest.raises(OrderError):
        parse_order_response({"fill_count": "5"}, "yes", 40)
    with pytest.raises(OrderError):
        parse_order_response("not a dict", "yes", 40)


def test_roundtrip_no_side_price_mapping_is_consistent():
    """build (no @ 60c -> ask @ 0.40) then parse a fill at that YES leg back to
    our 60c NO cost -- the two mappings must be inverses."""
    b = build_order_body("T", "no", 3, 60, TIF, STP, "c")
    yes_leg = b["price"]                        # "0.4000"
    r = parse_order_response(
        {"order_id": "o", "fill_count": "3", "average_fill_price": yes_leg},
        "no", 60)
    assert r["fill_price_cents"] == 60
