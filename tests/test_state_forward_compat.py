"""The state file is a contract between workers on different commits.

matchday-board runs from `main`. A repair may have been written by a branch, or
by a build from ten minutes later, and both of them save the same
`portfolio.json` back to the same branch. Adding a field is therefore not a
local change to one module: it is a change to a file that older code has to
keep being able to read.

`void` was added to the persisted record without being added to the dataclass
that reads it, and `PaperOrder(**raw)` turned an unknown key into a
`TypeError`. Three hours later matchday-board could not load the book at all.
"""
import json

from wc2026.paper.broker import (
    PaperOrder,
    PaperPortfolio,
    PaperPosition,
    _from_record,
    _to_record,
)

AN_ORDER = {
    "order_id": "o1", "case_id": "kalshi:X:no", "venue": "kalshi",
    "instrument_id": "X", "side": "no", "limit_price_cents": 65,
    "requested_size": 18.0, "status": "filled",
}
A_POSITION = {
    "position_id": "p1", "venue": "kalshi", "instrument_id": "X", "side": "no",
    "size": 18.0, "avg_cost_cents": 65.0, "fees_cents": 29,
}


def test_a_field_from_a_newer_writer_does_not_crash_an_older_reader():
    """The failure exactly: a key the running build has never heard of."""
    order = _from_record(PaperOrder, dict(AN_ORDER, invented_later="hello"))
    assert order.order_id == "o1"
    assert order.extra == {"invented_later": "hello"}


def test_an_unknown_field_survives_a_round_trip_through_a_build_that_ignores_it():
    """Dropping it would be worse in a quieter way: an older worker running
    once would silently erase a newer one's work on the next save."""
    order = _from_record(PaperOrder, dict(AN_ORDER, invented_later="hello"))
    back = _to_record(order)
    assert back["invented_later"] == "hello"
    assert "extra" not in back, "the file shape does not change"


def test_extra_never_shadows_a_field_the_build_does_know():
    order = _from_record(PaperOrder, dict(AN_ORDER, extra={"status": "open"}))
    assert _to_record(order)["status"] == "filled"


def test_a_void_now_round_trips_as_itself():
    order = _from_record(PaperOrder, dict(AN_ORDER, void=True,
                                          void_reason="wrong proposition",
                                          voided_at="2026-09-05T21:00:00+00:00"))
    position = _from_record(PaperPosition, dict(A_POSITION, void=True,
                                                void_reason="wrong proposition"))
    assert order.void is True and order.void_reason == "wrong proposition"
    assert position.void is True
    assert order.extra == {} and position.extra == {}
    assert _to_record(order)["voided_at"] == "2026-09-05T21:00:00+00:00"


def test_a_record_with_no_void_defaults_to_not_void():
    """Every order written before the field existed."""
    assert _from_record(PaperOrder, AN_ORDER).void is False
    assert _from_record(PaperPosition, A_POSITION).void is False


def test_a_whole_book_loads_and_saves_without_losing_an_unknown_key(tmp_path):
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps({
        "starting_cash_cents": 100_000, "cash_cents": 92_535,
        "reserved_cents": 0,
        "orders": {"o1": dict(AN_ORDER, void=True, void_reason="because",
                              from_the_future=42)},
        "positions": {"p1": dict(A_POSITION, settled=True,
                                 realized_pnl_cents=601.0)},
        "ledger": [], "boarded": {}, "saved_at": "2026-09-05T20:22:47+00:00",
    }), encoding="utf-8")

    book = PaperPortfolio.load(str(path))
    assert len(book.orders) == 1 and len(book.positions) == 1
    assert book.cash_cents == 92_535
    assert book.orders["o1"].void is True

    book.save(str(path))
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["orders"]["o1"]["from_the_future"] == 42
    assert written["orders"]["o1"]["void"] is True
    assert written["cash_cents"] == 92_535
