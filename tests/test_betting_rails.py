"""Safety rails: every execute-mode gate proven by a test with a fake client.
No test here touches the network or the real state/audit files."""
import dataclasses
import datetime as dt

import numpy as np
import pytest

import wc2026.betting.bankroll as bankroll_mod
import wc2026.betting.tracking as tracking_mod
from wc2026.betting import pipeline
from wc2026.betting.bankroll import BettingState
from wc2026.betting.config import BettingConfig
from wc2026.betting.ev import Candidate
from wc2026.betting.kalshi import OrderError
from wc2026.betting.markets import MappedMarket


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _redirect_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(bankroll_mod, "AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(tracking_mod, "RECS_LOG", str(tmp_path / "recs.jsonl"))


def _candidate(ticker="KXMLSGAME-26JUL25SJLAG-SJ", contracts=5, ask=40,
               verdict="approve"):
    ind = np.zeros((13, 13), dtype=bool)
    ind[1, 0] = True
    mkt = MappedMarket(ticker=ticker, event_ticker="KXMLSGAME-26JUL25SJLAG",
                       series="KXMLSGAME", title="t", sub_title="s",
                       home="San Jose Earthquakes", away="LA Galaxy",
                       kickoff_utc=None, claim="home_win", indicator=ind)
    c = Candidate(market=mkt, side="yes", claim="home_win", p_model=0.55,
                  p_calibrated=0.55, ask_cents=ask, depth_at_touch=100,
                  spread_cents=3, edge_raw=0.08, edge_calibrated=0.08)
    c.contracts = contracts
    c.fee_cents = 2
    c.stake_cents = contracts * ask + 2
    c.gate_verdict = verdict
    c.gate_multiplier = 1.0
    return c


class FakeClient:
    """Records orders; can be told to fail on the nth order or fill partially."""
    def __init__(self, authenticated=True, fail_on=None, balance=100_000, fill=None,
                 open_tickers=()):
        self._auth = authenticated
        self.fail_on = fail_on
        self.balance = balance
        self.fill = fill            # None = full fill; else the fill count
        self.open_tickers = set(open_tickers)
        self.orders = []

    def open_position_tickers(self):
        return set(self.open_tickers)

    @property
    def authenticated(self):
        return self._auth

    def require_auth(self, why):
        if not self._auth:
            from wc2026.betting.kalshi import ConfigError
            raise ConfigError(why)

    def get_balance_cents(self):
        return self.balance

    def place_limit_order(self, ticker, side, count, price_cents,
                          time_in_force="immediate_or_cancel",
                          self_trade_prevention="taker_at_cross"):
        if self.fail_on is not None and len(self.orders) + 1 == self.fail_on:
            raise OrderError("simulated API failure")
        self.orders.append((ticker, side, count, price_cents))
        filled = count if self.fill is None else min(self.fill, count)
        return {"order_id": f"o{len(self.orders)}", "filled": filled,
                "fill_price_cents": price_cents, "fee_cents": 2, "raw": {}}


@pytest.fixture
def state(tmp_path):
    return BettingState(bankroll_cents=100_000, path=str(tmp_path / "state.json"))


@pytest.fixture
def cfg():
    return dataclasses.replace(BettingConfig(), gate_enabled=False)


class Args:
    live = False
    min_edge = None
    skip_gate = False


def ALL(_prompt):
    """Selection stub: place every proposed order."""
    return "all"


def _patch_slate(monkeypatch, slate):
    monkeypatch.setattr(pipeline, "build_slate",
                        lambda *a, **k: [c for c in slate])


# --------------------------------------------------------------------------- #
# rails
# --------------------------------------------------------------------------- #
def test_kill_switch_blocks_everything(monkeypatch, state, cfg):
    cfg = dataclasses.replace(cfg, kill_switch=True)
    called = []
    monkeypatch.setattr(pipeline, "build_slate",
                        lambda *a, **k: called.append(1) or [])
    client = FakeClient()
    placed = pipeline.run_execute(Args(), cfg, client, state)
    assert placed == [] and client.orders == [] and called == []


def test_execute_requires_credentials(monkeypatch, state, cfg):
    from wc2026.betting.kalshi import ConfigError
    _patch_slate(monkeypatch, [_candidate()])
    with pytest.raises(ConfigError):
        pipeline.run_execute(Args(), cfg, FakeClient(authenticated=False), state)


def test_daily_loss_limit_halts_trading(monkeypatch, state, cfg):
    _patch_slate(monkeypatch, [_candidate()])
    state.ledger.append({"ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                         "ticker": "X", "pnl_cents": -int(cfg.daily_loss_limit_usd * 100)})
    client = FakeClient()
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state)
    assert placed == [] and client.orders == []


def test_weekly_loss_limit_halts_trading(monkeypatch, state, cfg):
    _patch_slate(monkeypatch, [_candidate()])
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=3)).isoformat()
    state.ledger.append({"ts": old, "ticker": "X",
                         "pnl_cents": -int(cfg.weekly_loss_limit_usd * 100)})
    client = FakeClient()
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state)
    assert placed == [] and client.orders == []


def test_dry_run_is_the_default_even_in_execute(monkeypatch, state, cfg):
    _patch_slate(monkeypatch, [_candidate()])
    client = FakeClient()
    placed = pipeline.run_execute(Args(), cfg, client, state)   # no --live
    assert placed == [] and client.orders == []


def test_live_without_tty_refuses(monkeypatch, state, cfg):
    """pytest runs without a tty, so --live must refuse before confirmation."""
    _patch_slate(monkeypatch, [_candidate()])
    client = FakeClient()
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state)
    assert placed == [] and client.orders == []


def test_live_with_declined_confirmation_places_nothing(monkeypatch, state, cfg):
    _patch_slate(monkeypatch, [_candidate()])
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: True)
    client = FakeClient()
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state,
                                  confirm_input=lambda _: "no thanks",
                                  select_input=ALL)
    assert placed == [] and client.orders == []


def test_live_with_confirmation_places_limit_orders(monkeypatch, state, cfg):
    _patch_slate(monkeypatch, [_candidate(contracts=5, ask=40)])
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: True)
    client = FakeClient()
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state,
                                  confirm_input=lambda _: "PLACE ORDERS", select_input=ALL)
    assert len(placed) == 1
    assert client.orders == [("KXMLSGAME-26JUL25SJLAG-SJ", "yes", 5, 40)]
    # fill recorded against bankroll
    assert state.n_open_positions == 1
    assert state.bankroll_cents == 100_000 - (5 * 40 + 2)


def test_partial_fill_records_only_what_filled(monkeypatch, state, cfg):
    """IOC can fill partially -- bankroll/positions must reflect the ACTUAL
    fill (3), not the requested size (5)."""
    _patch_slate(monkeypatch, [_candidate(contracts=5, ask=40)])
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: True)
    client = FakeClient(fill=3)
    args = Args(); args.live = True
    pipeline.run_execute(args, cfg, client, state, confirm_input=lambda _: "PLACE ORDERS", select_input=ALL)
    assert state.n_open_positions == 1
    assert state.positions[0]["contracts"] == 3
    assert state.bankroll_cents == 100_000 - (3 * 40 + 2)


def test_zero_fill_records_nothing(monkeypatch, state, cfg):
    """IOC with no liquidity at our limit fills 0 -> no position, no placement,
    bankroll untouched."""
    _patch_slate(monkeypatch, [_candidate(contracts=5, ask=40)])
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: True)
    client = FakeClient(fill=0)
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state,
                                  confirm_input=lambda _: "PLACE ORDERS", select_input=ALL)
    assert placed == [] and state.n_open_positions == 0
    assert state.bankroll_cents == 100_000       # untouched


def test_max_open_positions_enforced(monkeypatch, state, cfg):
    cfg = dataclasses.replace(cfg, max_open_positions=1)
    state.positions.append({"ticker": "X", "side": "yes", "contracts": 1,
                            "avg_cost_cents": 50.0, "opened_at": "t"})
    _patch_slate(monkeypatch, [_candidate()])
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: True)
    # Kalshi still holds X -> reconcile keeps it -> the cap still bites.
    client = FakeClient(open_tickers={"X"})
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state,
                                  confirm_input=lambda _: "PLACE ORDERS", select_input=ALL)
    assert placed == [] and client.orders == []


def test_api_error_aborts_run_no_retry(monkeypatch, state, cfg):
    """First order fails -> the whole run stops; the second order is never
    attempted and nothing is retried."""
    slate = [_candidate(ticker="T1"), _candidate(ticker="T2")]
    _patch_slate(monkeypatch, slate)
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: True)
    client = FakeClient(fail_on=1)
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state,
                                  confirm_input=lambda _: "PLACE ORDERS", select_input=ALL)
    assert placed == [] and client.orders == []
    assert state.n_open_positions == 0


def test_unscreened_bets_never_placed_when_gate_on(monkeypatch, state):
    """Gate enabled but bets unscreened (fail-closed) -> zero orders."""
    cfg = dataclasses.replace(BettingConfig(), gate_enabled=True)
    slate = [_candidate(verdict=None)]
    _patch_slate(monkeypatch, slate)
    import wc2026.betting.gate as gate_mod

    def _fail_screen(s, c, _invoke=None):
        for cand in s:
            cand.gate_verdict = "unscreened"
        return s
    monkeypatch.setattr(gate_mod, "screen_slate", _fail_screen)
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: True)
    client = FakeClient()
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state,
                                  confirm_input=lambda _: "PLACE ORDERS", select_input=ALL)
    assert placed == [] and client.orders == []


# --------------------------------------------------------------------------- #
# bankroll accounting
# --------------------------------------------------------------------------- #
def test_settlement_accounting(state):
    state.record_fill("T", "yes", 10, 40, 3)          # cost 403c
    assert state.bankroll_cents == 100_000 - 403
    pnl = state.record_settlement("T", "yes")         # YES wins -> payout 1000c
    assert pnl == 1000 - 403
    assert state.bankroll_cents == 100_000 - 403 + 1000
    assert state.n_open_positions == 0

    state.record_fill("U", "no", 10, 40, 3)
    pnl = state.record_settlement("U", "yes")         # result YES -> our NO loses
    assert pnl == -403
    # loss limits act on NET realized P&L: +597 then -403 nets positive
    assert state.realized_loss_cents(1.0) == 0


# --------------------------------------------------------------------------- #
# interactive order selection (how many of the proposed orders to place)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reply,expected", [
    ("3", 3), ("1", 1), ("5", 5),
    ("all", 5), ("ALL", 5), (" a ", 5),
    ("9", 5),                       # more than available -> clamped
    ("0", 0), ("", 0), ("   ", 0),  # explicit none / least-effort reply
    ("-2", 0), ("two", 0), ("3.5", 0), ("yes", 0), (None, 0),
])
def test_parse_order_selection(reply, expected):
    from wc2026.betting.pipeline import parse_order_selection
    assert parse_order_selection(reply, 5) == expected


def test_selection_places_only_the_chosen_count(monkeypatch, state, cfg):
    """Choosing 2 of 3 places the two best-ranked bets only."""
    slate = [_candidate(ticker="T1"), _candidate(ticker="T2"), _candidate(ticker="T3")]
    _patch_slate(monkeypatch, slate)
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: True)
    client = FakeClient()
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state,
                                  confirm_input=lambda _: "PLACE ORDERS",
                                  select_input=lambda _: "2")
    assert len(placed) == 2
    assert [o[0] for o in client.orders] == ["T1", "T2"]     # top-ranked first


def test_selection_all_places_everything(monkeypatch, state, cfg):
    slate = [_candidate(ticker="T1"), _candidate(ticker="T2")]
    _patch_slate(monkeypatch, slate)
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: True)
    client = FakeClient()
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state,
                                  confirm_input=lambda _: "PLACE ORDERS",
                                  select_input=ALL)
    assert len(placed) == 2


@pytest.mark.parametrize("reply", ["0", "", "garbage"])
def test_selection_none_or_invalid_places_nothing(monkeypatch, state, cfg, reply):
    """Cancelling (or fumbling) the count aborts BEFORE the confirmation --
    nothing is ordered even if 'PLACE ORDERS' would have been typed."""
    _patch_slate(monkeypatch, [_candidate(ticker="T1")])
    monkeypatch.setattr(pipeline.sys.stdin, "isatty", lambda: True)
    client = FakeClient()
    args = Args(); args.live = True
    placed = pipeline.run_execute(args, cfg, client, state,
                                  confirm_input=lambda _: "PLACE ORDERS",
                                  select_input=lambda _: reply)
    assert placed == [] and client.orders == []


def test_selection_not_prompted_in_dry_run(monkeypatch, state, cfg):
    """Dry run must never prompt -- it returns before the selection step."""
    def boom(_):
        raise AssertionError("selection prompt must not run in dry-run")
    _patch_slate(monkeypatch, [_candidate()])
    client = FakeClient()
    placed = pipeline.run_execute(Args(), cfg, client, state, select_input=boom)
    assert placed == [] and client.orders == []


# --------------------------------------------------------------------------- #
# position reconciliation against Kalshi (source of truth)
# --------------------------------------------------------------------------- #
class ReconcileClient:
    def __init__(self, open_tickers=(), markets=None, fail=False):
        self.open = set(open_tickers)
        self.markets = markets or {}          # ticker -> market dict
        self.fail = fail

    def open_position_tickers(self):
        if self.fail:
            raise RuntimeError("positions api down")
        return set(self.open)

    def get_market(self, ticker):
        return self.markets.get(ticker, {"status": "active", "result": ""})


def test_reconcile_settles_positions_kalshi_no_longer_holds(tmp_path):
    from wc2026.betting.bankroll import reconcile_positions
    state = BettingState(bankroll_cents=100_000, path=str(tmp_path / "s.json"))
    state.record_fill("T-YES", "yes", 10, 40, 3)      # our YES bet
    state.record_fill("U-NO", "no", 10, 40, 3)        # our NO bet
    # Kalshi holds neither; both markets settled YES (T wins, U -- a NO -- loses)
    client = ReconcileClient(open_tickers=set(), markets={
        "T-YES": {"status": "finalized", "result": "yes"},
        "U-NO": {"status": "finalized", "result": "yes"}})
    out = reconcile_positions(state, client, verbose=False)
    assert out["settled"] == 2 and state.n_open_positions == 0
    assert sorted(e["pnl_cents"] for e in state.ledger) == [-403, 597]


def test_reconcile_keeps_positions_still_open_on_kalshi(tmp_path):
    from wc2026.betting.bankroll import reconcile_positions
    state = BettingState(bankroll_cents=100_000, path=str(tmp_path / "s.json"))
    state.record_fill("T", "yes", 5, 40, 2)
    out = reconcile_positions(state, ReconcileClient(open_tickers={"T"}), verbose=False)
    assert out["settled"] == 0 and state.n_open_positions == 1


def test_reconcile_drops_flat_unsettled_position_without_pnl(tmp_path):
    from wc2026.betting.bankroll import reconcile_positions
    state = BettingState(bankroll_cents=100_000, path=str(tmp_path / "s.json"))
    state.record_fill("T", "yes", 5, 40, 2)
    client = ReconcileClient(open_tickers=set(),
                             markets={"T": {"status": "active", "result": ""}})
    out = reconcile_positions(state, client, verbose=False)
    assert out["dropped"] == 1 and state.n_open_positions == 0 and state.ledger == []


def test_reconcile_skips_safely_on_api_error(tmp_path):
    from wc2026.betting.bankroll import reconcile_positions
    state = BettingState(bankroll_cents=100_000, path=str(tmp_path / "s.json"))
    state.record_fill("T", "yes", 5, 40, 2)
    out = reconcile_positions(state, ReconcileClient(fail=True), verbose=False)
    assert out["skipped"] and state.n_open_positions == 1     # untouched (safe)


def test_reconcile_feeds_the_loss_limit(tmp_path, cfg):
    """The whole point: settled losses land in the ledger so the loss-limit
    rail (previously never fed) can actually fire."""
    from wc2026.betting.bankroll import reconcile_positions
    state = BettingState(bankroll_cents=100_000, path=str(tmp_path / "s.json"))
    assert state.loss_limits_breached(cfg) is None            # nothing yet
    state.record_fill("T", "no", 200, 60, 5)                  # $120 NO bet
    client = ReconcileClient(open_tickers=set(),
                             markets={"T": {"status": "finalized", "result": "yes"}})
    reconcile_positions(state, client, verbose=False)         # NO loses -> -$120
    assert state.loss_limits_breached(cfg) is not None        # daily limit fires
