"""Duplicate prevention in the recommendation log and tracker, and the
live-balance bankroll resolution. All offline."""
import json

import pytest

import wc2026.betting.bankroll as bankroll_mod
from wc2026.betting.bankroll import BettingState
from wc2026.betting.pipeline import resolve_bankroll
from wc2026.betting.tracking import (
    _load_lines,
    log_placement,
    log_recommendations,
    resolve,
    summarize,
)


@pytest.fixture(autouse=True)
def _redirect_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(bankroll_mod, "AUDIT_LOG", str(tmp_path / "audit.jsonl"))


PAST_KICK = "2020-06-01T00:00:00Z"      # safely in the past for "now >= kickoff"


def _rec(ticker="KXMLSGAME-26JUL25SJLAG-SJ", side="yes", price=40, contracts=5,
         kickoff=None):
    r = {"ticker": ticker, "side": side, "claim": "home_win",
         "price_cents": price, "contracts": contracts, "fee_cents": 2,
         "counterfactual_contracts": contracts, "edge": 0.05}
    if kickoff:
        r["kickoff_utc"] = kickoff
    return r


# --------------------------------------------------------------------------- #
# logging dedup
# --------------------------------------------------------------------------- #
def test_rerunning_recommend_logs_each_bet_once(tmp_path):
    p = str(tmp_path / "recs.jsonl")
    assert log_recommendations([_rec()], "recommend", p) == (1, 0)
    # same market/side again (e.g. recommend re-run before kickoff) -> skipped
    assert log_recommendations([_rec(price=42)], "recommend", p) == (0, 1)
    rows = _load_lines(p)
    assert len(rows) == 1 and rows[0]["price_cents"] == 40


def test_different_side_or_market_still_logged(tmp_path):
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec()], "recommend", p)
    n, dup = log_recommendations([_rec(side="no"), _rec(ticker="OTHER-T")],
                                 "recommend", p)
    assert (n, dup) == (2, 0)


def test_execute_mode_may_supersede_recommend_row(tmp_path):
    """An execute run records its own row once (gate verdicts, real placement);
    re-running execute does not duplicate it."""
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec()], "recommend", p)
    assert log_recommendations([_rec(contracts=3)], "execute", p) == (1, 0)
    assert log_recommendations([_rec(contracts=3)], "execute", p) == (0, 1)
    assert len(_load_lines(p)) == 2


# --------------------------------------------------------------------------- #
# tracking dedup
# --------------------------------------------------------------------------- #
def _dollars(cents):
    """Kalshi dollar-string price, or None for an untraded (0) book."""
    return f"{cents / 100:.4f}" if cents else None


class SettledClient:
    """A finalized market. `candle_c` is the YES price at kickoff that the
    candlestick history returns (None = the market never traded -> no candle)."""
    def __init__(self, result="yes", close_c=55, candle_c=None):
        self.result = result
        self.close_c = close_c
        self.candle_c = candle_c
        self.calls = 0
        self.candle_calls = 0

    def get_market(self, ticker):
        self.calls += 1
        return {"status": "finalized", "result": self.result,
                "last_price_dollars": "0.9900" if self.result == "yes" else "0.0100",
                "previous_price_dollars": _dollars(self.close_c)}

    def get_candlesticks(self, ticker, start_ts, end_ts, period_interval=1):
        self.candle_calls += 1
        if self.candle_c is None:
            return []
        return [{"end_period_ts": end_ts - 60,
                 "price": {"close_dollars": _dollars(self.candle_c)}}]


def test_each_bet_resolves_once_against_latest_row(tmp_path):
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec(contracts=5)], "recommend", p)
    log_recommendations([_rec(contracts=3, price=42)], "execute", p)

    client = SettledClient(result="yes")
    out = resolve(client=client, path=p, verbose=False)
    # ONE market fetch, ONE settled resolution, scored off the execute row
    assert client.calls == 1
    assert out["n_recommendations"] == 1 and out["n_settled"] == 1
    res = [r for r in _load_lines(p) if r.get("kind") == "resolution"]
    assert len(res) == 1
    assert res[0]["pnl_cents"] == 3 * (100 - 42) - 2   # latest (execute) row

    # a second track run adds nothing
    out2 = resolve(client=client, path=p, verbose=False)
    assert client.calls == 1                     # settled key not re-fetched
    assert len([r for r in _load_lines(p) if r.get("kind") == "resolution"]) == 1
    assert out2["n_settled"] == 1


def test_summary_counts_distinct_bets(tmp_path):
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec()], "recommend", p)
    log_recommendations([_rec()], "execute", p)
    log_recommendations([_rec(side="no")], "recommend", p)
    out = summarize(p, verbose=False)
    assert out["n_recommendations"] == 2         # (ticker,yes) and (ticker,no)


# --------------------------------------------------------------------------- #
# resolution upsert: exactly one row per bet
# --------------------------------------------------------------------------- #
class StatefulClient:
    """A single market whose state can be flipped to simulate progression.
    Prices are YES cents, emitted as the real dollar-string fields. `candle_c`
    is the YES price the candlestick history returns at kickoff."""
    def __init__(self, status="active", result="", last_c=55, prev_c=None,
                 candle_c=None):
        self.market = {"status": status, "result": result}
        self.set_last(last_c)
        self.set_prev(prev_c)
        self.candle_c = candle_c
        self.calls = 0

    def set_last(self, cents):
        self.market["last_price_dollars"] = _dollars(cents)

    def set_prev(self, cents):
        self.market["previous_price_dollars"] = _dollars(cents)

    def get_market(self, ticker):
        self.calls += 1
        return dict(self.market)

    def get_candlesticks(self, ticker, start_ts, end_ts, period_interval=1):
        if self.candle_c is None:
            return []
        return [{"end_period_ts": end_ts - 60,
                 "price": {"close_dollars": _dollars(self.candle_c)}}]


def test_resolution_upserts_one_row_per_bet(tmp_path):
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec(contracts=5)], "recommend", p)  # side yes, entry 40
    c = StatefulClient(last_c=55)

    resolve(client=c, path=p, verbose=False)                  # open @55
    c.set_last(58)
    resolve(client=c, path=p, verbose=False)                  # open @58 -> overwrite

    res = [r for r in _load_lines(p) if r["kind"] == "resolution"]
    assert len(res) == 1                                       # NOT appended
    assert res[0]["settled"] is False
    assert res[0]["closing_price_cents"] == 58                 # tracks latest price
    assert res[0]["clv_cents"] == 58 - 40


def test_settled_clv_uses_candlestick_at_kickoff(tmp_path):
    """The authoritative closing line is the candlestick price AT KICKOFF, read
    from history -- not any live market field (which is contaminated in-game)."""
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec(contracts=5, kickoff=PAST_KICK)], "recommend", p)
    # market's live price is the collapsed 0.99; the true kickoff close is 62c
    c = SettledClient(result="yes", close_c=99, candle_c=62)
    resolve(client=c, path=p, verbose=False)

    res = [r for r in _load_lines(p) if r["kind"] == "resolution"]
    assert res[0]["settled"] is True
    assert res[0]["clv_source"] == "candlestick"
    assert res[0]["closing_price_cents"] == 62                 # kickoff price
    assert res[0]["clv_cents"] == 62 - 40
    assert c.candle_calls == 1


def test_candlestick_close_is_frozen_and_not_refetched(tmp_path):
    """Once the kickoff close is locked from candlesticks, `track` neither
    re-fetches the market nor the candlesticks for that bet."""
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec(contracts=5, kickoff=PAST_KICK)], "recommend", p)
    c = SettledClient(result="yes", candle_c=55)
    resolve(client=c, path=p, verbose=False)
    assert (c.calls, c.candle_calls) == (1, 1)
    resolve(client=c, path=p, verbose=False)                  # frozen
    assert (c.calls, c.candle_calls) == (1, 1)


def test_settled_without_candlestick_is_clv_unavailable(tmp_path):
    """A settled bet with no kickoff close obtainable (untraded book) gets no
    CLV -- never a fabricated one from a contaminated field."""
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec(contracts=5, kickoff=PAST_KICK)], "recommend", p)
    c = SettledClient(result="yes", close_c=62, candle_c=None)  # no candles
    out = resolve(client=c, path=p, verbose=False)
    res = [r for r in _load_lines(p) if r["kind"] == "resolution"]
    assert res[0].get("clv_unavailable") is True
    assert "clv_cents" not in res[0]
    assert out["n_clv"] == 0 and out["mean_clv_cents"] is None


def test_open_market_shows_interim_clv_then_candlestick_locks(tmp_path):
    """Before kickoff, an open market shows a live 'interim' read; once past
    kickoff the candlestick close supersedes it as the authoritative CLV."""
    p = str(tmp_path / "recs.jsonl")
    # no kickoff yet in the record -> interim read from the live last price
    log_recommendations([_rec(contracts=5)], "recommend", p)
    out = resolve(client=StatefulClient(last_c=58), path=p, verbose=False)
    res = [r for r in _load_lines(p) if r["kind"] == "resolution"]
    assert res[0]["clv_source"] == "interim"
    assert res[0]["clv_cents"] == 58 - 40
    assert out["n_clv"] == 0 and out["n_clv_interim"] == 1     # interim not counted as closed


def test_price_parser_rejects_degenerate_and_bad_input():
    from wc2026.betting.tracking import _price_to_cents
    assert _price_to_cents("0.4800") == 48
    assert _price_to_cents("0.0100") == 1
    assert _price_to_cents("0.9900") == 99
    assert _price_to_cents("0.0000") is None      # untraded
    assert _price_to_cents("1.0000") is None      # settlement collapse
    assert _price_to_cents(None) is None
    assert _price_to_cents("") is None
    assert _price_to_cents("garbage") is None


def test_clv_distinct_match_count(tmp_path):
    """CLV's honest sample size counts distinct games, not bets (bets on the
    same match are correlated)."""
    p = str(tmp_path / "recs.jsonl")

    def _bet(ticker):
        return {"ticker": ticker, "side": "yes", "claim": "x", "price_cents": 40,
                "contracts": 5, "fee_cents": 2, "counterfactual_contracts": 5,
                "kickoff_utc": PAST_KICK}
    # two markets on the SAME game (shared date+teams token) + one other game
    log_recommendations([_bet("KXMLSGAME-26JUL22SKCMIN-MIN")], "recommend", p)
    log_recommendations([_bet("KXMLSTOTAL-26JUL22SKCMIN-3")], "recommend", p)
    log_recommendations([_bet("KXMLSGAME-26JUL22CINVAN-CIN")], "recommend", p)
    out = resolve(client=SettledClient(result="yes", candle_c=55), path=p, verbose=False)
    assert out["n_clv"] == 3            # three closed bets have a candlestick CLV
    assert out["n_clv_matches"] == 2    # but only two distinct games


def test_settled_row_missing_clv_is_backfilled_then_frozen(tmp_path):
    """The user's real situation: settled rows written before the CLV fix have
    no CLV. `track` must re-fetch them ONCE to backfill, then stop re-fetching."""
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec(contracts=5, kickoff=PAST_KICK)], "recommend", p)
    rec_id = [r for r in _load_lines(p) if r["kind"] == "recommendation"][0]["rec_id"]
    with open(p, "a", encoding="utf-8") as f:                  # old settled row, no CLV
        f.write(json.dumps({"kind": "resolution", "rec_id": rec_id,
                            "ticker": _rec()["ticker"], "side": "yes",
                            "market_status": "finalized", "settled": True,
                            "result": "yes", "pnl_cents": 100}) + "\n")

    c = SettledClient(result="yes", candle_c=62)               # candlestick close 62
    resolve(client=c, path=p, verbose=False)
    assert c.calls == 1                                        # re-fetched to backfill
    res = [r for r in _load_lines(p) if r["kind"] == "resolution"]
    assert len(res) == 1 and res[0]["clv_cents"] == 62 - 40
    assert res[0]["clv_source"] == "candlestick"

    resolve(client=c, path=p, verbose=False)                  # now frozen
    assert c.calls == 1                                        # not re-fetched again


def test_recompute_overrides_a_wrong_stored_kickoff(tmp_path):
    """Regression: bets logged before kickoff capture stored the Kalshi
    expiration (~game end) as kickoff, which reads an IN-GAME price as the
    close. A recompute with a dataset lookup must OVERRIDE that wrong kickoff
    (not just fill missing ones) and re-read the candlestick."""
    p = str(tmp_path / "recs.jsonl")
    # rec carries a WRONG kickoff (game-end); the dataset lookup knows the true one
    log_recommendations([_rec(contracts=5, kickoff="2026-07-23T02:30:00Z")], "recommend", p)
    good_kick = "2026-07-22T23:30:00Z"
    lookup = lambda rec: good_kick
    c = SettledClient(result="yes", candle_c=26)     # true pre-game close 26c
    out = resolve(client=c, path=p, verbose=False, recompute=True, kickoff_lookup=lookup)

    recs = [r for r in _load_lines(p) if r["kind"] == "recommendation"]
    assert recs[0]["kickoff_utc"] == good_kick        # overridden
    res = [r for r in _load_lines(p) if r["kind"] == "resolution"]
    assert res[0]["clv_source"] == "candlestick"
    assert res[0]["clv_cents"] == 26 - 40             # sane pre-game CLV, not in-game


def test_settled_without_kickoff_marks_clv_unavailable(tmp_path):
    """No stored kickoff -> a candlestick close cannot be located -> the settled
    bet is honestly clv-unavailable (never a fabricated value)."""
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec(contracts=5)], "recommend", p)   # no kickoff
    c = SettledClient(result="yes", candle_c=55)
    resolve(client=c, path=p, verbose=False)
    res = [r for r in _load_lines(p) if r["kind"] == "resolution"]
    assert res[0].get("clv_unavailable") is True and "clv_cents" not in res[0]
    assert c.candle_calls == 0                                 # never even queried
    resolve(client=c, path=p, verbose=False)                  # frozen
    assert c.calls == 1


def test_resolve_self_heals_stacked_interim_resolutions(tmp_path):
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec(contracts=5)], "recommend", p)
    rec_id = [r for r in _load_lines(p) if r["kind"] == "recommendation"][0]["rec_id"]
    with open(p, "a", encoding="utf-8") as f:                 # 3 old stacked snapshots
        for _ in range(3):
            f.write(json.dumps({"kind": "resolution", "rec_id": rec_id,
                                "ticker": _rec()["ticker"], "side": "yes",
                                "market_status": "active", "settled": False}) + "\n")
    assert len([r for r in _load_lines(p) if r["kind"] == "resolution"]) == 3

    resolve(client=StatefulClient(status="finalized", result="yes", prev_c=50),
            path=p, verbose=False)
    res = [r for r in _load_lines(p) if r["kind"] == "resolution"]
    assert len(res) == 1 and res[0]["settled"] is True


# --------------------------------------------------------------------------- #
# realized P&L only counts truly-executed (placed) bets
# --------------------------------------------------------------------------- #
def test_realized_is_zero_without_a_placement_even_when_settled(tmp_path):
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec(contracts=5)], "recommend", p)   # paper only
    out = resolve(client=SettledClient(result="yes", close_c=55), path=p,
                  verbose=False)
    assert out["n_settled"] == 1
    assert out["realized_pnl_usd"] == 0.0                      # nothing was placed
    assert out["counterfactual_pnl_usd"] != 0.0               # but paper P&L exists


def test_realized_populates_once_a_placement_row_exists(tmp_path):
    p = str(tmp_path / "recs.jsonl")
    log_recommendations([_rec(contracts=5)], "execute", p)     # side yes, entry 40
    resolve(client=SettledClient(result="yes", close_c=55), path=p, verbose=False)
    assert summarize(p, verbose=False)["realized_pnl_usd"] == 0.0

    log_placement({"ticker": _rec()["ticker"], "side": "yes", "contracts": 5,
                   "price_cents": 40, "fee_cents": 2, "order_id": "o1"}, p)
    out = summarize(p, verbose=False)
    # won: 5 contracts, entry 40 -> 5*(100-40) - fee 2 = 298c = $2.98
    assert out["realized_pnl_usd"] == pytest.approx(2.98)
    # a placement row must not disturb the one-resolution-per-bet invariant
    resolve(client=SettledClient(result="yes", close_c=55), path=p, verbose=False)
    assert len([r for r in _load_lines(p) if r["kind"] == "resolution"]) == 1


def test_dedupe_log_repairs_pre_dedup_history(tmp_path):
    """A log with stacked duplicates (written before prevention existed)
    collapses to the latest row per (ticker, side, mode), with a backup."""
    import os

    from wc2026.betting.tracking import dedupe_log
    p = str(tmp_path / "recs.jsonl")
    rows = []
    for k, price in enumerate((40, 41, 42)):        # 3 stacked duplicates
        rows.append({"kind": "recommendation", "rec_id": f"id{k}",
                     "ts": f"2026-07-2{k}T00:00:00", "mode": "recommend",
                     **_rec(price=price)})
    rows.append({"kind": "recommendation", "rec_id": "other",
                 "ts": "2026-07-22T01:00:00", "mode": "recommend",
                 **_rec(ticker="OTHER-T")})
    # resolutions: one for a dropped rec, two for the surviving one
    rows.append({"kind": "resolution", "rec_id": "id0", "ts": "t", "settled": False})
    rows.append({"kind": "resolution", "rec_id": "id2", "ts": "t1", "settled": False})
    rows.append({"kind": "resolution", "rec_id": "id2", "ts": "t2", "settled": True})
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = dedupe_log(p, verbose=False)
    kept = _load_lines(p)
    recs = [r for r in kept if r["kind"] == "recommendation"]
    res = [r for r in kept if r["kind"] == "resolution"]
    assert len(recs) == 2                            # latest dup + OTHER-T
    assert {r["rec_id"] for r in recs} == {"id2", "other"}
    assert [r["price_cents"] for r in recs if r["rec_id"] == "id2"] == [42]
    assert len(res) == 1 and res[0]["settled"]       # settled one preferred
    assert out["dropped"] == 4
    assert os.path.exists(out["backup"])


# --------------------------------------------------------------------------- #
# bankroll resolution
# --------------------------------------------------------------------------- #
class BalanceClient:
    def __init__(self, balance_cents=None, fail=False, authenticated=True):
        self.authenticated = authenticated
        self._balance = balance_cents
        self._fail = fail

    def get_balance_cents(self):
        if self._fail:
            raise RuntimeError("api down")
        return self._balance


def test_bankroll_uses_live_kalshi_balance(tmp_path):
    state = BettingState(bankroll_cents=100_000, path=str(tmp_path / "s.json"))
    got = resolve_bankroll(state, BalanceClient(balance_cents=42_137),
                           verbose=False)
    assert got == 42_137
    assert state.bankroll_cents == 42_137
    # persisted for the next unauthenticated run
    saved = json.load(open(state.path, encoding="utf-8"))
    assert saved["bankroll_cents"] == 42_137


def test_bankroll_without_credentials_uses_state(tmp_path):
    state = BettingState(bankroll_cents=55_500, path=str(tmp_path / "s.json"))
    got = resolve_bankroll(state, BalanceClient(authenticated=False),
                           verbose=False)
    assert got == 55_500


def test_bankroll_fetch_failure_falls_back_loudly(tmp_path):
    state = BettingState(bankroll_cents=77_700, path=str(tmp_path / "s.json"))
    got = resolve_bankroll(state, BalanceClient(fail=True), verbose=False)
    assert got == 77_700
    audit_rows = [json.loads(l) for l in
                  open(bankroll_mod.AUDIT_LOG, encoding="utf-8")]
    assert any(r["event"] == "balance_fetch_failed" for r in audit_rows)
