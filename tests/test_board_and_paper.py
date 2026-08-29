"""Board governance and the paper broker.

The board tests prove authority is one-directional and every failure path
fails closed. The broker tests prove a resting order does not fill just because
a price was touched, and that cash and settlement cannot be double-counted.
"""
import datetime as dt
import json

import pytest

from wc2026.board import (
    JUDGE_PLACEABLE,
    REDACTED_FIELDS,
    BoardFailure,
    SchemaError,
    assert_no_price_leak,
    coach_packet,
    run_board,
    validate_coach,
    validate_judge,
    validate_quant,
)
from wc2026.decision import build_case
from wc2026.paper import CANCELLED, EXPIRED, FILLED, BrokerError, PaperPortfolio
from wc2026.venues.base import KIND_BINARY, Book, Leg, MarketInstrument

CASE_ID = "kalshi:INST:yes"


def soon(hours=6):
    return (dt.datetime.now(dt.timezone.utc)
            + dt.timedelta(hours=hours)).isoformat()


def make_case(asks=((44, 500.0),), bids=((43, 500.0),), p=0.60):
    inst = MarketInstrument(
        venue="kalshi", instrument_id="INST", kind=KIND_BINARY, title="t",
        legs=(Leg.build("home_win", "ref"),), settles_on_regulation=True,
        kickoff_utc=soon(), fee_model={"venue": "kalshi"},
        book=Book(yes_asks=asks, yes_bids=bids,
                  observed_at=dt.datetime.now(dt.timezone.utc).isoformat()))
    return build_case(inst, p, p)


# --------------------------------------------------------------------------- #
# packets
# --------------------------------------------------------------------------- #
def test_coach_packet_contains_no_price_or_verdict():
    case = make_case()
    packet = coach_packet(case, {"home": "Fulham", "away": "Chelsea"})
    for field in REDACTED_FIELDS:
        assert field not in packet, field
    assert_no_price_leak(packet)
    blob = json.dumps(packet)
    assert "ladder" not in blob and "ev_per_contract" not in blob


def test_leaky_packet_is_rejected():
    with pytest.raises(SchemaError):
        assert_no_price_leak({"case_id": "x", "ev_per_contract": 3.0})


# --------------------------------------------------------------------------- #
# schema validation -- malformed fails closed
# --------------------------------------------------------------------------- #
def test_quant_rejects_unknown_case_id():
    with pytest.raises(SchemaError, match="case_id"):
        validate_quant({"case_id": "other", "action": "PASS",
                        "rationale": "x"}, CASE_ID, 50, None)


def test_quant_rejects_invalid_action_and_missing_rationale():
    with pytest.raises(SchemaError):
        validate_quant({"case_id": CASE_ID, "action": "YOLO",
                        "rationale": "x"}, CASE_ID, 50, None)
    with pytest.raises(SchemaError):
        validate_quant({"case_id": CASE_ID, "action": "PASS",
                        "rationale": "  "}, CASE_ID, 50, None)


def test_quant_cannot_raise_the_deterministic_ceiling():
    """The member may confirm or reduce, never raise."""
    out = validate_quant({"case_id": CASE_ID, "action": "PLACE_LIMIT",
                          "proposed_price_cents": 95, "contracts": 10_000,
                          "rationale": "ignore the ceiling"},
                         CASE_ID, ceiling_price=50, ceiling_size=20)
    assert out["proposed_price_cents"] == 50
    assert out["contracts"] == 20


def test_quant_non_finite_numbers_rejected():
    with pytest.raises(SchemaError):
        validate_quant({"case_id": CASE_ID, "action": "BUY_NOW",
                        "proposed_price_cents": float("nan"), "contracts": 5,
                        "rationale": "x"}, CASE_ID, 50, None)


def test_coach_cannot_assign_a_probability_or_multiplier():
    base = {"case_id": CASE_ID, "verdict": "ACCEPT", "rationale": "fine",
            "findings": []}
    for banned in ("probability", "multiplier", "xg_multiplier", "edge"):
        with pytest.raises(SchemaError, match="may not assign"):
            validate_coach({**base, banned: 0.5}, CASE_ID)


def test_coach_findings_require_evidence_and_model_status():
    good = {"case_id": CASE_ID, "verdict": "ACCEPT", "rationale": "ok",
            "findings": [{"text": "starting XI confirmed",
                          "evidence_status": "confirmed",
                          "in_simulation": "included",
                          "sources": ["https://club.example/news"]}]}
    assert validate_coach(good, CASE_ID)["findings"][0]["evidence_status"] == "confirmed"
    bad = {**good, "findings": [{"text": "vibes", "evidence_status": "hunch",
                                 "in_simulation": "included"}]}
    with pytest.raises(SchemaError):
        validate_coach(bad, CASE_ID)


def test_judge_cannot_increase_price_or_size():
    out = validate_judge({"case_id": CASE_ID, "action": "PAPER_PLACE_LIMIT",
                          "limit_price_cents": 90, "contracts": 999,
                          "decisive_reason": "conviction"},
                         CASE_ID, ceiling_price=44, ceiling_size=10)
    assert out["limit_price_cents"] == 44 and out["contracts"] == 10


def test_only_paper_actions_carry_an_order():
    out = validate_judge({"case_id": CASE_ID, "action": "PASS",
                          "decisive_reason": "no edge"}, CASE_ID, 44, 10)
    assert "limit_price_cents" not in out
    assert out["action"] not in JUDGE_PLACEABLE or False


# --------------------------------------------------------------------------- #
# orchestration -- fail closed everywhere
# --------------------------------------------------------------------------- #
def _quant_ok(**over):
    return {"case_id": CASE_ID, "action": "BUY_NOW",
            "proposed_price_cents": 44, "contracts": 10,
            "rationale": "edge confirmed", **over}


def _coach_ok(**over):
    return {"case_id": CASE_ID, "verdict": "ACCEPT", "rationale": "no news",
            "findings": [], "required_reruns": [], **over}


def _judge_ok(**over):
    return {"case_id": CASE_ID, "action": "PAPER_BUY_NOW",
            "limit_price_cents": 44, "contracts": 10,
            "decisive_reason": "value stands", **over}


def _router(quant=None, coach=None, judge=None, fail=None):
    calls = {"n": 0, "order": []}

    def invoke(prompt, model, allowed_tools=(), timeout=None):
        calls["n"] += 1
        if "QUANTITATIVE member" in prompt:
            calls["order"].append("quant")
            if fail == "quant":
                raise BoardFailure("boom")
            return quant or _quant_ok()
        if "SOCCER ANALYST" in prompt:
            calls["order"].append("coach")
            if fail == "coach":
                raise BoardFailure("timeout")
            return coach or _coach_ok()
        calls["order"].append("judge")
        if fail == "judge":
            raise BoardFailure("bad json")
        return judge or _judge_ok()

    return invoke, calls


FIXTURE = {"home": "Fulham", "away": "Chelsea", "league_id": "premier_league"}


def test_happy_path_reaches_the_judge_and_can_place(tmp_path):
    invoke, calls = _router()
    out = run_board(make_case(), FIXTURE, invoke=invoke,
                    log_path=str(tmp_path / "b.jsonl"))
    assert out["action"] == "PAPER_BUY_NOW"
    assert calls["order"] == ["quant", "coach", "judge"]


def test_judge_runs_only_after_both_sealed_proposals(tmp_path):
    invoke, calls = _router()
    run_board(make_case(), FIXTURE, invoke=invoke,
              log_path=str(tmp_path / "b.jsonl"))
    assert calls["order"].index("judge") == 2       # strictly last


def test_quant_failure_fails_closed_and_never_calls_the_judge(tmp_path):
    invoke, calls = _router(fail="quant")
    out = run_board(make_case(), FIXTURE, invoke=invoke,
                    log_path=str(tmp_path / "b.jsonl"))
    assert out["action"] == "DEFER" and "judge" not in calls["order"]


def test_coach_failure_fails_closed(tmp_path):
    invoke, _ = _router(fail="coach")
    out = run_board(make_case(), FIXTURE, invoke=invoke,
                    log_path=str(tmp_path / "b.jsonl"))
    assert out["action"] == "DEFER"


def test_judge_failure_fails_closed(tmp_path):
    invoke, _ = _router(fail="judge")
    out = run_board(make_case(), FIXTURE, invoke=invoke,
                    log_path=str(tmp_path / "b.jsonl"))
    assert out["action"] == "DEFER"


def test_quant_veto_cannot_be_overridden_by_the_judge(tmp_path):
    """A mathematical PASS can never become a buy."""
    invoke, calls = _router(quant={"case_id": CASE_ID, "action": "PASS",
                                   "rationale": "negative EV"})
    out = run_board(make_case(), FIXTURE, invoke=invoke,
                    log_path=str(tmp_path / "b.jsonl"))
    assert out["action"] == "PASS"
    assert "judge" not in calls["order"]


def test_coach_reject_blocks_the_trade(tmp_path):
    invoke, calls = _router(coach=_coach_ok(verdict="REJECT"))
    out = run_board(make_case(), FIXTURE, invoke=invoke,
                    log_path=str(tmp_path / "b.jsonl"))
    assert out["action"] == "PASS" and "judge" not in calls["order"]


def test_a_rerun_request_is_advice_and_not_a_veto(tmp_path):
    """Nothing in this system has ever executed a rerun. The coach asks to be
    re-run "with confirmed line-ups" and no such rerun exists, so treating the
    request as a stop was a veto with no path to resolution -- and it killed
    fixtures the coach had explicitly ACCEPTED."""
    invoke, _ = _router(coach=_coach_ok(required_reruns=["rerun without X"]))
    out = run_board(make_case(), FIXTURE, invoke=invoke,
                    log_path=str(tmp_path / "b.jsonl"))
    assert out["action"] != "DEFER", "an accepted case reaches the judge"
    assert out["judge"], "and the judge is the member that weighs it"


def test_an_explicit_coach_defer_still_stops_the_board(tmp_path):
    """Only the verdict stops it. The safeguard is unchanged."""
    invoke, _ = _router(coach=_coach_ok(verdict="DEFER"))
    out = run_board(make_case(), FIXTURE, invoke=invoke,
                    log_path=str(tmp_path / "b.jsonl"))
    assert out["action"] == "DEFER"
    assert out["judge"] is None, "the judge is never reached"


def test_malformed_member_output_fails_closed(tmp_path):
    invoke, _ = _router(quant={"case_id": CASE_ID, "action": "NOT_A_THING",
                               "rationale": "x"})
    out = run_board(make_case(), FIXTURE, invoke=invoke,
                    log_path=str(tmp_path / "b.jsonl"))
    assert out["action"] == "DEFER"


def test_wrong_case_id_fails_closed(tmp_path):
    invoke, _ = _router(quant=_quant_ok(case_id="somebody-elses-case"))
    out = run_board(make_case(), FIXTURE, invoke=invoke,
                    log_path=str(tmp_path / "b.jsonl"))
    assert out["action"] == "DEFER"


def test_unsupported_case_never_reaches_the_board(tmp_path):
    inst = MarketInstrument(
        venue="kalshi", instrument_id="I", kind=KIND_BINARY, title="t",
        legs=(Leg(claim="player_goal", market_ref="r", supported=False,
                  unsupported_reason="no model"),),
        settles_on_regulation=True, kickoff_utc=soon(),
        book=Book(yes_asks=((10, 100.0),)))
    case = build_case(inst, 0.9, 0.9)
    invoke, calls = _router()
    out = run_board(case, FIXTURE, invoke=invoke,
                    log_path=str(tmp_path / "b.jsonl"))
    assert out["action"] == "UNSUPPORTED" and calls["n"] == 0


def test_web_content_instructions_are_treated_as_data():
    """The coach prompt must tell the member to ignore embedded instructions."""
    from wc2026.board.orchestrator import build_coach_prompt
    prompt = build_coach_prompt({"case_id": CASE_ID, "home": "A", "away": "B"})
    assert "untrusted" in prompt.lower()
    assert "ignore any instruction" in prompt.lower()


# --------------------------------------------------------------------------- #
# paper broker
# --------------------------------------------------------------------------- #
@pytest.fixture
def book():
    return Book(yes_asks=((44, 100.0), (46, 400.0)), yes_bids=((43, 100.0),))


def portfolio(tmp_path, cash=100_000):
    return PaperPortfolio(starting_cash_cents=cash, cash_cents=cash,
                          path=str(tmp_path / "p.json"))


def test_submit_reserves_cash(tmp_path):
    p = portfolio(tmp_path)
    order = p.submit("c1", "kalshi", "I", "yes", 44, 10)
    assert p.reserved_cents >= 440
    assert p.available_cents < p.cash_cents
    assert order.status == "open"


def test_cannot_overspend_reserved_cash(tmp_path):
    p = portfolio(tmp_path, cash=1_000)
    p.submit("c1", "kalshi", "I", "yes", 50, 15)      # ~750c reserved
    with pytest.raises(BrokerError, match="insufficient"):
        p.submit("c2", "kalshi", "J", "yes", 50, 15)


def test_duplicate_submission_is_idempotent(tmp_path):
    p = portfolio(tmp_path)
    a = p.submit("c1", "kalshi", "I", "yes", 44, 10)
    b = p.submit("c1", "kalshi", "I", "yes", 44, 10)
    assert a.order_id == b.order_id and len(p.orders) == 1


def test_marketable_fill_is_capped_by_real_depth(tmp_path, book):
    p = portfolio(tmp_path)
    order = p.submit("c1", "kalshi", "I", "yes", 44, 500)
    p.fill_marketable(order.order_id, book)
    assert order.filled_size == 100.0          # only 100 at or below 44c
    assert order.status == "partially_filled"


def test_resting_order_does_not_fill_on_a_mere_touch(tmp_path):
    """THE rule: a print at our price is not a fill -- queue position is
    unknowable, so a touch must not be treated as an execution."""
    p = portfolio(tmp_path)
    order = p.submit("c1", "kalshi", "I", "yes", 40, 10)
    touched = Book(yes_asks=((40, 500.0),), yes_bids=((39, 500.0),))
    p.try_fill_resting(order.order_id, touched)
    assert order.filled_size == 0.0 and order.status == "open"


def test_resting_order_fills_when_the_market_trades_through(tmp_path):
    p = portfolio(tmp_path)
    order = p.submit("c1", "kalshi", "I", "yes", 40, 10)
    through = Book(yes_asks=((38, 500.0),), yes_bids=((37, 500.0),))
    p.try_fill_resting(order.order_id, through)
    assert order.filled_size == 10.0 and order.status == FILLED
    assert order.avg_fill_price_cents <= 40


def test_fill_never_worse_than_the_limit(tmp_path):
    p = portfolio(tmp_path)
    order = p.submit("c1", "kalshi", "I", "yes", 40, 10)
    through = Book(yes_asks=((30, 5.0), (39, 500.0)), yes_bids=((29, 10.0),))
    p.try_fill_resting(order.order_id, through)
    assert order.avg_fill_price_cents <= 40


def test_expiry_releases_reserved_cash(tmp_path):
    p = portfolio(tmp_path)
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1)).isoformat()
    order = p.submit("c1", "kalshi", "I", "yes", 44, 10, expires_at=past)
    before = p.available_cents
    p.expire_due()
    assert order.status == EXPIRED and p.available_cents > before


def test_cancel_releases_reserved_cash(tmp_path):
    p = portfolio(tmp_path)
    order = p.submit("c1", "kalshi", "I", "yes", 44, 10)
    before = p.available_cents
    p.cancel(order.order_id, reason="news")
    assert order.status == CANCELLED and p.available_cents > before


def test_settlement_pays_and_is_idempotent(tmp_path, book):
    p = portfolio(tmp_path)
    order = p.submit("c1", "kalshi", "I", "yes", 44, 10)
    p.fill_marketable(order.order_id, book)
    pnl = p.settle("I", "yes", "yes")
    assert pnl > 0 and len(p.ledger) == 1
    with pytest.raises(BrokerError, match="already settled"):
        p.settle("I", "yes", "yes")


def test_losing_settlement_costs_exactly_what_was_paid(tmp_path, book):
    p = portfolio(tmp_path)
    order = p.submit("c1", "kalshi", "I", "yes", 44, 10)
    p.fill_marketable(order.order_id, book)
    pnl = p.settle("I", "yes", "no")
    assert pnl == pytest.approx(-(order.avg_fill_price_cents * 10
                                  + order.fees_cents))


def test_pnl_uses_actual_fills_not_proposed_size(tmp_path):
    """A partially filled order must realise on what filled, not what was
    requested."""
    p = portfolio(tmp_path)
    thin = Book(yes_asks=((44, 3.0),), yes_bids=((43, 10.0),))
    order = p.submit("c1", "kalshi", "I", "yes", 44, 100)
    p.fill_marketable(order.order_id, thin)
    assert order.filled_size == 3.0
    pnl = p.settle("I", "yes", "yes")
    assert pnl == pytest.approx(100 * 3 - order.avg_fill_price_cents * 3
                                - order.fees_cents)


def test_portfolio_round_trips_through_disk(tmp_path, book):
    p = portfolio(tmp_path)
    order = p.submit("c1", "kalshi", "I", "yes", 44, 10)
    p.fill_marketable(order.order_id, book)
    path = p.save()
    again = PaperPortfolio.load(path)
    assert again.cash_cents == p.cash_cents
    assert len(again.orders) == 1 and len(again.positions) == 1


def test_summary_reports_fill_rate_and_realized(tmp_path, book):
    p = portfolio(tmp_path)
    order = p.submit("c1", "kalshi", "I", "yes", 44, 10)
    p.fill_marketable(order.order_id, book)
    p.settle("I", "yes", "yes")
    s = p.summary()
    assert s["n_settled"] == 1 and s["realized_pnl_usd"] > 0
    assert 0 < s["fill_rate"] <= 1


def test_a_crashed_member_records_why_it_crashed():
    """`claude -p --output-format json` writes its errors to STDOUT. Reading
    only stderr logged 37 consecutive failures as `member exited 1: ` with an
    empty cause -- 57% of slate chunks lost and no way to find out why."""
    import subprocess as sp

    from wc2026.board.orchestrator import BoardFailure, invoke_member

    class Proc:
        returncode = 1
        stdout = '{"type":"error","error":"usage limit reached"}'
        stderr = ""

    real = sp.run
    sp.run = lambda *a, **k: Proc()
    try:
        with pytest.raises(BoardFailure) as exc:
            invoke_member("a prompt", "claude-haiku-4-5")
    finally:
        sp.run = real
    message = str(exc.value)
    assert "usage limit reached" in message, "the cause must survive"
    assert "claude-haiku-4-5" in message, "which member failed"
    assert "8-char prompt" in message, "and how big the prompt was"


def test_a_crashed_member_says_so_even_with_no_output_at_all():
    import subprocess as sp

    from wc2026.board.orchestrator import BoardFailure, invoke_member

    class Silent:
        returncode = 137
        stdout = ""
        stderr = ""

    real = sp.run
    sp.run = lambda *a, **k: Silent()
    try:
        with pytest.raises(BoardFailure) as exc:
            invoke_member("p", "m")
    finally:
        sp.run = real
    assert "137" in str(exc.value)
    assert "no output on either stream" in str(exc.value)


# ── members that cannot run ───────────────────────────────────────────────────
# Measured 2026-08-28: 37 of 43 failed-closed sittings were the quant exiting 1
# with an empty message, because the CLI writes its errors to stdout. Two
# sessions burned 15 and 14 consecutive sittings in under a minute; a third ran
# 8 and then lost the next 8. That is an exhausted usage limit, not a verdict.
def _runner(results):
    """Stand in for `subprocess.run`, yielding one canned result per call."""
    seq = list(results)
    calls = []

    def run(*a, **k):
        calls.append(k.get("input") or (a[0] if a else None))
        return seq.pop(0) if len(seq) > 1 else seq[0]
    return run, calls


class _Proc:
    def __init__(self, code=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


def test_an_exhausted_usage_limit_is_not_a_crash(tmp_path):
    import subprocess as sp

    from wc2026.board import orchestrator as orch

    run, calls = _runner([_Proc(1, '{"is_error":true,'
                                   '"result":"Claude AI usage limit reached"}')])
    real = sp.run
    sp.run = run
    try:
        with pytest.raises(orch.BoardUnavailable):
            orch.invoke_member("p", "m")
    finally:
        sp.run = real
    assert len(calls) == 1, "an exhausted quota is never retried"


def test_a_transient_crash_is_retried_once(tmp_path):
    """A member that returns a VERDICT is never re-asked -- re-rolling until it
    says yes is how a board becomes decoration. One that never RAN has not
    decided anything, and failing closed silently drops its whole chunk."""
    import subprocess as sp

    from wc2026.board import orchestrator as orch

    run, calls = _runner([_Proc(1, "", "segfault"),
                          _Proc(0, '{"result":"{\\"ok\\": true}"}')])
    real = sp.run
    sp.run = run
    try:
        got = orch.invoke_member("p", "m")
    finally:
        sp.run = real
    assert got == {"ok": True}
    assert len(calls) == 2, "one retry, and it succeeded"


def test_a_member_that_keeps_crashing_still_fails_closed():
    import subprocess as sp

    from wc2026.board import orchestrator as orch

    run, calls = _runner([_Proc(1, "", "segfault")])
    real = sp.run
    sp.run = run
    try:
        with pytest.raises(orch.BoardFailure) as exc:
            orch.invoke_member("p", "m")
    finally:
        sp.run = real
    assert len(calls) == orch.MEMBER_ATTEMPTS
    assert "after %d attempts" % orch.MEMBER_ATTEMPTS in str(exc.value)


def test_an_unavailable_member_is_flagged_on_the_verdict(tmp_path):
    """The caller needs to tell 'this chunk broke' from 'every remaining call
    in this run will break', because the second means stop."""
    from wc2026.board import orchestrator as orch

    def gone(prompt, model, allowed_tools=(), timeout=None):
        raise orch.BoardUnavailable("member exited 1: usage limit reached")

    out = orch.run_board(make_case(), FIXTURE, invoke=gone,
                         log_path=str(tmp_path / "b.jsonl"))
    assert out["failed_closed"] and out["unavailable"] is True


def test_an_ordinary_crash_is_not_flagged_as_unavailable(tmp_path):
    from wc2026.board import orchestrator as orch

    def broke(prompt, model, allowed_tools=(), timeout=None):
        raise orch.BoardFailure("member exited 1: something else")

    out = orch.run_board(make_case(), FIXTURE, invoke=broke,
                         log_path=str(tmp_path / "b.jsonl"))
    assert out["failed_closed"] and out["unavailable"] is False
