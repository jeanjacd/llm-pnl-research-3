"""One board sitting per fixture, a verdict per market.

Boarding a single market per fixture discarded the rest: a quant PASS on one
price closed the whole match, though it said nothing about the totals or the
scorelines on it. The unit of MEASUREMENT is the fixture; the unit of ACTION is
the market. These tests pin that separation, and the safety rules that must
survive it -- a member may still only tighten a price, never raise one.
"""
import datetime as dt
import json

import pytest

from wc2026.board import orchestrator as orch
from wc2026.board.schemas import (
    SchemaError,
    quant_slate_packet,
    validate_judge_slate,
    validate_quant_slate,
)
from wc2026.decision import build_case
from wc2026.venues.base import KIND_BINARY, Book, Leg, MarketInstrument, utcnow_iso


def soon(hours=20):
    return (dt.datetime.now(dt.timezone.utc)
            + dt.timedelta(hours=hours)).isoformat()


def a_case(claim="home_win", ask=20, ident=None):
    ident = ident or claim
    book = Book(observed_at=utcnow_iso(), yes_asks=((ask, 2000),),
                yes_bids=((ask - 1, 2000),))
    inst = MarketInstrument(
        venue="kalshi", instrument_id=ident, kind=KIND_BINARY, title=ident,
        legs=(Leg.build(claim, ident, home="A", away="B", league_id="mls",
                        kickoff_utc=soon()),),
        league_id="mls", kickoff_utc=soon(), book=book,
        fee_model={"venue": "kalshi"})
    return build_case(inst, 0.55, 0.55, side="yes")


FIXTURE = {"home": "A", "away": "B", "league_id": "mls", "kickoff_utc": soon()}


def a_coach(verdict="ACCEPT", case_id="kalshi:home_win:yes"):
    return {"case_id": case_id, "verdict": verdict, "rationale": "ok",
            "evidence_cutoff": utcnow_iso(), "findings": [],
            "required_reruns": [], "sources": []}


class Invoker:
    """Replays canned member responses and records the prompts it was given."""

    def __init__(self, quant=None, coach=None, judge=None):
        self.quant, self.coach, self.judge = quant, coach, judge
        self.prompts = []

    def __call__(self, prompt, model, allowed_tools=(), timeout=None):
        self.prompts.append(prompt)
        if "SOCCER ANALYST" in prompt:
            return self.coach
        if "JUDGE" in prompt:
            return self.judge
        return self.quant


def quant_yes(cases, price=None, action="PLACE_LIMIT"):
    return {"decisions": [
        {"case_id": c.case_id, "action": action,
         "proposed_price_cents": price or c.max_limit_price_cents,
         "contracts": 10.0, "rationale": "ok", "counterarguments": [],
         "veto_codes": []} for c in cases]}


def judge_yes(cases, price=None, action="PAPER_PLACE_LIMIT"):
    return {"decisions": [
        {"case_id": c.case_id, "action": action,
         "limit_price_cents": price or c.max_limit_price_cents,
         "contracts": 10.0, "decisive_reason": "ok",
         "strongest_counterpoint": "none", "veto_codes": []} for c in cases]}


# --- the point of the change --------------------------------------------------
def test_every_market_on_a_fixture_gets_its_own_verdict():
    cases = [a_case("home_win"), a_case("btts"), a_case("total_over_2.5")]
    inv = Invoker(quant_yes(cases), a_coach(case_id=cases[0].case_id),
                  judge_yes(cases))
    out = orch.run_board_slate(cases, FIXTURE, invoke=inv)
    assert len(out["decisions"]) == 3
    assert {d["action"] for d in out["decisions"].values()} == {
        "PAPER_PLACE_LIMIT"}


def test_one_market_being_passed_does_not_close_the_fixture():
    """The exact failure this replaces: a PASS on one price closed the match."""
    cases = [a_case("home_win"), a_case("btts")]
    quant = quant_yes(cases)
    quant["decisions"][0]["action"] = "PASS"
    quant["decisions"][0].pop("proposed_price_cents")
    quant["decisions"][0].pop("contracts")
    inv = Invoker(quant, a_coach(case_id=cases[0].case_id), judge_yes([cases[1]]))
    out = orch.run_board_slate(cases, FIXTURE, invoke=inv)
    assert list(out["decisions"]) == [cases[1].case_id]
    assert out["decisions"][cases[1].case_id]["action"] == "PAPER_PLACE_LIMIT"


def test_the_whole_fixture_costs_three_member_calls_not_three_per_market():
    cases = [a_case("home_win"), a_case("btts"), a_case("draw"),
             a_case("total_over_2.5")]
    inv = Invoker(quant_yes(cases), a_coach(case_id=cases[0].case_id), judge_yes(cases))
    orch.run_board_slate(cases, FIXTURE, invoke=inv)
    assert len(inv.prompts) == 3, "quant, coach, judge -- once each"


# --- safety rules that must survive the change --------------------------------
def test_a_member_may_still_never_raise_a_price():
    case = a_case("home_win")
    ceiling = case.max_limit_price_cents
    cases = [case]
    inv = Invoker(quant_yes(cases, price=ceiling + 25), a_coach(),
                  judge_yes(cases, price=ceiling + 40))
    out = orch.run_board_slate(cases, FIXTURE, invoke=inv)
    assert out["decisions"][case.case_id]["limit_price_cents"] <= ceiling


def test_a_market_the_quant_refused_is_never_shown_to_the_judge():
    cases = [a_case("home_win"), a_case("btts")]
    quant = quant_yes(cases)
    quant["decisions"][1]["action"] = "PASS"
    quant["decisions"][1].pop("proposed_price_cents")
    quant["decisions"][1].pop("contracts")
    inv = Invoker(quant, a_coach(case_id=cases[0].case_id),
                  judge_yes([cases[0]]))
    out = orch.run_board_slate(cases, FIXTURE, invoke=inv)
    judge_prompt = next(p for p in inv.prompts if "JUDGE" in p)
    assert cases[1].case_id not in judge_prompt
    assert cases[1].case_id not in out["decisions"]


def test_a_judge_that_invents_an_approval_fails_the_whole_slate_closed():
    """Fails CLOSED rather than raising, and rather than partly applying: an
    unrecognised case_id means the response cannot be trusted at all."""
    cases = [a_case("home_win"), a_case("btts")]
    quant = quant_yes(cases)
    quant["decisions"][1]["action"] = "PASS"
    quant["decisions"][1].pop("proposed_price_cents")
    quant["decisions"][1].pop("contracts")
    # The judge answers for BOTH, including the one the quant refused.
    inv = Invoker(quant, a_coach(case_id=cases[0].case_id), judge_yes(cases))
    out = orch.run_board_slate(cases, FIXTURE, invoke=inv)
    assert out["decisions"] == {}
    assert out["failed_closed"] is True
    assert "judge" in out["failure"]


def test_a_coach_veto_stops_the_entire_slate():
    """The analyst's objection is to the MATCH, not to one price on it."""
    cases = [a_case("home_win"), a_case("btts")]
    inv = Invoker(quant_yes(cases), a_coach("REJECT", case_id=cases[0].case_id), judge_yes(cases))
    out = orch.run_board_slate(cases, FIXTURE, invoke=inv)
    assert out["decisions"] == {}
    assert "coach veto" in out["failure"]
    assert out["failed_closed"] is False, "a veto is a decision, not a breakage"


def test_a_coach_rerun_request_defers_the_entire_slate():
    cases = [a_case("home_win")]
    coach = a_coach(case_id=cases[0].case_id)
    coach["required_reruns"] = ["rerun without the suspended centre-back"]
    inv = Invoker(quant_yes(cases), coach, judge_yes(cases))
    out = orch.run_board_slate(cases, FIXTURE, invoke=inv)
    assert out["decisions"] == {} and "rerun" in out["failure"]


def test_a_broken_member_fails_closed_and_says_so():
    cases = [a_case("home_win")]

    def explode(prompt, model, allowed_tools=(), timeout=None):
        raise orch.BoardFailure("member invocation failed: no such file")

    out = orch.run_board_slate(cases, FIXTURE, invoke=explode)
    assert out["decisions"] == {}
    assert out["failed_closed"] is True


def test_the_coach_is_still_never_shown_a_price():
    cases = [a_case("home_win", ask=37)]
    inv = Invoker(quant_yes(cases), a_coach(case_id=cases[0].case_id), judge_yes(cases))
    orch.run_board_slate(cases, FIXTURE, invoke=inv)
    coach_prompt = next(p for p in inv.prompts if "SOCCER ANALYST" in p)
    for leaked in ("37", "ev_per_contract", "breakeven", "max_limit"):
        assert leaked not in coach_prompt, leaked


def test_the_coach_is_asked_once_per_fixture_not_once_per_market():
    cases = [a_case("home_win"), a_case("btts")]
    cache = {}
    inv = Invoker(quant_yes(cases), a_coach(case_id=cases[0].case_id), judge_yes(cases))
    orch.run_board_slate(cases, FIXTURE, invoke=inv, coach_cache=cache)
    orch.run_board_slate(cases, FIXTURE, invoke=inv, coach_cache=cache)
    assert sum(1 for p in inv.prompts if "SOCCER ANALYST" in p) == 1


# --- prompt content -----------------------------------------------------------
def test_the_quant_is_told_to_prefer_resting_to_crossing():
    """The difference between a profitable book and a losing one is mostly
    whether it pays the spread."""
    prompt = " ".join(orch.build_quant_slate_prompt(
        quant_slate_packet([a_case()]), FIXTURE).split())
    assert "PREFER A RESTING LIMIT TO CROSSING THE SPREAD" in prompt
    assert "BUY_NOW only when the touch itself clears the hurdle" in prompt


def test_the_judge_is_warned_that_one_fixture_is_one_risk():
    prompt = " ".join(orch.build_judge_slate_prompt(
        quant_slate_packet([a_case()]), {}, a_coach(), FIXTURE).split())
    assert "win or lose together" in prompt
    assert "Prefer a resting limit to crossing the spread" in prompt


def test_the_slate_packet_carries_the_numbers_and_not_prose():
    rows = quant_slate_packet([a_case("home_win"), a_case("btts")])
    assert len(rows) == 2
    for row in rows:
        for key in ("case_id", "claim", "p_lower", "breakeven_prob",
                    "ev_per_contract_cents", "max_limit_price_cents"):
            assert key in row, key


# --- validation ---------------------------------------------------------------
def test_an_unknown_case_id_is_rejected_not_ignored():
    ceilings = {"known": {"price": 40, "size": None}}
    payload = {"decisions": [{"case_id": "invented", "action": "PASS",
                              "rationale": "x"}]}
    with pytest.raises(SchemaError):
        validate_quant_slate(payload, ceilings)


def test_a_duplicated_case_id_is_rejected():
    case = a_case()
    ceilings = {case.case_id: {"price": 40, "size": None}}
    twice = quant_yes([case])["decisions"] * 2
    with pytest.raises(SchemaError):
        validate_quant_slate({"decisions": twice}, ceilings)


def test_a_market_the_member_omitted_is_simply_absent():
    """Silence is never consent: an omitted market yields no verdict at all."""
    cases = [a_case("home_win"), a_case("btts")]
    ceilings = {c.case_id: {"price": c.max_limit_price_cents, "size": None}
                for c in cases}
    only_first = {"decisions": quant_yes(cases)["decisions"][:1]}
    out = validate_quant_slate(only_first, ceilings)
    assert list(out) == [cases[0].case_id]


def test_a_malformed_slate_raises_rather_than_partially_applying():
    case = a_case()
    ceilings = {case.case_id: {"price": 40, "size": None}}
    with pytest.raises(SchemaError):
        validate_judge_slate({"decisions": "not a list"}, ceilings)
    with pytest.raises(SchemaError):
        validate_judge_slate(json.loads("{}"), ceilings)
