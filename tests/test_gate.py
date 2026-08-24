"""The news-check gate's three architectural laws, proven:
1. one-directional authority (clamps; can only shrink),
2. fail-closed on error/timeout/malformed output,
3. full audit + counterfactual hooks."""
import json

import numpy as np
import pytest

import wc2026.betting.gate as gate_mod
from wc2026.betting.config import BettingConfig
from wc2026.betting.ev import Candidate
from wc2026.betting.gate import (
    GateFailure,
    build_prompt,
    parse_and_validate,
    placeable,
    screen_slate,
)
from wc2026.betting.markets import MappedMarket


@pytest.fixture(autouse=True)
def _redirect_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_mod, "GATE_LOG", str(tmp_path / "gate.jsonl"))
    monkeypatch.setattr(gate_mod, "SUGGESTIONS_LOG", str(tmp_path / "sugg.jsonl"))


CFG = BettingConfig()
T1, T2 = "KXMLSGAME-26JUL25SJLAG-SJ", "KXMLSGAME-26JUL25SJLAG-TIE"


def _cand(ticker=T1, contracts=8):
    ind = np.zeros((13, 13), dtype=bool)
    ind[1, 0] = True
    mkt = MappedMarket(ticker=ticker, event_ticker="E", series="KXMLSGAME",
                       title="t", sub_title="s", home="San Jose Earthquakes",
                       away="LA Galaxy", kickoff_utc=None, claim="home_win",
                       indicator=ind)
    c = Candidate(market=mkt, side="yes", claim="home_win", p_model=0.5,
                  p_calibrated=0.5, ask_cents=40, depth_at_touch=100,
                  spread_cents=2, edge_raw=0.06, edge_calibrated=0.06)
    c.contracts = contracts
    c.stake_cents = contracts * 40 + 3
    c.fee_cents = 3
    return c


def _resp(bets, suggestions=None):
    d = {"bets": bets}
    if suggestions is not None:
        d["suggested_adjustments"] = suggestions
    return json.dumps(d)


def _entry(ticker=T1, verdict="approve", multiplier=None, rationale="ok",
           sources=("espn.com",)):
    e = {"ticker": ticker, "verdict": verdict, "rationale": rationale,
         "sources": list(sources)}
    if multiplier is not None:
        e["multiplier"] = multiplier
    return e


# --------------------------------------------------------------------------- #
# contract validation
# --------------------------------------------------------------------------- #
def test_valid_response_parses():
    v = parse_and_validate(_resp([_entry()]), [T1], CFG)
    assert v[T1]["verdict"] == "approve" and v[T1]["multiplier"] == 1.0


def test_malformed_json_fails():
    with pytest.raises(GateFailure):
        parse_and_validate("I think these bets look fine!", [T1], CFG)


def test_json_extracted_from_prose_and_fences():
    """Cheaper models wrap the JSON in prose / a ```json fence -- the payload
    must still be found and validated (Haiku behaviour, verified 2026-07-23)."""
    body = _resp([_entry(T1, "veto", rationale="star striker out")])
    # leading prose + a fenced block, exactly like a chatty model
    wrapped = f"Let me check the fixture first.\n\n```json\n{body}\n```"
    v = parse_and_validate(wrapped, [T1], CFG)
    assert v[T1]["verdict"] == "veto"
    # prose + bare object (no fence) also works
    v2 = parse_and_validate("Here is my verdict: " + body, [T1], CFG)
    assert v2[T1]["verdict"] == "veto"


def test_prose_without_any_json_still_fails_closed():
    with pytest.raises(GateFailure):
        parse_and_validate("I could not complete the research. Please advise.",
                           [T1], CFG)


def test_missing_ticker_fails():
    with pytest.raises(GateFailure, match="missing tickers"):
        parse_and_validate(_resp([_entry(T1)]), [T1, T2], CFG)


def test_unknown_ticker_fails():
    with pytest.raises(GateFailure, match="unknown ticker"):
        parse_and_validate(_resp([_entry("KXFAKE-1")]), [T1], CFG)


def test_duplicate_ticker_fails():
    with pytest.raises(GateFailure, match="duplicate"):
        parse_and_validate(_resp([_entry(T1), _entry(T1)]), [T1], CFG)


def test_invalid_verdict_fails():
    with pytest.raises(GateFailure, match="invalid verdict"):
        parse_and_validate(_resp([_entry(verdict="double down")]), [T1], CFG)


def test_reduce_without_multiplier_fails():
    with pytest.raises(GateFailure, match="without numeric multiplier"):
        parse_and_validate(_resp([_entry(verdict="reduce")]), [T1], CFG)


def test_missing_rationale_fails():
    with pytest.raises(GateFailure):
        parse_and_validate(_resp([_entry(rationale="")]), [T1], CFG)


def test_out_of_bounds_multipliers_are_clamped_in_code():
    # A 'reduce' claiming 3.0x is clamped to 1.0 -- the gate cannot increase.
    v = parse_and_validate(_resp([_entry(verdict="reduce", multiplier=3.0)]),
                           [T1], CFG)
    assert v[T1]["multiplier"] == CFG.gate_multiplier_max == 1.0
    # below the floor clamps up to the floor (0.25)
    v = parse_and_validate(_resp([_entry(verdict="reduce", multiplier=0.01)]),
                           [T1], CFG)
    assert v[T1]["multiplier"] == CFG.gate_multiplier_min
    # NaN is rejected outright
    with pytest.raises(GateFailure):
        parse_and_validate(_resp([_entry(verdict="reduce", multiplier=float("nan"))]),
                           [T1], CFG)


# --------------------------------------------------------------------------- #
# law 1: one-directional authority
# --------------------------------------------------------------------------- #
def test_stakes_can_only_shrink_never_grow():
    slate = [_cand(T1, contracts=8), _cand(T2, contracts=8)]
    resp = _resp([_entry(T1, "reduce", multiplier=0.5, rationale="rotation risk"),
                  _entry(T2, "approve", multiplier=99.0)])  # 99 ignored
    out = screen_slate(slate, CFG, _invoke=lambda p, c: resp)
    assert out[0].contracts == 4                    # floor(8 * 0.5)
    assert out[1].contracts == 8                    # approve never increases
    assert all(c.contracts <= 8 for c in out)
    assert out[0].confidence_parts["pre_gate_contracts"] == 8


def test_veto_zeroes_the_stake():
    slate = [_cand(T1)]
    resp = _resp([_entry(T1, "veto", rationale="starting XI rested per beat writer")])
    out = screen_slate(slate, CFG, _invoke=lambda p, c: resp)
    assert out[0].contracts == 0 and out[0].stake_cents == 0
    assert placeable(out) == []


# --------------------------------------------------------------------------- #
# law 2: fail closed
# --------------------------------------------------------------------------- #
def _screen_with_failure(slate, exc):
    def boom(prompt, cfg):
        raise exc
    return screen_slate(slate, CFG, _invoke=boom)


def test_invocation_error_fails_closed():
    out = _screen_with_failure([_cand(T1)], GateFailure("claude exploded"))
    assert out[0].gate_verdict == "unscreened"
    assert placeable(out) == []                     # never placed


def test_timeout_fails_closed():
    out = _screen_with_failure([_cand(T1)], GateFailure("gate timed out after 600s"))
    assert out[0].gate_verdict == "unscreened"
    assert placeable(out) == []


def test_malformed_output_fails_closed_not_partially_parsed():
    # T1's entry is valid, T2's is garbage -> NEITHER is trusted.
    slate = [_cand(T1), _cand(T2)]
    resp = json.dumps({"bets": [_entry(T1), {"ticker": T2, "verdict": "??"}]})
    out = screen_slate(slate, CFG, _invoke=lambda p, c: resp)
    assert all(c.gate_verdict == "unscreened" for c in out)
    assert placeable(out) == []


def test_unscreened_bets_stay_visible_for_recommend_mode():
    out = _screen_with_failure([_cand(T1, contracts=8)], GateFailure("x"))
    assert out[0].contracts == 8                    # display intact...
    assert placeable(out) == []                     # ...but unplaceable


# --------------------------------------------------------------------------- #
# law 3: audit + suggestions
# --------------------------------------------------------------------------- #
def test_every_verdict_is_audited(tmp_path):
    slate = [_cand(T1)]
    resp = _resp([_entry(T1, "reduce", multiplier=0.5, rationale="two starters on "
                         "international duty", sources=["mlssoccer.com"])])
    screen_slate(slate, CFG, _invoke=lambda p, c: resp)
    lines = [json.loads(x) for x in
             open(gate_mod.GATE_LOG, encoding="utf-8").read().splitlines()]
    v = [x for x in lines if x["event"] == "gate_verdict"]
    assert len(v) == 1
    assert v[0]["rationale"].startswith("two starters")
    assert v[0]["sources"] == ["mlssoccer.com"]
    assert v[0]["pre_contracts"] == 8 and v[0]["post_contracts"] == 4


def test_durable_findings_emit_adjustment_suggestions(tmp_path):
    slate = [_cand(T1)]
    resp = _resp([_entry(T1)],
                 suggestions=[{"team": "LA Galaxy", "attack_mult": 0.9,
                               "defense_mult": 1.0,
                               "note": "2026-07-20: striker out 6 weeks (club)"}])
    screen_slate(slate, CFG, _invoke=lambda p, c: resp)
    lines = [json.loads(x) for x in
             open(gate_mod.SUGGESTIONS_LOG, encoding="utf-8").read().splitlines()]
    assert lines[0]["team"] == "LA Galaxy"


# --------------------------------------------------------------------------- #
# prompt sanity
# --------------------------------------------------------------------------- #
def test_prompt_contains_slate_and_contract():
    p = build_prompt([{"ticker": T1, "match": "A v B"}])
    assert T1 in p and '"verdict"' in p and "veto" in p
    assert "cannot add bets" in p


def test_gate_command_grants_web_tools():
    """Headless `claude -p` must be handed WebSearch/WebFetch, or it cannot
    research and fails closed (regression for the real 2026-07-23 finding)."""
    from wc2026.betting.gate import _gate_command
    cmd = _gate_command(CFG)
    assert cmd[:2] == ["claude", "-p"]
    assert "--output-format" in cmd and "json" in cmd
    assert "--allowedTools" in cmd
    tail = cmd[cmd.index("--allowedTools"):]
    assert "WebSearch" in tail and "WebFetch" in tail
    # cost-effective model pinned for the gate task
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "claude-haiku-4-5"
