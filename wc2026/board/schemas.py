"""
board/schemas.py
================
Versioned response contracts for the three-member board, and the packet each
member is allowed to see.

Every schema is validated in code before anything is persisted or acted on.
Malformed output is a FAILURE, never something to partially parse: a response
that fails validation produces DEFER, and DEFER never becomes a trade.

The coach's packet is REDACTED. It must not contain price, EV, Kelly size, the
quantitative verdict, or a market-implied probability, because a coach who can
see the number being defended stops being an independent check on the sporting
assumptions and starts rationalising it.
"""
from __future__ import annotations

import math

PROMPT_VERSION = "board/v1"
SCHEMA_VERSION = "board-schema/v1"

# --- allowed verdicts ---------------------------------------------------------
QUANT_ACTIONS = ("BUY_NOW", "PLACE_LIMIT", "WAIT_FOR_QUOTE", "PASS", "DEFER",
                 "UNSUPPORTED")
COACH_VERDICTS = ("ACCEPT", "REJECT", "DEFER", "UNSUPPORTED")
JUDGE_ACTIONS = ("PAPER_BUY_NOW", "PAPER_PLACE_LIMIT", "WAIT_REPRICE", "PASS",
                 "DEFER", "UNSUPPORTED")
# Only these may reach the paper broker.
JUDGE_PLACEABLE = ("PAPER_BUY_NOW", "PAPER_PLACE_LIMIT")

# Fields the coach may never see.
REDACTED_FIELDS = ("touch_cents", "executable_avg_cents", "worst_cents",
                   "spread_cents", "ev_per_contract", "roi",
                   "uncertainty_adjusted_ev", "max_limit_price_cents",
                   "ladder", "breakeven_prob", "action", "reasons",
                   "market_implied_probability", "p_raw", "p_calibrated",
                   "p_lower", "p_upper", "adverse_selection_cents",
                   "slippage_cents")


class SchemaError(ValueError):
    """A board response violated its contract. Always fails closed."""


def _require(condition, message):
    if not condition:
        raise SchemaError(message)


def _finite_number(value, name):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             "%s must be a number" % name)
    _require(math.isfinite(float(value)), "%s must be finite" % name)
    return float(value)


# --- packets ------------------------------------------------------------------
def quant_packet(case, extra: dict | None = None) -> dict:
    """The full numeric packet: the quant member audits the arithmetic."""
    payload = case.as_dict()
    payload.update(extra or {})
    payload["prompt_version"] = PROMPT_VERSION
    return payload


def coach_packet(case, fixture: dict, extra: dict | None = None) -> dict:
    """The price-redacted packet handed to the soccer analyst."""
    payload = case.as_dict()
    for field in REDACTED_FIELDS:
        payload.pop(field, None)
    payload.update(extra or {})
    payload.update(fixture)
    payload["prompt_version"] = PROMPT_VERSION
    leaked = [f for f in REDACTED_FIELDS if f in payload]
    _require(not leaked, "coach packet leaked redacted fields: %s" % leaked)
    return payload


def assert_no_price_leak(packet: dict) -> None:
    """Belt-and-braces check used by tests and by the orchestrator."""
    leaked = [f for f in REDACTED_FIELDS if f in packet]
    if leaked:
        raise SchemaError("coach packet contains price/verdict fields: %s"
                          % leaked)


# --- response validation ------------------------------------------------------
def validate_quant(payload: dict, case_id: str, ceiling_price: int | None,
                   ceiling_size: float | None) -> dict:
    """Validate the quantitative member's response.

    Hard rule: the member may CONFIRM OR REDUCE the deterministic ceilings and
    may never raise them. Anything above the code-computed maximum is clamped
    down, not honoured.
    """
    _require(isinstance(payload, dict), "quant response must be an object")
    _require(payload.get("case_id") == case_id,
             "quant case_id mismatch: %r" % payload.get("case_id"))
    action = payload.get("action")
    _require(action in QUANT_ACTIONS, "invalid quant action %r" % action)

    out = {"case_id": case_id, "action": action,
           "rationale": str(payload.get("rationale") or "").strip(),
           "veto_codes": list(payload.get("veto_codes") or []),
           "counterarguments": list(payload.get("counterarguments") or []),
           "prompt_version": PROMPT_VERSION,
           "schema_version": SCHEMA_VERSION}
    _require(out["rationale"], "quant rationale is required")

    if action in ("BUY_NOW", "PLACE_LIMIT"):
        price = _finite_number(payload.get("proposed_price_cents"),
                               "proposed_price_cents")
        _require(0 < price < 100, "proposed price must be in 1..99")
        size = _finite_number(payload.get("contracts"), "contracts")
        _require(size > 0, "contracts must be positive")
        if ceiling_price is not None:
            price = min(price, ceiling_price)     # may only shrink
        if ceiling_size is not None:
            size = min(size, ceiling_size)
        out["proposed_price_cents"] = int(price)
        out["contracts"] = float(size)
    return out


def validate_coach(payload: dict, case_id: str) -> dict:
    """Validate the coach's response.

    The coach reports EVIDENCE and a verdict. It may not assign probabilities
    or multipliers -- if the simulation is missing something material, it must
    request an explicit rerun scenario instead of inventing a number.
    """
    _require(isinstance(payload, dict), "coach response must be an object")
    _require(payload.get("case_id") == case_id,
             "coach case_id mismatch: %r" % payload.get("case_id"))
    verdict = payload.get("verdict")
    _require(verdict in COACH_VERDICTS, "invalid coach verdict %r" % verdict)

    findings = payload.get("findings") or []
    _require(isinstance(findings, list), "findings must be a list")
    clean_findings = []
    for item in findings:
        _require(isinstance(item, dict), "each finding must be an object")
        status = item.get("evidence_status")
        _require(status in ("confirmed", "supported_inference", "uncertain",
                            "speculation"),
                 "invalid evidence_status %r" % status)
        modelled = item.get("in_simulation")
        _require(modelled in ("included", "partial", "absent", "unknown"),
                 "invalid in_simulation %r" % modelled)
        _require(str(item.get("text") or "").strip(), "finding text required")
        clean_findings.append({
            "text": str(item["text"]).strip(),
            "evidence_status": status,
            "in_simulation": modelled,
            "sources": [str(s) for s in (item.get("sources") or [])],
        })

    # A coach must not smuggle in a number.
    for banned in ("probability", "multiplier", "xg_multiplier", "fair_value",
                   "edge"):
        _require(banned not in payload,
                 "coach may not assign %r; request a rerun instead" % banned)

    return {
        "case_id": case_id, "verdict": verdict,
        "rationale": str(payload.get("rationale") or "").strip(),
        "findings": clean_findings,
        "required_reruns": [str(r) for r in (payload.get("required_reruns") or [])],
        "evidence_cutoff": payload.get("evidence_cutoff"),
        "sources": [str(s) for s in (payload.get("sources") or [])],
        "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION,
    }


def validate_judge(payload: dict, case_id: str, ceiling_price: int | None,
                   ceiling_size: float | None) -> dict:
    """Validate the judge's response.

    Hard rules enforced here regardless of what the judge claims:
      * price may only stay the same or become MORE conservative;
      * size may only stay the same or DECREASE;
      * only PAPER_BUY_NOW / PAPER_PLACE_LIMIT can reach the broker.
    """
    _require(isinstance(payload, dict), "judge response must be an object")
    _require(payload.get("case_id") == case_id,
             "judge case_id mismatch: %r" % payload.get("case_id"))
    action = payload.get("action")
    _require(action in JUDGE_ACTIONS, "invalid judge action %r" % action)

    out = {"case_id": case_id, "action": action,
           "decisive_reason": str(payload.get("decisive_reason") or "").strip(),
           "strongest_counterpoint":
               str(payload.get("strongest_counterpoint") or "").strip(),
           "veto_codes": list(payload.get("veto_codes") or []),
           "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION}
    _require(out["decisive_reason"], "judge decisive_reason is required")

    if action in JUDGE_PLACEABLE:
        price = _finite_number(payload.get("limit_price_cents"),
                               "limit_price_cents")
        size = _finite_number(payload.get("contracts"), "contracts")
        _require(0 < price < 100, "limit price must be in 1..99")
        _require(size > 0, "contracts must be positive")
        if ceiling_price is not None:
            price = min(price, ceiling_price)     # never richer
        if ceiling_size is not None:
            size = min(size, ceiling_size)        # never larger
        out["limit_price_cents"] = int(price)
        out["contracts"] = float(size)
        out["max_loss_cents"] = int(price) * float(size)
    return out


# --- slate: one board call for a whole fixture --------------------------------
# Boarding ONE market per fixture threw away every other market on it. A quant
# PASS on `not_away_win` being 1.22pp short of breakeven says nothing about the
# totals or the scorelines on the same match, yet it closed all of them.
#
# The unit of MEASUREMENT is still the fixture -- 34 bets on one match are one
# observation, and the statistics cluster on that. But the unit of ACTION is the
# market. Conflating the two was the error; the fix is cluster-robust stats
# (`eval.backtest.bootstrap_ci` already does exactly this for the model), not
# refusing to trade.
def quant_slate_packet(cases) -> list:
    """Compact per-market rows: the whole ladder in one prompt, not one market.

    Deliberately not the full per-case packet -- 34 of those would not fit, and
    the quant is auditing relative attractiveness across a fixture, for which
    the computed numbers are what matter.
    """
    rows = []
    for case in cases:
        rows.append({
            "case_id": case.case_id,
            "claim": getattr(case, "claim", None),
            "side": case.side,
            "venue": case.venue,
            "action": case.action,
            "touch_cents": case.touch_cents,
            "executable_avg_cents": case.executable_avg_cents,
            "depth_at_touch": case.depth_at_touch,
            "p_lower": None if case.p_lower is None else round(case.p_lower, 4),
            "breakeven_prob": None if case.breakeven_prob is None
                              else round(case.breakeven_prob, 4),
            "ev_per_contract_cents": None if case.ev_per_contract is None
                                     else round(case.ev_per_contract, 2),
            "roi": None if case.roi is None else round(case.roi, 4),
            "max_limit_price_cents": case.max_limit_price_cents,
            "adverse_selection_cents": round(case.adverse_selection_cents, 2),
            "ladder_rungs": len(case.ladder or []),
        })
    return rows


def _validate_slate(payload, key, ceilings, per_item, what):
    """Shared shape check for a slate response: one entry per case_id.

    A member that omits a case is not treated as approving it -- anything not
    explicitly returned is absent, and the caller treats absence as no verdict.
    """
    _require(isinstance(payload, dict), "%s response must be an object" % what)
    items = payload.get(key)
    _require(isinstance(items, list), "%s.%s must be a list" % (what, key))
    out, seen = {}, set()
    for item in items:
        _require(isinstance(item, dict), "%s entry must be an object" % what)
        case_id = item.get("case_id")
        _require(case_id in ceilings, "%s: unknown case_id %r" % (what, case_id))
        _require(case_id not in seen, "%s: duplicate case_id %r" % (what, case_id))
        seen.add(case_id)
        out[case_id] = per_item(item, case_id, ceilings[case_id])
    return out


def validate_quant_slate(payload: dict, ceilings: dict) -> dict:
    """case_id -> validated quant verdict. Same never-raise rule, per market."""
    def one(item, case_id, ceiling):
        return validate_quant(dict(item, case_id=case_id), case_id,
                              ceiling.get("price"), ceiling.get("size"))
    return _validate_slate(payload, "decisions", ceilings, one, "quant slate")


def validate_judge_slate(payload: dict, ceilings: dict) -> dict:
    """case_id -> validated judge verdict. Price/size may only tighten."""
    def one(item, case_id, ceiling):
        return validate_judge(dict(item, case_id=case_id), case_id,
                              ceiling.get("price"), ceiling.get("size"))
    return _validate_slate(payload, "decisions", ceilings, one, "judge slate")
