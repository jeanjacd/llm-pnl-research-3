"""
gate.py
=======
The news-check gate: the final stage between "bets proposed" and "orders
placed". An embedded Claude instance with web-search access reviews each
proposed bet for the soft factors the quantitative model deliberately cannot
see -- injuries, suspensions, international call-ups (MLS plays THROUGH FIFA
windows), announced rotation (Leagues Cup / Open Cup congestion), managerial
changes, travel/altitude/heat, and late-season rest decisions.

Three architectural laws, enforced in code and covered by tests:

  1. ONE-DIRECTIONAL AUTHORITY. The gate can only shrink or kill a stake.
     Multipliers are clamped to [gate_multiplier_min, 1.0] regardless of what
     the response claims; verdicts cannot add bets, change sides, markets, or
     prices; unknown tickers in the response are rejected. A hallucinated
     finding costs at most a missed bet, never a bad one.

  2. FAIL CLOSED. If the Claude invocation errors, times out, or returns
     anything that fails schema validation, the affected bets are marked
     `unscreened` -- and unscreened bets are NEVER placed (pipeline filter).
     Recommend mode still displays them, clearly flagged. Malformed output is
     failure, not something to partially parse. There is no retry that ends in
     "place it anyway".

  3. AUDIT THE GATE ITSELF. Every verdict, rationale, source list, timestamp,
     and the pre/post stake goes to gate_audit.jsonl. Pre-gate stakes are kept
     on each bet so counterfactual P&L of vetoed/reduced bets is trackable
     (tracking.py), making the gate's own contribution to EV measurable -- and
     removable if it proves to be noise.

Durable findings (e.g. a confirmed multi-week injury) additionally emit a
SUGGESTED entry for data/mls/adjustments.json into a review file -- the gate
screens today's bets; the adjustments file is how confirmed knowledge feeds
the model properly, via a human.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess

from .bankroll import audit
from .config import BETTING_DIR, GATE_LOG, BettingConfig
from .ev import Candidate

SUGGESTIONS_LOG = os.path.join(BETTING_DIR, "adjustment_suggestions.jsonl")

VALID_VERDICTS = {"approve", "reduce", "veto"}


class GateFailure(RuntimeError):
    """Any reason the gate's output cannot be FULLY trusted. Fail closed."""


# --------------------------------------------------------------------------- #
# prompt & invocation
# --------------------------------------------------------------------------- #
def build_prompt(bets: list[dict]) -> str:
    return f"""You are the news-check gate of an MLS betting pipeline. The quantitative
model behind these bets is blind to recent soft news. For EACH proposed bet
below, research the match (web search) for: confirmed/probable injuries,
suspensions, international call-ups, announced or likely lineup rotation
(especially around Leagues Cup / US Open Cup congestion), managerial changes,
travel/congestion context, Denver altitude, extreme heat, and playoff-seeding
rest decisions. Judge only whether the found information UNDERMINES the bet.

Proposed bets (JSON):
{json.dumps(bets, indent=1, default=str)}

Respond with ONLY a JSON object, no prose, no code fences, exactly this shape:
{{"bets": [{{"ticker": "<ticker from the slate>",
            "verdict": "approve" | "reduce" | "veto",
            "multiplier": <number in [0.25, 1.0]; required iff verdict is "reduce">,
            "rationale": "<one or two concise sentences>",
            "sources": ["<url or source name>", ...]}}, ...],
 "suggested_adjustments": [{{"team": "<canonical team name>",
                             "attack_mult": <number>, "defense_mult": <number>,
                             "note": "<dated, sourced note>"}}, ...]}}

Rules: every proposed ticker must appear exactly once; you cannot add bets,
change sides or prices, or increase stakes -- your only powers are approve,
reduce (with multiplier), and veto. "suggested_adjustments" is optional and
only for DURABLE, confirmed information (e.g. a multi-week injury). If you find
nothing material for a bet, approve it with rationale "no material news found".
"""


def _gate_command(cfg: BettingConfig) -> list[str]:
    """The headless Claude command line, including JSON output and the web
    tools the research step requires (granted non-interactively via
    --allowedTools, else `claude -p` cannot search)."""
    cmd = list(cfg.gate_command) + ["--output-format", "json"]
    if cfg.gate_model:
        cmd += ["--model", cfg.gate_model]
    if cfg.gate_allowed_tools:
        cmd += ["--allowedTools", *cfg.gate_allowed_tools]
    return cmd


def invoke_claude(prompt: str, cfg: BettingConfig) -> str:
    """Run the headless Claude invocation. Returns the raw result text.
    Raises GateFailure on any error or timeout -- never retried."""
    cmd = _gate_command(cfg)
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True,
                              text=True, encoding="utf-8",
                              timeout=cfg.gate_timeout_seconds)
    except subprocess.TimeoutExpired as e:
        raise GateFailure(f"gate timed out after {cfg.gate_timeout_seconds}s") from e
    except OSError as e:
        raise GateFailure(f"gate invocation failed: {e}") from e
    if proc.returncode != 0:
        raise GateFailure(f"gate exited {proc.returncode}: {proc.stderr[:500]}")
    try:
        outer = json.loads(proc.stdout)
        return outer["result"] if isinstance(outer, dict) and "result" in outer \
            else proc.stdout
    except json.JSONDecodeError as e:
        raise GateFailure(f"gate wrapper output not JSON: {e}") from e


# --------------------------------------------------------------------------- #
# strict validation (malformed = failure, never partially parsed)
# --------------------------------------------------------------------------- #
def _extract_json_object(raw: str) -> str:
    """Locate the JSON payload in a model response. Cheaper models often wrap
    the object in prose and/or a ```json fence despite the instruction not to;
    we extract the object and still validate it strictly below (this is finding
    the payload, NOT salvaging malformed JSON -- a bad object still fails)."""
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_and_validate(raw: str, slate_tickers: list[str],
                       cfg: BettingConfig) -> dict:
    """Parse the gate response against the strict contract. Any deviation
    raises GateFailure. Returns {ticker: {verdict, multiplier, rationale,
    sources}} plus '_suggestions'."""
    try:
        data = json.loads(_extract_json_object(raw))
    except json.JSONDecodeError as e:
        raise GateFailure(f"gate response is not valid JSON: {e}")
    if not isinstance(data, dict) or "bets" not in data \
            or not isinstance(data["bets"], list):
        raise GateFailure("gate response missing 'bets' list")

    out: dict[str, dict] = {}
    for item in data["bets"]:
        if not isinstance(item, dict):
            raise GateFailure("bet entry is not an object")
        t = item.get("ticker")
        v = item.get("verdict")
        if t not in slate_tickers:
            raise GateFailure(f"unknown ticker in gate response: {t!r}")
        if t in out:
            raise GateFailure(f"duplicate ticker in gate response: {t!r}")
        if v not in VALID_VERDICTS:
            raise GateFailure(f"invalid verdict {v!r} for {t}")
        m = item.get("multiplier")
        if v == "reduce":
            if not isinstance(m, (int, float)) or not math.isfinite(m):
                raise GateFailure(f"reduce without numeric multiplier for {t}")
            # LAW 1: clamp in code regardless of the claimed value.
            m = min(max(float(m), cfg.gate_multiplier_min), cfg.gate_multiplier_max)
        elif v == "approve":
            m = 1.0
        else:                                   # veto
            m = 0.0
        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise GateFailure(f"missing rationale for {t}")
        sources = item.get("sources", [])
        if not isinstance(sources, list):
            raise GateFailure(f"sources must be a list for {t}")
        out[t] = {"verdict": v, "multiplier": m,
                  "rationale": rationale.strip(),
                  "sources": [str(s) for s in sources]}

    missing = set(slate_tickers) - set(out)
    if missing:
        raise GateFailure(f"gate response missing tickers: {sorted(missing)}")

    sugg = data.get("suggested_adjustments", [])
    out["_suggestions"] = sugg if isinstance(sugg, list) else []
    return out


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def screen_slate(slate: list[Candidate], cfg: BettingConfig,
                 _invoke=invoke_claude) -> list[Candidate]:
    """Apply the news-check gate to a sized slate. Mutates and returns it.

    On success: each candidate gets verdict/multiplier/rationale; stakes are
    shrunk (floor) or vetoed. On ANY failure: every candidate is marked
    'unscreened' and its contracts left intact for DISPLAY -- the pipeline's
    execute path refuses to place anything not approved/reduced (fail closed).
    """
    if not slate:
        return slate
    bets = [{"ticker": c.market.ticker, "match": f"{c.market.home} v {c.market.away}",
             "kickoff_utc": str(c.market.kickoff_utc), "claim": c.claim,
             "side": c.side, "model_probability": round(c.p_model, 3),
             "market_price_cents": c.ask_cents, "edge": round(c.edge_raw, 3),
             "proposed_contracts": c.contracts,
             "proposed_stake_usd": round(c.stake_cents / 100, 2)} for c in slate]
    tickers = [b["ticker"] for b in bets]

    try:
        raw = _invoke(build_prompt(bets), cfg)
        verdicts = parse_and_validate(raw, tickers, cfg)
    except GateFailure as e:
        audit("gate_failure", {"error": str(e), "n_bets": len(bets)}, path=GATE_LOG)
        for c in slate:
            c.gate_verdict = "unscreened"
            c.gate_rationale = f"gate failed closed: {e}"
        return slate

    for c in slate:
        v = verdicts[c.market.ticker]
        pre = c.contracts
        c.confidence_parts["pre_gate_contracts"] = pre
        c.gate_verdict = v["verdict"]
        c.gate_multiplier = v["multiplier"]
        c.gate_rationale = v["rationale"]
        # LAW 1 again at the application site: never increase.
        new = min(pre, math.floor(pre * v["multiplier"]))
        c.contracts = max(new, 0)
        c.stake_cents = 0 if c.contracts == 0 else c.stake_cents
        if c.contracts >= 1:
            from .fees import trading_fee_cents
            c.fee_cents = trading_fee_cents(c.contracts, c.ask_cents,
                                            cfg.taker_fee_factor)
            c.stake_cents = c.contracts * c.ask_cents + c.fee_cents
        audit("gate_verdict", {
            "ticker": c.market.ticker, "verdict": v["verdict"],
            "multiplier": v["multiplier"], "rationale": v["rationale"],
            "sources": v["sources"], "pre_contracts": pre,
            "post_contracts": c.contracts}, path=GATE_LOG)

    for s in verdicts.get("_suggestions", []):
        if isinstance(s, dict) and s.get("team"):
            audit("adjustment_suggestion", s, path=SUGGESTIONS_LOG)

    return slate


def placeable(slate: list[Candidate]) -> list[Candidate]:
    """The execute-mode filter: only screened, non-vetoed, sized bets. This is
    where 'unscreened bets are never placed' is enforced."""
    return [c for c in slate
            if c.gate_verdict in ("approve", "reduce") and c.contracts >= 1]
