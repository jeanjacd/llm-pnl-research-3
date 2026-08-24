"""
bankroll.py
===========
Bankroll and position state, loss-limit accounting, and the append-only audit
log. All money is tracked in integer cents internally; the state file is JSON,
the audit log JSONL (one decision per line, timestamped, never rewritten).

Loss limits work on REALIZED P&L recorded in the ledger: before any order is
placed, the day's and week's realized losses are computed and compared against
the configured limits; a breach HALTS all trading (checked in pipeline.py, and
covered by tests).
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field

# AUDIT_LOG is resolved at CALL time via this module's attribute (so tests
# can redirect it), which static analysis cannot see -- hence the noqa.
from .config import AUDIT_LOG, STATE_JSON, BettingConfig  # noqa: F401


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def audit(event: str, payload: dict, path: str | None = None) -> None:
    """Append one audit record. Every decision that touches money goes here.
    The path resolves at call time so tests can redirect the log."""
    import wc2026.betting.bankroll as _self
    path = path or _self.AUDIT_LOG
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"ts": _utcnow().isoformat(), "event": event, **payload}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


@dataclass
class Position:
    ticker: str
    side: str
    contracts: int
    avg_cost_cents: float
    opened_at: str


@dataclass
class BettingState:
    bankroll_cents: int
    positions: list = field(default_factory=list)          # [Position dicts]
    ledger: list = field(default_factory=list)             # realized P&L events
    updated_at: str = ""

    path: str = STATE_JSON

    # ---------- persistence ----------
    @classmethod
    def load(cls, cfg: BettingConfig, path: str = STATE_JSON) -> "BettingState":
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return cls(bankroll_cents=int(d["bankroll_cents"]),
                       positions=d.get("positions", []),
                       ledger=d.get("ledger", []),
                       updated_at=d.get("updated_at", ""), path=path)
        return cls(bankroll_cents=int(round(cfg.default_bankroll_usd * 100)),
                   path=path)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.updated_at = _utcnow().isoformat()
        d = asdict(self)
        d.pop("path", None)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    # ---------- accounting ----------
    def record_fill(self, ticker: str, side: str, contracts: int,
                    price_cents: int, fee_cents: int) -> None:
        cost = contracts * price_cents + fee_cents
        self.bankroll_cents -= cost
        self.positions.append(asdict(Position(
            ticker=ticker, side=side, contracts=contracts,
            avg_cost_cents=(cost / contracts), opened_at=_utcnow().isoformat())))

    def record_settlement(self, ticker: str, result: str) -> int:
        """Settle every position on `ticker` given the market `result`
        ('yes'/'no'). A position wins iff its side matches the result (a YES and
        a NO position on the same ticker settle oppositely). A winning contract
        pays 100c, a loser 0. Books per-position realized P&L to the ledger and
        releases the positions. Returns total pnl_cents."""
        mine = [p for p in self.positions if p["ticker"] == ticker]
        if not mine:
            return 0
        total = 0
        for p in mine:
            contracts = p["contracts"]
            cost = round(contracts * p["avg_cost_cents"])
            payout = contracts * 100 if p["side"] == result else 0
            pnl = payout - cost
            self.bankroll_cents += payout
            total += pnl
            self.ledger.append({"ts": _utcnow().isoformat(), "ticker": ticker,
                                "side": p["side"], "result": result,
                                "pnl_cents": int(pnl)})
        self.positions = [p for p in self.positions if p["ticker"] != ticker]
        return total

    def drop_position(self, ticker: str) -> None:
        """Release a position WITHOUT booking P&L -- for a position Kalshi no
        longer holds but whose market has not settled (e.g. closed outside the
        automation). We can't reconstruct the realized amount, so we stop
        tracking it rather than invent a number."""
        self.positions = [p for p in self.positions if p["ticker"] != ticker]

    def realized_loss_cents(self, days: float) -> int:
        """Total realized LOSS (positive number) over the trailing window."""
        cutoff = _utcnow() - dt.timedelta(days=days)
        total = 0
        for e in self.ledger:
            if dt.datetime.fromisoformat(e["ts"]) >= cutoff:
                total += e["pnl_cents"]
        return max(0, -total)

    # ---------- rails ----------
    def loss_limits_breached(self, cfg: BettingConfig) -> str | None:
        daily = self.realized_loss_cents(1.0)
        weekly = self.realized_loss_cents(7.0)
        if daily >= int(cfg.daily_loss_limit_usd * 100):
            return (f"DAILY loss limit breached: lost "
                    f"${daily/100:.2f} >= ${cfg.daily_loss_limit_usd:.2f}")
        if weekly >= int(cfg.weekly_loss_limit_usd * 100):
            return (f"WEEKLY loss limit breached: lost "
                    f"${weekly/100:.2f} >= ${cfg.weekly_loss_limit_usd:.2f}")
        return None

    @property
    def n_open_positions(self) -> int:
        return len(self.positions)


def reconcile_positions(state: BettingState, client, verbose: bool = True) -> dict:
    """Sync locally-tracked positions against Kalshi (the source of truth).

    Any position Kalshi no longer holds open has left the book: if its market
    has settled we book realized P&L to the ledger (record_settlement), else it
    was closed outside the automation and we drop it without inventing a number.
    This keeps `n_open_positions` honest (so the position-cap rail stops
    blocking on stale, already-settled bets) and feeds the loss-limit ledger.

    Non-fatal by design: on any positions-fetch error it warns and leaves state
    untouched -- a stale-high open count only OVER-restricts (blocks trading),
    which is the safe direction. Returns a summary dict."""
    try:
        open_tickers = client.open_position_tickers()
    except Exception as e:                                    # noqa: BLE001
        if verbose:
            print(f"  ! position reconcile skipped (fetch failed: {e})")
        audit("reconcile_skipped", {"error": str(e)})
        return {"settled": 0, "dropped": 0, "skipped": True}

    settled = dropped = 0
    for ticker in {p["ticker"] for p in state.positions} - open_tickers:
        try:
            m = client.get_market(ticker)
        except Exception as e:                                # noqa: BLE001
            if verbose:
                print(f"  ! {ticker}: market fetch failed ({e}); leaving as-is")
            continue
        result = m.get("result") or ""
        if result in ("yes", "no"):
            pnl = state.record_settlement(ticker, result)
            settled += 1
            audit("position_settled", {"ticker": ticker, "result": result,
                                       "pnl_cents": pnl})
        else:
            state.drop_position(ticker)
            dropped += 1
            audit("position_dropped_unsettled",
                  {"ticker": ticker, "market_status": m.get("status")})

    if settled or dropped:
        state.save()
        if verbose:
            print(f"  reconciled positions: {settled} settled, {dropped} dropped"
                  f" -> {state.n_open_positions} still open")
    return {"settled": settled, "dropped": dropped, "skipped": False}
