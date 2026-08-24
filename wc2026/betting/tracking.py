"""
tracking.py
===========
The falsifiability loop: every recommendation and every placed bet is logged
and later scored against closing lines and outcomes, so the edge claim is
continuously testable -- including the news-check gate's contribution.

The log (recommendations.jsonl) holds three kinds of row, all keyed by a bet's
identity (ticker, side) -- one market, one direction:

  * recommendation -- one per bet per mode (recommend / execute): entry price,
    model prob, edge, proposed stake, gate verdict.
  * placement      -- written ONLY when a real order is actually placed in
    execute mode (ticker, side, executed contracts, price, fee, order_id).
    This is the sole signal that a bet was truly executed: realized P&L counts
    a bet only if it has a placement row, so paper-only runs realize nothing.
  * resolution     -- exactly ONE per bet, upserted (not appended) each time
    `track` runs. The CLOSING LINE is the price at kickoff, read from Kalshi
    candlesticks (price history) -- so `track` can run on any lazy schedule
    (even the next morning) and still reconstruct the true pre-game price; you
    never need to be present at the game. While a market is still open it holds
    a live "interim" price read; once past kickoff the candlestick close locks
    in; at settlement it also carries the outcome and P&L.

DUPLICATE PREVENTION.
  * recommendations are de-duplicated on write per (ticker, side, mode).
  * resolutions are upserted to one row per bet on every `track`, so the file
    self-heals if it accumulated interim snapshots before this existed.

P&L definitions (field names unchanged):
  * realized (pnl_cents)                       -- only bets with a placement row
    (i.e. actually executed). $0 until you place a real order.
  * counterfactual (counterfactual_pnl_cents)  -- what a recommendation WOULD
    have returned at its pre-gate size, for every settled bet.

`python -m wc2026 track` resolves what it can and prints the running summary.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from decimal import Decimal, InvalidOperation

from .config import RECS_LOG
from .kalshi import KalshiClient


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load_lines(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _key(row: dict) -> tuple[str, str]:
    return (row["ticker"], row["side"])


def _match_key(ticker: str) -> str:
    """The date+teams token shared by every market on one game, e.g.
    'KXMLSGAME-26JUL22SKCMIN-MIN' -> '26JUL22SKCMIN'. Used to count DISTINCT
    matches behind the CLV sample (bets on the same game are correlated, so the
    match count is the honest independent-sample size)."""
    parts = ticker.split("-")
    return parts[1] if len(parts) >= 2 else ticker


def _price_to_cents(dollar_str) -> int | None:
    """Parse a Kalshi dollar-string price ('0.4800') to integer cents in 1..99,
    or None if missing / blank / unparseable / degenerate.

    Kalshi quotes prices as dollar strings (there is NO integer `last_price`
    field). A value of 0 or 100 cents means 'no real market' -- an untraded
    book, or a `last_price` that has collapsed to the settlement value -- and is
    deliberately rejected so it never masquerades as a closing line."""
    if dollar_str in (None, ""):
        return None
    try:
        cents = int(round(Decimal(str(dollar_str)) * 100))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return cents if 0 < cents < 100 else None


def _to_epoch(ts) -> int | None:
    """Parse a stored UTC timestamp (ISO 'Z' or 'YYYY-MM-DD HH:MM:SS', assumed
    UTC if naive) to unix seconds, or None."""
    if not ts:
        return None
    s = str(ts).strip().replace(" ", "T").replace("Z", "+00:00")
    try:
        d = dt.datetime.fromisoformat(s)
    except ValueError:
        try:
            d = dt.datetime.fromisoformat(s[:19])
        except ValueError:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp())


def _candlestick_close_cents(candle: dict) -> int | None:
    """The YES price a candle represents, in cents (1..99), or None.

    Prefer the last traded price (`price.close_dollars`); if the candle had no
    trade, fall back to the bid/ask midpoint so thin-but-quoted markets still
    yield a closing line."""
    px = _price_to_cents((candle.get("price") or {}).get("close_dollars"))
    if px is not None:
        return px
    yb = _price_to_cents((candle.get("yes_bid") or {}).get("close_dollars"))
    ya = _price_to_cents((candle.get("yes_ask") or {}).get("close_dollars"))
    if yb is not None and ya is not None:
        return (yb + ya) // 2
    return None


def candlestick_yes_cents_at(client, ticker: str, kickoff_ts: int,
                             lookback_hours: int = 3) -> int | None:
    """The authoritative CLOSING line: the YES price at kickoff, read from
    price history. Fetches 1-minute candles in [kickoff-lookback, kickoff] and
    returns the last traded price at or before kickoff -- the market's final
    pre-game consensus, uncontaminated by in-game trading. None if the market
    never traded in the window."""
    candles = client.get_candlesticks(ticker, kickoff_ts - lookback_hours * 3600,
                                      kickoff_ts, period_interval=1)
    best = None
    for c in candles:
        if c.get("end_period_ts", 0) > kickoff_ts:
            continue
        px = _candlestick_close_cents(c)
        if px is not None:
            best = px               # keep the latest valid price at/<= kickoff
    return best


def _latest_rec_by_key(lines: list[dict]) -> dict[tuple[str, str], dict]:
    """Latest recommendation row per (ticker, side). File order is append
    order, so the last row seen for a key is the most recent."""
    out: dict[tuple[str, str], dict] = {}
    for r in lines:
        if r.get("kind") == "recommendation":
            out[_key(r)] = r
    return out


def log_recommendations(recs: list[dict], mode: str,
                        path: str | None = None) -> tuple[int, int]:
    """Append recommendations, skipping (ticker, side) pairs already logged in
    this mode. Returns (n_logged, n_skipped_as_duplicates)."""
    import wc2026.betting.tracking as _self
    path = path or _self.RECS_LOG
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = {(_key(r), r.get("mode")) for r in _load_lines(path)
                if r.get("kind") == "recommendation"}
    logged = skipped = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            if ((r["ticker"], r["side"]), mode) in existing:
                skipped += 1
                continue
            existing.add(((r["ticker"], r["side"]), mode))
            rec = {"kind": "recommendation", "rec_id": str(uuid.uuid4()),
                   "ts": _utcnow(), "mode": mode, **r}
            f.write(json.dumps(rec, default=str) + "\n")
            logged += 1
    return logged, skipped


def log_placement(order: dict, path: str | None = None) -> None:
    """Record a REAL placed order. This is the ONLY signal that a bet was
    actually executed -- summarize() counts realized P&L for a bet only if it
    has a placement row, so recommend-mode (paper) runs realize nothing."""
    import wc2026.betting.tracking as _self
    path = path or _self.RECS_LOG
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"kind": "placement", "ts": _utcnow(), **order}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _rewrite(path: str, rows: list[dict]) -> None:
    """Atomically replace the log with `rows` (temp file + os.replace)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, path)


def dedupe_log(path: str | None = None, verbose: bool = True) -> dict:
    """One-time/maintenance repair of a log written before duplicate
    prevention existed. Keeps the LATEST recommendation per (ticker, side,
    mode) and, for surviving rec_ids, the latest resolution (settled
    preferred). The original file is backed up alongside first."""
    import shutil

    import wc2026.betting.tracking as _self
    path = path or _self.RECS_LOG
    lines = _load_lines(path)
    if not lines:
        return {"kept": 0, "dropped": 0}

    backup = path + ".bak-" + _utcnow()[:10]
    shutil.copyfile(path, backup)

    latest_rec: dict[tuple, dict] = {}
    for r in lines:
        if r.get("kind") == "recommendation":
            latest_rec[(r["ticker"], r["side"], r.get("mode"))] = r
    surviving_ids = {r["rec_id"] for r in latest_rec.values()}

    best_res: dict[str, dict] = {}
    for r in lines:
        if r.get("kind") == "resolution" and r["rec_id"] in surviving_ids:
            prev = best_res.get(r["rec_id"])
            if prev is None or r.get("settled") or not prev.get("settled"):
                best_res[r["rec_id"]] = r

    kept = sorted(latest_rec.values(), key=lambda r: r["ts"]) \
        + sorted(best_res.values(), key=lambda r: r["ts"])
    with open(path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, default=str) + "\n")
    out = {"kept": len(kept), "dropped": len(lines) - len(kept),
           "backup": backup}
    if verbose:
        print(f"Deduped {path}: kept {out['kept']}, dropped {out['dropped']} "
              f"(backup: {backup})")
    return out


def _build_resolution(rec: dict, market: dict, prev: dict | None,
                      kickoff_yes_cents: int | None = None) -> dict:
    """One resolution row for a bet from its latest recommendation, the live
    market, and the prior resolution.

    CLV = our side's closing price - our entry (cents); positive = we beat the
    close. The closing line is the price AT KICKOFF, read from candlesticks
    (`kickoff_yes_cents`) -- the market's final pre-game consensus, uncontam-
    inated by in-game trading. Priority for the closing price:
      1. `kickoff_yes_cents` from candlesticks (authoritative)  -> clv_source "candlestick"
      2. a candlestick close already locked on the prior row    -> carried
      3. while still open: the live last trade (a running read)  -> clv_source "interim"
    A SETTLED bet keeps CLV only if it is candlestick-sourced; anything else at
    settlement (a stale interim, or nothing) is dropped as untrustworthy and
    the bet is marked clv_unavailable. previous_price is never used -- it is
    contaminated by in-game trades (verified 2026-07-23)."""
    status = market.get("status", "")
    result = market.get("result") or ""
    settled = status in ("finalized", "settled") or result in ("yes", "no")
    side = rec["side"]
    entry = rec.get("price_cents")
    row = {"kind": "resolution", "rec_id": rec["rec_id"], "ts": _utcnow(),
           "ticker": rec["ticker"], "side": side,
           "market_status": status, "settled": settled}

    def _set_yes(yes_cents: int, source: str):
        close_side = yes_cents if side == "yes" else 100 - yes_cents
        row["closing_price_cents"] = close_side
        row["clv_source"] = source
        if entry is not None:
            row["clv_cents"] = close_side - int(entry)

    carried = (prev and prev.get("clv_source") == "candlestick"
               and prev.get("clv_cents") is not None)
    if kickoff_yes_cents is not None:
        _set_yes(kickoff_yes_cents, "candlestick")
    elif carried:
        row["closing_price_cents"] = prev.get("closing_price_cents")
        row["clv_cents"] = prev.get("clv_cents")
        row["clv_source"] = "candlestick"
    elif not settled:
        live = _price_to_cents(market.get("last_price_dollars"))
        if live is not None:
            _set_yes(live, "interim")

    if settled:
        won = (result == side)
        n = rec.get("contracts", 0)
        fee = rec.get("fee_cents", 0)
        pnl_per = (100 if won else 0) - entry
        row.update({
            "result": result, "won": won,
            "pnl_cents": n * pnl_per - fee if n else None,
            "counterfactual_pnl_cents":
                rec.get("counterfactual_contracts", 0) * pnl_per,
        })
        if row.get("clv_source") != "candlestick":
            # No authoritative kickoff close -> no trustworthy CLV. Drop any
            # stale interim value and freeze the bet as clv-unavailable.
            for f in ("clv_cents", "closing_price_cents", "clv_source"):
                row.pop(f, None)
            row["clv_unavailable"] = True
    return row


def _clv_final(prev: dict | None) -> bool:
    """A settled bet needs no further work once its CLV is a candlestick close
    or confirmed unavailable; anything else (a pre-fix row, a stale interim) is
    re-processed to compute the real closing line."""
    if not prev or not prev.get("settled"):
        return False
    if prev.get("clv_unavailable"):
        return True
    return prev.get("clv_source") == "candlestick" and prev.get("clv_cents") is not None


def resolve(client: KalshiClient | None = None, path: str = RECS_LOG,
            verbose: bool = True, recompute: bool = False,
            kickoff_lookup=None) -> dict:
    """Upsert exactly one resolution per bet from live market + price history,
    then summarize.

    For each bet past kickoff, the authoritative closing line is read from
    candlesticks at the stored kickoff time. `kickoff_lookup(rec) -> iso|None`
    optionally backfills a kickoff onto older recommendation rows that predate
    kickoff capture. `recompute=True` re-resolves every bet (e.g. to rebuild
    CLV under improved logic) instead of skipping finalized ones.

    Re-running OVERWRITES each bet's single resolution row rather than
    appending, so `track` never stacks duplicates and old logs self-heal."""
    client = client or KalshiClient()
    lines = _load_lines(path)
    non_res = [r for r in lines if r.get("kind") != "resolution"]

    # (Re)derive kickoff from the dataset onto recommendation rows (persisted
    # below). This OVERRIDES an existing value, because bets logged before
    # kickoff capture stored the Kalshi expiration (~game end) in this field --
    # an authoritative ESPN kickoff from the dataset replaces that.
    if kickoff_lookup is not None:
        for r in non_res:
            if r.get("kind") == "recommendation":
                k = kickoff_lookup(r)
                if k:
                    r["kickoff_utc"] = k

    latest: dict[tuple[str, str], dict] = {}
    for r in non_res:
        if r.get("kind") == "recommendation":
            latest[_key(r)] = r
    id_to_key = {r["rec_id"]: _key(r) for r in non_res
                 if r.get("kind") == "recommendation"}

    # Best existing resolution per bet (settled preferred, else latest seen).
    existing: dict[tuple[str, str], dict] = {}
    for r in lines:
        if r.get("kind") == "resolution" and r.get("rec_id") in id_to_key:
            k = id_to_key[r["rec_id"]]
            prev = existing.get(k)
            if prev is None or r.get("settled") or not prev.get("settled"):
                existing[k] = r

    now_ep = int(dt.datetime.now(dt.timezone.utc).timestamp())
    updated = dict(existing)
    for key, rec in latest.items():
        prev = existing.get(key)
        if not recompute and _clv_final(prev):
            continue
        try:
            m = client.get_market(rec["ticker"])
        except Exception as e:                                # noqa: BLE001
            if verbose:
                print(f"  ! {rec['ticker']}: fetch failed ({e}); will retry")
            continue
        # Read the kickoff closing line once, when past kickoff and not already
        # locked from a candlestick.
        kickoff_yc = None
        # A recompute must re-read the candlestick (a locked value may have come
        # from a wrong kickoff); normal runs reuse an already-locked close.
        have_candle = (not recompute and prev
                       and prev.get("clv_source") == "candlestick"
                       and prev.get("clv_cents") is not None)
        if not have_candle:
            kt = _to_epoch(rec.get("kickoff_utc"))
            if kt is not None and now_ep >= kt:
                kickoff_yc = candlestick_yes_cents_at(client, rec["ticker"], kt)
        updated[key] = _build_resolution(rec, m, prev, kickoff_yes_cents=kickoff_yc)

    _rewrite(path, non_res + [updated[k] for k in updated])
    return summarize(path, verbose=verbose)


def summarize(path: str = RECS_LOG, verbose: bool = True) -> dict:
    """Running totals, counted once per (ticker, side)."""
    lines = _load_lines(path)
    latest = _latest_rec_by_key(lines)
    id_to_key = {r["rec_id"]: _key(r) for r in lines
                 if r.get("kind") == "recommendation"}
    # latest resolution per key (file order = chronological)
    res_by_key: dict[tuple[str, str], dict] = {}
    for r in lines:
        if r.get("kind") == "resolution" and r["rec_id"] in id_to_key:
            k = id_to_key[r["rec_id"]]
            if r.get("settled") or not res_by_key.get(k, {}).get("settled"):
                res_by_key[k] = r

    # A bet is "placed" (and thus eligible for REALIZED P&L) only if a real
    # order was recorded for it -- i.e. it has a placement row. Recommend-mode
    # (paper) bets never get one, so realized stays $0 until you truly execute.
    placed_keys = {_key(r) for r in lines if r.get("kind") == "placement"}
    settled = {k: r for k, r in res_by_key.items() if r.get("settled")}
    placed = [r for k, r in settled.items() if k in placed_keys]
    # Headline CLV uses ONLY authoritative candlestick closing lines; live
    # "interim" reads on still-open markets are counted separately as in-progress.
    candle_rows = [r for r in res_by_key.values()
                   if r.get("clv_cents") is not None
                   and r.get("clv_source") == "candlestick"]
    clvs = [r["clv_cents"] for r in candle_rows]
    clv_matches = {_match_key(r["ticker"]) for r in candle_rows}
    n_interim = sum(1 for r in res_by_key.values()
                    if r.get("clv_source") == "interim")

    out = {
        "n_recommendations": len(latest),
        "n_settled": len(settled),
        "realized_pnl_usd": sum(r.get("pnl_cents") or 0 for r in placed) / 100,
        "counterfactual_pnl_usd": sum(r.get("counterfactual_pnl_cents") or 0
                                      for r in settled.values()) / 100,
        "mean_clv_cents": (sum(clvs) / len(clvs)) if clvs else None,
        "n_clv": len(clvs),
        "n_clv_matches": len(clv_matches),
        "n_clv_interim": n_interim,
    }
    if verbose:
        print(f"Tracking: {out['n_recommendations']} distinct bets, "
              f"{out['n_settled']} settled")
        print(f"  realized P&L (placed bets): ${out['realized_pnl_usd']:+.2f}")
        print(f"  counterfactual P&L (unplaced/vetoed): "
              f"${out['counterfactual_pnl_usd']:+.2f}")
        if out["mean_clv_cents"] is not None:
            print(f"  mean CLV: {out['mean_clv_cents']:+.2f}c over {out['n_clv']} "
                  f"closed bets ({out['n_clv_matches']} distinct matches)")
        else:
            print("  mean CLV: n/a (no closing line captured yet)")
        if n_interim:
            print(f"  ({n_interim} open bets have a live in-progress CLV read)")
    return out
