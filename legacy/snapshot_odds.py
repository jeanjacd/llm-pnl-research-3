"""
snapshot_odds.py — capture Kalshi's OPENING 1X2 odds per World Cup game, once each,
and feed them into results.csv so backtest.py can tune MODEL_TRUST against the market
you actually bet into.

How it works
------------
1. Reads every live KXWCGAME (match-winner) market from Kalshi.
2. For each game it hasn't recorded yet, pulls the order-book mid price for
   home / draw / away and converts to implied probabilities.
3. DEDUP: each game is stored exactly once -> the FIRST time it has a full, live
   3-way book. That first capture is the "opening" line and is never overwritten,
   so re-running this all day only ever ADDS new games.
4. Merges the stored opening odds onto matching completed rows in results.csv.

Run it a few times across the day (e.g. each time you check in before kickoffs):
    python snapshot_odds.py
Then:
    python backtest.py        # blend sweep now uses real Kalshi opening odds

Durable store: data/incoming/kalshi_odds_snapshots.csv  (this is the source of truth;
if you re-download results.csv it wipes the odds columns, so just re-run this to
re-attach them from the store).
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import time
from datetime import datetime, timedelta, timezone

from kalshi_client import KalshiClient
from merge_odds import _norm   # reuse the alias-aware name normalizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INCOMING = os.path.join(DATA_DIR, "incoming")
RESULTS_CSV = os.path.join(INCOMING, "results.csv")
STORE_CSV = os.path.join(INCOMING, "kalshi_odds_snapshots.csv")
STORE_COLS = ["game_key", "date", "home", "away",
              "mkt_home", "mkt_draw", "mkt_away", "snapshot_time"]
MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def _game_date(game_key: str) -> str:
    """'26JUN24SUICAN' -> '2026-06-24' (the match date encoded in the ticker)."""
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})", game_key)
    if not m:
        return ""
    return f"20{m.group(1)}-{MONTHS.get(m.group(2), 0):02d}-{int(m.group(3)):02d}"


def _mid_implied(ob: dict) -> float | None:
    """Order-book mid implied probability for a YES contract (falls back to ask)."""
    ya, yb = ob.get("yes_ask"), ob.get("yes_bid")
    if ya and yb:
        return (ya + yb) / 200.0
    if ya:
        return ya / 100.0
    return None


def _shift(date_str: str, days: int) -> str:
    try:
        return (datetime.strptime(date_str, "%Y-%m-%d") +
                timedelta(days=days)).strftime("%Y-%m-%d")
    except ValueError:
        return date_str


def load_store() -> tuple[list[dict], set]:
    if not os.path.exists(STORE_CSV):
        return [], set()
    with open(STORE_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, {r["game_key"] for r in rows}


def snapshot(client: KalshiClient, dry: bool) -> list[dict]:
    rows, seen = load_store()
    markets = client.get_markets(series_ticker="KXWCGAME", status="open", limit=1000)
    by_event: dict[str, list] = {}
    for m in markets:
        by_event.setdefault(m.raw.get("event_ticker", ""), []).append(m)

    added = 0
    for event, ms in by_event.items():
        if "-" not in event:
            continue
        game_key = event.split("-", 1)[1]
        if game_key in seen:
            continue  # DEDUP: opening odds for this game already captured -> never overwrite
        tm = re.match(r"(.+?)\s+vs\s+(.+?)\s+Winner", ms[0].title or "")
        if not tm:
            continue
        home, away = tm.group(1).strip(), tm.group(2).strip()
        oc: dict[str, float] = {}
        for mk in ms:
            sub = (mk.raw.get("yes_sub_title") or "")
            imp = _mid_implied(client.orderbook_prices(mk.ticker))
            time.sleep(0.05)  # be gentle on the API
            if imp is None:
                continue
            sl = sub.lower()
            if "tie" in sl or "draw" in sl:
                oc["draw"] = imp
            elif _norm(sub) == _norm(home):
                oc["home"] = imp
            elif _norm(sub) == _norm(away):
                oc["away"] = imp
        if all(k in oc for k in ("home", "draw", "away")):
            rows.append({
                "game_key": game_key, "date": _game_date(game_key),
                "home": home, "away": away,
                "mkt_home": round(oc["home"], 4), "mkt_draw": round(oc["draw"], 4),
                "mkt_away": round(oc["away"], 4),
                "snapshot_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            seen.add(game_key)
            added += 1
            print(f"  + {home} {oc['home']:.2f} / draw {oc['draw']:.2f} / "
                  f"{away} {oc['away']:.2f}   ({game_key})")

    print(f"snapshot: +{added} new games captured; store now holds {len(rows)} games "
          f"(games with empty books were skipped and will be retried next run)")
    if not dry and added:
        with open(STORE_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STORE_COLS)
            w.writeheader()
            w.writerows(rows)
    return rows


def merge_into_results(store_rows: list[dict], dry: bool) -> None:
    if not os.path.exists(RESULTS_CSV):
        print("No results.csv to merge into (run backtest.py --from-statsbomb or add martj42).")
        return
    with open(RESULTS_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fns = list(rows[0].keys())
    for c in ("mkt_home", "mkt_draw", "mkt_away"):
        if c not in fns:
            fns.append(c)

    idx = {}
    for s in store_rows:
        key = (s["date"], frozenset({_norm(s["home"]), _norm(s["away"])}))
        idx[key] = {"home": _norm(s["home"]),
                    "o": (float(s["mkt_home"]), float(s["mkt_draw"]), float(s["mkt_away"]))}

    cd = next((c for c in fns if c.lower() in ("date", "match_date")), "date")
    ch = next((c for c in fns if "home" in c.lower() and "score" not in c.lower()), "home_team")
    ca = next((c for c in fns if "away" in c.lower() and "score" not in c.lower()), "away_team")

    matched = 0
    for row in rows:
        hn, an = _norm(row.get(ch, "")), _norm(row.get(ca, ""))
        pair = frozenset({hn, an})
        hit = None
        for dd in (0, 1, -1):
            hit = idx.get((_shift(row.get(cd, ""), dd), pair))
            if hit:
                break
        if not hit:
            continue
        h, d, a = hit["o"]
        mh, md, ma = (h, d, a) if hit["home"] == hn else (a, d, h)
        row["mkt_home"], row["mkt_draw"], row["mkt_away"] = mh, md, ma
        matched += 1

    print(f"merge: attached opening odds to {matched} completed result rows")
    if not dry:
        with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fns)
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="don't write store or results")
    ap.add_argument("--merge-only", action="store_true",
                    help="skip the Kalshi fetch; just re-attach the existing store to results.csv")
    args = ap.parse_args()

    if args.merge_only:
        rows, _ = load_store()
        print(f"merge-only: {len(rows)} games in store")
        merge_into_results(rows, args.dry_run)
        return

    client = KalshiClient()
    print("Snapshotting live Kalshi KXWCGAME opening odds...")
    rows = snapshot(client, args.dry_run)
    merge_into_results(rows, args.dry_run)
    print("\nDone. Run: python backtest.py   (blend sweep uses Kalshi opening odds)")


if __name__ == "__main__":
    main()
