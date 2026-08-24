"""
merge_odds.py — add 1X2 betting odds to results.csv so the backtest can tune MODEL_TRUST.

Pulls World Cup match odds from the BALLDONTLIE FIFA API (2018 + 2022) and merges them
onto your existing data/incoming/results.csv (matched by date + team names). Only the
rows it can match get mkt_home/mkt_draw/mkt_away columns; everything else is untouched,
which is exactly what the backtest wants (full history for Elo, odds on the recent subset).

Setup:
  1. Get a key at https://balldontlie.io  (FIFA World Cup API).
  2. Paste it below.
  3. python merge_odds.py            (use --dry-run first to preview the match rate)
  4. python backtest.py             -> the MODEL_TRUST blend sweep now runs.

Odds are American moneylines; we convert to implied probabilities and average across
vendors (the backtest normalizes out the vig). Run with your key to verify the match rate.
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta

import requests

# ============================ API KEY (fill in) ============================
BALLDONTLIE_API_KEY = "64160bf8-e26f-44fa-96af-250dc963ee27"
# ===========================================================================

BASE = "https://api.balldontlie.io/fifa/worldcup/v1"
SEASONS = [2018, 2022, 2026]   # BALLDONTLIE only carries odds for 2026 (live tournament)
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "data", "incoming", "results.csv")
REQUEST_DELAY = 1.5   # seconds between requests (All-Star = 60/min; raise if still 429'd)

# Normalized-name aliases so BALLDONTLIE names match the martj42 results.csv names.
_ALIASES = {
    "usa": "unitedstates", "us": "unitedstates",
    "korearepublic": "southkorea", "republicofkorea": "southkorea",
    "iriran": "iran", "china": "chinapr",
    "ivorycoast": "cotedivoire",
    "bosniaherzegovina": "bosniaandherzegovina",
}


def _norm(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z]", "", s.lower())
    return _ALIASES.get(s, s)


def _american_to_prob(o) -> float | None:
    try:
        o = float(o)
    except (ValueError, TypeError):
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return None


def _request(path: str, params: dict, max_retries: int = 6) -> dict:
    """Single GET with 429 backoff (honors Retry-After, else exponential)."""
    headers = {"Authorization": BALLDONTLIE_API_KEY}
    for attempt in range(max_retries):
        r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=40)
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After") or 0) or min(60, 5 * (2 ** attempt))
            print(f"  [rate-limited] waiting {wait:.0f}s and retrying...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()  # give up -> raise the last 429
    return {}


def _get(path: str, params: dict) -> list:
    """Paginated GET against the BALLDONTLIE FIFA API (cursor-based), rate-limited."""
    out, cursor = [], None
    for _ in range(200):  # safety cap
        p = dict(params, per_page=100)
        if cursor is not None:
            p["cursor"] = cursor
        body = _request(path, p)
        out += body.get("data", [])
        cursor = (body.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(REQUEST_DELAY)
    return out


def fetch_matches() -> dict[int, dict]:
    """match_id -> {date, home, away}."""
    rows = _get("/matches", {"seasons[]": SEASONS})
    out = {}
    for m in rows:
        dt = (m.get("datetime") or "")[:10]
        ht = (m.get("home_team") or {}).get("name")
        at = (m.get("away_team") or {}).get("name")
        if dt and ht and at:
            out[m["id"]] = {"date": dt, "home": ht, "away": at}
    return out


def fetch_odds(match_ids: list[int]) -> dict[int, tuple[float, float, float]]:
    """match_id -> consensus implied (home, draw, away), averaged across vendors."""
    try:
        rows = _get("/odds", {"seasons[]": SEASONS})
    except requests.HTTPError:
        rows = []
    if not rows:  # fall back to fetching odds per match if seasons[] isn't supported
        rows = []
        for i in range(0, len(match_ids), 50):
            rows += _get("/odds", {"match_ids[]": match_ids[i:i + 50]})
            time.sleep(REQUEST_DELAY)
    acc: dict[int, list] = defaultdict(list)
    for o in rows:
        h = _american_to_prob(o.get("moneyline_home_odds"))
        d = _american_to_prob(o.get("moneyline_draw_odds"))
        a = _american_to_prob(o.get("moneyline_away_odds"))
        if None not in (h, d, a):
            acc[o["match_id"]].append((h, d, a))
    out = {}
    for mid, lst in acc.items():
        n = len(lst)
        h, d, a = (sum(x[0] for x in lst) / n, sum(x[1] for x in lst) / n,
                   sum(x[2] for x in lst) / n)
        s = h + d + a
        if s <= 0 or max(h, d, a) / s > 0.97:
            continue  # near-certain line => likely a settled/post-match price; skip
        out[mid] = (h, d, a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report match rate, don't write")
    args = ap.parse_args()

    if "PASTE_" in BALLDONTLIE_API_KEY:
        print("Set BALLDONTLIE_API_KEY at the top of this file first.")
        return
    if not os.path.exists(RESULTS_CSV):
        print(f"No {RESULTS_CSV}. Run: python backtest.py --from-statsbomb  (or add martj42 CSV).")
        return

    print("Fetching BALLDONTLIE matches + odds (WC 2018, 2022)...")
    matches = fetch_matches()
    odds = fetch_odds(list(matches.keys()))
    print(f"  {len(matches)} matches, {len(odds)} with odds")

    # Index BALLDONTLIE games with odds by (date, frozenset of normalized team names).
    bd_index: dict[tuple, dict] = {}
    for mid, info in matches.items():
        if mid not in odds:
            continue
        key = (info["date"], frozenset({_norm(info["home"]), _norm(info["away"])}))
        bd_index[key] = {"home": _norm(info["home"]), "odds": odds[mid]}

    # Load results.csv, attach odds to matched rows (try exact date, then +/- 1 day).
    with open(RESULTS_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fns = list(rows[0].keys())
    c_date = next((c for c in fns if c.lower() in ("date", "match_date")), "date")
    c_h = next((c for c in fns if "home" in c.lower() and "score" not in c.lower()), "home_team")
    c_a = next((c for c in fns if "away" in c.lower() and "score" not in c.lower()), "away_team")
    for c in ("mkt_home", "mkt_draw", "mkt_away"):
        if c not in fns:
            fns.append(c)

    def shifted(date_str, days):
        try:  # datetime handles any year (the file goes back to 1872; mktime can't)
            return (datetime.strptime(date_str, "%Y-%m-%d") +
                    timedelta(days=days)).strftime("%Y-%m-%d")
        except ValueError:
            return date_str

    matched = 0
    for row in rows:
        hn, an = _norm(row.get(c_h, "")), _norm(row.get(c_a, ""))
        pair = frozenset({hn, an})
        hit = None
        for dd in (0, 1, -1):
            hit = bd_index.get((shifted(row.get(c_date, ""), dd), pair))
            if hit:
                break
        if not hit:
            continue
        h, d, a = hit["odds"]
        # Orient odds to this row's home/away.
        if hit["home"] == hn:
            mh, md, ma = h, d, a
        else:
            mh, md, ma = a, d, h
        row["mkt_home"], row["mkt_draw"], row["mkt_away"] = round(mh, 4), round(md, 4), round(ma, 4)
        matched += 1

    print(f"Matched odds onto {matched} of {len(rows)} result rows.")
    if args.dry_run:
        print("[dry-run] not writing.")
        return
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote odds into {RESULTS_CSV}. Now run: python backtest.py  (blend sweep will run).")


if __name__ == "__main__":
    main()
