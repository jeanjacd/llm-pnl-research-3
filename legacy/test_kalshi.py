"""
test_kalshi.py — diagnose the Kalshi connection independently of the model.

Read-only: checks key loading, public connectivity, authenticated signature, and
scans open markets for World Cup / soccer so you can fill KALSHI_TICKER_MAP.
Run:  python test_kalshi.py
"""
from __future__ import annotations
import sys
import requests

from kalshi_client import KalshiClient, _to_market

KEYWORDS = ["WORLD CUP", "WORLDCUP", "SOCCER", "FIFA",
            "SWITZERLAND", "CANADA", "BRAZIL", "ARGENTINA", "FRANCE"]


def main():
    c = KalshiClient()

    print("=" * 60)
    print("[1] Credentials")
    print(f"    key id loaded     : {bool(c.key_id and 'PASTE_' not in c.key_id)}")
    print(f"    private key loaded: {c._private_key is not None}")
    if c._private_key is None:
        print("    -> No private key. Fill KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH.")

    print("\n[2] Public connectivity: GET /exchange/status")
    try:
        st = c._get("/exchange/status")
        print(f"    OK -> {st}")
    except Exception as e:
        print(f"    FAILED: {_err(e)}")

    print("\n[3] Authenticated request (proves your signature works): "
          "GET /portfolio/balance")
    try:
        bal = c._get("/portfolio/balance")
        print(f"    OK -> balance = {bal}")
        print("    => AUTH IS WORKING.")
    except Exception as e:
        print(f"    FAILED: {_err(e)}")
        print("    => 401/403 here means the key id, path, or signature is wrong.")

    print("\n[4] Market data: scanning open markets for World Cup / soccer ...")
    found, scanned, cursor = [], 0, None
    try:
        for _ in range(8):  # up to 8 pages
            params = {"limit": 1000, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            data = c._get("/markets", params=params)
            ms = data.get("markets", [])
            scanned += len(ms)
            for m in ms:
                blob = f"{m.get('ticker','')} {m.get('title','')} " \
                       f"{m.get('event_ticker','')}".upper()
                if any(k in blob for k in KEYWORDS):
                    found.append(_to_market(m))
            cursor = data.get("cursor")
            if not cursor or not ms:
                break
        print(f"    scanned {scanned} open markets; matched {len(found)}")
        for m in found[:25]:
            print(f"      {m.ticker:34s} YES {m.yes_price:>2}c "
                  f"(impl {m.implied_prob()*100:3.0f}%) | {m.title[:50]}")
        if found:
            print("\n    Copy the tickers you want into KALSHI_TICKER_MAP in main.py.")
        else:
            print("    No WC/soccer markets in the sampled pages. Either they're not "
                  "listed yet, or you need the exact series_ticker — find it on the "
                  "Kalshi site (the URL), then call get_markets(series_ticker=...).")
    except Exception as e:
        print(f"    FAILED: {_err(e)}")
    print("=" * 60)


def _err(e: Exception) -> str:
    if isinstance(e, requests.HTTPError) and e.response is not None:
        body = e.response.text[:300]
        return f"HTTP {e.response.status_code} — {body}"
    return f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    main()
