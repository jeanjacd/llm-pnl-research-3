"""
site/build.py
=============
Build the published page from the ledger on disk.

Deliberately tolerant about the ledger being absent or thin: the workflow runs
this on every board and maintenance cycle, including the ones that do nothing,
and a build that fails when there is nothing to report would take the site down
on exactly the quiet days it is supposed to sit through. An empty book renders
an honest empty page.

Deliberately INTOLERANT about the ledger being unreadable. A truncated or
malformed portfolio is a different thing from an empty one, and silently
publishing a page that says "nothing here" when the record actually exists
would be the worst failure available -- it looks like a fact about the trading
rather than a fact about the build.
"""
from __future__ import annotations

import datetime as dt
import json
import os

from . import render

DEFAULT_LEDGER = os.path.join("data", "paper", "portfolio.json")
DEFAULT_OUT = os.path.join("site", "index.html")

EMPTY = {"starting_cash_cents": 100_000, "cash_cents": 100_000,
         "reserved_cents": 0, "positions": {}, "orders": {}, "boarded": {},
         "ledger": [], "saved_at": None}


class LedgerUnreadable(RuntimeError):
    """The ledger exists but could not be parsed. Never treated as empty."""


def load_ledger(path: str = DEFAULT_LEDGER) -> dict:
    if not os.path.exists(path):
        return dict(EMPTY)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise LedgerUnreadable("%s: %s" % (path, exc)) from exc
    if not isinstance(data, dict):
        raise LedgerUnreadable("%s: expected an object, got %s"
                               % (path, type(data).__name__))
    merged = dict(EMPTY)
    merged.update(data)
    return merged


def build(ledger_path: str = DEFAULT_LEDGER, out_path: str = DEFAULT_OUT,
          now=None) -> dict:
    """Render the page and write it. Returns what was built, for the log."""
    portfolio = load_ledger(ledger_path)
    html = render.page(portfolio, now=now or dt.datetime.now(dt.timezone.utc))
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    # `.nojekyll` stops GitHub Pages running the output through Jekyll, which
    # would otherwise drop any file or directory beginning with an underscore.
    with open(os.path.join(parent or ".", ".nojekyll"), "w",
              encoding="utf-8") as fh:
        fh.write("")

    from . import model
    summary = model.summary(portfolio)
    return {
        "out": out_path,
        "bytes": len(html),
        "fixtures": summary["board"]["n_fixtures"],
        "declined": summary["board"]["n_declined"],
        "positions": len(summary["fixtures"]),
        "settled": summary["pnl"]["n_settled"],
        "saved_at": portfolio.get("saved_at"),
    }


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build the JJ's Journal page")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    built = build(args.ledger, args.out)
    print("built %(out)s (%(bytes)d bytes) — %(fixtures)d fixtures boarded, "
          "%(declined)d declined, %(settled)d markets settled" % built)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
