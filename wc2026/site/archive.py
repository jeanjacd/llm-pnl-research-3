"""
site/archive.py
===============
The settlement archive: one page per settled fixture, and an index of all of
them.

WHY IT EXISTS. The front page shows a rolling window -- 48 fixtures, which is
what fits at a legible column width -- and a record that only publishes its
recent history is asking to be taken on trust. Every settlement the book has
ever made gets a permanent address here, and every settled column on the front
page links to it.

WHAT A FIXTURE PAGE IS FOR. The front page answers "how is it going". This
answers "show me the working": every market taken on the match, the price paid,
the size, where the market closed, which side won, what it returned, and what
the board said when it approved the fixture. It is the level at which a wrong
number is actually findable -- the Real Betis mispricing was visible in one row
of one fixture and invisible in every aggregate above it.

STATIC PAGES, NOT A QUERY INTERFACE. This publishes to GitHub Pages from a
workflow, so the archive is a directory of flat files: `settlements/index.html`
plus one `settlements/<slug>.html` per fixture. No server, no database, and
nothing that can be down when the front page is up.
"""
from __future__ import annotations

import datetime as dt

from . import ledger, render
from .render import cents, claim_label, dollars, esc, money, or_dash, pct


def _day(value) -> str:
    stamp = render.model._parse(value)
    return stamp.strftime("%a %d %b %Y") if stamp else str(value or "")


def _month(value) -> str:
    stamp = render.model._parse(value)
    return stamp.strftime("%B %Y") if stamp else "Undated"


def shell(title: str, body: str, depth: int = 1, description: str = "") -> str:
    """The chrome every archive page shares.

    Deliberately the same stylesheet as the front page rather than a trimmed
    copy: a second sheet is a second thing to keep in step, and these pages
    print the same figures in the same faces.
    """
    up = "../" * depth
    links = "\n".join('<link href="%s" rel="stylesheet">' % f
                      for f in render.FONTS)
    return """<!DOCTYPE html>
<html lang="en" data-mode="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%s &middot; JJ&rsquo;s Journal</title>
<meta name="description" content="%s">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
%s
<style>%s</style>
</head>
<body>
<div class="sheet">
  <i class="crop tl"></i><i class="crop tr"></i><i class="crop bl"></i><i class="crop br"></i>
  <header class="subhead">
    <div class="rule-double"></div>
    <div class="dateline">
      <a class="back" href="%sindex.html">&larr; JJ&rsquo;s Journal</a>
      <span class="spacer"></span>
      <span>The settlement archive</span>
      <div class="modes" role="group" aria-label="Reading light">
        <button id="mLight" aria-pressed="true">Printed</button>
        <button id="mDark" aria-pressed="false">Lit</button>
      </div>
    </div>
    <div class="rule-double flip"></div>
  </header>
%s
</div>
<script>%s</script>
</body>
</html>
""" % (esc(title), esc(description or title), links, render.CSS, up, body,
       render.JS)


# --- one fixture --------------------------------------------------------------
def _market_row(market: dict) -> str:
    clv = market["clv_cents"]
    close = market["closing_price_cents"]
    return """
      <tr class="%s">
        <td class="what">%s</td>
        <td class="tnum">%s</td>
        <td class="tnum">%s</td>
        <td class="tnum">%s</td>
        <td class="tnum">%s</td>
        <td class="tnum %s">%s</td>
        <td class="verdict">%s</td>
        <td class="tnum %s">%s</td>
      </tr>""" % (
        "won" if market["won"] else "lost",
        esc(claim_label(str(market["claim"] or ""), market["home"],
                        market["away"])),
        esc(str(market["venue"] or "")[:4]),
        esc("%d¢" % round(market["cost_cents"])),
        esc("%d" % round(market["size"])),
        esc("—" if close is None else "%d¢" % round(float(close))),
        "up" if (clv or 0) >= 0 else "dn",
        esc(or_dash(clv, lambda v: cents(float(v)))),
        "won" if market["won"] else "lost",
        "pos" if market["pnl_cents"] >= 0 else "neg",
        esc(dollars(market["pnl_cents"])))


def fixture_page(row: dict, board_record: dict = None) -> str:
    """Everything the book did on one match, and what the board said first."""
    markets = row["markets"]
    won = row["n_won"]
    body_rows = "".join(_market_row(m) for m in markets)
    reason = (board_record or {}).get("reason")
    considered = (board_record or {}).get("markets_considered")
    verdict = (board_record or {}).get("action")

    note = ""
    if reason or verdict:
        note = """
    <aside class="verdictbox">
      <p class="kicker">What the board said</p>
      <p class="said">%s</p>
      <p class="saidfoot">%s</p>
    </aside>""" % (
            esc(reason or "No reason recorded."),
            esc("%s · %s market%s read"
                % (str(verdict or "decision not recorded").replace("_", " "),
                   "{:,}".format(int(considered or 0)),
                   "" if considered == 1 else "s")))

    return shell(
        "%s v %s" % (row["home"], row["away"]),
        """
  <article class="fixture">
    <p class="kicker">%s &middot; %s</p>
    <h1 class="fixture-h">%s <span>v</span> %s</h1>
    <dl class="scoreline">
      <div><dt>Settled</dt><dd class="tnum %s">%s</dd></div>
      <div><dt>Staked</dt><dd class="tnum">%s</dd></div>
      <div><dt>Markets</dt><dd class="tnum">%d won of %d</dd></div>
      <div><dt>Closing line</dt><dd class="tnum">%s</dd></div>
    </dl>
    %s
    <div class="tablewrap">
      <table class="settlements">
        <caption>Every market settled on this fixture</caption>
        <thead><tr>
          <th scope="col">What it backed</th><th scope="col">Venue</th>
          <th scope="col">Paid</th><th scope="col">Size</th>
          <th scope="col">Close</th><th scope="col">CLV</th>
          <th scope="col">Result</th><th scope="col">Net</th>
        </tr></thead>
        <tbody>%s</tbody>
      </table>
    </div>
    <p class="backlink"><a href="index.html">&larr; Every settlement</a></p>
  </article>""" % (
            esc(_day(row["kickoff_utc"])),
            esc(str(row["league_id"] or "").replace("_", " ")),
            esc(row["home"]), esc(row["away"]),
            "pos" if row["pnl_cents"] >= 0 else "neg",
            esc(dollars(row["pnl_cents"])),
            esc(money(row["staked_cents"])), won, row["n_markets"],
            esc(or_dash(row["clv_cents"], lambda v: cents(float(v)))),
            note, body_rows),
        description="Every market settled on %s v %s, %s."
                    % (row["home"], row["away"], _day(row["kickoff_utc"])))


# --- the index ----------------------------------------------------------------
def index_page(rows: list, totals: dict) -> str:
    """Every settled fixture ever, newest first, grouped by month."""
    if not rows:
        return shell("The settlement archive", """
  <article class="fixture">
    <h1 class="fixture-h">The settlement archive</h1>
    <p class="nothing">Nothing has settled yet. The archive fills itself the
    first time a market pays out.</p>
    <p class="backlink"><a href="../index.html">&larr; JJ&rsquo;s Journal</a></p>
  </article>""")

    groups, month = [], None
    for row in rows:
        this = _month(row["kickoff_utc"])
        if this != month:
            groups.append((this, []))
            month = this
        groups[-1][1].append(row)

    blocks = []
    for name, items in groups:
        lines = []
        for row in items:
            lines.append("""
        <li class="%s"><a href="%s.html">
          <span class="when tnum">%s</span>
          <span class="who">%s <i>v</i> %s</span>
          <span class="count tnum">%d of %d</span>
          <span class="clv tnum %s">%s</span>
          <span class="net tnum %s">%s</span>
        </a></li>""" % (
                "won" if row["pnl_cents"] >= 0 else "lost",
                esc(row["slug"]),
                esc(row["date"]), esc(row["home"]), esc(row["away"]),
                row["n_won"], row["n_markets"],
                "up" if (row["clv_cents"] or 0) >= 0 else "dn",
                esc(or_dash(row["clv_cents"], lambda v: cents(float(v)))),
                "pos" if row["pnl_cents"] >= 0 else "neg",
                esc(dollars(row["pnl_cents"]))))
        blocks.append('<section class="monthblock"><h2>%s</h2>'
                      '<ul class="archive-list">%s</ul></section>'
                      % (esc(name), "".join(lines)))

    return shell("The settlement archive", """
  <article class="fixture">
    <p class="kicker">The complete record</p>
    <h1 class="fixture-h">Every settlement</h1>
    <p class="standfirst-long">%d fixture%s have settled %d market%s, for
    %s on %s staked. The front page shows the most recent 48 fixtures; this
    is all of them, and each one links to its own working.</p>
    <dl class="scoreline">
      <div><dt>Net</dt><dd class="tnum %s">%s</dd></div>
      <div><dt>Markets won</dt><dd class="tnum">%d of %d</dd></div>
      <div><dt>Strike rate</dt><dd class="tnum">%s</dd></div>
      <div><dt>Closing line</dt><dd class="tnum">%s</dd></div>
    </dl>
    %s
    <p class="backlink"><a href="../index.html">&larr; JJ&rsquo;s Journal</a></p>
  </article>""" % (
        totals["n_fixtures"], "" if totals["n_fixtures"] == 1 else "s",
        totals["n_markets"], "" if totals["n_markets"] == 1 else "s",
        esc(dollars(totals["pnl_cents"])), esc(money(totals["staked_cents"])),
        "pos" if totals["pnl_cents"] >= 0 else "neg",
        esc(dollars(totals["pnl_cents"])),
        totals["n_won"], totals["n_markets"],
        esc(or_dash(totals["strike_rate"], pct)),
        esc(or_dash(totals["clv_cents"], lambda v: cents(float(v)))),
        "".join(blocks)),
        description="Every market this paper book has settled, newest first.")


def totals_for(rows: list) -> dict:
    markets = [m for row in rows for m in row["markets"]]
    scored = [float(m["clv_cents"]) for m in markets
              if m["clv_cents"] is not None]
    won = sum(1 for m in markets if m["won"])
    return {
        "n_fixtures": len(rows),
        "n_markets": len(markets),
        "n_won": won,
        "strike_rate": (won / len(markets)) if markets else None,
        "pnl_cents": sum(m["pnl_cents"] for m in markets),
        "staked_cents": sum(m["stake_cents"] for m in markets),
        "clv_cents": (sum(scored) / len(scored)) if scored else None,
    }


def pages(portfolio: dict, now=None) -> dict:
    """{relative path: html} for the whole archive.

    Returned rather than written so the caller owns the filesystem, which is
    what keeps this testable without a temp directory.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    rows = ledger.settled_fixtures(portfolio)
    boarded = portfolio.get("boarded") or {}
    out = {"settlements/index.html": index_page(rows, totals_for(rows))}
    for row in rows:
        record = boarded.get("|".join(str(p) for p in row["key"])) or {}
        out["settlements/%s.html" % row["slug"]] = fixture_page(row, record)
    return out
