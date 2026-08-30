"""
site/render.py
==============
Render the paper record as a single self-contained page.

DESIGN PROVENANCE. Tokens, type roles, rule vocabulary and motion values come
from the design brief and its reference implementation (`the-form-book.html`)
and are used unchanged. The material rules -- salmon newsprint under the
printed reading, a tote board under the lit one -- are a token swap plus four
overrides, not two designs.

THE MASTHEAD IS THE ONE DELIBERATE DEPARTURE. The brief argues a masthead must
be drawn rather than set, because a typeface is a general solution and general
solutions read as general. The wordmark here is `JJ's Journal`, supplied as
artwork set in UnifrakturMaguntia with Libre Baskerville for the dateline and
tagline, and it is kept as type on purpose -- it is the identity that exists.
What the supplied SVG could not do, this does:

  - the baked `#f4f1e8` ground is dropped. It was the cream the brief bans,
    and it sat wrong on the salmon sheet.
  - fills become `currentColor`, so the masthead flips with the reading light
    instead of staying ink-black on a dark board.
  - the volume, number and date were baked into the artwork. They are live
    here, driven by the record, because a masthead that prints a fixed date is
    wrong every day but one.

NOTHING ON THIS PAGE IS DECORATIVE ABOUT ITS NUMBERS. Every rate carries the
sample it was computed over, and where there is no measurement the page says so
rather than printing a zero.
"""
from __future__ import annotations

import datetime as dt
import html
import json

from . import model

MINUS = "−"


# --- formatting ---------------------------------------------------------------
def esc(text) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def signed(value: float, places: int = 2, suffix: str = "") -> str:
    """Always an explicit sign, always U+2212 for negatives.

    A hyphen is narrower than a plus even in a tabular face, so a signed column
    set with hyphens visibly steps in and out. U+2212 is drawn to the same
    width as the plus.
    """
    sign = "+" if value >= 0 else MINUS
    return "%s%.*f%s" % (sign, places, abs(value), suffix)


def dollars(cents_value: float, places: int = 2) -> str:
    """Signed money. The page reports P&L in dollars, not units.

    Units earn their keep when the bankroll moves, so that "+18u" means the
    same thing at any size. `paper.cycle` sizes every position against
    `starting_cash_cents`, so this bankroll does not move and a unit was a flat
    $10 -- every `u` figure was the dollar figure divided by ten, which is a
    second representation carrying no second fact.

    Closing line value stays in cents: it is a price difference per contract,
    not an amount of money.
    """
    sign = "+" if cents_value >= 0 else MINUS
    return "%s$%s" % (sign, "{:,.{p}f}".format(abs(cents_value) / 100.0,
                                               p=places))


def cents(value: float, places: int = 1) -> str:
    return signed(value, places, "¢")


def money(cents_value: float) -> str:
    whole = cents_value / 100.0
    sign = MINUS if whole < 0 else ""
    return "%s$%s" % (sign, "{:,.2f}".format(abs(whole)))


def pct(fraction, places: int = 1) -> str:
    """A percentage, with a true minus when it is negative.

    `-7.5%` slipped through here while every other signed figure on the page
    used U+2212. In a tabular face the hyphen is narrower than the plus, so a
    column mixing the two steps in and out by a fraction of a character.
    """
    if fraction is None:
        return "—"
    return ("%.*f%%" % (places, 100.0 * fraction)).replace("-", MINUS)


def n_of(count) -> str:
    """The parenthetical nobody else prints. `62.1%` is decoration."""
    return '<i>n=%s</i>' % esc(count)


def or_dash(value, formatter):
    """No measurement is not the same claim as a measurement of zero."""
    return "—" if value is None else formatter(value)


# --- the lede -----------------------------------------------------------------
def lede(s: dict) -> str:
    """A written summary, composed from the record rather than an LLM call.

    The brief suggests generating this with a model on every load. It is
    written from the data instead: the page is rebuilt by a workflow that
    already spends model time on the board itself, and a build that can fail
    for want of an API key is a worse page than one whose prose is assembled.
    What matters is that it ends on a conclusion, and a conclusion drawn from
    thresholds is still a conclusion.
    """
    board, clv, pnl = s["board"], s["clv"], s["pnl"]
    acted = board["n_acted"]
    considered = board["markets_considered"]

    first = (
        "The board sat on {fx} fixture{fxs} and read {mk:,} market{mks} to do "
        "it. It acted on {act} and declined the rest — {rate} of the "
        "record so far is a decision not to bet.".format(
            fx=board["n_fixtures"], fxs="" if board["n_fixtures"] == 1 else "s",
            mk=considered, mks="" if considered == 1 else "s",
            act=acted or "none",
            rate=or_dash(board["decline_rate"], lambda v: pct(v, 0)))
        if board["n_fixtures"] else
        "The board has not sat yet. Nothing here is a measurement.")

    if clv["mean_cents"] is None:
        second = ("No position has reached a closing line, so there is no "
                  "closing line value to report and none is invented below.")
    else:
        second = (
            "Closing line value runs {clv} a contract across {n} settled "
            "fixture{s}, which is far too short a sample to mean anything "
            "and is printed with its denominator for exactly that reason."
            .format(clv=cents(clv["mean_cents"]), n=clv["n_fixtures"],
                    s="" if clv["n_fixtures"] == 1 else "s"))
        if clv["n_bets"] > clv["n_fixtures"]:
            second += (
                " The {b} bets behind it sit on {f} match{es}: correlated "
                "markets off one scoreline grid are one view written several "
                "ways, not several views.".format(
                    b=clv["n_bets"], f=clv["n_fixtures"],
                    es="" if clv["n_fixtures"] == 1 else "es"))

    if pnl["n_settled"]:
        third = ("The book is {net} on {n} settled market{s} against "
                 "{st} staked.".format(
                     net=dollars(pnl["realized_cents"]), n=pnl["n_settled"],
                     s="" if pnl["n_settled"] == 1 else "s",
                     st=dollars(pnl["staked_cents"])))
    else:
        third = "Nothing has settled yet."

    # The conclusion. Thresholds, but stated as a judgement rather than hedged.
    if pnl["n_settled"] < 20:
        verdict = ("Nothing here argues for changing the process, because "
                   "nothing here is yet a sample. The number worth watching "
                   "is the decline rate, not the profit.")
    elif clv["mean_cents"] is not None and clv["mean_cents"] > 0:
        verdict = ("The forecasts are beating the number the market closed "
                   "at. That is the only result on this page that has begun "
                   "to earn its sample.")
    else:
        verdict = ("Closing line value is not yet positive. Until it is, the "
                   "profit column is variance wearing a result's clothes.")

    body = "".join("<p>%s</p>" % esc(p) for p in (first, second, third))
    return body + "<p>%s</p>" % esc(verdict)


# --- sections -----------------------------------------------------------------
def kicker(text: str) -> str:
    return '<p class="kicker">%s</p>' % esc(text)


def masthead(s: dict, now: dt.datetime) -> str:
    """JJ's Journal, rewired: live dateline, currentColor, no baked ground."""
    # The supplied artwork stacks: double rule, dateline, wordmark, tagline
    # between flanking rules, double rule. That order is kept exactly; only the
    # content of the dateline becomes live.
    fixtures = s["board"]["n_fixtures"]
    volume = 1 + (now.year - 2026)
    dateline = [
        "Vol. %s" % _roman(volume),
        "No. %d" % max(1, len(s["daily"])),
        "%d fixture%s boarded" % (fixtures, "" if fixtures == 1 else "s"),
        "%s on the book" % money(s["cash_cents"] or 0),
    ]
    return """
  <header class="masthead">
    <div class="rule-double"></div>
    <div class="dateline">
      <span class="edition">%s</span>
      <span class="spacer"></span>
      %s
      <div class="modes" role="group" aria-label="Reading light">
        <button id="mLight" aria-pressed="true">Printed</button>
        <button id="mDark" aria-pressed="false">Lit</button>
      </div>
    </div>
    <h1 class="wordmark">JJ&rsquo;s Journal</h1>
    <p class="tagline"><span>A site for sore eyes</span></p>
    <div class="rule-double flip"></div>
  </header>""" % (
        esc(now.strftime("%A, %B %d, %Y").upper()),
        "".join("<span>%s</span>" % esc(d) for d in dateline))


def _roman(n: int) -> str:
    numerals = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    out = ""
    for value, sign in numerals:
        while n >= value:
            out += sign
            n -= value
    return out or "I"


def form_band(s: dict) -> str:
    figures = s["form"]
    if not figures:
        body = ('<p class="empty">No fixture has been boarded yet. The record '
                'starts at the first sitting.</p>')
    else:
        cells = []
        for f in figures:
            if f["kind"] == "brk":
                cells.append('<span class="fig brk" aria-hidden="true">%s</span>'
                             % esc(f["char"]))
            else:
                # No entry animation and therefore no stagger delay: the
                # record is looked at daily, and the frequency rule says a
                # thing seen that often gets no motion at all.
                cells.append(
                    '<button class="fig" data-r="%s" data-d="%s">%s</button>'
                    % (esc(f["kind"]), esc(f["detail"]), esc(f["char"])))
        body = '<div class="figures" id="figures">%s</div>' % "".join(cells)
    return """
  <section class="formband">
    %s
    %s
    <div class="figkey">
      <span>&middot; = boarded, declined</span>
      <span>digit = markets that cashed on a fixture that finished down</span>
      <span>&#10003; = fixture finished up</span>
      <span>/ = month</span>
      <span class="figread" id="figread" data-rest="1" aria-live="polite">Hover or focus a figure to read its fixture</span>
    </div>
  </section>""" % (kicker("Form · %d fixtures, oldest first" % len(
        [f for f in figures if f["kind"] != "brk"])), body)


def leaders(s: dict) -> str:
    clv, pnl, fills, board = s["clv"], s["pnl"], s["fills"], s["board"]
    rows = [
        ("Closing line value",
         or_dash(clv["mean_cents"], cents), n_of(clv["n_fixtures"]), True),
        ("Declined to bet", or_dash(board["decline_rate"], lambda v: pct(v, 0)),
         n_of(board["n_fixtures"]), False),
        ("Net", dollars(pnl["realized_cents"]),
         "<i>over %d settled markets</i>" % pnl["n_settled"], False),
        ("Return on stake", or_dash(pnl["roi"], pct),
         "<i>%s staked</i>" % esc(money(pnl["staked_cents"])), False),
        ("Strike rate", or_dash(pnl["strike_rate"], pct),
         n_of(pnl["n_settled"]), False),
        ("Resting orders filled", or_dash(fills["rate"], pct),
         n_of(fills["n_resolved"]), False),
        ("Markets read", "{:,}".format(board["markets_considered"]),
         "<i>median %s</i>" % esc(
             or_dash(board["median_considered"], lambda v: "%.0f" % v)), False),
    ]
    out = []
    for label, value, note, head in rows:
        out.append(
            '<div%s><dt>%s</dt><span class="dots"></span><dd>%s%s</dd></div>'
            % (' class="head"' if head else "", esc(label), esc(value), note))
    return '<dl class="leaders">%s</dl>' % "".join(out)


def chart(s: dict) -> str:
    """Cumulative actual against cumulative expected.

    The zero line is the heaviest rule in the design; every other line on the
    plot is a hairline. Nothing is animated on load -- this is a page its owner
    will open daily, and an animation watched daily is a tax.
    """
    curve = s["equity"]
    if len(curve) < 2:
        return """
  <section class="panel">
    <div class="panel-head"><h2>Cumulative units — actual against expected</h2></div>
    <p class="empty">Two settlements are needed before a curve means anything.
    %d recorded so far.</p>
  </section>""" % len(curve)

    W, H = 720.0, 180.0
    actual = [p["actual_cents"] for p in curve]
    expected = [p["expected_cents"] for p in curve]
    lo = min(min(actual), min(expected), 0.0)
    hi = max(max(actual), max(expected), 0.0)
    span = (hi - lo) or 1.0
    pad = span * 0.12
    lo, hi = lo - pad, hi + pad
    span = hi - lo

    def x(i):
        return (i / max(1, len(curve) - 1)) * W

    def y(v):
        return H - ((v - lo) / span) * H

    def path(series):
        return " ".join(("M" if i == 0 else "L") + "%.2f %.2f" % (x(i), y(v))
                        for i, v in enumerate(series))

    zero = y(0.0)
    grid = "".join(
        '<line x1="0" y1="%.2f" x2="%.0f" y2="%.2f" stroke="var(--rule)" '
        'stroke-width=".5"/>' % (y(v), W, y(v))
        for v in (hi - pad / 2, lo + pad / 2) if abs(y(v) - zero) > 8)

    labels = "".join(
        '<text x="-10" y="%.2f" class="axis" text-anchor="end">%s</text>'
        % (y(v) + 3.5, esc(dollars(v, 0)))
        for v in (hi - pad / 2, 0.0, lo + pad / 2))

    return """
  <section class="panel">
    <div class="panel-head">
      <h2>Cumulative units — actual against expected</h2>
      <div class="legend">
        <span><i style="background:var(--series)"></i>Actual</span>
        <span><i style="background:var(--ink-soft)"></i>Expected at close</span>
        <span class="readout tnum" id="readout">%s</span>
      </div>
    </div>
    <svg class="chart" id="chart" viewBox="-58 -12 800 214" aria-hidden="true">
      %s
      %s
      <path d="%s" fill="none" stroke="var(--ink-soft)" stroke-width="1"
        stroke-dasharray="3 3"/>
      <path d="%s" fill="none" stroke="var(--series)" stroke-width="1.75"
        stroke-linejoin="round" stroke-linecap="round"/>
      <line class="zero" x1="-58" y1="%.2f" x2="742" y2="%.2f"
        stroke="var(--ink)" stroke-width="1.5"/>
      <line class="cross" id="cross" x1="0" y1="-6" x2="0" y2="%.0f"
        stroke="var(--ink)" stroke-width=".75" stroke-dasharray="2 2"/>
      <circle class="crossdot" id="crossdot" cx="0" cy="0" r="3.4"
        fill="var(--series)"/>
    </svg>
    <p class="caption">The dashed line is what the positions were worth at the
    number the market closed at; the solid line is what actually happened. The
    gap between them is variance, and at %d settlement%s it is nearly all of
    what you can see.</p>
  </section>""" % (
        esc("%s · %s" % (dollars(actual[-1]), cents(
            s["clv"]["mean_cents"] or 0.0))),
        grid, labels, path(expected), path(actual), zero, zero, H,
        len(curve), "" if len(curve) == 1 else "s")


def ledger_strip(s: dict) -> str:
    days = s["daily"]
    if not days:
        return ""
    peak = max(abs(d["pnl_cents"]) for d in days) or 1.0
    stake_peak = max(d["stake_cents"] for d in days) or 1.0
    # Width is stake, in PIXELS rather than flex-grow. With flex-grow a single
    # day expands to the full measure and the strip reads as a solid rule --
    # a filled bar where the design means one mark.
    span = max(3, min(22, round(880 / max(len(days), 40))))
    marks = []
    for d in days:
        height = max(2, round(abs(d["pnl_cents"]) / peak * 25))
        width = max(3, round(span * max(0.35, d["stake_cents"] / stake_peak)))
        marks.append(
            '<span class="mark %s" style="height:%dpx;width:%dpx" '
            'title="%s"></span>'
            % ("up" if d["pnl_cents"] >= 0 else "dn", height, width,
               esc("%s · %s · %s staked" % (d["date"], dollars(d["pnl_cents"]),
                                            money(d["stake_cents"])))))
    return "%s\n<div class=\"ledger\" role=\"img\" aria-label=\"%s\">%s</div>" % (
        kicker("Daily ledger · %d day%s · width = stake, height = result"
               % (len(days), "" if len(days) == 1 else "s")),
        esc("Daily profit and loss over %d days" % len(days)),
        "".join(marks))


CLAIM_WORDS = {
    "home_win": "%(home)s win", "away_win": "%(away)s win", "draw": "draw",
    "btts": "both teams score",
}


def claim_label(claim: str, home: str, away: str) -> str:
    """Say what the contract actually is, in words a reader parses at a glance.

    `not_1h_total_over_2.5` is precise and unreadable. The page is dense on
    purpose, but density only reads as a luxury signal when the reader can
    decode it -- otherwise it is just noise wearing a form book's clothes.
    """
    negated = claim.startswith("not_")
    base = claim[4:] if negated else claim
    half = base.startswith("1h_")
    if half:
        base = base[3:]
    names = {"home": home, "away": away}

    if base in CLAIM_WORDS:
        text = CLAIM_WORDS[base] % names
    elif base.startswith("total_over_"):
        text = "over %s goals" % base.rsplit("_", 1)[1]
    elif base.startswith("total_under_"):
        text = "under %s goals" % base.rsplit("_", 1)[1]
    # Order matters: `home_wins_by_over_` also starts with `home_`, so the
    # spread test has to come before the team-total one.
    elif base.startswith(("home_wins_by_over_", "away_wins_by_over_")):
        side = "home" if base.startswith("home") else "away"
        text = "%s by over %s" % (names[side], base.rsplit("_", 1)[1])
    elif base.startswith(("home_over_", "away_over_")):
        side, line = base.split("_over_")
        text = "%s over %s" % (names[side], line)
    elif base.startswith("score_"):
        text = "score %s" % base.split("_", 1)[1]
    else:
        text = base.replace("_", " ")

    if half:
        text = "1st half " + text
    return ("not " + text) if negated else text


STAMP = """
      <svg class="stamp" viewBox="0 0 118 54" aria-label="Paid">
        <defs>
          <filter id="rough"><feTurbulence type="fractalNoise" baseFrequency=".05" numOctaves="3" seed="9" result="n"/>
            <feDisplacementMap in="SourceGraphic" in2="n" scale="2.1" xChannelSelector="R" yChannelSelector="G"/></filter>
          <filter id="roughT"><feTurbulence type="fractalNoise" baseFrequency=".11" numOctaves="3" seed="4" result="n"/>
            <feDisplacementMap in="SourceGraphic" in2="n" scale="1.0" xChannelSelector="R" yChannelSelector="G"/></filter>
        </defs>
        <g filter="url(#rough)" fill="none" stroke="currentColor">
          <rect x="3" y="3" width="112" height="48" stroke-width="2.4"/>
          <rect x="8.5" y="8.5" width="101" height="37" stroke-width=".9"/>
        </g>
        <text filter="url(#roughT)" x="59" y="34.5" text-anchor="middle"
          font-size="19.5" fill="currentColor" stroke="none">PAID</text>
      </svg>"""


def ticket(row: dict) -> str:
    """One fixture as one slip; the markets held on it are the legs.

    A parlay's story is where the chain broke. This book has no chains -- every
    contract settles alone -- so the story is which of the correlated markets
    off one scoreline grid landed and which did not. A fixture that took six
    positions and cashed two is a different object from one that cashed none,
    and the spine is what shows it.
    """
    held = sorted(row["positions"],
                  key=lambda p: (bool(p.get("settled")),
                                 -(float(p.get("realized_pnl_cents") or 0))))
    cashed = row["settled_fixture"] and row["pnl_cents"] > 0
    live = row["n_open"] > 0

    if live:
        tag = "Live · %d of %d open" % (row["n_open"], row["n_markets"])
        # The brief's rule for a running ticket: the hero figure is what is
        # still to play for, not what has been banked. `pnl_cents` counts
        # SETTLED markets and a live fixture has none, so printing it read
        # "+0.00u" against "if every open market holds" -- a true number
        # answering a question nobody asked.
        figure, figclass = dollars(row["open_upside_cents"]), " live"
    else:
        tag = "%s · %s" % ("Cashed" if cashed else "Settled", row["date"])
        figure, figclass = dollars(row["pnl_cents"]), ""

    legs = []
    for p in held:
        pnl = float(p.get("realized_pnl_cents") or 0)
        if not p.get("settled"):
            state, glyph = "open", "&#9679;"
        elif pnl > 0:
            state, glyph = "", "&#10003;"
        else:
            state, glyph = "dead", "&#10007;"
        clv_note = ("" if p.get("clv_cents") is None
                    else " · clv %s" % cents(float(p["clv_cents"])))
        legs.append(
            '<li class="%s"><span class="dot">%s</span>'
            '<span>%s <em>%s</em></span>'
            '<span class="odds">%s%s</span></li>'
            % (state, glyph,
               esc(claim_label(str(p.get("claim") or ""), row["home"],
                               row["away"])),
               esc(str(p.get("venue") or "")[:4]),
               esc("%d¢" % round(float(p.get("avg_cost_cents") or 0))),
               esc(clv_note)))

    foot_left = ("%d markets staked" % row["n_open"] if live else
                 "%d of %d markets cashed" % (row["n_cashed"],
                                              row["n_settled"]))
    return """
    <article class="card">%s
      <div class="card-head">
        <div><span class="tag">%s</span><br><span class="big tnum%s">%s</span></div>
        <span class="tag tnum">%s</span>
      </div>
      <p class="match">%s <span>v</span> %s</p>
      <ol class="spine">%s</ol>
      <div class="card-foot"><span>%s</span><strong class="%s tnum">%s</strong></div>
    </article>""" % (
        STAMP if cashed else "", esc(tag), figclass, esc(figure),
        esc("%s · %d markets" % (str(row["league_id"]).replace("_", " "),
                                 row["n_markets"])),
        esc(row["home"]), esc(row["away"]), "".join(legs), esc(foot_left),
        "" if live else ("pos" if row["pnl_cents"] >= 0 else "neg"),
        esc(money(row["open_staked_cents"]) if live
            else dollars(row["pnl_cents"])))


def tickets(s: dict) -> str:
    acted = [r for r in s["fixtures"] if r["acted"]]
    if not acted:
        return kicker("Tickets") + (
            '<p class="empty">No position has been taken yet. Every fixture '
            'boarded so far was declined, and the reasoning is below.</p>')
    return kicker("Tickets · %d fixture%s acted on"
                  % (len(acted), "" if len(acted) == 1 else "s")) + \
        '<div class="cards">%s</div>' % "".join(ticket(r) for r in acted[:6])


def abstentions(s: dict) -> str:
    """The decision this system makes most often, given its own section.

    Nothing in the category shows what it passed on, because a tracker is built
    to display action. But a record that prints only the bets it took describes
    a more reckless system than the one that produced it, and on this book the
    declines are the overwhelming majority of the evidence.
    """
    declined = [r for r in s["fixtures"] if r["boarded"] and not r["acted"]]
    if not declined:
        return ""
    rows = []
    for r in declined[:14]:
        rows.append(
            '<tr><td>%s</td><td>%s v %s</td><td>%s</td><td class="r">%s</td>'
            '<td class="r">%s</td><td class="why">%s</td></tr>'
            % (esc(r["date"]), esc(r["home"]), esc(r["away"]),
               esc(str(r["league_id"]).replace("_", " ")),
               esc(r["markets_considered"] or "—"),
               esc("%.1fh" % r["hours_to_kickoff"]
                   if r.get("hours_to_kickoff") is not None else "—"),
               esc(model._clip(r.get("reason") or "—", 190))))
    return """
  %s
  <p class="standfirst">%s of %s boarded fixtures were passed over. That is the
  system working rather than the system idle: a market is taken only when the
  model disagrees with the price by more than the fee and the cost of crossing
  the spread.</p>
  <div class="tablewrap">
    <table>
      <thead><tr><th>Date</th><th>Fixture</th><th>League</th>
        <th class="r">Markets read</th><th class="r">To kick-off</th>
        <th>Why not</th></tr></thead>
      <tbody>%s</tbody>
    </table>
  </div>""" % (kicker("Declined"), s["board"]["n_declined"],
               s["board"]["n_fixtures"], "".join(rows))


def _clv_cell(row) -> str:
    body = or_dash(row["clv_cents"], cents)
    return body + (" (n=%d)" % row["n_clv"] if row["n_clv"] else "")


def breakdown(s: dict) -> str:
    """Where the exposure sits, by league and by market family."""
    if not s["leagues"]:
        return ""
    league_rows = "".join(
        '<tr><td>%s</td><td class="r">%s</td><td class="r">%s</td>'
        '<td class="r">%s</td><td class="r %s">%s</td></tr>'
        % (esc(str(r["league_id"]).replace("_", " ")), esc(r["n_fixtures"]),
           esc(r["n_markets"]), esc(_clv_cell(r)),
           "w" if r["pnl_cents"] >= 0 else "l", esc(dollars(r["pnl_cents"])))
        for r in s["leagues"])
    family_rows = "".join(
        '<tr><td>%s</td><td>%s</td><td class="r">%s</td>'
        '<td class="r">%s</td><td class="r %s">%s</td></tr>'
        % (esc(r["period"]), esc(r["family"]), esc(r["n"]), esc(_clv_cell(r)),
           "w" if r["pnl_cents"] >= 0 else "l", esc(dollars(r["pnl_cents"])))
        for r in s["families"])
    return """
  <div class="grid2 tight">
    <div>
      %s
      <div class="tablewrap">
        <table><thead><tr><th>League</th><th class="r">Fixtures</th>
          <th class="r">Markets</th><th class="r">CLV</th><th class="r">Net</th>
        </tr></thead><tbody>%s</tbody></table>
      </div>
    </div>
    <div>
      %s
      <div class="tablewrap">
        <table><thead><tr><th>Half</th><th>Family</th><th class="r">Markets</th>
          <th class="r">CLV</th><th class="r">Net</th>
        </tr></thead><tbody>%s</tbody></table>
      </div>
    </div>
  </div>""" % (kicker("By league"), league_rows,
               kicker("By market family"), family_rows)


def bet_table(s: dict) -> str:
    held = []
    for row in s["fixtures"]:
        for p in row["positions"]:
            held.append((row, p))
    if not held:
        return ""
    held.sort(key=lambda rp: str(rp[1].get("opened_at") or ""), reverse=True)
    rows = []
    for row, p in held[:24]:
        settled = bool(p.get("settled"))
        pnl = float(p.get("realized_pnl_cents") or 0)
        result = dollars(pnl) if settled else "open"
        cls = ("" if not settled else "w" if pnl >= 0 else "l")
        rows.append(
            '<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
            '<td class="r">%s</td><td class="r">%s</td>'
            '<td class="r">%s</td><td class="r %s">%s</td></tr>'
            % (esc(row["date"]), esc("%s v %s" % (row["home"], row["away"])),
               esc(claim_label(str(p.get("claim") or ""), row["home"],
                               row["away"])),
               esc(p.get("venue")),
               esc("%d¢" % round(float(p.get("avg_cost_cents") or 0))),
               esc("%g" % float(p.get("size") or 0)),
               esc("—" if p.get("clv_cents") is None
                   else cents(float(p["clv_cents"]))),
               cls, esc(result)))
    return kicker("Every market held") + """
  <div class="tablewrap">
    <table>
      <thead><tr><th>Date</th><th>Fixture</th><th>Market</th><th>Venue</th>
        <th class="r">Paid</th><th class="r">Size</th><th class="r">CLV</th>
        <th class="r">Result</th></tr></thead>
      <tbody>%s</tbody>
    </table>
  </div>""" % "".join(rows)


def colophon(s: dict) -> str:
    return """
  <footer class="colophon">
    <p><b>Colophon</b></p>
    <p>The wordmark is set in UnifrakturMaguntia, the dateline and tagline in
    Libre Baskerville. Everything else is Archivo across its width and weight
    axes, narrow for the tables, with Martian Mono reserved for the form line
    where fixed width carries meaning rather than mood. Figures use tabular
    lining sets throughout, so no column shifts as a value updates, and signed
    values take a true minus rather than a hyphen, because the two are not the
    same width. Rules follow a print vocabulary: hairline for row separation,
    thin for structure, thick for section breaks, and an Oxford rule — thick
    over hairline — beneath the masthead alone. The ground is the salmon of the
    sporting and financial press. Losses take the only colour on the page;
    profit is carried by weight and by position above the zero rule, which is
    the heaviest line in the design. Under the lit reading the same sheet
    becomes a tote board, and amber replaces ink.</p>
    <p>Every rate prints the sample it was computed over. Rates average over
    fixtures rather than bets, because one match can produce thirty correlated
    markets off a single scoreline grid and counting them separately would
    overstate the evidence roughly thirtyfold. Where there is no measurement
    the page prints an em dash rather than a zero. The record is paper: no
    order here was ever placed with a venue.</p>
    <p>Rebuilt from the ledger on every board and maintenance run.
    Ledger last written %s.</p>
  </footer>""" % esc(s.get("saved_at") or "—")


# Tokens, type roles and motion values are the brief's, unchanged. The masthead
# block is the only addition: the supplied artwork's proportions, rebuilt so the
# dateline can be live and the whole thing can flip with the reading light.
CSS = """
:root{
  --display:Archivo,system-ui,sans-serif;
  --ui:Archivo,system-ui,sans-serif;
  --dense:"Archivo Narrow",Archivo,system-ui,sans-serif;
  --fig:"Martian Mono",ui-monospace,"SFMono-Regular",monospace;
  --black:"UnifrakturMaguntia","Libre Baskerville",Georgia,serif;
  --serif:"Libre Baskerville",Georgia,serif;
  --rule-hair:.5px; --rule-thin:1px; --rule-thick:3px;
  --ease-out:cubic-bezier(.23,1,.32,1);
  --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:24px; --s6:36px; --s7:56px;
  --max:1040px;
}
html[data-mode="light"]{
  --paper:#FFF1E5; --inset:#FDFCFA; --ink:#16181C; --ink-mid:#5C6068;
  --ink-soft:#A79C93; --rule:#E2D3C6; --rule-soft:#F0E5DA;
  --loss:#C1372E; --live:#00595A; --win:#16181C; --series:#16181C;
  --panel-bg:#FDFCFA; --panel-line:var(--ink); --panel-shadow:none;
  --radius:0px; --grain:.05; --grain-blend:multiply; --glow:none;
}
html[data-mode="dark"]{
  --paper:#0A0A0B; --inset:#131317; --ink:#E9E4D9; --ink-mid:#98938A;
  --ink-soft:#5E5952; --rule:#2A2A30; --rule-soft:#1B1B20;
  --loss:#E5484D; --live:#F2A93B; --win:#F2A93B; --series:#F2A93B;
  --panel-bg:linear-gradient(180deg,#18181D 0%,#131317 55%);
  --panel-line:#2A2A30;
  --panel-shadow:0 0 0 1px rgb(255 255 255/.05),0 2px 6px rgb(0 0 0/.5),
    0 20px 48px rgb(0 0 0/.35);
  --radius:2px; --grain:.055; --grain-blend:overlay;
  --glow:0 0 22px rgb(242 169 59/.16);
}
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--ui);
  font-size:15px;line-height:1.5;
  transition:background 260ms var(--ease-out),color 260ms var(--ease-out)}
.tnum,table,.leaders dd{font-variant-numeric:tabular-nums lining-nums slashed-zero}
body::after{content:"";position:fixed;inset:0;pointer-events:none;z-index:99;
  opacity:var(--grain);mix-blend-mode:var(--grain-blend);
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}
.sheet{max-width:var(--max);margin:0 auto;padding:var(--s6) var(--s5) var(--s6);
  position:relative}
.crop{position:absolute;width:15px;height:15px;pointer-events:none;
  border-color:var(--ink-soft);opacity:.55}
.crop.tl{top:14px;left:6px;border-top:var(--rule-thin) solid;border-left:var(--rule-thin) solid}
.crop.tr{top:14px;right:6px;border-top:var(--rule-thin) solid;border-right:var(--rule-thin) solid}
.crop.bl{bottom:14px;left:6px;border-bottom:var(--rule-thin) solid;border-left:var(--rule-thin) solid}
.crop.br{bottom:14px;right:6px;border-bottom:var(--rule-thin) solid;border-right:var(--rule-thin) solid}

/* ── MASTHEAD ── the supplied artwork, rebuilt live ──────────── */
.rule-double{border-top:var(--rule-thick) solid var(--ink);position:relative;
  padding-top:5px}
.rule-double::after{content:"";position:absolute;left:0;right:0;top:8px;
  height:var(--rule-thin);background:var(--ink)}
.rule-double.flip{border-top:var(--rule-thin) solid var(--ink);
  margin-top:var(--s4);padding-top:0}
.rule-double.flip::after{top:5px;height:var(--rule-thick)}
.dateline{display:flex;align-items:center;gap:var(--s3);flex-wrap:wrap;
  margin:15px 0 0;font-family:var(--serif);font-size:10.5px;
  letter-spacing:.19em;text-transform:uppercase;color:var(--ink-mid)}
.dateline .spacer{flex:1 1 auto}
.dateline .edition{letter-spacing:.22em}
.wordmark{font-family:var(--black);font-weight:400;
  font-size:clamp(46px,11.5vw,132px);line-height:1.04;text-align:center;
  margin:var(--s4) 0 0;color:var(--ink);letter-spacing:.01em;
  text-wrap:balance}
.tagline{margin:var(--s3) 0 0;display:flex;align-items:center;gap:var(--s4);
  color:var(--ink-mid)}
.tagline::before,.tagline::after{content:"";flex:1 1 auto;
  height:var(--rule-thin);background:var(--ink)}
.tagline span{font-family:var(--serif);font-style:italic;font-size:15px;
  letter-spacing:.09em;white-space:nowrap}
.modes{display:flex;border:var(--rule-thin) solid var(--ink);
  border-radius:var(--radius)}
.modes button{appearance:none;background:none;border:0;cursor:pointer;
  color:var(--ink-mid);font-family:var(--dense);font-size:10px;font-weight:600;
  letter-spacing:.12em;text-transform:uppercase;padding:6px 11px;
  transition:background 140ms var(--ease-out),color 140ms var(--ease-out)}
.modes button[aria-pressed="true"]{background:var(--ink);color:var(--paper)}
.modes button:focus-visible{outline:2px solid var(--live);outline-offset:2px}

/* ── KICKERS + FORM ─────────────────────────────────────────── */
.kicker{display:flex;align-items:center;gap:10px;margin:var(--s6) 0 var(--s3);
  font-family:var(--dense);font-size:10.5px;font-weight:700;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-mid)}
.kicker::before{content:"";width:22px;height:var(--rule-thick);
  background:var(--ink);flex:none}
.figures{display:grid;grid-template-columns:repeat(24,minmax(0,1fr));gap:4px;
  font-family:var(--fig);font-size:clamp(12px,1.9vw,22px);font-weight:500;
  line-height:1}
.fig{position:relative;padding:9px 0;min-width:0;width:100%;text-align:center;
  border:0;background:none;color:var(--ink-soft);cursor:pointer;font:inherit;
  border-radius:var(--radius);
  transition:color 140ms var(--ease-out),background 140ms var(--ease-out)}
.fig[data-r="cash"]{color:var(--ink);font-weight:700}
.fig[data-r="late"]{color:var(--ink)}
.fig[data-r="early"]{color:var(--ink)}
.fig[data-r="declined"]{color:var(--ink-soft)}
.fig.brk{color:var(--ink-soft);cursor:default;opacity:.5;align-self:center}
.fig:hover,.fig:focus-visible{background:var(--rule-soft);color:var(--ink);
  outline:none}
.fig[data-r="cash"]::after{content:"";position:absolute;left:22%;right:22%;
  bottom:1px;height:2px;background:var(--live)}
.figkey{margin:var(--s3) 0 0;font-family:var(--dense);font-size:11px;
  letter-spacing:.03em;color:var(--ink-mid);display:flex;gap:var(--s4);
  flex-wrap:wrap;align-items:baseline}
.figread{font-family:var(--fig);font-size:11.5px;color:var(--ink);
  flex:1 1 100%;min-height:2.6em;display:flex;align-items:center;
  padding:var(--s2) var(--s3);margin-top:var(--s3);
  background:var(--inset);border:var(--rule-thin) solid var(--rule);
  border-radius:var(--radius);letter-spacing:0;text-wrap:pretty}
.figread[data-rest="1"]{color:var(--ink-soft);font-family:var(--dense);
  letter-spacing:.06em;text-transform:uppercase;font-size:10.5px}
.formband{margin-bottom:var(--s6)}
html[data-mode="dark"] .fig:not(.brk){background:#1D1D22;
  box-shadow:inset 0 0 0 1px #26262C}
html[data-mode="dark"] .fig:not(.brk)::before{content:"";position:absolute;
  left:0;right:0;top:50%;height:1px;background:#0A0A0B;opacity:.85}
html[data-mode="dark"] .fig[data-r="cash"]{color:var(--live);
  text-shadow:0 0 14px rgb(242 169 59/.45)}
html[data-mode="dark"] .fig:hover{background:#26262C}

/* ── LEDE + LEADERS ─────────────────────────────────────────── */
.grid2{display:grid;grid-template-columns:1.55fr 1fr;gap:var(--s6);
  align-items:start;margin-bottom:var(--s5)}
.grid2.tight{grid-template-columns:1fr 1fr;gap:var(--s5)}
.lede{font-size:15px;line-height:1.66;text-wrap:pretty;
  hanging-punctuation:first;max-width:92ch;margin-bottom:var(--s6)}
.lede p{margin:0 0 .9em}
.lede p:first-of-type::first-letter{float:left;font-family:var(--black);
  font-size:3.4em;line-height:.82;margin:.02em .10em -.06em 0;color:var(--ink)}
dl.leaders{margin:0 0 var(--s6);font-family:var(--dense)}
dl.leaders>div{display:flex;align-items:baseline;gap:7px;padding:9px 0;
  border-bottom:var(--rule-hair) solid var(--rule)}
dl.leaders>div:first-child{border-top:var(--rule-thin) solid var(--ink);
  padding-top:11px}
dl.leaders dt{margin:0;font-size:11.5px;font-weight:600;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-mid);white-space:nowrap}
dl.leaders .dots{flex:1 1 auto;
  border-bottom:var(--rule-hair) dotted var(--ink-soft);
  transform:translateY(-3px);min-width:14px}
dl.leaders dd{margin:0;font-size:16px;font-weight:700;white-space:nowrap}
dl.leaders dd i{font-style:normal;font-size:10.5px;font-weight:400;
  color:var(--ink-soft);margin-left:5px}
dl.leaders>div.head dd{font-family:var(--fig);font-size:24px;font-weight:600;
  letter-spacing:-.03em}

/* ── PANELS + CHART ─────────────────────────────────────────── */
.panel{background:var(--panel-bg);border:var(--rule-thin) solid var(--panel-line);
  border-radius:var(--radius);box-shadow:var(--panel-shadow);
  padding:var(--s5);margin-bottom:var(--s5)}
.panel-head{display:flex;justify-content:space-between;align-items:baseline;
  gap:var(--s4);flex-wrap:wrap;margin-bottom:var(--s5)}
.panel-head h2{margin:0;font-family:var(--dense);font-size:12px;font-weight:700;
  letter-spacing:.14em;text-transform:uppercase}
.legend{display:flex;gap:var(--s4);font-family:var(--dense);font-size:10.5px;
  letter-spacing:.09em;text-transform:uppercase;color:var(--ink-mid);
  align-items:baseline}
.legend i{display:inline-block;width:15px;height:2px;margin-right:6px;
  vertical-align:middle}
.chart{width:100%;height:auto;display:block;overflow:visible;touch-action:none}
.chart .axis{font-family:var(--dense);font-size:9px;fill:var(--ink-soft);
  font-variant-numeric:tabular-nums lining-nums}
.cross,.crossdot{opacity:0;transition:opacity 120ms var(--ease-out)}
.chart.on .cross,.chart.on .crossdot{opacity:1}
.readout{font-family:var(--fig);font-size:11px;color:var(--ink-mid);
  min-width:22ch;text-align:right;text-transform:none;letter-spacing:0}
.caption,.standfirst{margin:var(--s4) 0 0;font-size:13px;line-height:1.55;
  color:var(--ink-mid);max-width:70ch;text-wrap:pretty}
.standfirst{margin:0 0 var(--s4)}
.empty{margin:0 0 var(--s5);padding:var(--s4);font-size:13.5px;
  color:var(--ink-mid);border:var(--rule-thin) dashed var(--rule);
  border-radius:var(--radius);text-wrap:pretty}

/* ── LEDGER ─────────────────────────────────────────────────── */
.ledger{position:relative;height:58px;display:flex;align-items:center;gap:2px;
  margin-bottom:var(--s5)}
.ledger::before{content:"";position:absolute;left:0;right:0;top:50%;
  height:var(--rule-thin);background:var(--ink)}
.mark{position:relative;flex:0 0 auto;border-radius:1px}
.mark.up{background:var(--win);align-self:flex-end;margin-bottom:29px;
  box-shadow:var(--glow)}
.mark.dn{background:var(--loss);align-self:flex-start;margin-top:29px}

/* ── TICKETS ────────────────────────────────────────────────── */
/* auto-FILL with a max track, not auto-fit with 1fr: a lone ticket must
   stay a slip rather than stretching the width of the sheet. */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,384px));
  gap:var(--s4);margin-bottom:var(--s5)}
.card{display:flex;flex-direction:column;position:relative;overflow:hidden;
  background:var(--panel-bg);border:var(--rule-thin) solid var(--panel-line);
  border-radius:var(--radius);box-shadow:var(--panel-shadow);padding:var(--s4)}
.card-head{display:flex;justify-content:space-between;align-items:baseline;
  gap:var(--s3);padding-bottom:10px;
  border-bottom:var(--rule-thin) solid var(--ink);margin-bottom:var(--s3)}
.card-head .tag{font-family:var(--dense);font-size:10px;font-weight:700;
  letter-spacing:.13em;text-transform:uppercase;color:var(--ink-mid)}
.card-head .big{font-family:var(--fig);font-size:19px;font-weight:600;
  letter-spacing:-.025em}
.card-head .big.live{color:var(--live)}
.match{margin:0 0 var(--s3);font-family:var(--dense);font-size:14px;
  font-weight:700;letter-spacing:.01em}
.match span{color:var(--ink-soft);font-weight:400;font-style:italic;
  padding:0 3px}
.spine{list-style:none;margin:0 0 var(--s4);padding:0}
.spine li{position:relative;padding-left:25px;min-height:33px;display:flex;
  align-items:center;justify-content:space-between;gap:var(--s3);
  font-size:13.5px}
.spine li::before{content:"";position:absolute;left:6px;top:17px;bottom:-1px;
  width:var(--rule-thin);background:var(--rule)}
.spine li:last-child::before{display:none}
.spine .dot{position:absolute;left:0;top:10px;width:13px;height:13px;
  border-radius:99px;border:1.5px solid var(--win);background:var(--win);
  color:var(--paper);display:grid;place-items:center;font-size:8px;
  line-height:1;font-family:var(--fig)}
.spine .odds{font-family:var(--dense);font-size:11px;color:var(--ink-mid);
  white-space:nowrap;font-variant-numeric:tabular-nums lining-nums}
/* Two venues quoting one claim is two positions at two prices, not a repeated
   row. Without the venue the lines render identically and read as a bug. */
.spine em{font-style:normal;font-family:var(--dense);font-size:9.5px;
  font-weight:600;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-soft);margin-left:6px}

.spine li.dead,.spine li.dead .odds{color:var(--ink-soft)}
.spine li.dead .dot{background:none;border-color:var(--loss);
  color:var(--loss)}
.spine li.dead::before{background:var(--rule-soft)}
.spine li.open{color:var(--live);font-weight:500}
.spine li.open .dot{background:none;border-color:var(--live);color:var(--live);
  box-shadow:var(--glow)}
.card-foot{margin-top:auto;padding-top:10px;
  border-top:var(--rule-hair) solid var(--rule);display:flex;
  justify-content:space-between;align-items:baseline;gap:var(--s3);
  font-family:var(--dense);font-size:11px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--ink-mid)}
.card-foot strong{font-size:16px;font-weight:700;letter-spacing:0;
  text-transform:none;font-variant-numeric:tabular-nums lining-nums}
.pos{color:var(--win)} .neg{color:var(--loss)}
/* A stamp overlaps what it is stamped on -- that is what makes it read as a
   stamp rather than a badge -- but it runs off the edge of the slip so most of
   the leg stays legible, and multiply keeps the ink beneath it readable. */
.stamp{position:absolute;right:-42px;top:100px;width:132px;height:60px;
  transform:rotate(-9.5deg);color:var(--loss);opacity:.72;
  mix-blend-mode:multiply;pointer-events:none}
html[data-mode="dark"] .stamp{color:var(--live);mix-blend-mode:screen;
  opacity:.9;filter:drop-shadow(0 0 10px rgb(242 169 59/.4))}
.stamp text{font-family:var(--display);font-weight:800;letter-spacing:.06em}

/* ── TABLES ─────────────────────────────────────────────────── */
.tablewrap{border:var(--rule-thin) solid var(--panel-line);
  border-radius:var(--radius);background:var(--panel-bg);
  box-shadow:var(--panel-shadow);overflow-x:auto;margin-bottom:var(--s5)}
table{width:100%;border-collapse:collapse;font-family:var(--dense);
  font-size:13px}
thead th{text-align:left;font-size:10px;font-weight:700;letter-spacing:.13em;
  text-transform:uppercase;color:var(--ink-mid);padding:11px var(--s4);
  border-bottom:var(--rule-thin) solid var(--ink);white-space:nowrap;
  position:sticky;top:0;background:var(--panel-bg)}
thead th.r,tbody td.r{text-align:right}
tbody td{padding:10px var(--s4);
  border-bottom:var(--rule-hair) solid var(--rule-soft);white-space:nowrap}
tbody td.why{white-space:normal;min-width:280px;color:var(--ink-mid);
  font-size:12px;line-height:1.45;text-wrap:pretty}
tbody tr:last-child td{border-bottom:0}
tbody tr{transition:background 120ms var(--ease-out)}
tbody tr:hover{background:var(--rule-soft)}
td.w{font-weight:700} td.l{color:var(--loss);font-weight:700}
html[data-mode="dark"] td.w{color:var(--win)}

/* ── COLOPHON ───────────────────────────────────────────────── */
.colophon{margin-top:var(--s7);padding-top:var(--s4);
  border-top:var(--rule-thick) solid var(--ink);position:relative}
.colophon::before{content:"";position:absolute;left:0;right:0;top:5px;
  height:var(--rule-thin);background:var(--ink)}
.colophon p{margin:var(--s3) 0 0;font-family:var(--dense);font-size:11.5px;
  line-height:1.65;color:var(--ink-mid);max-width:78ch;text-wrap:pretty}
.colophon b{color:var(--ink);font-weight:700;letter-spacing:.12em;
  text-transform:uppercase;font-size:10.5px}

@media (max-width:900px){ .grid2,.grid2.tight{grid-template-columns:1fr;
  gap:var(--s5)} .lede{columns:1} }
@media (max-width:760px){ .figures{grid-template-columns:repeat(12,minmax(0,1fr))} }
@media (max-width:430px){ .figures{grid-template-columns:repeat(8,minmax(0,1fr));
  gap:3px} .sheet{padding:var(--s5) var(--s4)} .crop{display:none}
  .tagline::before,.tagline::after{display:none} }
@media (prefers-reduced-motion:reduce){
  *{animation-duration:.01ms !important;transition-duration:.01ms !important}
}
"""

JS = """
(function(){
  var read=document.getElementById('figread'),
      wrap=document.getElementById('figures'),
      REST=read?read.textContent:'';
  if(wrap){
    function show(e){var b=e.target.closest('.fig[data-d]');
      if(b){read.textContent=b.dataset.d; read.removeAttribute('data-rest');}}
    wrap.addEventListener('pointerover',show);
    wrap.addEventListener('focusin',show);
    wrap.addEventListener('pointerleave',function(){
      read.textContent=REST; read.setAttribute('data-rest','1');});
  }

  /* The readout is fixed-width with tabular figures, so the panel head does
     not reflow by a pixel as the cursor moves across the plot. */
  var chart=document.getElementById('chart');
  if(chart&&window.CURVE&&CURVE.length>1){
    var cross=document.getElementById('cross'),
        dot=document.getElementById('crossdot'),
        out=document.getElementById('readout'),
        BASE=out.textContent, W=720;
    chart.addEventListener('pointermove',function(e){
      var r=chart.getBoundingClientRect(),
          t=(-58+((e.clientX-r.left)/r.width)*800)/W,
          i=Math.max(0,Math.min(CURVE.length-1,Math.round(t*(CURVE.length-1)))),
          p=CURVE[i];
      cross.setAttribute('x1',p.x); cross.setAttribute('x2',p.x);
      dot.setAttribute('cx',p.x);   dot.setAttribute('cy',p.y);
      chart.classList.add('on');
      out.textContent=p.label;
    });
    chart.addEventListener('pointerleave',function(){
      chart.classList.remove('on'); out.textContent=BASE;});
  }

  var root=document.documentElement,
      bL=document.getElementById('mLight'), bD=document.getElementById('mDark');
  function light(m){
    root.dataset.mode=m;
    bL.setAttribute('aria-pressed',String(m==='light'));
    bD.setAttribute('aria-pressed',String(m==='dark'));
    try{localStorage.setItem('jj-mode',m);}catch(err){}
  }
  bL.onclick=function(){light('light');}; bD.onclick=function(){light('dark');};
  var saved=null;
  try{saved=localStorage.getItem('jj-mode');}catch(err){}
  if(saved==='light'||saved==='dark') light(saved);
  else if(matchMedia('(prefers-color-scheme: dark)').matches) light('dark');
})();
"""


def crosshair_data(s: dict) -> str:
    """Precomputed plot coordinates, so hover does no arithmetic."""
    curve = s["equity"]
    if len(curve) < 2:
        return "[]"
    W, H = 720.0, 180.0
    actual = [p["actual_cents"] for p in curve]
    expected = [p["expected_cents"] for p in curve]
    lo = min(min(actual), min(expected), 0.0)
    hi = max(max(actual), max(expected), 0.0)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.12, hi + span * 0.12
    span = hi - lo
    points = []
    for i, p in enumerate(curve):
        points.append({
            "x": round((i / max(1, len(curve) - 1)) * W, 2),
            "y": round(H - ((p["actual_cents"] - lo) / span) * H, 2),
            "label": "settle %d · %s actual · %s at close"
                     % (i + 1, dollars(p["actual_cents"]),
                        dollars(p["expected_cents"])),
        })
    return json.dumps(points)


FONTS = (
    "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&display=swap",
    "https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@400;500;600;700&display=swap",
    "https://fonts.googleapis.com/css2?family=Martian+Mono:wght@200..700&display=swap",
    "https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap",
    "https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&display=swap",
)


def page(portfolio: dict, now=None) -> str:
    """The whole sheet.

    Each family is requested separately. A single combined request with one
    malformed axis range returns a 400 and takes every other face down with it.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    s = model.summary(portfolio)
    links = "\n".join(
        '<link href="%s" rel="stylesheet">' % f for f in FONTS)

    return """<!DOCTYPE html>
<html lang="en" data-mode="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JJ&rsquo;s Journal</title>
<meta name="description" content="A paper trading record: closing line value, \
what was bet, and what was declined.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
%s
<style>%s</style>
</head>
<body>
<div class="sheet">
  <i class="crop tl"></i><i class="crop tr"></i><i class="crop bl"></i><i class="crop br"></i>
%s
%s
  <div class="lede">%s</div>
  %s
%s
%s
%s
%s
%s
%s
%s
</div>
<script>window.CURVE=%s;</script>
<script>%s</script>
</body>
</html>
""" % (links, CSS, masthead(s, now), form_band(s), lede(s), leaders(s),
       chart(s), ledger_strip(s), tickets(s), bet_table(s), breakdown(s),
       abstentions(s), colophon(s), crosshair_data(s), JS)
