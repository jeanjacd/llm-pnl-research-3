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

from . import ledger, model

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
                     st=money(pnl["staked_cents"])))
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


def ledger_view(portfolio: dict, now) -> dict:
    """The ledger's own slice of the record, which depends on the clock.

    `model.summary` is pure over the portfolio and deliberately has no notion
    of "now". The ledger does: whether a position is in play, awaiting a
    result, or has not kicked off is a fact about the time the page was built,
    and is printed as such.
    """
    rows = ledger.open_tickets(portfolio, now=now)
    cols = ledger.columns(portfolio, now=now)
    return {"open": rows, "exposure": ledger.exposure(rows),
            "columns": cols, "scale": ledger.scale(cols),
            "run": ledger.run(cols),
            "settled_fixtures": ledger.settled_fixtures(portfolio),
            "settlements": ledger.settlements(portfolio)}


def _daystamp(col) -> str:
    """"Sep 4", without the leading zero, on a platform-independent path."""
    stamp = col.get("stamp") or col.get("kickoff")
    if not stamp:
        return str(col.get("date") or "")
    return "%s %d" % (stamp.strftime("%b"), stamp.day)


def _when(row, now) -> str:
    """The state of a fixture, in words, honest about being a static build."""
    minutes = row.get("minutes_to_kickoff")
    if row["state"] == "live":
        played = int(-(minutes or 0))
        return "In play · about %d min gone at press time" % max(0, played)
    if row["state"] == "awaiting":
        return "Full time · waiting on the result"
    if minutes is None:
        return "Kick-off not recorded"
    if minutes < 60:
        return "Kicks off in %d min" % int(minutes)
    if minutes < 48 * 60:
        return "Kicks off in %dh %02dm" % (int(minutes // 60),
                                           int(minutes % 60))
    return "Kicks off %s" % (row["kickoff"].strftime("%a %d %b, %H:%M UTC")
                             if row.get("kickoff") else "later")


def open_rail(s: dict, now) -> str:
    """Zone 1. What is riding right now, and what has to happen.

    THE FIXTURE IS THE CARD, NOT THE BET. This book takes up to four correlated
    markets off one scoreline grid, and `paper/clv.py` establishes the fixture
    as the unit of independence for exactly that reason. Four cards for one
    match would show the same opinion four times and read as four opinions.
    """
    rows = s["open"]
    exposure = s["exposure"]
    if not rows:
        settled = s["pnl"]["n_settled"]
        return ("""
  <section class="open empty" aria-labelledby="open-h">
    %s
    <p class="nothing">Nothing open. <span>%d market%s settled, and the book is
    flat until the next sitting.</span></p>
  </section>""" % (kicker("Open"), settled, "" if settled == 1 else "s"))

    parts = []
    for label, count in (("in play", exposure["n_live"]),
                         ("awaiting settlement", exposure["n_awaiting"]),
                         ("still to kick off", exposure["n_pending"])):
        if count:
            parts.append("%d %s" % (count, label))
    standfirst = " · ".join(parts)

    cards = []
    for row in rows:
        legs = []
        for leg in row["legs"]:
            price = float(leg.get("avg_cost_cents") or 0)
            size = float(leg.get("size") or 0)
            clv = leg.get("clv_cents")
            legs.append(
                '<div class="leg"><dt>%s<small>%s</small></dt>'
                '<dd class="tnum">%s<span class="times">&times;</span>%s'
                '<b>%s</b>%s</dd></div>'
                % (esc(claim_label(str(leg.get("claim") or ""), row["home"],
                                   row["away"])),
                   esc(str(leg.get("venue") or "")[:4]),
                   esc("%d¢" % round(price)), esc("%d" % round(size)),
                   esc(money(size * price)),
                   "" if clv is None
                   else '<i class="clvtag %s">%s</i>'
                        % ("up" if float(clv) >= 0 else "dn",
                           esc(cents(float(clv))))))
        # One sentence, from the largest position on the fixture. The others
        # are listed above it; repeating a sentence per leg would bury it.
        instruction = row["legs"][0].get("needs") or ""
        cards.append("""
      <article class="ticket %s" role="listitem" tabindex="0"
               aria-label="%s">
        <p class="state"><i></i>%s</p>
        <h3 class="pair">%s <span>v</span> %s</h3>
        <p class="comp">%s · %d market%s</p>
        <dl class="legs">%s</dl>
        <p class="needs">%s</p>
        <div class="tfoot"><span>%s at risk</span><strong>%s if it all lands</strong></div>
      </article>""" % (
            esc(row["state"]),
            esc("%s versus %s, %s. %s at risk across %d markets."
                % (row["home"], row["away"], _when(row, now),
                   money(row["staked_cents"]), row["n_legs"])),
            esc(_when(row, now)), esc(row["home"]), esc(row["away"]),
            esc(str(row["league_id"] or "").replace("_", " ")),
            row["n_legs"], "" if row["n_legs"] == 1 else "s",
            "".join(legs), esc(instruction),
            esc(money(row["staked_cents"])),
            esc(dollars(row["upside_cents"]))))

    return """
  <section class="open" aria-labelledby="open-h">
    %s
    <p class="standfirst" id="open-h"><span>%s</span>
      <span class="figs"><b class="tnum">%s</b> at risk
      <b class="tnum">%s</b> to win</span></p>
    <div class="rail" role="list">%s</div>
  </section>""" % (kicker("Open"), esc(standfirst),
                   esc(money(exposure["staked_cents"])),
                   esc(dollars(exposure["upside_cents"])),
                   "".join(cards))


def run_curve(s: dict) -> str:
    """Zone 2. Cumulative dollars on the book's own x-positions.

    Drawn on the same 0..1000 grid the book's columns sit on, so a peak in the
    curve is directly above the fixture that made it. The cone at the right is
    the open position, floor to ceiling -- exposure, not a forecast.
    """
    cols, curve = s["columns"], s["run"]
    if len(cols) < 2:
        return ""
    points = curve["points"]
    values = [p["cents"] for p in points] + [0.0]
    lo, hi = min(values), max(values)
    # The record sets the scale. The cone may extend it by half again and no
    # further; past that it simply runs off the plate, which is what an
    # exposure four times the size of the realised record actually looks like.
    room = ((hi - lo) or 1.0) * 0.5
    if curve["has_open"]:
        lo = min(lo, max(curve["floor_cents"], lo - room))
        hi = max(hi, min(curve["ceiling_cents"], hi + room))
    span = (hi - lo) or 1.0
    height_px = 96.0

    def x_at(i):
        return 1000.0 * (i + 0.5) / len(cols)

    def y_at(value):
        return height_px - (float(value) - lo) / span * height_px

    line = " ".join("%.1f,%.2f" % (x_at(p["i"]), y_at(p["cents"]))
                    for p in points)
    zero = y_at(0.0)
    cone = ""
    if curve["has_open"]:
        x_end = x_at(len(cols) - 1)
        cone = ('<polygon class="cone" points="%.1f,%.2f 1000,%.2f 1000,%.2f"/>'
                % (x_end, y_at(curve["final_cents"]),
                   y_at(curve["ceiling_cents"]), y_at(curve["floor_cents"])))
    note = ("floor %s if every open market loses, ceiling %s if every one lands"
            % (dollars(curve["floor_cents"]), dollars(curve["ceiling_cents"]))
            if curve["has_open"] else "nothing open")
    return """
  <section class="run">
    <div class="runhead"><span>Run</span>
      <b class="tnum %s">%s</b>
      <span class="runnote">%s</span></div>
    <svg class="runplot" viewBox="0 0 1000 %.0f" preserveAspectRatio="none"
         role="img" aria-label="%s">
      <line class="zero" x1="0" y1="%.2f" x2="1000" y2="%.2f"/>
      %s
      <polyline class="curve" points="%s"/>
    </svg>
  </section>""" % (
        "pos" if curve["final_cents"] >= 0 else "neg",
        esc(dollars(curve["final_cents"])), esc(note), height_px,
        esc("Cumulative profit and loss across %d fixtures, ending %s. %s"
            % (len(cols), dollars(curve["final_cents"]), note)),
        zero, zero, cone, line)


BOOK_STATES = {
    "won": "won", "lost": "lost", "push": "level", "void": "voided",
    "declined": "declined", "unfilled": "ordered, never filled",
    "open": "open", "live": "in play",
}


def _column_sentence(col) -> str:
    """The aria-label, read as a sentence rather than a set of attributes."""
    when = col["stamp"].strftime("%d %b") if col["stamp"] else col["date"]
    head = "%s, %s versus %s" % (when, col["home"], col["away"])
    state = BOOK_STATES.get(col["outcome"], col["outcome"])
    if col["outcome"] in ("won", "lost"):
        body = ("%s %s on %d markets, %s of %s staked"
                % (state, money(abs(col["net_cents"])), col["n_markets"],
                   dollars(col["net_cents"]), money(col["staked_cents"])))
    elif col["outcome"] in ("open", "live"):
        body = "%s, %s staked on %d markets" % (
            state, money(col["open_staked_cents"]), col["n_markets"])
    elif col["outcome"] == "unfilled":
        body = "ordered %d markets, filled none" % col["n_orders"]
    elif col["outcome"] == "void":
        body = "%d market%s voided, none counted" % (
            col.get("n_void") or 0, "" if col.get("n_void") == 1 else "s")
    else:
        body = state
    clv = col["clv_cents"]
    tail = ("" if clv is None else
            ", %s the close by %s"
            % ("beat" if clv >= 0 else "missed",
               cents(abs(float(clv))).lstrip("+")))
    return "%s: %s%s." % (head, body, tail)


def _column(col, scale_info, settled_slugs) -> str:
    clv = col["clv_cents"]
    track = ""
    if clv is not None:
        track = ('<i class="clvmark %s" aria-hidden="true"></i>'
                 % ("up" if float(clv) >= 0 else "dn"))

    outcome = col["outcome"]
    if outcome in ("won", "lost"):
        frac, over = ledger.height(col["net_cents"], scale_info)
        bar = ('<i class="bar %s%s" style="--h:%.4f" aria-hidden="true"></i>'
               % (outcome, " over" if over else "", frac))
    elif outcome in ("open", "live"):
        payout = (float(col["open_staked_cents"] or 0)
                  + float(col["upside_cents"] or 0))
        frac, over = ledger.height(payout, scale_info)
        # Filled to the price paid, which is the market's OWN probability at
        # entry -- a fact the record holds. A live win probability is not: no
        # part of this system polls an in-play price, and drawing one would be
        # the only number on the page nothing could check.
        stake = float(col["open_staked_cents"] or 0)
        implied = (stake / payout) if payout > 0 else 0.0
        bar = ('<i class="bar %s%s" style="--h:%.4f;--p:%.4f" '
               'aria-hidden="true"></i>'
               % (outcome, " over" if over else "", frac,
                  max(0.0, min(1.0, implied))))
    else:
        bar = '<i class="bar %s" aria-hidden="true"></i>' % outcome

    href = ("settlements/%s.html" % col["slug"]) if col["slug"] in settled_slugs else ""
    tag, attrs, close = "span", "", "</span>"
    if href:
        tag, attrs, close = "a", ' href="%s"' % esc(href), "</a>"
    sentence = _column_sentence(col)
    classes = "col %s%s%s" % (outcome, " linked" if href else "",
                              " newmonth" if col["new_month"] else "")
    return ('<%s class="%s"%s tabindex="-1" role="listitem" data-d="%s" '
            'aria-label="%s"><span class="clvtrack">%s</span>'
            '<span class="barbox">%s</span>%s'
            % (tag, classes, attrs, esc(sentence), esc(sentence), track, bar,
               close))


def book(s: dict) -> str:
    """Zone 3. One column per fixture, oldest at the left.

    The direct replacement for the glyph strip. Every fact has one channel:
    position says won or lost, length says how much, fill says settled or
    still running, and the track above says what the closing line thought.
    Nothing is double-encoded, and the legend below is a courtesy rather than
    a decoder ring -- delete it and the shape still reads.
    """
    cols = s["columns"]
    if not cols:
        return ('<section class="book empty"><p class="nothing">No fixture has '
                'been boarded yet. The record starts at the first sitting.'
                '</p></section>')
    scale_info = s["scale"]
    settled_slugs = {r["slug"] for r in s["settled_fixtures"]}
    first, last = cols[0], cols[-1]
    when = "%s to %s" % (_daystamp(first), _daystamp(last))
    marks = "".join(_column(c, scale_info, settled_slugs) for c in cols)
    months = "".join(
        '<span style="--i:%d">%s</span>' % (i, esc(c["month"]))
        for i, c in enumerate(cols) if c["new_month"] or i == 0)

    over = scale_info.get("clamped") or 0
    key = [
        "above the line won, below lost",
        "bar length is dollars",
        "hollow = still running",
        "hairline = boarded and declined",
    ]
    if over:
        key.append("notched = past the scale")
    return """
  <section class="book" aria-labelledby="book-h">
    <div class="bookhead" id="book-h"><span>Book</span>
      <b>%d fixtures, %s</b>
      <a class="archive" href="settlements/index.html">Every settlement &rarr;</a>
    </div>
    <div class="grid" id="book" role="list" style="--n:%d"
         aria-label="%s">%s</div>
    <div class="axis" style="--n:%d">%s</div>
    <p class="readout" id="bookread" data-rest="1" aria-live="polite">%s</p>
  </section>""" % (
        len(cols), esc(when), len(cols),
        esc("The book, %d fixtures from %s, oldest first" % (len(cols), when)),
        marks, len(cols), months, esc(" · ".join(key)))


def ledger_band(s: dict, now) -> str:
    """The whole ledger: what is open, how the run stands, and the book."""
    return "%s\n%s\n%s" % (open_rail(s, now), run_curve(s), book(s))


def cascade(s: dict) -> str:
    """Where the candidates went, as a shape rather than a list of numbers.

    Answers the question the book raises and cannot itself answer: the
    record is mostly dots and dashes, and this is why. The two big losses are
    the board declining and the market never reaching our price, and they are
    different failures -- one is judgement, the other is the price we chose.
    """
    rows = [r for r in s.get("funnel") or [] if r["n"] or True]
    if not rows:
        return ""
    top = max(r["n"] for r in rows) or 1
    out = []
    for r in rows:
        share = r["n"] / top
        out.append(
            '<div><dt>%s<small>%s</small></dt>'
            '<span class="bar" style="--w:%.4f"><i></i></span>'
            '<dd>%s</dd></div>'
            % (esc(r["stage"]), esc(r["note"]), share,
               esc("{:,}".format(r["n"]))))
    board = s["board"]
    # The fixture-level facts sit BESIDE the ladder, not in it: they are a
    # different unit and would invite a comparison of bar lengths that means
    # nothing.
    aside = ("%d fixtures boarded, %d of them declined outright. Every figure "
             "in the ladder counts markets, so the bars are comparable to each "
             "other and to nothing else."
             % (board["n_fixtures"], board["n_declined"]))
    return (kicker("How the record narrows")
            + '<dl class="cascade">%s</dl>' % "".join(out)
            + '<p class="cascade-note">%s</p>' % esc(aside))


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
    "home_win": ("%(home)s win", "%(home)s do not win"),
    "away_win": ("%(away)s win", "%(away)s do not win"),
    "draw": ("draw", "no draw"),
    "btts": ("both teams score", "not both teams score"),
}


def claim_label(claim: str, home: str, away: str) -> str:
    """Say what the position BACKS, in words, read at a glance.

    THE CLAIM ALREADY CARRIES THE SIDE, AND FOLDING `side` IN AGAIN NEGATED IT
    TWICE. `paper.cycle` builds the no side of a market as `"not_" + leg.claim`
    and prices it with `probability_for` of that same negated string, so a
    position's `claim` is the proposition THAT POSITION PAYS ON -- exactly what
    `outcomes.winning_side` settles it against. `side` is the venue mechanic,
    not a second negation: across the live book all 792 `no` orders carry a
    `not_` claim and all 621 `yes` orders do not, so it holds no information
    the claim does not already have.

    Reading it as one made 46 of 71 positions print as their own opposite. On
    Real Betis v Real Madrid the page showed a first-half Betis win and a
    first-half draw both settling as winners -- two mutually exclusive
    propositions, which is what sent us looking.

    Negation is phrased, not prefixed. `not_total_over_2.5` really is
    "under 2.5 goals" -- the lines are all halves, so there is no push and the
    complement is exact. Spreads keep the "not" because the complement of
    "wins by over 1.5" includes losing, and calling that "by under 1.5" would
    be a different, friendlier, wrong claim.
    """
    negated = str(claim).startswith("not_")
    base = claim[4:] if negated else str(claim)
    half = base.startswith("1h_")
    if half:
        base = base[3:]
    names = {"home": home, "away": away}

    if base in CLAIM_WORDS:
        text = CLAIM_WORDS[base][1 if negated else 0] % names
    elif base.startswith(("total_over_", "total_under_")):
        over = base.startswith("total_over_")
        word = "under" if over == negated else "over"
        text = "%s %s goals" % (word, base.rsplit("_", 1)[1])
    # Order matters: `home_wins_by_over_` also starts with `home_`, so the
    # spread test has to come before the team-total one.
    elif base.startswith(("home_wins_by_over_", "away_wins_by_over_")):
        who = "home" if base.startswith("home") else "away"
        text = "%s by over %s" % (names[who], base.rsplit("_", 1)[1])
        return ("not " + text) if negated else text
    elif base.startswith(("home_over_", "away_over_")):
        who, line = base.split("_over_")
        text = "%s %s %s" % (names[who], "under" if negated else "over", line)
    elif base.startswith("score_"):
        line = base.split("_", 1)[1]
        text = ("score not %s" % line) if negated else ("score %s" % line)
    else:
        text = base.replace("_", " ")
        if negated:
            text = "not " + text
    return ("1st half " + text) if half else text


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
    if row.get("n_unfilled"):
        foot_left += " · %d never filled" % row["n_unfilled"]
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
    declined = [r for r in s["fixtures"] if r.get("declined")]
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
    axes, narrow for the tables, with Martian Mono reserved for the ledger
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
    <p>In the ledger every fact takes exactly one channel. A bar sits above the
    baseline when the fixture won and below it when it lost; its length is
    dollars; its fill says whether the bet has settled, is still resting, or is
    being played out. Closing line value is lifted clear of the bars into a
    track of its own, and teal and red mean that and nothing else on the page —
    a bar is ink whichever way it went. The board declining is a hairline at
    the baseline, which holds the rhythm of the calendar without claiming any
    of the attention: it is the commonest thing in the record and the least
    worth looking at.</p>
    <p>Every rate prints the sample it was computed over. Rates average over
    fixtures rather than bets, because one match can produce thirty correlated
    markets off a single scoreline grid and counting them separately would
    overstate the evidence roughly thirtyfold. Where there is no measurement
    the page prints an em dash rather than a zero. The record is paper: no
    order here was ever placed with a venue.</p>
    <p>Nothing here polls a price while a match is running, so no card shows
    an in-play score, a live win probability or a cash-out value. What an open
    ticket shows is what it cost, what it returns if it lands, where the market
    closed if that was captured, and what has to happen — every one of which
    the record actually holds. The front page carries the most recent 48
    fixtures; the rest are in <a href="settlements/index.html">the settlement
    archive</a>, a page per fixture, with the board&rsquo;s own words beside
    its arithmetic.</p>
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
body{margin:0;overflow-x:clip;background:var(--paper);color:var(--ink);
  font-family:var(--ui);
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
/* ══ THE LEDGER ════════════════════════════════════════════════════════════
   Three zones on one column grid. Every fact gets exactly one channel and
   nothing is double-encoded:

     position above / below the baseline   won or lost
     bar length                            dollars
     fill style                            settled, open, or in play
     the CLV track                         closing line value, alone

   Bars are ink. Won and lost are told apart by which side of the rule they
   sit on, NOT by colour, which is what leaves teal and red free to mean one
   thing each in the track above. The moment a bar goes red the track stops
   being readable, so hold this. */

/* ── ZONE 1 · OPEN ── the hero, and the only place to spend any boldness ── */
.open{margin:0 0 var(--s5)}
.standfirst{display:flex;align-items:baseline;justify-content:space-between;
  gap:var(--s2) var(--s4);flex-wrap:wrap;
  margin:var(--s2) 0 var(--s3);font-family:var(--dense);font-size:12px;
  letter-spacing:.02em;color:var(--ink-mid)}
.standfirst .figs{display:flex;gap:var(--s3);white-space:nowrap}
.standfirst b{color:var(--ink);font-weight:700;padding-right:3px}
.open .nothing{margin:var(--s2) 0 0;font-family:var(--serif);font-size:15px}
.open .nothing span{color:var(--ink-mid)}

.rail{display:flex;gap:var(--s3);overflow-x:auto;overscroll-behavior-x:contain;
  scroll-snap-type:x proximity;padding:2px 2px var(--s3);margin:0 -2px;
  scrollbar-width:thin;scrollbar-color:var(--ink-soft) transparent}
.rail::-webkit-scrollbar{height:5px}
.rail::-webkit-scrollbar-track{background:var(--rule-soft)}
.rail::-webkit-scrollbar-thumb{background:var(--ink-soft)}
.ticket{flex:0 0 clamp(236px,26vw,292px);scroll-snap-align:start;
  display:flex;flex-direction:column;gap:var(--s2);
  padding:var(--s3) var(--s3) var(--s2);background:var(--inset);
  border:var(--rule-thin) solid var(--rule);border-radius:0;
  transition:border-color 160ms var(--ease-out)}
.ticket:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
/* The live card is the one thing on the page allowed to raise its voice. */
.ticket.live{border-color:var(--ink);border-top:var(--rule-thick) solid var(--ink)}
.ticket.awaiting{border-color:var(--ink-soft)}
.state{display:flex;align-items:center;gap:6px;margin:0;
  font-family:var(--dense);font-size:10px;font-weight:700;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-mid)}
.state i{width:6px;height:6px;flex:none;background:var(--ink-soft)}
.ticket.live .state{color:var(--ink)}
.ticket.live .state i{background:var(--live);animation:beat 2.6s var(--ease-out) infinite}
@keyframes beat{0%,100%{opacity:1}50%{opacity:.28}}
.ticket .pair{margin:0;font-family:var(--display);font-size:19px;
  font-weight:700;line-height:1.16;letter-spacing:-.012em;text-wrap:balance}
.ticket .pair span{font-family:var(--serif);font-weight:400;font-size:13px;
  font-style:italic;color:var(--ink-mid);padding:0 2px}
.ticket .comp{margin:0;font-family:var(--dense);font-size:10.5px;
  letter-spacing:.13em;text-transform:uppercase;color:var(--ink-soft)}
.legs{margin:var(--s1) 0 0;padding:var(--s2) 0 0;
  border-top:var(--rule-hair) solid var(--rule)}
.leg{display:flex;align-items:baseline;gap:var(--s2);padding:3px 0;
  border-bottom:var(--rule-hair) solid var(--rule-soft)}
.leg:last-child{border-bottom:0}
.leg dt{flex:1 1 auto;min-width:0;font-size:12.5px;line-height:1.3;
  text-wrap:pretty}
.leg dt small{display:block;font-family:var(--dense);font-size:9.5px;
  letter-spacing:.12em;text-transform:uppercase;color:var(--ink-soft)}
.leg dd{flex:0 1 auto;margin:0;font-family:var(--fig);font-size:10.5px;
  color:var(--ink-mid);display:flex;align-items:baseline;gap:4px;
  white-space:nowrap}
.leg dd b{color:var(--ink);font-weight:500}
.leg .times{opacity:.5}
.clvtag{font-style:normal;font-size:9.5px;padding-left:2px}
.clvtag.up{color:var(--live)}
.clvtag.dn{color:var(--loss)}
/* The instruction. A claim is a proposition; this is the thing to watch. */
.needs{margin:auto 0 0;padding:var(--s2) 0 0;
  border-top:var(--rule-hair) solid var(--rule);
  font-family:var(--serif);font-size:13px;line-height:1.45;text-wrap:pretty}
.tfoot{display:flex;justify-content:space-between;align-items:baseline;
  gap:var(--s2);padding-top:var(--s2);white-space:nowrap;
  border-top:var(--rule-thin) solid var(--ink);
  font-family:var(--fig);font-size:9.5px;letter-spacing:-.01em;
  color:var(--ink-soft)}
.tfoot strong{color:var(--ink);font-weight:500;font-size:11px}
.tfoot span{font-size:10px}

/* ── ZONE 2 · RUN ── the same x-positions as the book, so they read as one ── */
.run{margin:0 0 2px}
.runhead{display:flex;align-items:baseline;gap:var(--s2);flex-wrap:wrap;
  font-family:var(--dense);font-size:10.5px;font-weight:700;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-mid)}
.runhead b{font-family:var(--fig);font-size:13px;letter-spacing:0;
  font-weight:500}
.runhead b.pos{color:var(--ink)}
.runhead b.neg{color:var(--loss)}
.runnote{font-weight:400;letter-spacing:.04em;text-transform:none;
  font-size:10.5px;color:var(--ink-soft)}
.runplot{display:block;width:100%;height:92px;margin-top:var(--s1);
  overflow:visible}
.runplot .zero{stroke:var(--ink);stroke-width:1;opacity:.3;
  vector-effect:non-scaling-stroke}
.runplot .curve{fill:none;stroke:var(--ink);stroke-width:1.5;
  stroke-linejoin:round;stroke-linecap:round;vector-effect:non-scaling-stroke}
/* Exposure drawn as a range. It is not a forecast and must not look like one,
   so it has no edge stroke and no midline. */
.runplot .cone{fill:var(--ink);opacity:.11}

/* ── ZONE 3 · BOOK ── one column per fixture, oldest at the left ── */
.book{margin:0 0 var(--s6)}
.bookhead{display:flex;align-items:baseline;gap:var(--s2);flex-wrap:wrap;
  font-family:var(--dense);font-size:10.5px;font-weight:700;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-mid);
  padding-bottom:var(--s1)}
.bookhead b{font-weight:400;letter-spacing:.04em;text-transform:none;
  color:var(--ink-soft)}
.bookhead{padding-bottom:var(--s2)}
.bookhead .archive{margin-left:auto;letter-spacing:.1em;color:var(--ink);
  text-decoration:none;border-bottom:var(--rule-thin) solid var(--ink)}
.bookhead .archive:hover{background:var(--ink);color:var(--paper)}
.book .grid{display:grid;grid-template-columns:repeat(var(--n),minmax(0,1fr));
  gap:2px;align-items:start}
.col{position:relative;display:flex;flex-direction:column;gap:3px;min-width:0;
  color:inherit;text-decoration:none}
.col.linked{cursor:pointer}
.col:focus-visible{outline:2px solid var(--ink);outline-offset:2px;
  border-radius:1px}
/* A month is a full-height hairline and a word at the axis, not a character
   in the same alphabet as the data. */
.col.newmonth::before{content:"";position:absolute;left:-2px;top:0;bottom:-18px;
  width:var(--rule-hair);background:var(--rule);pointer-events:none}

/* The single biggest legibility gain in the redesign: CLV lifted out of the
   bars into its own scannable row. Teal above the midline, red below, so the
   distinction survives greyscale and colour is never the only carrier. */
.clvtrack{position:relative;height:7px;flex:none}
.clvtrack::after{content:"";position:absolute;left:0;right:0;top:50%;
  height:var(--rule-hair);background:var(--rule)}
.clvmark{position:absolute;left:12%;right:12%;height:2px}
.clvmark.up{bottom:50%;margin-bottom:1px;background:var(--live)}
.clvmark.dn{top:50%;margin-top:1px;background:var(--loss)}

.barbox{position:relative;height:86px;flex:none}
.barbox::after{content:"";position:absolute;left:0;right:0;top:50%;
  height:var(--rule-hair);background:var(--rule)}
.bar{position:absolute;left:0;right:0;display:block}
.bar.won{bottom:50%;height:calc(var(--h) * 42px);background:var(--ink)}
.bar.lost{top:50%;height:calc(var(--h) * 42px);background:var(--ink)}
.bar.push{top:calc(50% - 1px);height:2px;background:var(--ink)}
/* Voided: a result the record keeps and the measurement does not. A dash at
   the baseline, hollow, so it reads as "this happened and counts for nothing"
   rather than as a fixture that broke even. */
.bar.void{top:calc(50% - 2px);height:4px;background:none;
  box-shadow:inset 0 0 0 1px var(--ink)}
/* Boarded and declined: holds the rhythm of the calendar, claims nothing. */
.bar.declined{top:calc(50% - .5px);height:1px;background:var(--ink-soft)}
/* Ordered and never reached is the MARKET's answer, not the board's. Same
   weight, hollow, so the two never read as one fact. */
.bar.unfilled{top:calc(50% - 2.5px);height:5px;background:none;
  box-shadow:inset 0 0 0 1px var(--ink-soft)}
/* Still running: the outline is what it pays, the fill is what it cost --
   which is the market's own probability at entry, and a number this record
   actually holds. Nothing here polls an in-play price, so nothing here draws
   a live win probability. */
.bar.open,.bar.live{bottom:50%;height:calc(var(--h) * 42px);background:none;
  box-shadow:inset 0 0 0 1px var(--ink)}
.bar.live::before,.bar.open::before{content:"";position:absolute;
  left:0;right:0;bottom:0;height:calc(var(--p) * 100%);background:var(--ink);
  opacity:.82}
.bar.live{box-shadow:inset 0 0 0 1.5px var(--ink)}
.bar.live::after{content:"";position:absolute;left:-1px;right:-1px;top:-1px;
  bottom:-1px;box-shadow:0 0 0 1px var(--live);
  animation:beat 2.6s var(--ease-out) infinite}
/* Past the 95th percentile. One outlier should not flatten the rest, and
   hiding that it is an outlier would be worse. */
.bar.over.won::after,.bar.over.lost::after{content:"";position:absolute;
  left:0;right:0;height:0;border-left:3px solid transparent;
  border-right:3px solid transparent}
.bar.over.won::after{top:-4px;border-bottom:4px solid var(--ink)}
.bar.over.lost::after{bottom:-4px;border-top:4px solid var(--ink)}

.col:hover .bar.won,.col:hover .bar.lost,.col:hover .bar.push{background:var(--live)}
.col:hover .bar.declined,.col:hover .bar.unfilled{background:var(--ink);
  box-shadow:inset 0 0 0 1px var(--ink)}

.book .axis{position:relative;height:15px;margin-top:3px;
  font-family:var(--dense);font-size:9.5px;font-weight:700;letter-spacing:.15em;
  text-transform:uppercase;color:var(--ink-soft)}
.book .axis span{position:absolute;top:0;
  left:calc(var(--i) / var(--n) * 100%);white-space:nowrap}
.book .readout{margin:var(--s2) 0 0;padding:var(--s2) 0 0;text-align:left;
  border-top:var(--rule-hair) solid var(--rule);min-height:2.1em;
  font-family:var(--fig);font-size:11.5px;line-height:1.5;color:var(--ink);
  text-wrap:pretty}
/* The legend is a courtesy, not a decoder ring: delete it and the shape still
   reads, which is the acceptance test this replacement had to pass. */
.book .readout[data-rest="1"]{font-family:var(--dense);font-size:10.5px;
  letter-spacing:.06em;text-transform:uppercase;color:var(--ink-soft)}

html[data-mode="dark"] .ticket{background:var(--panel-bg);
  border-color:var(--panel-line)}
html[data-mode="dark"] .ticket.live{border-color:var(--live);
  border-top-color:var(--live);box-shadow:var(--glow)}
/* Under the lit reading the bars stay INK -- which is cream here -- rather
   than taking the tote board's amber. Amber is what the CLV track's up-mark
   uses, and a book drawn in the same colour as its own closing-line track is
   the exact overload this component exists to undo. */
html[data-mode="dark"] .bar.won,html[data-mode="dark"] .bar.lost,
html[data-mode="dark"] .bar.push{background:var(--ink)}
html[data-mode="dark"] .bar.declined{background:var(--ink-soft)}
html[data-mode="dark"] .bar.open,html[data-mode="dark"] .bar.live{
  box-shadow:inset 0 0 0 1px var(--ink)}
html[data-mode="dark"] .bar.live{box-shadow:inset 0 0 0 1.5px var(--ink)}
html[data-mode="dark"] .bar.live::before,
html[data-mode="dark"] .bar.open::before{background:var(--ink)}
html[data-mode="dark"] .bar.over.won::after{border-bottom-color:var(--ink)}
html[data-mode="dark"] .bar.over.lost::after{border-top-color:var(--ink)}
html[data-mode="dark"] .runplot .curve{stroke:var(--series)}
html[data-mode="dark"] .runplot .cone{fill:var(--series);opacity:.14}
html[data-mode="dark"] .col:hover .bar.won,
html[data-mode="dark"] .col:hover .bar.lost{background:var(--live)}

@media (prefers-reduced-motion:reduce){
  .ticket.live .state i,.bar.live::after{animation:none}
  .bar.live::after{box-shadow:none;outline:1px dashed var(--live);
    outline-offset:0}
}
@media (max-width:560px){
  .sheet{padding-left:var(--s3);padding-right:var(--s3)}
  .ticket{flex-basis:min(78vw,268px)}
  .book .grid{gap:1px}
  .barbox{height:70px}
  .bar.won,.bar.lost,.bar.open,.bar.live{height:calc(var(--h) * 34px)}
  .bookhead .archive{margin-left:0;flex-basis:100%}
}


/* THE CASCADE. The record is mostly abstentions and misses, so a reader
   looking at a row of dots is owed the arithmetic behind them. Each stage is
   a rule whose length is its share of the widest, which makes the attrition
   the shape of the thing rather than a column of numbers to compare by eye. */
.cascade{margin:0;display:grid;gap:0}
.cascade div{display:grid;grid-template-columns:22ch 1fr 10ch;
  align-items:center;gap:var(--s4);padding:9px 0;
  border-bottom:var(--rule-hair) solid var(--rule)}
.cascade div:first-child{border-top:var(--rule-thin) solid var(--ink)}
.cascade dt{margin:0;font-family:var(--dense);font-size:11px;font-weight:600;
  letter-spacing:.09em;text-transform:uppercase;color:var(--ink);
  white-space:nowrap;line-height:1.25}
/* The note sits under its stage rather than trailing the bar, so a bar at
   full width has nothing to push off the sheet. */
.cascade dt small{display:block;font-size:10px;font-weight:400;
  letter-spacing:.02em;text-transform:none;color:var(--ink-soft);
  white-space:normal}
.cascade .bar{position:relative;height:10px;min-width:2px}
.cascade .bar i{position:absolute;left:0;top:0;bottom:0;background:var(--ink);
  width:calc(var(--w) * 100%);min-width:2px}
.cascade-note{margin:var(--s3) 0 var(--s6);font-size:12px;
  line-height:1.55;color:var(--ink-mid);max-width:70ch;text-wrap:pretty}
.cascade dd{margin:0;text-align:right;font-family:var(--fig);font-size:14px;
  font-variant-numeric:tabular-nums lining-nums;white-space:nowrap}
@media (max-width:560px){
  .cascade div{grid-template-columns:1fr 9ch;gap:var(--s3)}
  .cascade .bar{grid-column:1 / -1;order:3}
}
/* ── LEDE + LEADERS ─────────────────────────────────────────── */
/* A grid track's automatic minimum is its content's min-content, and a wide
   table's min-content is the table. Without this the column blew past the
   container by 80px on a phone and the page scrolled sideways to reach it --
   `.tablewrap` scrolls for exactly that reason and never got the chance. */
.grid2 > *{min-width:0}
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


/* ── THE SETTLEMENT ARCHIVE ─────────────────────────────────────
   Sub-pages, sharing this sheet rather than a trimmed copy of it: a second
   stylesheet is a second thing to keep in step, and these print the same
   figures in the same faces. */
.subhead{margin-bottom:var(--s5)}
.subhead .dateline{margin:15px 0 0}
.back{color:var(--ink);text-decoration:none;letter-spacing:.16em;
  border-bottom:var(--rule-thin) solid var(--ink)}
.back:hover{background:var(--ink);color:var(--paper)}
.fixture{max-width:100%}
.fixture-h{margin:var(--s2) 0 var(--s4);font-family:var(--display);
  font-size:clamp(30px,5.4vw,54px);font-weight:800;line-height:1.04;
  letter-spacing:-.028em;text-wrap:balance}
.fixture-h span{font-family:var(--serif);font-style:italic;font-weight:400;
  font-size:.44em;color:var(--ink-mid);padding:0 .12em}
.standfirst-long{margin:0 0 var(--s4);font-family:var(--serif);font-size:15px;
  line-height:1.62;max-width:66ch;text-wrap:pretty}
.scoreline{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:0;margin:0 0 var(--s5);border-top:var(--rule-thick) solid var(--ink);
  border-bottom:var(--rule-thin) solid var(--ink)}
.scoreline > div{padding:var(--s3) var(--s3) var(--s3) 0;
  border-right:var(--rule-hair) solid var(--rule)}
.scoreline > div:last-child{border-right:0}
.scoreline dt{font-family:var(--dense);font-size:10px;font-weight:700;
  letter-spacing:.15em;text-transform:uppercase;color:var(--ink-mid)}
.scoreline dd{margin:5px 0 0;font-family:var(--fig);font-size:19px;
  font-weight:500;letter-spacing:-.01em}
.scoreline dd.pos{color:var(--ink)}
.scoreline dd.neg{color:var(--loss)}

/* The board's own words, kept beside its arithmetic. The reasoning is the
   part of this record that cannot be recomputed. */
.verdictbox{margin:0 0 var(--s5);padding:var(--s3) var(--s4);
  border-left:var(--rule-thick) solid var(--ink);background:var(--inset)}
.verdictbox .kicker{margin:0 0 var(--s2)}
.said{margin:0;font-family:var(--serif);font-size:14px;line-height:1.6;
  max-width:70ch;text-wrap:pretty}
.saidfoot{margin:var(--s2) 0 0;font-family:var(--dense);font-size:10.5px;
  letter-spacing:.12em;text-transform:uppercase;color:var(--ink-soft)}

table.settlements caption{text-align:left;padding:11px var(--s4) 0;
  font-family:var(--dense);font-size:10px;font-weight:700;letter-spacing:.13em;
  text-transform:uppercase;color:var(--ink-soft)}
table.settlements td.what{white-space:normal;min-width:190px}
table.settlements td.verdict{font-weight:700;text-transform:capitalize}
table.settlements td.verdict.lost{color:var(--loss)}
table.settlements tr.void td{color:var(--ink-soft)}
table.settlements tr.void td.what{text-decoration:line-through;
  text-decoration-color:var(--rule)}
table.settlements td.verdict.void{color:var(--ink-soft);font-weight:400;
  letter-spacing:.1em;text-transform:uppercase;font-size:10.5px}
table.settlements tr.voidwhy td{padding-top:0;font-size:11.5px;
  color:var(--ink-mid);white-space:normal;text-wrap:pretty;
  border-bottom:var(--rule-hair) solid var(--rule-soft)}
table.settlements tr.voidwhy:hover{background:none}
/* Its own line under the figure it qualifies, rather than trailing off the
   end of one and wrapping mid-phrase. */
.scoreline dd small{display:block;margin-top:3px;font-family:var(--dense);
  font-size:10px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--ink-soft)}
table.settlements td.up{color:var(--live)}
table.settlements td.dn{color:var(--loss)}
table.settlements td.pos{font-weight:700}
table.settlements td.neg{color:var(--loss);font-weight:700}
html[data-mode="dark"] table.settlements td.pos{color:var(--win)}
html[data-mode="dark"] table.settlements td.verdict.won{color:var(--win)}

.monthblock{margin:0 0 var(--s5)}
.monthblock h2{margin:0 0 var(--s2);font-family:var(--dense);font-size:11px;
  font-weight:700;letter-spacing:.18em;text-transform:uppercase;
  color:var(--ink-mid);padding-bottom:var(--s1);
  border-bottom:var(--rule-thin) solid var(--ink)}
.archive-list{list-style:none;margin:0;padding:0}
.archive-list a{display:grid;
  grid-template-columns:auto minmax(0,1fr) auto auto auto;
  align-items:baseline;gap:var(--s3);padding:9px 2px;color:inherit;
  text-decoration:none;border-bottom:var(--rule-hair) solid var(--rule-soft)}
.archive-list a:hover,.archive-list a:focus-visible{background:var(--rule-soft);
  outline:none}
.archive-list a:focus-visible{outline:2px solid var(--ink);outline-offset:-2px}
.archive-list .when{font-family:var(--fig);font-size:10.5px;
  color:var(--ink-soft)}
.archive-list .who{font-size:14px;min-width:0;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.archive-list .who i{font-family:var(--serif);font-style:italic;font-size:11px;
  color:var(--ink-mid);padding:0 3px}
.archive-list .count{font-family:var(--dense);font-size:11px;
  color:var(--ink-mid)}
.archive-list .clv{font-family:var(--fig);font-size:11px;min-width:5ch;
  text-align:right}
.archive-list .clv.up{color:var(--live)}
.archive-list .clv.dn{color:var(--loss)}
.archive-list .net{font-family:var(--fig);font-size:12px;min-width:8ch;
  text-align:right;font-weight:500}
.archive-list .net.neg{color:var(--loss)}
html[data-mode="dark"] .archive-list .net.pos{color:var(--win)}
.backlink{margin:var(--s5) 0 0;font-family:var(--dense);font-size:11.5px;
  letter-spacing:.1em;text-transform:uppercase}
.backlink a{color:var(--ink)}
.fixture .nothing{font-family:var(--serif);font-size:15px;max-width:60ch}

@media (max-width:560px){
  .archive-list a{grid-template-columns:auto minmax(0,1fr) auto;
    row-gap:2px}
  .archive-list .count{display:none}
}

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
  .tagline::before,.tagline::after{display:none} }
@media (prefers-reduced-motion:reduce){
  *{animation-duration:.01ms !important;transition-duration:.01ms !important}
}
"""

JS = """
(function(){
  /* THE BOOK. Hover and keyboard focus open the SAME readout, so no fact on
     this page is reachable by pointer alone. Roving tabindex: the book is one
     tab stop and the arrows walk it, rather than 48 stops between the header
     and the next section. */
  var read=document.getElementById('bookread'),
      wrap=document.getElementById('book'),
      REST=read?read.textContent:'';
  if(wrap&&read){
    var cols=[].slice.call(wrap.querySelectorAll('.col'));
    if(cols.length) cols[cols.length-1].tabIndex=0;
    function show(e){var b=e.target.closest('.col[data-d]');
      if(b){read.textContent=b.dataset.d; read.removeAttribute('data-rest');}}
    function rest(){read.textContent=REST; read.setAttribute('data-rest','1');}
    wrap.addEventListener('pointerover',show);
    wrap.addEventListener('focusin',show);
    wrap.addEventListener('pointerleave',rest);
    wrap.addEventListener('focusout',function(e){
      if(!wrap.contains(e.relatedTarget)) rest();});
    wrap.addEventListener('keydown',function(e){
      var i=cols.indexOf(document.activeElement); if(i<0) return;
      var month=function(dir){
        var j=i;
        while(true){ j+=dir;
          if(j<=0) return 0;
          if(j>=cols.length-1) return cols.length-1;
          if(cols[j].classList.contains('newmonth')) return j; } };
      var to={ArrowLeft:i-1,ArrowRight:i+1,Home:0,End:cols.length-1,
              PageUp:month(-1),PageDown:month(1)}[e.key];
      if(to===undefined) return;
      e.preventDefault();
      to=Math.max(0,Math.min(cols.length-1,to));
      cols[i].tabIndex=-1; cols[to].tabIndex=0; cols[to].focus();
    });
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
    s.update(ledger_view(portfolio, now))
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
%s
</div>
<script>window.CURVE=%s;</script>
<script>%s</script>
</body>
</html>
""" % (links, CSS, masthead(s, now), ledger_band(s, now), lede(s), leaders(s),
       cascade(s), chart(s), ledger_strip(s), tickets(s), bet_table(s),
       breakdown(s), abstentions(s), colophon(s), crosshair_data(s), JS)
