"""The published page.

This HTML goes on the public internet with the user's name on it, so the
failures worth guarding are the ones that would make it state something untrue:
a rate without its denominator, a zero standing in for a missing measurement, a
board's reasoning injected into the markup, a stale ledger rendered as if it
were current.
"""
import datetime as dt
import json

import pytest
from paper_fixtures import book, pos, verdict

from wc2026.site import build, render

NOW = dt.datetime(2026, 8, 29, 12, 0, tzinfo=dt.timezone.utc)


def page(**kw) -> str:
    return render.page(book(**kw), now=NOW)


# --- the page must not lie -----------------------------------------------------
def test_an_empty_book_renders_a_page_that_says_so():
    """The workflow rebuilds on every cycle, including the ones that do
    nothing. A quiet day must produce an honest page, not a crash."""
    html = page()
    assert "JJ" in html and "<html" in html
    assert "The board has not sat yet" in html
    assert "—" in html, "absent measurements print an em dash"


def test_a_missing_measurement_is_a_dash_and_never_a_zero():
    """`0.0%` is a claim that something was measured and came out at zero."""
    html = page(positions=[pos(clv=None, pnl=None)])
    assert "Closing line value" in html
    # no CLV exists, so no cents figure may be asserted anywhere near it
    assert "+0.0¢" not in html and "−0.0¢" not in html


def test_every_rate_carries_its_denominator():
    html = page(positions=[pos(clv=2.0, pnl=500)], boarded=[verdict()])
    for label in ("Closing line value", "Strike rate", "Declined to bet"):
        assert label in html
    assert "n=" in html, "rates print the sample they were computed over"


def test_the_clv_sample_is_fixtures_not_bets():
    """Three correlated markets on one match are one observation, and the page
    must print 1 -- printing 3 would treble the apparent evidence."""
    html = page(positions=[pos("home_over_1.5", clv=4.0, pnl=100),
                           pos("away_over_0.5", clv=1.0, pnl=100),
                           pos("draw", clv=1.0, pnl=100)])
    assert "n=1</i>" in html


def test_signed_figures_use_a_true_minus():
    """U+2212 is drawn to the same width as `+` in a tabular face; a hyphen is
    narrower, so a signed column set with hyphens visibly steps."""
    html = page(positions=[pos(pnl=-2500, clv=-3.0)])
    assert "−2.50u" in html and "−3.0¢" in html
    assert "-2.50u" not in html and "-3.0¢" not in html


# --- injection -----------------------------------------------------------------
def test_a_board_reason_cannot_inject_markup():
    """The reason text is model output. It is data on this page, never markup."""
    nasty = '<script>alert(1)</script><img src=x onerror=alert(2)>'
    html = page(boarded=[verdict(reason=nasty)])
    # The angle brackets and quotes are what make it markup. Escaped, the
    # payload survives as inert text -- which is correct, it IS what the board
    # said -- and cannot open a tag or close the attribute holding it. The page
    # has script tags of its own, so the assertion is on the payload itself.
    assert nasty not in html
    assert "<script>alert(1)" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html


def test_a_team_name_cannot_inject_markup():
    html = page(positions=[pos(home='" onmouseover="alert(1)')])
    assert 'onmouseover="alert(1)"' not in html


def test_the_form_readout_escapes_its_detail_attribute():
    html = page(boarded=[verdict(reason='" autofocus onfocus="alert(1)')])
    assert 'onfocus="alert(1)"' not in html


# --- the sections the brief asks for ------------------------------------------
def test_the_page_carries_every_section():
    html = page(positions=[pos(clv=2.0, pnl=900)],
                boarded=[verdict(), verdict(home="C", action="PAPER_PLACE_LIMIT")],
                orders=[{"status": "filled", "kind": "limit"}],
                ledger=[{"ts": "2026-08-28T20:00:00+00:00",
                         "instrument_id": "draw-A", "side": "yes",
                         "pnl_cents": 900, "won": True}])
    for marker in ("Form ·", "Declined", "By league", "By market family",
                   "Every market held", "Colophon", "Tickets"):
        assert marker in html, marker


def test_the_abstentions_are_shown_rather_than_hidden():
    """The decision this book makes most often is the decision not to bet."""
    html = page(boarded=[verdict(home=str(i)) for i in range(9)])
    assert "Declined" in html
    assert "9 of 9 boarded fixtures were passed over" in html


def test_the_zero_line_is_the_heaviest_rule_on_the_chart():
    curve = [{"ts": "2026-08-2%dT12:00:00+00:00" % i,
              "instrument_id": "draw-A", "side": "yes",
              "pnl_cents": 500 * (i - 1), "won": True} for i in (1, 2, 3)]
    html = page(positions=[pos(clv=1.0, pnl=500)], ledger=curve)
    assert 'class="zero"' in html and 'stroke-width="1.5"' in html


def test_nothing_animates_the_headline_or_the_chart_on_load():
    """Frequency rule: this page is opened daily, and an animation watched
    daily is a tax."""
    html = page(positions=[pos(clv=1.0, pnl=500)])
    assert "@keyframes" not in html, "no load animation is defined at all"
    assert "animation-delay" not in html, "no stagger survives on the figures"
    # Nothing counts the hero up, and the chart path is emitted complete
    # rather than drawn: no dash-offset trick, no requestAnimationFrame.
    assert "stroke-dashoffset" not in html
    assert "requestAnimationFrame" not in html


def test_the_three_font_families_are_requested_separately():
    """One combined request with a malformed axis returns 400 and takes every
    face down with it."""
    html = page()
    assert html.count('fonts.googleapis.com/css2?family=') == len(render.FONTS)
    assert "UnifrakturMaguntia" in html and "Libre+Baskerville" in html


def test_figures_are_tabular_everywhere():
    html = page(positions=[pos(clv=1.0, pnl=500)])
    assert "tabular-nums lining-nums slashed-zero" in html


# --- the masthead --------------------------------------------------------------
def test_the_masthead_carries_no_baked_ground_and_no_baked_date():
    """The supplied artwork had both. A masthead printing a fixed date is
    wrong every day but one, and its cream ground is the colour the brief
    bans."""
    html = page()
    assert "#f4f1e8" not in html.lower()
    assert "SATURDAY, AUGUST 29, 2026" in html.upper()
    later = render.page(book(), now=dt.datetime(2027, 1, 2, tzinfo=dt.timezone.utc))
    assert "SATURDAY, JANUARY 02, 2027" in later.upper()
    assert "Vol. II" in later, "the volume follows the year"


def test_the_masthead_inherits_the_reading_light():
    html = page()
    assert "var(--ink)" in html.split(".wordmark")[1][:200]


# --- the build ----------------------------------------------------------------
def test_a_missing_ledger_builds_an_empty_page_rather_than_failing(tmp_path):
    out = tmp_path / "site" / "index.html"
    built = build.build(str(tmp_path / "nope.json"), str(out))
    assert out.exists() and built["fixtures"] == 0
    assert (tmp_path / "site" / ".nojekyll").exists()


def test_an_unreadable_ledger_stops_the_build(tmp_path):
    """A truncated ledger is not an empty one. Publishing 'nothing here' would
    read as a fact about the trading rather than a fact about the build."""
    bad = tmp_path / "portfolio.json"
    bad.write_text('{"positions": {', encoding="utf-8")
    with pytest.raises(build.LedgerUnreadable):
        build.build(str(bad), str(tmp_path / "out.html"))


def test_a_ledger_that_is_not_an_object_stops_the_build(tmp_path):
    bad = tmp_path / "portfolio.json"
    bad.write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(build.LedgerUnreadable):
        build.build(str(bad), str(tmp_path / "out.html"))


def test_the_build_writes_the_real_record(tmp_path):
    src = tmp_path / "portfolio.json"
    src.write_text(json.dumps(book([pos(clv=2.0, pnl=900)],
                                   [verdict(), verdict(home="C")])),
                   encoding="utf-8")
    out = tmp_path / "index.html"
    built = build.build(str(src), str(out))
    assert built["fixtures"] == 2 and built["settled"] == 1
    assert "Declined" in out.read_text(encoding="utf-8")


def test_the_page_is_self_contained(tmp_path):
    """Google Fonts is the only external host; nothing else may be fetched."""
    html = page(positions=[pos(clv=1.0, pnl=500)])
    import re
    hosts = set(re.findall(r'https?://([^/"\')\s]+)', html))
    assert hosts <= {"fonts.googleapis.com", "fonts.gstatic.com",
                     "www.w3.org"}, hosts


# --- a live ticket ------------------------------------------------------------
def live_pos(**kw):
    p = pos(**kw)
    p["settled"] = False
    p["realized_pnl_cents"] = None
    return p


def test_a_running_ticket_shows_what_it_still_pays_not_a_settled_zero():
    """`pnl_units` counts SETTLED markets and a live fixture has none, so the
    footer printed "+0.00u" under "if every open market holds" -- a true
    number answering a question nobody asked."""
    html = page(positions=[live_pos(claim="draw", cost=20.0, size=100.0),
                           live_pos(claim="btts", cost=40.0, size=50.0)])
    # 100 x (100-20) + 50 x (100-40) = 8000 + 3000 = 11000c = 11u
    assert "+11.00u" in html
    assert "Live · 2 of 2 open" in html


def test_a_running_ticket_reports_its_stake_separately_from_its_upside():
    html = page(positions=[live_pos(claim="draw", cost=20.0, size=100.0)])
    assert "+8.00u" in html, "upside is the hero figure"
    assert "+2.00u" in html, "and the stake is reported beside it"
    assert "1 markets staked" in html


def test_a_settled_ticket_still_reports_realised_money():
    html = page(positions=[pos(claim="draw", pnl=4000, clv=1.0)])
    assert "+4.00u" in html and "1 of 1 markets cashed" in html


def test_two_venues_quoting_one_claim_are_not_one_repeated_row():
    """They are two positions at two prices. Rendered without the venue they
    are identical lines and read as a duplication bug."""
    html = page(positions=[
        pos(claim="total_over_2.5", venue="kalshi", cost=64.0, pnl=100),
        pos(claim="total_over_2.5", venue="polymarket", cost=66.0, pnl=100)])
    assert "kals" in html and "poly" in html
    assert "64¢" in html and "66¢" in html


# --- what a unit is -----------------------------------------------------------
def test_the_page_says_what_a_unit_is():
    """`u` is the primary figure everywhere. An unstated convention is just a
    number nobody can check."""
    html = page(positions=[pos(clv=1.0, pnl=500)])
    assert "One unit is 1% of the starting bankroll" in html
    assert "$10.00 on this book" in html


def test_units_follow_the_bankroll_rather_than_a_literal():
    html = render.page(book([pos(pnl=5000)], start=500_000), now=NOW)
    assert "$50.00 on this book" in html
    assert "+1.00u" in html, "5000c against a 5000c unit"


def test_a_negative_percentage_uses_a_true_minus():
    html = page(positions=[pos(pnl=-2000, cost=40.0, size=100.0)])
    assert "−" in html
    assert "-7.5%" not in html and "-5.0%" not in html
