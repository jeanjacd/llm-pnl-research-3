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
    assert "−$25.00" in html and "−3.0¢" in html
    assert "-$25.00" not in html and "-3.0¢" not in html


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
    for marker in ("Open", "Run", "Book", "Every settlement", "Declined",
                   "By league", "By market family", "Every market held",
                   "Colophon", "Tickets"):
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
    daily is a tax. The one exception is the live pulse, which is a STATE and
    only exists while a match is actually being played -- and it is the first
    thing dropped under `prefers-reduced-motion`."""
    html = page(positions=[pos(clv=1.0, pnl=500)])
    assert html.count("@keyframes") == 1, "only the live pulse is defined"
    assert "@keyframes beat" in html
    assert "prefers-reduced-motion" in html
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
    # 100 x (100-20) + 50 x (100-40) = 8000 + 3000 = 11000c
    assert "+$110.00" in html
    assert "Live · 2 of 2 open" in html


def test_a_running_ticket_reports_its_stake_separately_from_its_upside():
    html = page(positions=[live_pos(claim="draw", cost=20.0, size=100.0)])
    assert "+$80.00" in html, "upside is the hero figure"
    assert "$20.00" in html, "and the stake is reported beside it"
    assert "1 markets staked" in html


def test_a_settled_ticket_still_reports_realised_money():
    html = page(positions=[pos(claim="draw", pnl=4000, clv=1.0)])
    assert "+$40.00" in html and "1 of 1 markets cashed" in html


def test_two_venues_quoting_one_claim_are_not_one_repeated_row():
    """They are two positions at two prices. Rendered without the venue they
    are identical lines and read as a duplication bug."""
    html = page(positions=[
        pos(claim="total_over_2.5", venue="kalshi", cost=64.0, pnl=100),
        pos(claim="total_over_2.5", venue="polymarket", cost=66.0, pnl=100)])
    assert "kals" in html and "poly" in html
    assert "64¢" in html and "66¢" in html


# --- money is money -----------------------------------------------------------
def test_the_page_reports_dollars_and_never_units():
    """A unit is only worth printing when the bankroll moves. This one does
    not, so `u` was the dollar figure divided by ten and one more number that
    could disagree with the first."""
    html = page(positions=[pos(clv=1.0, pnl=500)],
                boarded=[verdict()],
                ledger=[{"ts": "2026-08-28T20:00:00+00:00",
                         "instrument_id": "draw-A", "side": "yes",
                         "pnl_cents": 500, "won": True}])
    assert "+$5.00" in html
    import re
    assert not re.search(r"[+−]\d[\d,.]*u", html), "no unit figures"


def test_closing_line_value_stays_in_cents():
    """It is a price difference per contract, not an amount of money."""
    html = page(positions=[pos(clv=4.0, pnl=500)])
    assert "+4.0¢" in html


def test_a_negative_percentage_uses_a_true_minus():
    html = page(positions=[pos(pnl=-2000, cost=40.0, size=100.0)])
    assert "−" in html
    assert "-7.5%" not in html and "-5.0%" not in html


# --- the claim already carries the side ---------------------------------------
# Real Betis v Real Madrid, 2026-09-04. The page printed "1st half Real Betis
# win" AND "1st half draw", both settled as winners, on the same fixture. Two
# propositions that cannot both be true, so one of them was being named wrong.
#
# `paper.cycle` builds the no side as `"not_" + leg.claim` and prices it with
# `probability_for` of that same negated string, so the claim on a position IS
# the proposition it pays on -- which is what `outcomes.winning_side` settles
# it against, and what `test_money_correctness` already asserts. Folding `side`
# in on the way to the page applied the negation a second time.
def test_a_position_is_named_by_the_claim_it_settles_on():
    """The label and the settlement must read the claim the same way. They did
    not: settlement paid `not_score_4-0` when the score was anything else, and
    the page called that same position a bet ON 4-0."""
    assert render.claim_label("not_score_4-0", "A", "B") == "score not 4-0"
    assert render.claim_label("score_4-0", "A", "B") == "score 4-0"
    assert render.claim_label("not_1h_score_3-0", "A", "B") == \
        "1st half score not 3-0"


def test_the_two_betis_rows_can_no_longer_both_read_as_a_win():
    """The screenshot that surfaced this. On a drawn half BOTH positions do
    win -- `1h_draw` and `not_1h_home_win` are compatible -- but only once the
    second is named for what it backs."""
    a = render.claim_label("1h_draw", "Real Betis", "Real Madrid")
    b = render.claim_label("not_1h_home_win", "Real Betis", "Real Madrid")
    assert a == "1st half draw"
    assert b == "1st half Real Betis do not win"
    assert a != b


def test_negation_is_phrased_rather_than_prefixed():
    """`not_total_over_2.5` is exactly "under 2.5": every line is a half, so
    there is no push and the complement is clean."""
    assert render.claim_label("not_total_over_2.5", "A", "B") == \
        "under 2.5 goals"
    assert render.claim_label("not_total_under_2.5", "A", "B") == \
        "over 2.5 goals"
    assert render.claim_label("not_draw", "A", "B") == "no draw"
    assert render.claim_label("not_home_win", "Betis", "Madrid") == \
        "Betis do not win"


def test_a_spread_keeps_its_not_because_the_complement_includes_losing():
    """"Not by over 1.5" covers a one-goal win, a draw and a defeat. Calling
    it "by under 1.5" would be a different and friendlier claim."""
    assert render.claim_label("not_home_wins_by_over_1.5", "Betis", "M") == \
        "not Betis by over 1.5"


def test_the_page_names_a_no_side_holding_by_its_own_proposition():
    sold = pos(claim="not_score_4-0", home="Real Madrid", away="Málaga",
               pnl=-8900, clv=2.0)
    sold["side"] = "no"
    html = page(positions=[sold])
    assert "score not 4-0" in html


def test_most_of_the_score_book_wins_when_the_score_is_something_else():
    """The inverse of the story this file used to tell. A book holding 24
    `not_score_X-Y` positions wins 23 of them: paying ~92c for a near-certainty
    is what the record shows, and exactly one of them can lose."""
    held = []
    for score, pnl in (("4-0", -8900), ("0-1", 700), ("2-1", 700)):
        p = pos(claim="not_score_%s" % score, home="Real Madrid",
                away="Málaga", pnl=pnl, clv=2.0)
        p["side"] = "no"
        p["instrument_id"] = "inst-%s" % score
        held.append(p)
    html = page(positions=held)
    for score in ("4-0", "0-1", "2-1"):
        assert "score not %s" % score in html


# --- the band is an instrument, not a line -------------------------------------
def test_the_book_carries_a_closing_line_track_of_its_own():
    """CLV is the hero metric and was a two-pixel vertical offset inside the
    figures. Lifting it into its own row is the single biggest legibility gain
    in the redesign."""
    html = page(positions=[pos(claim="draw", pnl=-500, clv=3.0)])
    assert 'class="clvtrack"' in html
    assert 'class="clvmark up"' in html or 'class="clvmark dn"' in html


def test_a_mark_sits_above_the_track_when_the_market_closed_our_way():
    """Above and below the track's midline, so the distinction survives
    greyscale and colour is never the only carrier."""
    up = page(positions=[pos(claim="draw", pnl=-500, clv=4.0)])
    dn = page(positions=[pos(claim="draw", pnl=-500, clv=-4.0)])
    assert 'class="clvmark up"' in up and 'class="clvmark dn"' not in up
    assert 'class="clvmark dn"' in dn and 'class="clvmark up"' not in dn


def test_a_bar_is_ink_whether_it_won_or_lost():
    """The encoding discipline the whole component rests on: won and lost are
    told apart by WHICH SIDE OF THE RULE they sit on, never by colour, which is
    what leaves teal and red free to mean one thing each in the track above.
    The moment a bar goes red the track stops being readable."""
    html = page(positions=[pos(claim="draw", home="A", away="B", pnl=900,
                               clv=1.0),
                           pos(claim="draw", home="C", away="D", pnl=-900,
                               clv=1.0)])
    import re
    # Anchored: a hover affordance may tint a bar, an ENCODING may not, and
    # the two are told apart by the selector the rule starts with.
    for rule in re.findall(r"(?m)^\.bar\.(?:won|lost|push)[^{]*\{[^}]*\}",
                           html):
        assert "--loss" not in rule and "--live" not in rule, rule
    assert ".bar.won{bottom:50%" in html, "won sits above the rule"
    assert ".bar.lost{top:50%" in html, "lost sits below it"


def test_a_fixture_with_no_closing_line_gets_no_mark():
    html = page(boarded=[verdict()])
    assert 'class="clvtrack"' in html, "the track still runs"
    assert '<i class="clvmark' not in html, "but nothing is drawn on it"


def test_the_cascade_explains_where_the_candidates_went():
    html = page(positions=[pos(claim="draw", pnl=100)],
                boarded=[verdict(considered=120)],
                orders=[{"status": "expired", "kind": "limit"}])
    assert "How the record narrows" in html
    assert "prices the board looked at" in html
    assert "fixtures boarded" in html, "the fixture counts sit beside it"
