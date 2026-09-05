"""The settlement archive: one page per settled fixture, plus an index.

The front page shows a rolling 48-fixture window. These are the pages that make
the rest of the record reachable, and the level at which a wrong number is
actually findable -- the Real Betis mispricing was one row of one fixture and
invisible in every aggregate above it.
"""
import json
import os

from paper_fixtures import book, pos, verdict

from wc2026.site import archive, build, ledger


def settled_book():
    return book(
        [pos(claim="1h_draw", home="Real Betis", away="Real Madrid",
             league="la_liga", kickoff="2026-09-04", pnl=2592, clv=0.5,
             size=39.0, cost=32.0),
         pos(claim="not_1h_home_win", home="Real Betis", away="Real Madrid",
             league="la_liga", kickoff="2026-09-04", pnl=601, clv=-16.5,
             size=18.0, cost=65.0),
         pos(claim="draw", home="Celta Vigo", away="Athletic Club",
             league="la_liga", kickoff="2026-08-30", pnl=-1700, clv=-3.5)],
        boarded=[verdict(league="la_liga", home="Real Betis",
                         away="Real Madrid", day="2026-09-04",
                         action="PAPER_PLACE_LIMIT",
                         reason="approved 3 of 12 markets: negative EV at touch",
                         considered=103)])


def test_every_settled_fixture_gets_its_own_page():
    pages = archive.pages(settled_book())
    slugs = {r["slug"] for r in ledger.settled_fixtures(settled_book())}
    assert "settlements/index.html" in pages
    for slug in slugs:
        assert "settlements/%s.html" % slug in pages
    assert len(pages) == len(slugs) + 1


def test_the_index_links_only_to_pages_that_exist():
    """A build that wrote one and not the other would publish a page full of
    dead links, and nothing on a static host would notice."""
    pages = archive.pages(settled_book())
    index = pages["settlements/index.html"]
    import re
    hrefs = set(re.findall(r'href="([^"]+\.html)"', index))
    hrefs.discard("../index.html")
    for href in hrefs:
        assert "settlements/%s" % href in pages, href
    assert hrefs, "the index has to link somewhere"


def test_a_fixture_page_shows_the_working_for_every_market_on_it():
    pages = archive.pages(settled_book())
    page = pages["settlements/2026-09-04-la-liga-real-betis-real-madrid.html"]
    # every column of the working
    for fragment in ("1st half draw", "1st half Real Betis do not win",
                     "32¢", "65¢", "+$25.92", "+$6.01", "−16.5¢",
                     "Won"):
        assert fragment in page, fragment


def test_the_two_rows_that_looked_impossible_now_read_as_compatible():
    """The screenshot that started this: a first-half Betis win and a
    first-half draw, both settled as winners, on one fixture. They are the same
    two positions; only the second was named as its own opposite."""
    page = archive.pages(settled_book())[
        "settlements/2026-09-04-la-liga-real-betis-real-madrid.html"]
    assert "1st half Real Betis do not win" in page
    assert ">1st half Real Betis win<" not in page


def test_the_boards_own_words_are_kept_beside_its_arithmetic():
    """The reasoning is the part of this record that cannot be recomputed."""
    page = archive.pages(settled_book())[
        "settlements/2026-09-04-la-liga-real-betis-real-madrid.html"]
    assert "What the board said" in page
    assert "approved 3 of 12 markets" in page
    assert "103 markets read" in page


def test_a_fixture_the_board_never_recorded_still_gets_a_page():
    """Settlement and boarding are separate records, and the board's is the one
    that can be missing."""
    page = archive.pages(settled_book())[
        "settlements/2026-08-30-la-liga-celta-vigo-athletic-club.html"]
    assert "Celta Vigo" in page
    assert "What the board said" not in page


def test_the_index_totals_the_whole_record_not_the_visible_window():
    pages = archive.pages(settled_book())
    index = pages["settlements/index.html"]
    assert "2 fixtures have settled 3 markets" in index
    assert "Every settlement" in index


def test_an_empty_book_says_so_rather_than_drawing_an_empty_axis():
    pages = archive.pages(book())
    assert len(pages) == 1
    assert "Nothing has settled yet" in pages["settlements/index.html"]


def test_archive_pages_carry_the_same_stylesheet_as_the_front_page():
    """A second sheet is a second thing to keep in step."""
    page = archive.pages(settled_book())["settlements/index.html"]
    assert ".archive-list" in page and ".bar.won" in page


# --- the build writes them ----------------------------------------------------
def test_the_build_writes_the_archive_beside_the_front_page(tmp_path):
    ledger_path = tmp_path / "portfolio.json"
    ledger_path.write_text(json.dumps(settled_book()), encoding="utf-8")
    out = tmp_path / "site" / "index.html"
    built = build.build(str(ledger_path), str(out))

    # the front page, the archive index, and one page per settled fixture
    assert built["pages"] == 4
    assert os.path.exists(str(tmp_path / "site" / "settlements" / "index.html"))
    assert os.path.exists(str(
        tmp_path / "site" / "settlements"
        / "2026-09-04-la-liga-real-betis-real-madrid.html"))
    # and the front page reaches them
    assert 'href="settlements/index.html"' in out.read_text(encoding="utf-8")


def test_the_front_page_links_every_settled_column_to_its_page(tmp_path):
    ledger_path = tmp_path / "portfolio.json"
    ledger_path.write_text(json.dumps(settled_book()), encoding="utf-8")
    out = tmp_path / "site" / "index.html"
    build.build(str(ledger_path), str(out))
    html = out.read_text(encoding="utf-8")
    import re
    linked = set(re.findall(r'href="settlements/([^"]+)\.html"', html))
    linked.discard("index")
    for slug in linked:
        assert (tmp_path / "site" / "settlements" / (slug + ".html")).exists()
    assert linked == {r["slug"] for r in
                      ledger.settled_fixtures(settled_book())}
