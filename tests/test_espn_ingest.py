"""League-aware ESPN ingestion: parsing, point-in-time columns, identity, and
fail-safe validation. Entirely offline -- synthetic fixtures shaped like the
verified live payloads (see docs/BASELINE.md)."""
import pandas as pd
import pytest

from wc2026.data.espn import (
    PROVIDER_PAGE_LIMIT,
    REGULATION_KNOWN_AFTER_MIN,
    SCHEMA_COLUMNS,
    IngestError,
    IngestReport,
    fetch_window,
    parse_event,
    validate_season,
)
from wc2026.leagues import get_league

EPL = get_league("premier_league")
MLS = get_league("mls")
RETRIEVED = "2026-08-01T00:00:00+00:00"


def event(eid="1", date="2024-08-16T19:00Z", slug="2024-25-english-premier-league",
          status="STATUS_FULL_TIME", home=("359", "Arsenal", "2"),
          away=("382", "Manchester City", "1"), neutral=None, venue_city="London"):
    return {
        "id": eid, "date": date, "name": "%s at %s" % (away[1], home[1]),
        "season": {"year": 2024, "slug": slug},
        "competitions": [{
            "status": {"type": {"name": status}},
            "neutralSite": neutral,
            "venue": {"address": {"city": venue_city, "country": "England"}},
            "competitors": [
                {"homeAway": "home", "score": home[2],
                 "team": {"id": home[0], "displayName": home[1]}},
                {"homeAway": "away", "score": away[2],
                 "team": {"id": away[0], "displayName": away[1]}},
            ],
        }],
    }


def parse(ev, spec=EPL, report=None):
    return parse_event(ev, spec, spec.primary_feed, report or IngestReport("x"),
                       RETRIEVED)


# --- schema and point-in-time -------------------------------------------------
def test_row_matches_the_declared_schema():
    row = parse(event())
    assert set(row) == set(SCHEMA_COLUMNS)


def test_point_in_time_columns_are_distinct_and_ordered():
    """known_after (when the result became knowable) must trail kickoff, and
    retrieved_at (when we fetched) is recorded separately."""
    row = parse(event())
    assert row["kickoff_utc"] == pd.Timestamp("2024-08-16 19:00:00")
    assert row["known_after_utc"] == row["kickoff_utc"] + pd.Timedelta(
        minutes=REGULATION_KNOWN_AFTER_MIN)
    assert row["known_after_utc"] > row["kickoff_utc"]
    assert row["retrieved_at_utc"] == RETRIEVED


def test_unplayed_fixture_has_no_known_after():
    row = parse(event(status="STATUS_SCHEDULED",
                      home=("359", "Arsenal", None),
                      away=("382", "Manchester City", None)))
    assert row["home_score"] is None and row["known_after_utc"] is None


def test_date_is_day_only_but_kickoff_keeps_the_time():
    row = parse(event(date="2024-08-16T19:00Z"))
    assert row["date"] == pd.Timestamp("2024-08-16 00:00:00")
    assert row["kickoff_utc"].hour == 19


# --- competition classification ----------------------------------------------
def test_european_league_rows_get_the_league_tier():
    row = parse(event())
    assert row["tournament"] == "English Premier League"
    assert row["tier"] == "league"


def test_mls_playoff_rows_get_the_playoff_tier():
    row = parse(event(slug="mls-cup"), spec=MLS)
    assert row["tier"] == "playoff"
    assert row["tournament"] == "MLS Cup Playoffs"


def test_excluded_competitions_return_none():
    assert parse(event(slug="all-star-game"), spec=MLS) is None


def test_unrecognised_season_slug_fails_loud():
    with pytest.raises(Exception):
        parse(event(slug="mystery-cup"))


# --- statuses -----------------------------------------------------------------
def test_extra_time_result_is_flagged_not_hidden():
    row = parse(event(status="STATUS_FINAL_AET", slug="mls-cup"), spec=MLS)
    assert row["went_to_extra_time"] is True
    assert row["home_score"] == 2


def test_penalty_shootout_keeps_the_regulation_draw():
    row = parse(event(status="STATUS_FINAL_PEN", slug="mls-cup",
                      home=("1", "A", "2"), away=("2", "B", "2")), spec=MLS)
    assert row["home_score"] == row["away_score"] == 2
    assert row["went_to_extra_time"] is False


def test_postponed_match_is_skipped_and_recorded():
    rep = IngestReport("x")
    assert parse(event(status="STATUS_POSTPONED"), report=rep) is None
    assert len(rep.skipped) == 1


def test_unknown_status_fails_loud():
    with pytest.raises(IngestError):
        parse(event(status="STATUS_INVENTED"))


def test_played_match_without_score_fails_loud():
    with pytest.raises(IngestError):
        parse(event(home=("359", "Arsenal", None)))


def test_implausible_score_fails_loud():
    with pytest.raises(IngestError):
        parse(event(home=("359", "Arsenal", "45")))


def test_neutral_site_is_preserved():
    assert parse(event(neutral=True))["neutral"] is True
    assert parse(event(neutral=None))["neutral"] is False


# --- identity -----------------------------------------------------------------
def test_rename_merges_to_the_latest_name_by_stable_id():
    rep = IngestReport("x")
    parse(event(eid="1", home=("110", "Montreal Impact", "1"),
                away=("359", "Arsenal", "1")), report=rep)
    parse(event(eid="2", home=("110", "CF Montreal", "2"),
                away=("359", "Arsenal", "0")), report=rep)
    assert rep.team_names["110"] == "CF Montreal"
    assert rep.renames["110"] == ["Montreal Impact", "CF Montreal"]


def test_competitor_without_id_fails_loud():
    ev = event()
    ev["competitions"][0]["competitors"][0]["team"] = {"displayName": "X"}
    with pytest.raises(IngestError):
        parse(ev)


def test_accented_names_survive_verbatim():
    row = parse(event(home=("1", "Bayer Leverkusen", "1"),
                      away=("2", "Borussia Mönchengladbach", "0")))
    rep = IngestReport("x")
    parse(event(away=("2", "Borussia Mönchengladbach", "0")), report=rep)
    assert rep.team_names["2"] == "Borussia Mönchengladbach"


# --- season validation --------------------------------------------------------
def _league_season(n_teams, matches_per_pair=2, spec=EPL):
    """Build a complete double round-robin season of synthetic rows."""
    rep = IngestReport("x")
    rows, eid = [], 0
    ids = [str(100 + i) for i in range(n_teams)]
    for i in ids:
        for j in ids:
            if i == j:
                continue
            for _ in range(matches_per_pair - 1):
                eid += 1
                rows.append(parse(event(eid=str(eid), home=(i, "T" + i, "1"),
                                        away=(j, "T" + j, "0")), spec=spec,
                                  report=rep))
    return rows, rep


def test_complete_double_round_robin_validates():
    rows, rep = _league_season(20)
    assert len(rows) == 380
    counts = validate_season(EPL, 2024, rows, rep, is_current=False)
    assert counts["league_matches"] == 380
    assert counts["expected_league_matches"] == 380


def test_truncated_season_is_refused():
    """A silently truncated upstream window must fail, not ingest quietly."""
    rows, rep = _league_season(20)
    with pytest.raises(IngestError, match="partial season"):
        validate_season(EPL, 2024, rows[:300], rep, is_current=False)


def test_in_progress_season_is_not_size_checked():
    rows, rep = _league_season(20)
    counts = validate_season(EPL, 2026, rows[:120], rep, is_current=True)
    assert counts["league_matches"] == 120     # allowed while the season runs


def test_smaller_league_uses_its_own_expected_size():
    """Bundesliga/Ligue 1 (18 teams) must validate at 306, not 380."""
    bl = get_league("bundesliga")
    rows, rep = _league_season(18, spec=bl)
    assert len(rows) == 306
    counts = validate_season(bl, 2024, rows, rep, is_current=False)
    assert counts["expected_league_matches"] == 306


def test_duplicate_event_ids_are_refused():
    rep = IngestReport("x")
    rows = [parse(event(eid="7"), report=rep), parse(event(eid="7"), report=rep)]
    with pytest.raises(IngestError, match="duplicate"):
        validate_season(EPL, 2024, rows, rep, is_current=False)


def test_mls_historical_game_count_warns_rather_than_failing():
    """MLS played 30 games per club in 2010 and 34 today; the historical
    format difference must warn, not block ingestion."""
    rep = IngestReport("x")
    rows, eid = [], 0
    ids = [str(200 + i) for i in range(16)]
    for k in range(240):                      # 2010-style season
        eid += 1
        h, a = ids[k % 16], ids[(k + 1) % 16]
        rows.append(parse(event(eid=str(eid), slug="regular-season-2010",
                                home=(h, "T" + h, "1"), away=(a, "T" + a, "0")),
                          spec=MLS, report=rep))
    validate_season(MLS, 2010, rows, rep, is_current=False)
    assert any("games per club" in w for w in rep.warnings)


def test_grossly_short_mls_season_is_refused():
    rep = IngestReport("x")
    rows, eid = [], 0
    ids = [str(200 + i) for i in range(16)]
    for k in range(20):                       # far too few -> partial response
        eid += 1
        h, a = ids[k % 16], ids[(k + 1) % 16]
        rows.append(parse(event(eid=str(eid), slug="regular-season",
                                home=(h, "T" + h, "1"), away=(a, "T" + a, "0")),
                          spec=MLS, report=rep))
    with pytest.raises(IngestError, match="partial season"):
        validate_season(MLS, 2019, rows, rep, is_current=False)


# --- provider paging ----------------------------------------------------------
class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _Session:
    def __init__(self, n_events):
        self.n = n_events

    def get(self, *a, **k):
        return _Resp({"events": [event(eid=str(i)) for i in range(self.n)]})


def test_response_at_the_page_cap_is_treated_as_truncated():
    with pytest.raises(IngestError, match="response cap"):
        fetch_window("eng.1", "20240101", "20241231",
                     session=_Session(PROVIDER_PAGE_LIMIT))


def test_normal_response_passes_through():
    evs = fetch_window("eng.1", "20240101", "20241231", session=_Session(10))
    assert len(evs) == 10
