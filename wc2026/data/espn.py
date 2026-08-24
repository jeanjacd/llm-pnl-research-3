"""
espn.py
=======
League-aware ingestion from the ESPN public scoreboard API.

Generalises the former MLS-only `data/mls.py` to every league in the registry.
The loader-contract schema is preserved so the frozen model consumes the output
unchanged, with point-in-time columns added:

    date, kickoff_utc, known_after_utc, home_team, away_team, home_id, away_id,
    home_score, away_score, tournament, tier, city, country, neutral,
    status, went_to_extra_time, event_id, retrieved_at_utc

Point-in-time discipline (mission rule 2) -- two distinct timestamps:
  * `known_after_utc`  when the information became EFFECTIVE. For a played
    match the result cannot be known before the final whistle, estimated
    conservatively as kickoff + REGULATION_KNOWN_AFTER_MIN. Null while unplayed.
  * `retrieved_at_utc` when THIS system fetched the row.
Historical evaluation may only use rows whose `known_after_utc` precedes the
simulated decision time.

Identity: teams are keyed by the provider's stable numeric ID and mapped to a
canonical display name. Renames are merged to the latest name and recorded in
the manifest. A team ID that cannot be resolved is an error, never a silent
league-average default.

Every ingest is idempotent, schema-validated, checksummed and accompanied by a
provenance manifest, and refuses to write on a suspected partial response.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass, field

import pandas as pd
import requests

from ..leagues import CompetitionFeed, LeagueSpec, classify_season_slug, get_league

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"

# ESPN caps a scoreboard response; hitting it means the window was truncated.
PROVIDER_PAGE_LIMIT = 1000

# A regulation match plus half-time and stoppage. Used to bound when a result
# could first have been known. Deliberately generous: over-estimating this
# delay can only make point-in-time evaluation more conservative.
REGULATION_KNOWN_AFTER_MIN = 150

# Statuses we understand. Anything else raises rather than being coerced.
PLAYED_STATUSES = {"STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FINAL_PEN",
                   "STATUS_FINAL_AET"}
EXTRA_TIME_STATUSES = {"STATUS_FINAL_AET"}
UPCOMING_STATUSES = {"STATUS_SCHEDULED", "STATUS_DELAYED", "STATUS_FIRST_HALF",
                     "STATUS_SECOND_HALF", "STATUS_HALFTIME",
                     "STATUS_IN_PROGRESS", "STATUS_END_OF_REGULATION",
                     "STATUS_OVERTIME", "STATUS_SHOOTOUT",
                     "STATUS_END_OF_EXTRATIME", "STATUS_END_OF_PERIOD",
                     "STATUS_RESCHEDULED"}
SKIPPED_STATUSES = {"STATUS_CANCELED", "STATUS_POSTPONED", "STATUS_ABANDONED",
                    "STATUS_SUSPENDED", "STATUS_FORFEIT"}

SCHEMA_COLUMNS = ["date", "kickoff_utc", "known_after_utc", "home_team",
                  "away_team", "home_id", "away_id", "home_score", "away_score",
                  "tournament", "tier", "city", "country", "neutral", "status",
                  "went_to_extra_time", "event_id", "retrieved_at_utc"]


class IngestError(RuntimeError):
    """Any anomaly in upstream data. Never swallowed, never coerced."""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


@dataclass
class IngestReport:
    league_id: str
    seasons: dict = field(default_factory=dict)
    team_names: dict = field(default_factory=dict)     # provider id -> name
    renames: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        for season in sorted(self.seasons):
            c = self.seasons[season]
            lines.append("  %-8s %4d played / %3d upcoming  (%s)"
                         % (c["label"], c["played"], c["upcoming"],
                            ", ".join("%s=%d" % (k, v)
                                      for k, v in sorted(c["tiers"].items()))))
        for tid, names in self.renames.items():
            lines.append("  rename merged (id %s): %s" % (tid, " -> ".join(names)))
        for w in self.warnings:
            lines.append("  WARNING: " + w)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def fetch_window(slug: str, start: str, end: str, session=None,
                 timeout: int = 60) -> list:
    """Raw events for a provider date window. Raises on a truncated response."""
    sess = session or requests
    resp = sess.get(SCOREBOARD.format(slug=slug),
                    params={"dates": "%s-%s" % (start, end),
                            "limit": str(PROVIDER_PAGE_LIMIT)},
                    timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    events = payload.get("events", [])
    if len(events) >= PROVIDER_PAGE_LIMIT:
        raise IngestError(
            "%s %s-%s: hit the %d-event response cap; window may be truncated. "
            "Chunk the request before trusting it."
            % (slug, start, end, PROVIDER_PAGE_LIMIT))
    return events


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_event(event: dict, spec: LeagueSpec, feed: CompetitionFeed,
                report: IngestReport, retrieved_at: str) -> dict | None:
    """One provider event -> one schema row, or None when deliberately skipped."""
    comps = event.get("competitions") or []
    if not comps:
        raise IngestError("event without competitions: %r" % event.get("id"))
    comp = comps[0]
    status = (((comp.get("status") or {}).get("type") or {}).get("name") or "")
    if status in SKIPPED_STATUSES:
        report.skipped.append("%s %s [%s]" % (event.get("date"),
                                              event.get("name"), status))
        return None
    if status not in PLAYED_STATUSES | UPCOMING_STATUSES:
        raise IngestError("unknown event status %r on %r"
                          % (status, event.get("name")))

    if feed.is_primary:
        classified = classify_season_slug(
            spec, (event.get("season") or {}).get("slug", ""))
        if classified is None:
            return None
        label, tier = classified
    else:
        label, tier = feed.label, feed.tier

    sides = {}
    for competitor in comp.get("competitors") or []:
        team = competitor.get("team") or {}
        tid = str(team.get("id") or "")
        name = team.get("displayName") or ""
        if not tid or not name:
            raise IngestError("competitor without id/name in %r"
                              % event.get("name"))
        previous = report.team_names.get(tid)
        if previous is not None and previous != name:
            report.renames.setdefault(tid, [previous]).append(name)
        report.team_names[tid] = name          # latest name wins (canonical)
        sides[competitor.get("homeAway")] = (tid, competitor.get("score"))
    if set(sides) != {"home", "away"}:
        raise IngestError("event without a home/away pair: %r"
                          % event.get("name"))

    played = status in PLAYED_STATUSES

    def score(side):
        raw = sides[side][1]
        if not played:
            return None
        if raw is None or raw == "":
            raise IngestError("played match missing a score: %r"
                              % event.get("name"))
        value = int(raw)
        if not 0 <= value <= 30:
            raise IngestError("implausible score %d in %r"
                              % (value, event.get("name")))
        return value

    kickoff = pd.Timestamp(event["date"]).tz_convert(None)
    known_after = (kickoff + pd.Timedelta(minutes=REGULATION_KNOWN_AFTER_MIN)
                   if played else None)
    venue = comp.get("venue") or {}
    address = venue.get("address") or {}
    return {
        "date": kickoff.normalize(),
        "kickoff_utc": kickoff,
        "known_after_utc": known_after,
        "home_team": None,                 # filled after canonicalisation
        "away_team": None,
        "home_id": sides["home"][0],
        "away_id": sides["away"][0],
        "home_score": score("home"),
        "away_score": score("away"),
        "tournament": label,
        "tier": tier,
        "city": address.get("city", ""),
        "country": address.get("country", ""),
        "neutral": bool(comp.get("neutralSite")),
        "status": status,
        "went_to_extra_time": status in EXTRA_TIME_STATUSES,
        "event_id": str(event.get("id") or ""),
        "retrieved_at_utc": retrieved_at,
    }


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_season(spec: LeagueSpec, season: int, rows: list,
                    report: IngestReport, is_current: bool,
                    n_skipped: int = 0) -> dict:
    """Boundary validation for one season of the primary feed."""
    primary_tiers = {spec.primary_feed.tier}
    league_rows = [r for r in rows if r["tier"] in primary_tiers]
    played = [r for r in rows if r["home_score"] is not None]
    upcoming = [r for r in rows if r["home_score"] is None]

    ids = [r["event_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise IngestError("%s %s: duplicate event ids in season payload"
                          % (spec.league_id, spec.season_label(season)))
    if upcoming and not is_current:
        report.warnings.append(
            "%s %s: %d unplayed fixtures in a completed season"
            % (spec.league_id, spec.season_label(season), len(upcoming)))

    teams = ({r["home_id"] for r in league_rows}
             | {r["away_id"] for r in league_rows})
    expected = spec.schedule.expected_matches(len(teams))
    if league_rows and not is_current:
        # A completed season must be consistent with its schedule shape. This
        # is what catches a silently truncated upstream window.
        if spec.schedule.kind == "double_round_robin":
            # Exact: every pair meets home and away. Matches legitimately
            # dropped upstream (abandoned, postponed, forfeited -- e.g. the
            # 2016-17 Ligue 1 Bastia tie awarded after crowd trouble) still
            # count toward completeness; they simply have no playable score.
            if len(league_rows) != expected:
                if len(league_rows) + n_skipped == expected:
                    report.warnings.append(
                        "%s %s: %d league matches + %d skipped (abandoned/"
                        "postponed) = %d expected"
                        % (spec.league_id, spec.season_label(season),
                           len(league_rows), n_skipped, expected))
                else:
                    raise IngestError(
                        "%s %s: %d league matches (+%d skipped) but %d teams "
                        "implies %d. Refusing to ingest a suspected partial "
                        "season."
                        % (spec.league_id, spec.season_label(season),
                           len(league_rows), n_skipped, len(teams), expected))
        else:
            # Unbalanced schedules (MLS) legitimately changed games-per-team
            # over the years (30 in 2010, 34 today), so the invariant is that
            # every club played a SIMILAR number of matches. A truncated
            # window collapses that count and is caught here.
            per_team = {}
            for r in league_rows:
                per_team[r["home_id"]] = per_team.get(r["home_id"], 0) + 1
                per_team[r["away_id"]] = per_team.get(r["away_id"], 0) + 1

            counts = sorted(per_team.values())
            modal = counts[len(counts) // 2] if counts else 0
            configured = spec.schedule.games_per_team or 0
            if configured and modal < 0.6 * configured:
                raise IngestError(
                    "%s %s: median %d games per club against a configured %d. "
                    "Refusing to ingest a suspected partial season."
                    % (spec.league_id, spec.season_label(season), modal,
                       configured))
            if configured and modal != configured:
                report.warnings.append(
                    "%s %s: %d games per club (configured %d) -- historical "
                    "format difference, ingested as-is"
                    % (spec.league_id, spec.season_label(season), modal,
                       configured))
    tiers = {}
    for r in rows:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    return {"label": spec.season_label(season), "played": len(played),
            "upcoming": len(upcoming), "teams": len(teams), "tiers": tiers,
            "expected_league_matches": expected,
            "league_matches": len(league_rows)}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def ingest_league(spec: LeagueSpec, first_season: int | None = None,
                  last_season: int | None = None, session=None,
                  verbose: bool = True) -> dict:
    """Full idempotent refresh of one league. Returns the manifest dict."""
    today = _utcnow().date()
    # A season is "current" if its window has not closed yet.
    first = first_season if first_season is not None else spec.first_season
    if last_season is None:
        last_season = today.year if today.month >= 7 else today.year - 1
        if spec.season_format == "calendar_year":
            last_season = today.year
    retrieved_at = _utcnow().isoformat()
    report = IngestReport(league_id=spec.league_id)
    sess = session or requests.Session()
    rows: list = []

    for season in range(first, last_season + 1):
        windows = spec.season_windows(season)
        last_end = windows[-1][1]
        end_date = dt.date(int(last_end[:4]), int(last_end[4:6]),
                           int(last_end[6:]))
        is_current = end_date >= today
        season_rows: list = []
        seen_events: set = set()
        skipped_before = len(report.skipped)
        for feed in spec.feeds:
            for start, end in windows:
                events = fetch_window(feed.provider_slug, start, end,
                                      session=sess)
                for event in events:
                    eid = str(event.get("id") or "")
                    if eid in seen_events:
                        continue          # overlapping windows -> same match
                    # Bucket by the PROVIDER's season year so a season that
                    # overruns its calendar window is still attributed
                    # correctly, and the next season is not absorbed.
                    if feed.is_primary and spec.season_format != "calendar_year":
                        year = (event.get("season") or {}).get("year")
                        if year is not None and int(year) != season:
                            continue
                    row = parse_event(event, spec, feed, report, retrieved_at)
                    seen_events.add(eid)
                    if row is not None:
                        season_rows.append(row)
        if not season_rows:
            report.warnings.append("%s %s: no matches returned"
                                   % (spec.league_id, spec.season_label(season)))
            continue
        counts = validate_season(spec, season, season_rows, report, is_current,
                                 n_skipped=len(report.skipped) - skipped_before)
        report.seasons[season] = counts
        rows.extend(season_rows)
        if verbose:
            print("  %-8s %s: %d played, %d upcoming"
                  % (spec.league_id, spec.season_label(season),
                     counts["played"], counts["upcoming"]))

    if not rows:
        raise IngestError("%s: ingest produced no rows" % spec.league_id)

    frame = pd.DataFrame(rows)
    frame["home_team"] = frame["home_id"].map(report.team_names)
    frame["away_team"] = frame["away_id"].map(report.team_names)
    if frame["home_team"].isna().any() or frame["away_team"].isna().any():
        raise IngestError("%s: team id without a canonical name" % spec.league_id)

    duplicated = frame["event_id"].duplicated()
    if duplicated.any():
        raise IngestError("%s: %d duplicate event ids across seasons"
                          % (spec.league_id, int(duplicated.sum())))

    out = frame[SCHEMA_COLUMNS].sort_values(["kickoff_utc", "event_id"]).copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["kickoff_utc"] = out["kickoff_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["known_after_utc"] = out["known_after_utc"].apply(
        lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(t) else "")

    os.makedirs(spec.raw_dir, exist_ok=True)
    out.to_csv(spec.matches_csv, index=False)

    manifest = {
        "league_id": spec.league_id,
        "display_name": spec.display_name,
        "provider": spec.provider,
        "feeds": [{"key": f.key, "slug": f.provider_slug, "label": f.label,
                   "tier": f.tier, "role": f.role} for f in spec.feeds],
        "retrieved_at_utc": retrieved_at,
        "seasons": {str(k): v for k, v in report.seasons.items()},
        "n_rows": int(len(out)),
        "n_teams": int(len({*out["home_id"], *out["away_id"]})),
        "team_id_to_name": report.team_names,
        "renames_merged": report.renames,
        "skipped_events": report.skipped,
        "warnings": report.warnings,
        "schema_columns": SCHEMA_COLUMNS,
        "matches_csv_sha256": _sha256(spec.matches_csv),
        "known_after_minutes": REGULATION_KNOWN_AFTER_MIN,
    }
    with open(spec.manifest_json, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if verbose:
        print(report.summary())
        print("  wrote %s (%d rows, sha %s)"
              % (spec.matches_csv, len(out), manifest["matches_csv_sha256"][:12]))
    return manifest


def ingest_all(league_ids=None, verbose: bool = True) -> dict:
    """Ingest every registered league (or a subset). Returns id -> manifest."""
    from ..leagues import league_ids as all_ids
    out = {}
    for lid in (league_ids or all_ids()):
        out[lid] = ingest_league(get_league(lid), verbose=verbose)
    return out
