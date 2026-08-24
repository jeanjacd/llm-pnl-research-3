"""
build_database.py
=================
Builds the player database for STRONGEST model accuracy using the best free xG data
(FBref / Opta), blending CURRENT 2026 World Cup form with a recent CLUB-SEASON prior.

Why this design
---------------
- FBref carries Opta-grade xG and updates live, but it BLOCKS automated scraping
  (Cloudflare 403). It DOES let you export any table to CSV in-browser. Since you
  refresh daily by hand, you export the CSVs and this script blends them -- best
  data quality, no scraping fight.
- Only ~2 WC games per team have been played, so WC xG alone is a tiny, noisy sample.
  A player's recent CLUB season (~30-50 games) is a far larger, still-recent sample.
  We blend them with MINUTES-WEIGHTED SHRINKAGE and a recency multiplier on WC form.

Daily workflow
--------------
1. On FBref, open the 2026 World Cup "Standard Stats" player table
   (https://fbref.com/en/comps/1/stats/World-Cup-Stats), click
   "Share & Export -> Get table as CSV", paste into:
        data/incoming/wc_players.csv
2. Open a recent club-season "Player Standard Stats" table (e.g. Big-5 leagues
   https://fbref.com/en/comps/Big5/stats/players/), export to CSV, paste into:
        data/incoming/club_players.csv
3. (Optional) Export the WC "Squad Standard Stats" (for/against) to:
        data/incoming/wc_teams.csv
4. Run:  python build_database.py
        python build_database.py --recency-weight 3 --expand-players
        python build_database.py --statsbomb-fallback   # if you have no CSVs

Team strength backbone = current Elo in teams.json (updates after each match). We do
NOT overwrite it with noisy 2-game WC team xG unless wc_teams.csv is supplied (and
even then it is heavily shrunk toward the tournament average).
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles choke on accents
except Exception:
    pass

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INCOMING_DIR = os.path.join(DATA_DIR, "incoming")

# Blend / shrinkage knobs
DEFAULT_RECENCY_WEIGHT = 2.0   # how many times to weight a WC minute vs a club minute
PRIOR_PSEUDO_90S = 5.0         # shrink small samples toward team/league mean
LEAGUE_AVG_XG90 = 0.20         # weak prior an attacker regresses toward with tiny data
TOP_PLAYERS_PER_TEAM = 6       # for --expand-players

# Team-rating shrinkage (only used if wc_teams.csv provided)
TEAM_PRIOR_MATCHES = 4.0
TEAM_AVG_XG = 1.35

# StatsBomb historical fallback (free, no key) -- 2022 WC = competition 43 / season 106
SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
SB_COMPETITION_ID = 43
SB_SEASON_ID = 106

# Country name aliases: FBref/StatsBomb -> our teams.json names (only where different).
TEAM_ALIASES = {
    "Korea Republic": "South Korea", "South Korea": "South Korea",
    "United States": "USA", "USA": "USA", "US": "USA",
    "IR Iran": "Iran", "Iran": "Iran",
    "Turkey": "Turkiye", "Türkiye": "Turkiye", "Turkiye": "Turkiye",
    "Czech Republic": "Czechia", "Czechia": "Czechia",
    "Cote d'Ivoire": "Ivory Coast", "Côte d'Ivoire": "Ivory Coast",
    "IvoryCoast": "Ivory Coast",
    "DR Congo": "DR Congo", "Congo DR": "DR Congo",
    "Curaçao": "Curacao", "Curacao": "Curacao",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _norm_name(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z ]", " ", s.lower())
    s = s.replace(" and ", " ")          # "Bosnia and Herzegovina" ~ "Bosnia & ..."
    return re.sub(r"\s+", " ", s).strip()


def normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def _find_col(fieldnames: list[str], *cands: str) -> str | None:
    """Find a CSV column by exact (case-insensitive) match, then by 'contains'."""
    low = {f.lower().strip(): f for f in fieldnames if f}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    for c in cands:
        for f in fieldnames:
            if f and c.lower() in f.lower():
                return f
    return None


def _to_float(v) -> float:
    try:
        return float(str(v).replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0.0


# ---------------------------------------------------------------------------
# FBref CSV parsing  (PRIMARY)
# ---------------------------------------------------------------------------
def parse_fbref_csv(path: str) -> dict[str, dict]:
    """
    Parse an FBref 'Standard Stats' CSV export.
    Returns {norm_name: {"name","squad","min","nineties","gls","xg"}}.
    Robust to FBref's column naming; needs Player, a minutes column, Gls and xG.
    """
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        # FBref CSV sometimes has a leading separator/title line; sniff the header row.
        lines = f.read().splitlines()
    # find the header line (the one containing 'Player')
    start = 0
    for i, ln in enumerate(lines[:5]):
        if "Player" in ln:
            start = i
            break
    reader = csv.DictReader(lines[start:])
    fns = reader.fieldnames or []
    c_player = _find_col(fns, "Player")
    c_squad = _find_col(fns, "Squad", "Team", "Nation")
    c_min = _find_col(fns, "Min", "Minutes")
    c_90 = _find_col(fns, "90s")
    c_gls = _find_col(fns, "Gls", "Goals")
    c_xg = _find_col(fns, "xG", "Expected Goals")
    if not c_player or not c_xg or not (c_min or c_90):
        print(f"  [csv] {os.path.basename(path)}: required columns not found "
              f"(need Player, xG, Min/90s). Found: {fns}")
        return {}

    out: dict[str, dict] = {}
    for row in reader:
        name = (row.get(c_player) or "").strip()
        if not name or name.lower() == "player":
            continue
        nineties = _to_float(row.get(c_90)) if c_90 else _to_float(row.get(c_min)) / 90.0
        minutes = _to_float(row.get(c_min)) if c_min else nineties * 90.0
        if nineties <= 0:
            continue
        rec = {
            "name": name,
            "squad": (row.get(c_squad) or "").strip() if c_squad else "",
            "min": minutes,
            "nineties": nineties,
            "gls": _to_float(row.get(c_gls)) if c_gls else 0.0,
            "xg": _to_float(row.get(c_xg)),
        }
        # If a player appears twice (e.g. multi-club season), accumulate.
        key = _norm_name(name)
        if key in out:
            for k in ("min", "nineties", "gls", "xg"):
                out[key][k] += rec[k]
        else:
            out[key] = rec
    print(f"  [csv] {os.path.basename(path)}: parsed {len(out)} players")
    return out


def blend_rate(wc: dict | None, club: dict | None, recency_weight: float) -> float:
    """
    Minutes-weighted blend of WC and club xG-per-90, shrunk toward LEAGUE_AVG_XG90
    by PRIOR_PSEUDO_90S to tame tiny samples. Returns blended xG90.
    """
    num = LEAGUE_AVG_XG90 * PRIOR_PSEUDO_90S
    den = PRIOR_PSEUDO_90S
    if club and club["nineties"] > 0:
        num += club["xg"]            # club xG over club 90s
        den += club["nineties"]
    if wc and wc["nineties"] > 0:
        num += wc["xg"] * recency_weight        # WC counted recency_weight times
        den += wc["nineties"] * recency_weight
    return num / den if den > 0 else LEAGUE_AVG_XG90


def apply_fbref_blend(players: list[dict], wc: dict, club: dict,
                      recency_weight: float, expand: bool,
                      teams_by_name: dict) -> int:
    # team_shot_share denominator = sum of squad xG in the WC dataset (xG share).
    team_xg_total: dict[str, float] = {}
    for rec in wc.values():
        t = normalize_team(rec["squad"])
        team_xg_total[t] = team_xg_total.get(t, 0.0) + rec["xg"]

    def shot_share(rec):
        t = normalize_team(rec["squad"])
        denom = team_xg_total.get(t, 0.0)
        return round(min(rec["xg"] / denom, 0.5), 3) if denom > 0 else None

    updated = 0
    for p in players:
        key = _norm_name(p["name"])
        w, c = wc.get(key), club.get(key)
        if not w and not c:
            continue  # no real data for this player; keep seed
        p["xG90"] = round(blend_rate(w, c, recency_weight), 3)
        if w:
            sh = shot_share(w)
            if sh:
                p["team_shot_share"] = sh
        p["source"] = "fbref blend (WC+club)" if (w and c) else \
                      ("fbref WC" if w else "fbref club")
        updated += 1
    print(f"[fbref] blended xG90 onto {updated} existing players "
          f"(recency_weight={recency_weight})")

    if expand:
        seen = {_norm_name(p["name"]) for p in players}
        by_team: dict[str, list] = {}
        for key, w in wc.items():
            if key in seen:
                continue
            team = normalize_team(w["squad"])
            if team not in teams_by_name:
                continue
            xg90 = round(blend_rate(w, club.get(key), recency_weight), 3)
            sh = shot_share(w) or 0.1
            by_team.setdefault(team, []).append({
                "name": w["name"], "team": team, "position": "FW",
                "xG90": xg90, "team_shot_share": sh, "penalty_taker": False,
                "expected_minutes": 80,
                "source": "fbref blend (WC+club)" if club.get(key) else "fbref WC",
            })
        added = 0
        for team, plist in by_team.items():
            plist.sort(key=lambda r: r["xG90"] * r["team_shot_share"], reverse=True)
            for new in plist[:TOP_PLAYERS_PER_TEAM]:
                players.append(new)
                added += 1
        print(f"[fbref] added {added} new players (--expand-players)")
    return updated


def apply_elo_csv(teams: list[dict], path: str) -> int:
    """
    Update team Elo from data/incoming/elo.csv. Needs columns like Team,Elo
    (header names are matched flexibly). Matched teams get elo_source='manual'.
    """
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = f.read().splitlines()
    start = next((i for i, ln in enumerate(lines[:5])
                  if ("team" in ln.lower() or "elo" in ln.lower())), 0)
    reader = csv.DictReader(lines[start:])
    fns = reader.fieldnames or []
    c_team = _find_col(fns, "Team", "Country", "Nation", "Squad", "Name")
    c_elo = _find_col(fns, "Elo", "Rating", "EloRating")
    if not (c_team and c_elo):
        print(f"  [csv] elo.csv: need Team and Elo columns; found {fns}")
        return 0
    by_norm = {_norm_name(normalize_team(t["name"])): t for t in teams}
    n = 0
    for row in reader:
        raw = (row.get(c_team) or "").strip()
        t = by_norm.get(_norm_name(normalize_team(raw)))
        elo = _to_float(row.get(c_elo))
        if t and elo > 0:
            t["elo"] = round(elo, 1)
            t["elo_source"] = "manual"
            n += 1
    print(f"[elo] updated Elo on {n} teams from elo.csv")
    return n


def apply_fbref_team_ratings(teams: list[dict], path: str) -> int:
    """Optional: set attack/defense from WC squad CSV, heavily shrunk toward average."""
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = f.read().splitlines()
    start = next((i for i, ln in enumerate(lines[:5]) if "Squad" in ln), 0)
    reader = csv.DictReader(lines[start:])
    fns = reader.fieldnames or []
    c_sq = _find_col(fns, "Squad", "Team")
    c_mp = _find_col(fns, "MP", "90s", "Matches")
    c_xg = _find_col(fns, "xG", "Expected Goals")
    c_xga = _find_col(fns, "xGA", "xG Against")
    if not (c_sq and c_xg):
        print("  [csv] wc_teams.csv: need Squad and xG columns; skipping team ratings")
        return 0
    teams_by_name = {t["name"]: t for t in teams}
    n = 0
    for row in reader:
        name = normalize_team((row.get(c_sq) or "").strip())
        t = teams_by_name.get(name)
        if not t:
            continue
        mp = _to_float(row.get(c_mp)) if c_mp else 0.0
        xg = _to_float(row.get(c_xg))
        xga = _to_float(row.get(c_xga)) if c_xga else None
        if mp <= 0:
            continue
        # Shrink per-match rate toward TEAM_AVG_XG with TEAM_PRIOR_MATCHES pseudo-games.
        t["attack_rating"] = round((xg + TEAM_AVG_XG * TEAM_PRIOR_MATCHES) /
                                   (mp + TEAM_PRIOR_MATCHES), 3)
        if xga is not None:
            t["defense_rating"] = round((xga + TEAM_AVG_XG * TEAM_PRIOR_MATCHES) /
                                        (mp + TEAM_PRIOR_MATCHES), 3)
        t["ratings_source"] = "fbref WC (shrunk)"
        n += 1
    print(f"[fbref] set shrunk attack/defense on {n} teams from wc_teams.csv")
    return n


# ---------------------------------------------------------------------------
# StatsBomb historical fallback (free, no key)  -- only with --statsbomb-fallback
# ---------------------------------------------------------------------------
def _sb_get(path: str):
    r = requests.get(f"{SB_BASE}/{path}", timeout=60)
    r.raise_for_status()
    return r.json()


def _minutes_from_positions(positions: list) -> float:
    def to_min(ts, default):
        if not ts:
            return default
        mm, ss = ts.split(":")
        return int(mm) + int(ss) / 60.0
    return sum(max(0.0, to_min(p.get("to"), 95.0) - to_min(p.get("from"), 0.0))
               for p in (positions or []))


def statsbomb_fallback(players: list[dict], teams_by_name: dict, max_matches=None):
    print(f"[statsbomb] HISTORICAL fallback: {SB_COMPETITION_ID}/{SB_SEASON_ID} "
          f"(this is 2022 data -- a stale prior, use only if you have no CSVs)")
    matches = _sb_get(f"matches/{SB_COMPETITION_ID}/{SB_SEASON_ID}.json")
    if max_matches:
        matches = matches[:max_matches]
    agg: dict[int, dict] = {}
    team_xg: dict[str, float] = {}
    for i, mt in enumerate(matches, 1):
        mid = mt["match_id"]
        try:
            events, lineups = _sb_get(f"events/{mid}.json"), _sb_get(f"lineups/{mid}.json")
        except requests.HTTPError:
            continue
        for side in lineups:
            for p in side["lineup"]:
                rec = agg.setdefault(p["player_id"], {"name": p.get("player_nickname")
                                     or p["player_name"], "team": side["team_name"],
                                     "xg": 0.0, "min": 0.0})
                rec["min"] += _minutes_from_positions(p.get("positions"))
        for e in events:
            if e.get("type", {}).get("name") != "Shot":
                continue
            xg = float(e.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0)
            team_xg[e["team"]["name"]] = team_xg.get(e["team"]["name"], 0.0) + xg
            pid = e.get("player", {}).get("id")
            if pid is not None:
                rec = agg.setdefault(pid, {"name": e["player"].get("name"),
                                     "team": e["team"]["name"], "xg": 0.0, "min": 0.0})
                rec["xg"] += xg
        print(f"  [{i}/{len(matches)}] aggregated", end="\r")
    by_norm = {}
    for rec in agg.values():
        if rec.get("name"):
            by_norm.setdefault(_norm_name(rec["name"]), rec)
    n = 0
    for p in players:
        rec = by_norm.get(_norm_name(p["name"]))
        if rec and rec["min"] >= 90 and rec["xg"] > 0:
            p["xG90"] = round(rec["xg"] / (rec["min"] / 90.0), 3)
            p["source"] = "statsbomb 2022 (stale fallback)"
            n += 1
    print(f"\n[statsbomb] updated {n} players from stale 2022 prior")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recency-weight", type=float, default=DEFAULT_RECENCY_WEIGHT,
                    help="how many times to weight a WC minute vs a club minute")
    ap.add_argument("--expand-players", action="store_true",
                    help="add top WC attackers per team, not just update seed")
    ap.add_argument("--statsbomb-fallback", action="store_true",
                    help="use stale 2022 StatsBomb if you have no FBref CSVs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(DATA_DIR, "teams.json"), encoding="utf-8") as f:
        tdata = json.load(f)
    with open(os.path.join(DATA_DIR, "players.json"), encoding="utf-8") as f:
        pdata = json.load(f)
    teams_by_name = {t["name"]: t for t in tdata["teams"]}

    # Team Elo from your CSV (independent of player data).
    elo_n = apply_elo_csv(tdata["teams"], os.path.join(INCOMING_DIR, "elo.csv"))

    # Player rates from FBref CSVs (or stale StatsBomb fallback).
    wc = parse_fbref_csv(os.path.join(INCOMING_DIR, "wc_players.csv"))
    club = parse_fbref_csv(os.path.join(INCOMING_DIR, "club_players.csv"))
    did_players = False
    if wc or club:
        apply_fbref_blend(pdata["players"], wc, club, args.recency_weight,
                          args.expand_players, teams_by_name)
        apply_fbref_team_ratings(tdata["teams"],
                                 os.path.join(INCOMING_DIR, "wc_teams.csv"))
        did_players = True
    elif args.statsbomb_fallback:
        statsbomb_fallback(pdata["players"], teams_by_name)
        did_players = True

    if not (elo_n or did_players):
        print("Nothing imported: add data/incoming/elo.csv and/or wc_players.csv + "
              "club_players.csv (or use --statsbomb-fallback). See README.")
        return

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    tdata.setdefault("_meta", {})["last_build"] = time.strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(DATA_DIR, "teams.json"), "w", encoding="utf-8") as f:
        json.dump(tdata, f, indent=2, ensure_ascii=False)
    with open(os.path.join(DATA_DIR, "players.json"), "w", encoding="utf-8") as f:
        json.dump(pdata, f, indent=2, ensure_ascii=False)
    print("\nDone. Database refreshed from current data (seed kept where no data).")


if __name__ == "__main__":
    main()
