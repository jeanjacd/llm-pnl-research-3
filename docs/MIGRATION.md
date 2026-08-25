# Migration notes

From the single-league (MLS) betting tool to the multi-league, board-governed
paper-trading system.

## Private state — read this before pushing

`data/betting/` holds **real money history** from the earlier phase: bankroll,
positions, realised P&L, order and gate audit logs, plus `.bak` copies.

* It is **git-ignored** and must never be committed.
* It has **not** been deleted — it stays on this machine.
* `scripts/check_no_private_state.py` fails CI if it (or a key, `.env`, or a
  state file containing a balance/P&L/order id) ever becomes tracked.

Before the first push, verify:

```bash
python scripts/check_no_private_state.py
git status --porcelain | grep -i "data/betting" && echo "STOP: private state staged"
```

The paper ledger (`data/paper/portfolio.json`) is also ignored in the working
tree: it lives on the dedicated `automation-state` branch, written by
`scripts/state_sync.py`, so the public tree never carries mutable state.

## Data layout

| Before | After |
|---|---|
| `data/mls/raw/matches.csv` | `data/leagues/mls/raw/matches.csv` |
| `data/mls/fitted_params.json` | `data/leagues/<id>/fitted_params.json` |
| `data/mls/calibration.json` | `data/leagues/<id>/calibration/<family>.json` |
| `data/mls/adjustments.json` | `data/leagues/<id>/adjustments.json` |
| — | `data/leagues/<id>/evaluation.json` |

The legacy `data/mls/` directory is left in place, unused. Nothing reads it.
Regenerate everything with:

```bash
python -m wc2026 update --all
python -m wc2026 evaluate --all
```

## Code changes that will break external callers

* **`config.py` no longer exports** `RESULTS_CSV`, `FITTED_PARAMS_JSON`,
  `ADJUSTMENTS_JSON`, `MLS_*`. Use `get_league(<id>)` and its path properties.
* **`loader.load_matches(path)` requires an explicit path.** Use
  `loader.load_league(spec)` for a league.
* **`data/mls.py` and `data/ingest.py` were removed**, superseded by
  `data/espn.py` (league-aware, point-in-time, checksummed).
* **`build_team_strength(..., apply_soft_factors=True)` became
  `adjustments_path=<path|None>`.** The soft-factor layer is now opt-in per
  league; historical fits and backtests stay purely quantitative by default.
* **`DCRatings.expected_goals` raises `UnknownTeamError`** for an unrated team
  instead of silently substituting the league average. Pass
  `allow_unknown=True` only for exploratory/reporting paths.
* **Competition tiers are generic** (`league`, `playoff`, `domestic_cup`,
  `continental_cup`), not MLS-specific strings. `tier` is now a column written
  at ingestion rather than re-derived from a label.
* Every CLI command that touches data takes `--league` (or `--all`).

## Behavioural changes worth knowing

* **Tuning no longer reports its own selection score as performance.** The
  headline comes from a holdout that is opened exactly once.
* **Cups are `training_only`.** The old MLS fit included cups at weight 1.0,
  which mixed 286 non-MLS clubs (US Open Cup reaches amateur sides) into a
  single-league fit with no league-strength term. Inclusion must now be
  re-earned per league by walk-forward test.
* **Untuned leagues cannot be traded** — `require_tradeable` raises.
* The legacy `recommend` / `execute` / `track` commands still exist for the
  Kalshi single-league workflow. New work should use `paper-cycle`, which is
  paper-only by construction.
* **The loop is now two commands, not one.** `paper-cycle` discovers markets,
  builds cases and boards one fixture at a time; `paper-maintain` settles,
  replays resting fills and captures closing lines with no model calls and no
  market discovery. Run the second every day -- it is what turns submitted
  orders into a P&L, and it is nearly free.

```bash
python -m wc2026 paper-maintain --state data/paper/portfolio.json
```

## Verifying a migration

```bash
python -m wc2026 leagues            # all five present, tuned, tradeable
python scripts/validate_manifests.py  # checksums match the CSVs
pytest -q                            # 362 tests
```
