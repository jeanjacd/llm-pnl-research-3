# Baseline audit (pre-refactor)

Established before any code change, per the mission's "inspect first" rule.
Environment: Python 3.11.9, existing `.venv`. **176 tests collected, 176 passing.**
The project is **not a Git repository** — no history exists and none is fabricated here.

Package code ~4,900 LOC; tests ~1,060 LOC.

## Reusable components (preserve, generalize)

| Component | Why it survives |
|---|---|
| `model/dixon_coles.py` | Time-weighted DC bivariate-Poisson, penalised MLE, analytic gradients. Sign convention correct (rho<0), tested. |
| `model/elo.py`, `model/ratings.py` | Transparent Elo + team-specific shrinkage `w = n_eff/(n_eff+k)`. Directly useful for promoted/thin-history clubs. |
| `sim/match.py` | Scoreline grid → all match-level claims from one joint distribution. Correct, but see "not exact" below. |
| `eval/backtest.py` | Expanding-window walk-forward with monthly refit; proper scoring rules. Good skeleton, insufficient for headline claims (see gaps). |
| `betting/fees.py` | Integer-exact Kalshi fee math, verified cell-by-cell against the published fee table. |
| `betting/kelly.py` | Joint expected-log-wealth optimisation over the score grid — correct correlated sizing. |
| `betting/gate.py` | Fail-closed schema validation, one-directional authority, audit. Sound governance pattern to extend to a 3-member board. |
| `betting/tracking.py` | Upsert-once resolution rows, candlestick closing line, placement-gated realized P&L. |

## MLS-specific assumptions (must generalize)

- **Global singletons** in `config.py`: `RESULTS_CSV`, `FITTED_PARAMS_JSON`, `ADJUSTMENTS_JSON`, `MLS_CALIBRATION_JSON` — one active league, hard-wired.
- `data/mls.py` season-window fetch is **calendar-year** (`YYYY0101-YYYY1231`); European seasons span Aug–May.
- `_classify_mls_slug` understands `regular-season`, MLS playoff vocabulary, `all-star-game`. European slugs are `2024-25-english-premier-league` — unrecognised, would raise.
- `_expected_regular_season_size` hard-codes `n_teams * 34 / 2` (MLS unbalanced schedule). European double round-robin is `n*(n-1)`.
- `loader.classify_tier` maps MLS/US competition names only.
- `betting/markets.py` `_TEAMS` is a 30-club MLS alias table; `KXMLS*` series only.
- `MLS_CONFIG` importance weights keyed to `mls_regular`/`mls_playoffs`/cups.

## Verified provider facts (checked live, not assumed)

| League | ESPN slug | 2024-25 matches | Teams | Schedule |
|---|---|---|---|---|
| MLS | `usa.1` | 412 (incl. playoffs) | 30 | 34 games/team |
| Premier League | `eng.1` | 380 | 20 | double round-robin |
| La Liga | `esp.1` | 380 | 20 | double round-robin |
| Bundesliga | `ger.1` | 306 | 18 | double round-robin |
| Ligue 1 | `fra.1` | 306 | 18 | double round-robin |

Ligue 1 **changed size**: 20 teams/380 matches through 2022-23, 18/306 from 2023-24 — a fixed expected-size rule would reject valid seasons.
EPL 2024-25 promoted Ipswich/Leicester/Southampton and relegated Burnley/Luton/Sheffield United: **3 clubs per season arrive with no top-flight history.**

## Behaviour claimed in docs but NOT enforced in code

1. **"Exact" scoreline grid.** README and `sim/match.py` call the grid exact. It is a truncated 13×13 grid, renormalised — excluded tail mass is never quantified. Defensible for 1X2, materially wrong for high-line totals/exact scores.
2. **Slippage / book walking.** `betting/ev.py` docstring: a fill averaging more than `max_slippage_cents` above the touch "is trimmed". In reality `OrderBook.avg_fill_price` is **never called from the package** (only a test) and `max_slippage_cents` is **never read**. EV uses the touch price; size is capped at touch depth. Conservative in effect, but the documented multi-level walk does not happen.
3. **`rho_bounds`** — config says rho is "bounded to a sane negative range during fitting". The field is never referenced; rho is held fixed, never bounded.
4. **`maker_fee_factor`** — declared and documented; no maker path exists. All EV assumes taker (conservative, but the claim is unimplemented).
5. **Unknown teams silently become league average.** `DCRatings.expected_goals` uses `.get(team, 0.0)`. A promoted club, a rename miss, or a typo yields a confident average-strength forecast with no signal that identity resolution failed.
6. **`host_nations`** — dead World-Cup-era field.

## Known limitations

- Evaluation uses one expanding-window pass; **hyperparameters were tuned on the same window** later reported as the headline result. There is no untouched final holdout, and no sentinel test that a future row cannot change a past forecast.
- Calibration is a **single pooled trust curve** built from 1X2 outcomes, then applied to totals, spreads, team totals and exact scores alike — unvalidated transfer across market families.
- Backtest beats a base-rate baseline only (log-loss 1.0460 vs 1.0641 ≈ 1.7%). **No market-relative claim is supported**: there are no stored point-in-time market prices.
- CLV sample is tiny and consistent with zero (mean −0.19c over 43 bets / 12 matches, t≈−0.43).
- Kalshi-only; no Polymarket, no combo/parlay representation, no cross-venue matching.
- No point-in-time separation of "effective at" vs "retrieved at" anywhere in the data model.

## Private/live state — must never be published

`data/betting/` holds **real money history**: `state.json` (bankroll $78.80, ledger of settled P&L), `recommendations.jsonl` (262 rows incl. 20 real placements), `audit.jsonl`, `gate_audit.jsonl`, `adjustment_suggestions.jsonl`, plus `.bak`/`.pre-democlean` copies.
`.gitignore` currently covers only `data/raw/`, `__pycache__/`, `.venv/`, `.pytest_cache/`, `*.egg-info/`.
**On push, the entire personal trading history and bankroll would be committed.** Fixing this precedes any repository work.
Also unpublishable: `.env`, Kalshi private key path/ID, local Claude credentials, raw authenticated API responses, full model transcripts.
