# Multi-league refactor — status against the specification

Honest accounting. Anything not listed as **DONE** is not implemented, and no
claim in this repository should be read as implying otherwise.


---

## DONE — all eight phases

Test suite: **362 passing**, lint clean, private-state check clean.
Still **not a git repository**; no history created or claimed.

### Phase 0 — baseline (`docs/BASELINE.md`)
Architecture split into reusable / MLS-specific / limitations /
claimed-but-unenforced / private state. Found: the "exact" grid is truncated,
slippage was documented but never executed, `rho_bounds` and `maker_fee_factor`
were dead, unknown teams silently became league-average, and hyperparameters
were tuned on the reported window.

### Repository hygiene (Phase 8)
`.gitignore` previously covered only `data/raw/`, so the personal bankroll and
trade history would have been published on first push. Now excluded, with
`scripts/check_no_private_state.py` failing CI if anything private is tracked.

### Phase 1 — league registry and ingestion
`leagues.py` (`LeagueSpec`, disjoint paths and parameter objects, verified
slugs, schedule shapes, competition roles) and `data/espn.py` (point-in-time
`known_after_utc` + `retrieved_at_utc`, stable team ids, SHA-256 manifests,
fail-loud validation). Handles the COVID season overrun, Ligue 1's 20->18
change, and abandoned/cancelled matches. All five leagues ingested.

### Phase 2 — leak-resistant evaluation
`eval/nested.py`: DEV / CAL / HOLDOUT, holdout opened exactly once
(`SplitViolation` on a second attempt). Cluster-bootstrap CIs, calibration
slope/intercept, ECE. Sentinel tests prove a future row cannot change a past
forecast. Unknown teams are skipped and counted. Results in `docs/EVALUATION.md`.

### Phase 3 — venue-independent market ingestion
`venues/`: normalised `MarketInstrument` (binary / native combo / bundle, not
interchangeable), read-only Kalshi + Polymarket providers, decision hashing so
unchanged markets aren't re-reviewed, cross-venue equivalence that refuses an
unknown settlement basis. Full category audit in `docs/VENUES.md`.

### Phase 4 — deterministic fair value and limit prices
`decision/`: conservative probability bounds, real book walking, venue-exact
fees, adverse-selection reserve, a **limit ladder** over every viable tick, six
action states, and joint combo valuation that never multiplies dependent
marginals (measured 19.6% error on a same-match parlay).

### Phase 5 — sealed three-member board
`board/`: versioned prompts and strict schemas; quant and coach independent and
sealed; judge only after both validate; price-redacted coach packet; every
failure path resolves to DEFER; no stage can raise a price or size.

### Phase 6 — paper broker
`paper/broker.py`: cash reservation, idempotent submission, marketable fills
capped by real depth, resting fills only on trade-through, expiry/cancel,
once-only settlement, P&L on actual fills.

### Phase 7 — GitHub Actions
`ci.yml`, `hourly-board.yml` (cron 17 * * * *, concurrency-guarded, paper-only),
`daily-data.yml`, `weekly-evaluation.yml`. Durable state on a dedicated branch
via `scripts/state_sync.py` — never a dependency cache.

### Phase 8 — packaging and reporting
`pyproject.toml` (deps, extras, ruff, coverage, console script), rewritten
README, `docs/MIGRATION.md`, `scripts/build_report.py` producing a report that
separates backtest from paper from counterfactual.

---

## Completion checklist (mission section "Before declaring completion")

| # | Item | Result |
|---|---|---|
| 1 | Full suite from a clean environment | 362 passed |
| 2 | Formatting / static checks | ruff clean |
| 3 | Offline five-league fixture | manifests validate, checksums match |
| 4 | Mocked Kalshi + Polymarket snapshots | covered in `test_venues.py` |
| 5 | Current-price buy | BUY_NOW verified |
| 6 | Conditional limit that fills later | trade-through fills; touch does not |
| 7 | Conditional limit that expires | expired, cash released |
| 8 | Coach-required rerun | DEFER |
| 9 | Malformed board output | fails closed to DEFER |
| 10 | Scalar settlement | pays correctly; double-settle blocked |
| 11 | Two scheduled cycles | idempotent, no double-submit |
| 12 | Compare behaviour to spec | this table |
| 13 | Unsupported families listed honestly | `docs/VENUES.md` |

---

## Honest limitations (unchanged by this work)

- **No market-alpha claim is supported.** Every evaluation figure is model vs
  base rate; no point-in-time market prices are stored yet.
- **MLS is weak and over-confident** (1.8% improvement, calibration slope
  0.792); its CI barely clears its baseline.
- **~54% of discovered markets are abstained from** (corners, halves, player
  props, futures) — the data exists, the models do not. Decision recorded.
- Venue coverage is a snapshot; a newly added modellable family would be
  abstained from until a parser is written.
- The board has been exercised end-to-end with mocked members and live-verified
  only on the older single-league gate. A live three-member run has not been
  executed here.
- Second-division history for promoted-team priors was not evaluated.
