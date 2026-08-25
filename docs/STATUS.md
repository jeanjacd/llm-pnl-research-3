# Multi-league refactor — status against the specification

Honest accounting. Anything not listed as **DONE** is not implemented, and no
claim in this repository should be read as implying otherwise.


---

## DONE — all eight phases

Test suite: **528 passing**, lint clean, private-state check clean.

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
`ci.yml`, `matchday-board.yml` (cron `17 */3 * * 5,6,0` -- every 3h on
Fri/Sat/Sun UTC, where 89.5% of fixtures fall; concurrency-guarded, paper-only),
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

---

## Live board runs (2026-08-24) — what they found

Three runs against real books and a real fitted model, using
`CLAUDE_CODE_OAUTH_TOKEN`. Each one surfaced a defect that the 362-test suite
did not, which is the point of the exercise.

| # | venue | market | board outcome | defect exposed |
|---|---|---|---|---|
| 1 | Polymarket | Crystal Palace v Man City, draw | DEFER (quant) | Polymarket fee read as a fraction — 10,000× — giving `breakeven_prob = 2.56` |
| 2 | Kalshi | Fulham v Chelsea, draw | PASS (coach REJECT) | `expected_expiration_time` used as kick-off; the match was already 72 min old |
| 3 | Kalshi | Crystal Palace v Man City, draw | DEFER (coach) | none — clean run, 94h to kick-off |

Run 1's quant member wrote: *"Ladder reports fee_cents=1000 on a 1¢ contract
(10x the notional) … Breakeven_prob=2.56% is mathematically incoherent for a
23¢ price."* Run 2's analyst reported first-half shot counts for a match the
deterministic layer believed had not started.

In every run the judge was correctly **not reached** — a quant or coach veto
resolves before it, and the board failed closed each time.

### What those runs changed

* Polymarket fees: correct formula shape, integer-cent symmetry, maker/taker
  roles, rate taken from `/fee-rate` (see `docs/VENUES.md`).
* Kalshi fixture identity: rules-text parsing, home-first ordering verified
  255/255, full claim mapping for all six families — **0 → 1,086 priceable
  instruments**.
* Kick-off: sourced from our fixture table; unknown kick-off now DEFERs.
* `paper/cycle.py` withheld the fixture table from Kalshi specifically, which
  is what made the venue contribute nothing. Both providers now receive it.
* Club-name aliases (Köln/Cologne, PSG, M´gladbach, Bilbao, LAFC/LA Galaxy),
  each added only because a measured resolution failure demanded it.

An alias asserts a specific club, so it only counts against a name that really
is that club: "PSG" expanded to "Paris Saint-Germain" scored **0.67** against
**Paris FC** — a different Ligue 1 side — and on a day PSG was idle there was
no runner-up for the margin rule to catch. Raising the acceptance threshold was
not an option: correct matches score as low as 0.62 ("DC United" →
"D.C. United", "Saint Louis" → "St. Louis CITY SC").

---

## The measurement loop (2026-08-25)

Until this, the paper loop could not produce a P&L **at any cadence**.
`PaperPortfolio.try_fill_resting` and `.settle` had no production caller, so a
limit order rested until it expired and `realized_pnl_usd` was structurally
0.00. The suite was green throughout, because it exercised the broker's methods
directly rather than through the cycle. Unit coverage on a component says
nothing about whether the component is plugged in.

### What was wired

| piece | file | what it does |
|---|---|---|
| claim outcomes | `paper/outcomes.py` | what a claim PAID, given a scoreline |
| settlement | `paper/settlement.py` | resolves positions against finished matches |
| resting fills | `paper/fills.py` | replays venue price history across the gap |
| closing line | `paper/clv.py` | CLV at kick-off, averaged over fixtures |
| selection | `paper/selection.py` | one board decision per fixture, at T-24h |

**Settlement answers the same question the forecast did.** `test_outcomes.py`
does not check this by inspection: for every claim it sums the model's own
scoreline grid over the cells settlement calls a win and asserts the total
equals `probability_for`. A divergence in either file fails the suite.

**Fills are a counterfactual, not an observation.** Nothing here reaches a
venue -- `paper/broker.py` imports no network library -- so no matching engine
knows our orders exist. Comparing book snapshots each run would make the
measured fill rate a property of the cron expression, so the tape is replayed
across the whole gap instead. Kalshi's per-minute candlesticks give the ask low
exactly; Polymarket publishes only the mid (verified: bid 0.600 / ask 0.610 ->
history 0.605), so its ask is bounded rather than observed and its fills are
biased low relative to Kalshi. The two must not be compared naively.

A resting fill pays **our limit**, not the price the tape printed: we are the
passive side and the aggressor takes the improvement. Quoting the fill at the
observed price handed us a free cent per contract and inflated P&L.

### Verified end to end on real data

Real Kalshi markets, real candlesticks, real ESPN result -- Fulham 2-3 Chelsea,
2026-08-24:

| bet | fill | fee | outcome |
|---|---|---|---|
| Chelsea win @ 50c x100 | traded through to 48c | 175c | **+$48.25** |
| draw @ 26c x100 | traded through to 25c | 135c | **-$27.35** |

Realized P&L **+$20.90**; both fees match Kalshi's published table exactly.
CLV: -2.50c, +1.50c -- reported as **one** fixture, not two observations.

### Why one board decision per fixture

Measured on the live venues 2026-08-24: **307 placeable cases across 9 distinct
fixtures**, 34 per match. One fixture produced `away_win`, `away_over_0.5/1.5/
2.5` and `away_wins_by_over_1.5/2.5` -- a single directional view written six
ways off one scoreline grid, which loses six times together.

Boarding all 307 costs ~28 hours of model time per run (measured: quant 87s,
coach 180s) against a 45-minute job, and buys 9 independent observations.

`top-N by EV` is the obvious cap and the wrong one: the highest apparent EV is
where the model's error is most likely positive, so it samples the model
precisely where it is most over-confident. Selection is by claim FAMILY instead
-- 1X2 first, exact scorelines last -- with EV only as a tie-break inside a
family, which also keeps observations comparable across fixtures.

The coach is cached per fixture. Its packet was diffed across two markets on
one match: identical on home, away, league and kick-off, differing only in
`case_id`, `claim`, `instrument_id`, `observed_at` and `hours_to_kickoff`. It
was repeating the same 180s of web research per market, and could return
contradictory findings for one fixture.

### Schedule, split by cost profile

* `matchday-board.yml` -- `17 */6 * * *`, **every day**. Each fixture is boarded
  once at ~T-24h, so a run with nothing in the window is nearly free. This
  supersedes the Fri/Sat/Sun-only schedule: 89.5% of fixtures fall on those
  three days, but a Wednesday match has its T-24h on **Tuesday**, and the 162
  midweek fixtures are 10.5% of the sample that any significance claim rests on.
* `paper-maintenance.yml` -- `20 6,18 * * *`, every day, **no model calls**.
  Settlement, fill replay and closing lines cost a handful of HTTP requests.

`matchday-board` had **no data-fetch step** and `data/leagues/*/raw/` is
gitignored, so every scheduled run had been skipping every league as a silent
no-op. Both workflows now refresh results first.

### CLV is the headline, P&L the secondary reading

| metric | edge | independent bets | at ~48 fixtures/wk |
|---|---|---|---|
| CLV (sigma 2.8c) | 0.5c | 246 | ~5 weeks |
| P&L (sigma ~40c on a 20c binary) | 4c | 785 | ~16 weeks |

The summary leads with CLV, prints the **fixture** count beside the bet count,
computes the headline over per-fixture means, and states outright that a
reading below ~246 fixtures is noise.

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
- **MLS is weak** — see above; the same caveat applies to every venue.
- Second-division history for promoted-team priors was not evaluated. This is
  live and visible: 17 Bundesliga instruments are discovered, resolved and
  claim-mapped but cannot be priced because **SV Elversberg** has no fitted
  rating. `UnknownTeamError` is raised rather than substituting a league
  average.
- **Polymarket's fee is charged at twice the published rate** because the venue
  reports twice the published rate on all three of its own surfaces. Every
  Polymarket EV figure is therefore conservative by that amount. See
  `docs/VENUES.md`.
- The board has now been run live end-to-end on both venues (above), but three
  runs is not a sample. No claim is made about its decision quality.
- **Fill SIZE is assumed, not observed.** Price history carries no depth, so a
  replayed fill takes the order's whole remaining size. Every such fill records
  `basis="history"` so they can be separated later.
- **No CLV or P&L reading yet approaches significance.** At the time of
  writing the book holds a handful of fixtures against the ~246 needed.
