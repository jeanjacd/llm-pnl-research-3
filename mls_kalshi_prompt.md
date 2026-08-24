# Mission: Retire the World Cup, Stand Up an MLS EV-Betting Engine on Kalshi

You are working in the `worldcup2026` project. The 2026 World Cup chapter of this project is over. Your mission has four phases, in strict order: **(1) archive all World Cup and international data**, **(2) import MLS data into the existing pipeline without changing the statistical engine**, **(3) build an expected-value betting layer for Kalshi with dynamic fractional-Kelly sizing**, and **(4) build a news-check gate — an embedded Claude review step that screens proposed bets for injuries, call-ups, rotation, and other soft factors the model cannot see — as the final stage before any order is placed**. MLS is the only league in scope for now — the architecture should anticipate more leagues later, but do not import any other league's data yet.

Before writing any code, read `README.md`, `wc2026/config.py`, `wc2026/data/loader.py`, `wc2026/data/sources.py`, `wc2026/data/ingest.py`, `wc2026/cli.py`, and skim `wc2026/model/` and `wc2026/sim/match.py` so you understand exactly what you are preserving. `legacy/kalshi_client.py` contains a working Kalshi REST client (RSA-PSS request signing, market lookup, name aliasing) — mine it for reference, but treat legacy code as read-only history.

---

## Guiding principles (non-negotiable)

1. **Mathematical precision and accuracy.** Every probability, EV, fee, and stake calculation must be exact and unit-tested. No approximations where closed forms exist. Kalshi prices are integer cents; handle rounding explicitly and in the house's favor when conservative.
2. **Optimal data manipulation.** Vectorized pandas/numpy throughout, typed schemas, idempotent ingestion, no silent coercions. Validate every dataset at the boundary (row counts, date ranges, null checks, duplicate detection) and fail loudly on anomalies.
3. **High-level statistical analysis.** Nothing ships on faith. Every modeling choice for MLS must be justified by walk-forward, leakage-free validation — the same standard the current backtest already sets (log-loss, Brier, RPS vs. a base-rate baseline, calibration curves).
4. **Thoroughness. No lackadaisical mistakes.** Run the full test suite after every phase. Write new tests for everything you add. Double-check sign conventions, date boundaries, and team-name mappings — the legacy code's Dixon-Coles sign error is this project's cautionary tale. Before declaring any phase done, re-verify its acceptance criteria explicitly.

## The prime directive: the simulation engine is frozen

`wc2026/model/` (elo, dixon_coles, ratings, adjustments) and `wc2026/sim/match.py` are the validated statistical core. **Do not modify their mathematics.** The Dixon-Coles time-weighted bivariate-Poisson fit, the Elo-prior shrinkage, the exact scoreline grid — all of it stays byte-for-byte identical unless a genuine bug is discovered (in which case: stop, document it, and flag it before touching anything).

What *is* fair game: configuration values (they exist to be re-tuned per dataset), the data layer, the CLI, and new modules. The design insight that makes this clean: **the loader's output schema is the contract.** If MLS data is transformed into the exact schema the loader already emits (`date, home_team, away_team, home_score, away_score, tournament, city, country, neutral` → typed table with `played` and `tier`), the frozen engine consumes it without knowing anything changed.

---

## Phase 1 — Archive the World Cup / international era

Create an `archive/wc2026/` directory and move into it, preserving structure: `data/raw/results.csv` (the international results dataset), `data/format_2026.json` (the WC bracket), `data/fitted_params.json` (CV-tuned on international data — these params are WC-era artifacts and must not leak into MLS fits), and the WC-era `data/adjustments.json`. Add a short `archive/wc2026/ARCHIVE.md` recording what was archived, when, why, and how to restore it. Keep `wc2026/sim/tournament.py` and `data/format_2026.json`-dependent code paths intact in the codebase (they are part of the frozen engine's history and may serve future tournaments) — they just won't be exercised by the MLS workflow. Nothing gets deleted; this is an archive, not a purge. If the project is a git repo, make the archive move its own clean commit before any new work.

**Acceptance:** the archive is complete and documented; no active code path reads WC/international data files from their old locations; the test suite still passes (adjust test fixtures/paths as needed without weakening any test).

## Phase 2 — MLS data ingestion

**Source selection is yours, and it must be rigorous.** Evaluate candidate sources (e.g., FBref, ESPN's public endpoints, football-data.org, API-Football, jfjelstul's American Soccer datasets on GitHub, or others you find) against these criteria: verifiable accuracy, update latency (results must be available within ~a day of matches, and upcoming fixtures must be listed), historical depth (target 8+ seasons for fitting and walk-forward validation), licensing that permits this use, reliability/reproducibility of access, and schema completeness. Document the evaluation and your choice in a `sources.py`-style provenance entry — this project keeps every data source auditable in one place. Prefer a source (or combination) that yields both historical results **and** forward fixtures, mirroring how `results.csv` served double duty.

Build the ingestion so MLS data lands in the loader-contract schema noted above. League specifics to get right, since they differ from internationals and the config was tuned for a different world:

- **Home advantage is real and large in MLS.** Internationals defaulted to neutral venues; MLS matches are almost never neutral. `neutral=False` everywhere except genuinely neutral fixtures, and predictions must apply the fitted home term.
- **Tier/importance weighting:** regular-season MLS games are one tier; decide (and document) how to treat MLS Cup playoffs, US Open Cup, Leagues Cup, and CONCACAF Champions Cup appearances by MLS clubs — include them in fitting only if you validate that they help out-of-sample.
- **Re-tune everything tunable on MLS data:** half-life, rho, blend_k, Elo constants, importance weights — via the existing `tune`/walk-forward machinery, writing MLS-specific fitted params to a new file (never overwriting the archived WC params). The in-tournament goal-calibration mechanism (`tournament_calib_k`) was a WC-specific fix; disable it unless MLS validation independently justifies an analogous seasonal calibration.
- **MLS structural quirks:** franchise expansion (new teams with thin histories — the Elo-prior shrinkage exists for exactly this), name changes/rebrands over the years, conference structure, and the regular-season point that **MLS regular-season matches can end in draws but playoff matches cannot** — make sure fitting and prediction handle competition context correctly.
- **Roster/soft factors** stay exactly as designed: explicit, human-curated, dated entries in an adjustments file. Never fabricate automated injury feeds.

**Acceptance:** `python -m wc2026 update/fit/match/backtest/tune` (or a cleanly parallel league-aware CLI) run end-to-end on MLS data; a walk-forward MLS backtest report shows calibration and beats the base-rate baseline meaningfully on log-loss/Brier/RPS; new loader and ingestion tests pass alongside the entire existing suite.

## Phase 3 — Kalshi EV engine with dynamic fractional Kelly

Build a new module (e.g., `wc2026/betting/`) that connects model probabilities to Kalshi MLS markets. First step: **verify via the Kalshi API which MLS markets actually exist** (moneyline/1X2-style, totals, series/futures — whatever is listed), their ticker structure, and their liquidity. Do not assume; inspect real market data. Reuse the legacy client's auth pattern (RSA-PSS signing) but move all credentials to environment variables — **no keys or key paths in source, ever** (note: the legacy file currently hardcodes them; do not propagate that).

### EV computation — exact, fee-aware, execution-aware

- Model probability comes from the frozen engine's exact scoreline grid, mapped precisely onto each market's settlement rules (read the market rules text — e.g., does a market settle on regulation, or include extra time/shootouts for playoff games?).
- Cost basis is the **ask** you would actually pay (plus slippage awareness against order-book depth), not the midpoint. Include Kalshi's trading fees in EV per the current published fee schedule — fetch/verify it, don't recall it — and account for fee rounding per contract.
- Edge = model probability − fee-adjusted breakeven probability. Only positive-edge markets past a configurable minimum-edge threshold are candidates.

### Dynamic half-to-quarter Kelly

Stake sizing scales continuously between **half Kelly (maximum) and quarter Kelly (minimum)** as a function of confidence in the edge, never exceeding half Kelly. Design the confidence function explicitly and document it — reasonable ingredients: size of the edge relative to model calibration error on that probability bin (from the MLS backtest), data richness for the teams involved (n_eff), market liquidity, and time to match. Handle the real-world constraints exactly: integer contract counts, per-market position limits, bankroll as a tracked state, and **simultaneous correlated bets** (multiple markets on the same match are highly correlated — size them jointly, e.g., via simultaneous-Kelly on the joint outcome distribution from the scoreline grid, not independently).

### Two modes, hard safety rails

- **Recommend mode (default):** pulls live markets, computes EV and stakes, and emits a ranked, timestamped report — model prob, market price, fee-adjusted EV, confidence, Kelly fraction used, recommended contracts — plus a persistent log. Places nothing.
- **Execute mode:** places real orders, gated by strict, config-defined safety measures, all of which must be enforced in code and tested: an explicit `--live` flag with interactive confirmation (dry-run is the default even in execute mode), per-bet maximum stake (both absolute dollars and % of bankroll), daily and weekly loss limits that halt trading when breached, a maximum number of open positions, a minimum-liquidity requirement before any order, limit orders only (never cross a spread beyond a configured tolerance), full audit logging of every decision and order, and a kill-switch config flag that disables execution globally. Any API error or unexpected market state aborts the trade, never retries blindly into a position.
- **Verification loop:** track every recommendation and placed bet against closing lines and outcomes (CLV and realized P&L), so the edge claim is continuously falsifiable.

**Acceptance:** unit tests cover EV math (including fees and rounding), Kelly sizing (including the dynamic scaling boundaries and correlated-bet handling), and every safety rail (e.g., a test proving the loss-limit halt fires); recommend mode runs end-to-end against live Kalshi data; execute mode demonstrably refuses to place orders without the full gauntlet of confirmations.

## Phase 4 — The news-check gate (embedded Claude review of proposed bets)

The quantitative model is deliberately blind to soft factors — injuries, suspensions, lineup rotation, and the rest live in a human-curated adjustments file by design. For autonomous operation, that blind spot gets patched by a **news-check gate**: the final pipeline stage between "bets proposed" and "orders placed," in which an embedded Claude instance with web-search access reviews each proposed bet for real-world information the model cannot see. This matters more in MLS than most leagues: the regular season plays *through* FIFA international windows (teams lose starters to call-ups mid-season), squads rotate heavily during Leagues Cup / US Open Cup congestion, and travel distance, Denver altitude, summer heat, and late-season playoff-seeding rest decisions all move outcomes.

**Implementation:** invoke Claude headlessly from the pipeline — `claude -p` with `--output-format json` (or the Claude Agent SDK if a programmatic interface is cleaner) — passing the day's proposed bet slate (teams, market, model probability, market price, edge, proposed stake) and instructing it to research each match: confirmed and probable injuries, suspensions, international call-ups, announced or expected lineup rotation, managerial changes, travel/congestion context, and weather where material. The response must conform to a **strict JSON contract**, validated in code: per bet, a verdict of `approve` | `reduce` (with a multiplier in a bounded range, e.g. 0.25–1.0) | `veto`, plus a concise rationale and the sources consulted. Malformed output is treated as failure, not partially parsed.

Three rules are architectural law for this gate, and all three must be enforced in code and covered by tests:

1. **One-directional authority.** The gate can only shrink or kill a stake — never increase one, never add a bet, never alter the market, side, or price. A hallucinated finding therefore costs at most a missed bet, never a bad one. The stake multiplier is clamped in code regardless of what the response claims.
2. **Fail closed.** If the Claude invocation errors, times out (set an explicit timeout), or returns anything that fails schema validation, the affected bets do not get placed. No retry loop that ends in "place it anyway." Recommend mode may still display the bets, clearly flagged as unscreened.
3. **Audit the gate itself.** Log every verdict with its rationale, sources, and timestamp. Track the counterfactual P&L of vetoed and reduced bets (what would they have returned?) alongside realized P&L of approved bets, so that after a meaningful sample the gate's contribution to EV is measurable — and removable if it proves to be noise. The gate is subject to the same falsifiability standard as the model.

The gate runs in both modes: in recommend mode its verdicts annotate the report; in execute mode it is a mandatory pre-order stage. A config flag can disable it explicitly (logged loudly), but the default for any execute-mode run is gate-on. Where the gate surfaces durable information (e.g., a confirmed multi-week injury), prefer also emitting a suggested dated entry for the adjustments file for human review — the gate screens today's bets; the adjustments file is how confirmed knowledge feeds the model properly.

**Acceptance:** the JSON contract is schema-validated with tests for malformed, out-of-bounds, and timeout cases proving fail-closed behavior; clamping tests prove stakes can never increase through the gate; an end-to-end recommend-mode run shows annotated verdicts with sources; the audit log captures verdict, rationale, and counterfactual tracking hooks.

---

## Final deliverables and verification

Update `README.md` to reflect the project's new identity (archival note, MLS pipeline, betting engine usage, news-check gate, safety documentation). Ensure the complete test suite — old and new — passes. Then perform an explicit final self-review against this document, phase by phase, confirming each acceptance criterion with evidence (test output, backtest metrics, sample recommend-mode report). List anything deferred or uncertain honestly rather than papering over it. Precision over speed, always: a wrong number in this system costs real money.
