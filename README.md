# wc2026 — multi-league soccer market research & paper trading

A board-governed research system for five soccer leagues. It models each league
independently, discovers Kalshi and Polymarket contracts, computes fair value and
conditional limit prices **in code**, routes candidates through a sealed
three-member Claude board, and **paper trades** the approved decisions.

> **PAPER TRADING ONLY.** Live trading is zero and unsupported in this phase.
> No workflow, script or module in this repository can place a real order: the
> venue providers are read-only and the broker writes to its own ledger.

**362 tests passing.** Not a git repository yet — the workflows in
`.github/workflows/` are ready for when it is pushed.

## Leagues

| League | Provider | Matches | Tuned | Holdout log-loss vs base |
|---|---|---|---|---|
| premier_league | `eng.1` | 4,940 | ✅ | 1.0009 vs 1.0769 (7.1%) |
| la_liga | `esp.1` | 4,940 | ✅ | 0.9748 vs 1.0629 (8.3%) |
| bundesliga | `ger.1` | 3,978 | ✅ | 0.9960 vs 1.0851 (8.2%) |
| ligue_1 | `fra.1` | 4,541 | ✅ | 0.9844 vs 1.0618 (7.3%) |
| mls | `usa.1` | 8,012 | ✅ | 1.0515 vs 1.0705 (1.8%) |

Each league owns its data, parameters, calibration and evaluation report. There
is **no** "active league" global — every command takes `--league`.

## Quick start

```bash
pip install -e ".[dev]"
python -m wc2026 leagues                       # registry + status
python -m wc2026 update   --all                # ingest results + fixtures
python -m wc2026 evaluate --all                # nested DEV/CAL/HOLDOUT
python -m wc2026 match --league premier_league "Arsenal" "Chelsea"
python -m wc2026 paper-cycle --no-board        # one deterministic paper cycle
```

## How a decision is made

1. **Ingest** (`data/espn.py`) — league-aware, checksummed, point-in-time
   (`known_after_utc` *and* `retrieved_at_utc`), fails loud on partial data.
2. **Model** (`model/`, `sim/match.py`) — frozen Dixon–Coles bivariate Poisson
   producing one joint scoreline grid per fixture.
3. **Evaluate** (`eval/nested.py`) — hyperparameters on DEV only, calibrator on
   CAL only, holdout opened **exactly once** (a second attempt raises).
4. **Discover** (`venues/`) — Kalshi + Polymarket, read-only, normalised to
   `MarketInstrument`, hashed so unchanged markets aren't re-reviewed.
5. **Decide** (`decision/`) — deterministic fair value, fees, book walking, an
   adverse-selection reserve, and a **limit ladder** of every viable tick.
6. **Govern** (`board/`) — sealed quant + coach, then judge.
7. **Paper trade** (`paper/`) — cash reservation, conservative fills,
   settlement, P&L.

### The deterministic layer owns every number

Probabilities, fees, EV, limit prices and sizes are computed in code *before*
any model is consulted. The board may **audit and shrink** them; it can never
replace them with its own arithmetic, raise a price, or increase a size.

Action states: `BUY_NOW` · `PLACE_LIMIT` · `WAIT_FOR_QUOTE` · `PASS` · `DEFER` ·
`UNSUPPORTED`. An unattractive quote becomes a conditional limit rather than
being discarded — unless the required price is so far below the bid that it is
not realistically obtainable, which is a `PASS`.

### The board

| Member | Sees | Cannot |
|---|---|---|
| Quantitative | full numeric case | raise the computed ceiling |
| Soccer analyst | **price-redacted** packet + web research | assign any probability or multiplier |
| Judge | packet + both sealed proposals | research, or increase price/size |

Quant and coach run independently in fresh sessions; the judge runs **only**
after both validate. Every failure — invalid JSON, unknown/duplicate case id,
timeout, schema violation — resolves to `DEFER`. Retrieved web content is
treated as untrusted data, never as instructions.

### Paper fills are deliberately pessimistic

A resting order does **not** fill because a price was touched — queue position
is unknowable, so it requires the market to trade *through* the limit with real
size. Marketable orders fill only against captured depth. P&L uses actual fills,
never proposed size.

## What is deliberately not supported

Only the six families derivable from the regulation scoreline grid are priced:
1X2, totals, BTTS, team totals, spreads, exact scores.

Corners, first/second halves, player props, method-of-victory, first-team-to-score
and season futures are **discovered, recorded and abstained from**. The data for
some of them exists; the *model* does not, and per the operating rules nothing is
priced without its own validated model. That is ~54% of discovered markets —
see [docs/VENUES.md](docs/VENUES.md) for audited counts and the decision record.

## Honest limitations

- **No market-alpha claim is supported.** Every evaluation number is model vs
  base rate. There are no stored point-in-time market prices yet.
- **MLS is the weakest league** (1.8% improvement) and **over-confident**
  (calibration slope 0.792). Its interval barely clears its baseline.
- Untuned leagues are refused for trading, by code, not convention.
- Promoted clubs with no rating are **skipped and counted**, never silently
  scored as league-average.
- Cross-venue equivalence requires a *known and identical* settlement basis;
  Polymarket does not expose one, so it currently blocks by design.
- Venue coverage is a snapshot — a newly added modellable family would be
  abstained from until a parser exists.

## Layout

```
wc2026/
  leagues.py          the league registry (all per-league paths)
  data/     espn.py · loader.py           point-in-time ingestion
  model/ · sim/                           FROZEN statistical core
  eval/     nested.py · backtest.py · tune.py
  venues/   base.py · kalshi_provider.py · polymarket.py   (read-only)
  decision/ calculator.py · joint.py      deterministic EV + limit ladder
  board/    schemas.py · orchestrator.py  sealed three-member board
  paper/    broker.py · cycle.py          paper trading + scheduled cycle
data/leagues/<id>/    raw · fitted_params · calibration · evaluation
docs/                 BASELINE · EVALUATION · VENUES · STATUS · MIGRATION
.github/workflows/    ci · matchday-board · paper-maintenance · daily-data ·
                      weekly-evaluation
```

## Privacy

`data/betting/` (real bankroll and trade history from the earlier single-league
phase) is git-ignored and **must never be published**;
`scripts/check_no_private_state.py` fails CI if anything private becomes
tracked. See [docs/MIGRATION.md](docs/MIGRATION.md).

## Tests

```bash
pytest -q          # 362 tests
ruff check wc2026 tests
python scripts/check_no_private_state.py
```
