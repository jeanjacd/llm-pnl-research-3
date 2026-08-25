# Venue API reference (enumerated live, 2026-08-24)

Every fact here was read from the live APIs. Two earlier claims in this project
were wrong because the wrong query was used; both are corrected below and
pinned by tests so they cannot silently regress.

---

## Kalshi

### Discovery
`GET /series?category=Sports`, then filter by league prefix. Full observed sets:

| League | Prefix | Series |
|---|---|---|
| premier_league | `KXEPL` | 38 |
| la_liga | `KXLALIGA` | 36 |
| bundesliga | `KXBUNDESLIGA` | 30 |
| ligue_1 | `KXLIGUE1` | 25 |
| mls | `KXMLS` | 28 |

### Supported families (6) — map onto the regulation scoreline grid
`GAME` (1X2) · `TOTAL` · `SPREAD` · `BTTS` · `TEAMTOTAL` · `SCORE`

Settlement wording that must be present: *"90 minutes plus stoppage time (does
not include extra time or penalties)"*. A market whose rules do not confirm this
is abstained from.

### Unsupported families — recorded, never priced
`1H`,`1HBTTS`,`1HSPREAD`,`1HTOTAL`,`1HSCORE`,`2H`,`2HBTTS`,`2HSPREAD`,`2HTOTAL`
(halves) · `GOAL`,`ANYGOAL`,`FIRSTGOAL`,`SOA`,`AST` (player props) ·
`CORNERS`,`TCORNERS` · `MOV`,`FTTS`,`ADVANCE` · `LEADER`,`LAST`,`TOP`,`TOP2`,
`TOP4`,`TOP6`,`RELEGATION`,`POINTMARGIN`,`TEAMPOINTS`,`SEASONSTAT`,`POY`,
`H2H`,`H2HFINISH`

### Two ticker traps
* `KXLALIGA2*`, `KXBUNDESLIGA2*` are the **second division** (LaLiga 2,
  2. Bundesliga) — `2GAME`, `2BTTS`, `2SPREAD`, `2TOTAL`, `2ADVANCE`, `2PROMO`.
* `KXMLSAST*` is the **All-Star game** (`ASTGAME`, `ASTSPREAD`, …).

Both would be swept in by a prefix scan. Series tickers are therefore built as
exact `prefix + suffix` strings, asserted by a test.

### Fixture identity — parsed from the rules text, not the ticker
The event ticker (`KXEPLGAME-26SEP06ARSCFC`) encodes both clubs as **variable
length** codes: `KXMLSGAME-26SEP05RSLLAFC` is `RSL` + `LAFC`, not `RSL` + `LAF`.
It cannot be split reliably. `rules_primary`/`rules_secondary` name both clubs
in full plus the scheduled date, and are parsed instead — 1,113/1,113 open
soccer markets parse.

Two wordings exist and both must be handled:

| family | wording |
|---|---|
| most | `…the <HOME> vs <AWAY> professional <LABEL> soccer game originally scheduled for <Mon D, YYYY>` |
| BTTS | `…the <HOME> vs <AWAY> <LABEL> match originally scheduled for …` — **no "soccer"** |

The sentence usually opens *"If Tie is the result of the …"*, so a regex
anchored on the **first** `the` captures `result of the Arsenal`. It must bind
to the last one. `<LABEL>` is exactly one of `EPL`, `MLS`, `Bundesliga`,
`La Liga`, `Ligue 1`; leaving it unconstrained lets it swallow the tail of the
away team's name (`Los Angeles FC` → away `Los Angeles`, label `FC … MLS`).

**Ordering: the rules list the HOME side first.** Verified two independent ways
on 2026-08-24 — against Kalshi's own `custom_strike` `home_team_id`/
`away_team_id` on SCORE markets (4/4), and against our ESPN fixture table
across all five leagues (255/255, no counterexamples). `resolve_fixture` still
reports `flipped` and callers honour it, so a convention change degrades to a
corrected claim rather than a wrong one.

Claims are derived from `yes_sub_title` (plus `floor_strike`), which names the
team in full, so ordering never affects claim correctness:

| family | `yes_sub_title` | claim |
|---|---|---|
| GAME | `Tie` / `Chelsea` | `draw` / `away_win` |
| TOTAL | `Over 2.5 goals scored` | `total_over_2.5` |
| SPREAD | `Fulham wins by more than 1.5 goals` | `home_wins_by_over_1.5` |
| BTTS | `Both Teams To Score` | `btts` |
| TEAMTOTAL | `Chelsea over 1.5 goals` | `away_over_1.5` |
| SCORE | `Chelsea wins 3-1` (winner's goals first) | `score_1-3` |

Result: **1,086 priceable instruments of 2,156 discovered**, zero unresolved
fixtures. Before this, the Kalshi path returned 223 "supported" instruments and
produced **zero** cases — legs carried no home/away, so no claim ever resolved.

### `expected_expiration_time` is NOT the kick-off
Measured across 63 open EPL markets: `expected_expiration_time` equals
`occurrence_datetime` exactly, and both sit **3 hours after** kick-off — they
are when the market settles. Reading either as the kick-off presented
Fulham vs Chelsea (19:00Z kick-off) as *"kickoff in 1.86 hours"* at 20:11Z,
i.e. offered a match already 72 minutes old as a pre-match trade. The board's
soccer analyst caught it by reporting first-half shot counts.

Kick-off comes from **our own fixture table** via the resolved fixture. A leg
with no known kick-off now DEFERs rather than skipping the freshness check,
because a pre-match model has no in-play validity.

### Parlays — multivariate event collections
Open collections: `KXMVESPORTSMULTIGAMEEXTENDED-R`, `KXMVECROSSCATEGORY-R`,
`KXMVECROSSCATEGORY-SHARD1-R`.

* `size_min=2`, `size_max=0` (unbounded), `is_all_yes=False`,
  `is_single_market_per_event=False`.
* Functional description: *"the resulting market will only resolve to YES if
  every associated market resolves to YES; scalar outcomes are multiplied"*.
* `GET /multivariate_event_collections/{ticker}` returns
  `associated_event_tickers` — the **eligible legs**. Of 1,462 eligible legs,
  **217 belong to our five leagues** across 58 fixtures:

  | League | fixtures | supported legs | unsupported legs |
  |---|---|---|---|
  | premier_league | 10 | GAME/SPREAD/TOTAL/BTTS ×10 | 1H*, CORNERS, TCORNERS |
  | la_liga | 14 | ×14 | 1H*, CORNERS, TCORNERS |
  | bundesliga | 9 | ×9 | — |
  | ligue_1 | 9 | ×9 | — |
  | mls | 16 | GAME ×16, others ×1 | 1H* |

* Because `is_single_market_per_event=False`, **two legs from the same fixture
  can be combined** — strongly dependent, so a joint probability must come from
  the shared scoreline grid, never from multiplying marginals.

**Verified limitation.** Pre-created combo markets (200 open) expose only
repeated leg *descriptions* (`"no Over 5.5 goals scored"` ×36) with **no fixture
identifiers** — absent from the market payload, from the event payload even with
`with_nested_markets=true`, and from `rules_primary` (empty). Their order books
are empty. Pricing a chosen combination requires the lookup/RFQ path, which
creates or confirms a quote and is forbidden in paper mode.

→ Combos are **discovered and valuable only when we construct them ourselves**
from eligible legs. A pre-created combo is recorded as UNSUPPORTED, and no combo
ever receives a claimed executable price.

### Orders
`POST /portfolio/orders` is **dead** (410 `deprecated_v1_order_endpoint`).
Current: `POST /portfolio/events/orders`, YES-leg single-book model —
`side:"bid"` = buy YES, `side:"ask"` = sell YES (= buy NO at `1 − price`).

---

## Polymarket

### Discovery — use the venue's own league registry
`GET /sports` returns the league registry with tag IDs. **These must be used.**

| League | tag id(s) |
|---|---|
| premier_league | 82, 306 |
| la_liga | 780 |
| bundesliga | 1494 |
| ligue_1 | 102070 |
| mls | 100100 |

Then `GET /events?tag_id=<id>&closed=false` (offset pagination, 100/page).

> **Corrected error.** An earlier sweep used `tag_slug=soccer` plus a naive
> team-name substring match and concluded these leagues had ~zero liquidity.
> Both were wrong: the generic soccer tag returns a different, largely obscure
> population, and ESPN display names ("Manchester City") do not substring-match
> venue names ("Man City").

### Measured liquidity (upcoming fixtures, `acceptingOrders`, real CLOB books)

| League | upcoming events | markets | median liq | ask depth (soonest) | spread |
|---|---|---|---|---|---|
| premier_league | 95 | 1,074 | $1,329 | $93,605 | 1c |
| la_liga | 96 | 1,124 | $950 | $23,418 | 1c |
| mls | 82 | 516 | $385 | $173,688 | 6c |
| bundesliga | 93 | 975 | $76 | $7,791 | 4c |
| ligue_1 | 89 | 1,035 | $39 | $5,762 | 4c |

Liquidity concentrates near kickoff, so depth is read per fixture from the CLOB
book and never inferred from an aggregate field.

### Category containers (one event per category, per fixture)

| Container | markets/fixture | Supported |
|---|---|---|
| `<A> vs. <B>` | 3 | ✅ 1X2 |
| `… - Exact Score` | ~17 | ✅ exact scores (`Any Other Score` ❌) |
| `… - More Markets` | 33 | ✅ totals, team totals, spreads, BTTS (half variants ❌) |
| `… - Halftime Result` | — | ❌ |
| `… - Second Half Result` | — | ❌ |
| `… - First Team to Score` | — | ❌ |
| `… - Total Corners` | — | ❌ |
| `… - Player Props` | — | ❌ |

### Question wordings and outcome semantics
Outcome labels differ by category and **`outcomes[0]` is not always "Yes"** —
the first CLOB token corresponds to `outcomes[0]`, so every claim is written
from that outcome's point of view.

| Wording | outcomes | claim |
|---|---|---|
| `Will <T> win on <date>?` | Yes/No | `home_win` / `away_win` |
| `Will <A> vs. <B> end in a draw?` | Yes/No | `draw` |
| `Exact Score: <A> h - a <B>?` | Yes/No | `score_h-a` (re-oriented) |
| `Spread: <T> (-1.5)` | **[T, other]** | `<side>_wins_by_over_1.5` |
| `<A> vs. <B>: O/U 2.5` | **[Over, Under]** | `total_over_2.5` |
| `<A> vs. <B>: <T> O/U 1.5` | [Over, Under] | `<side>_over_1.5` |
| `<A> vs. <B>: Both Teams to Score` | Yes/No | `btts` |

> **Corrected error.** Full-match totals are spelled bare (`O/U 2.5`), **not**
> "Total Goals". Searching for the word "goals" returns nothing and wrongly
> implies the venue has no totals. Corner totals share the O/U wording, so
> `corner`, `1st/2nd half`, `player`, `any other score` and `first team to
> score` are excluded explicitly.

### Field traps
* `startDate` = market **creation** time. Kickoff is `gameStartTime`.
* `liquidityNum` / `liquidity` / `liquidityClob` are aggregates, **not**
  executable depth. Cost basis comes from `GET /book?token_id=…` or not at all.
* `orderMinSize` is **5**, `orderPriceMinTickSize` is 0.01.
* `acceptingOrders` is the string `"True"`/`"False"`.

### Resulting coverage (supported / discovered)

| League | events | instruments | supported | % |
|---|---|---|---|---|
| mls | 163 | 1,593 | 770 | 48.3% |
| ligue_1 | 71 | 917 | 396 | 43.2% |
| bundesliga | 88 | 1,096 | 468 | 42.7% |
| la_liga | 137 | 1,695 | 720 | 42.5% |
| premier_league | 134 | 1,625 | 684 | 42.1% |

The unsupported remainder is fully accounted for and legitimately out of model
scope — EPL example: corners 437, first half 209, second half 209,
any-other-score 19, player props 10, first-to-score 7. **No unclassified
residue**, i.e. no silent parser gaps.


---

## Fees — both venues use the same formula shape

`fee = C × rate × p × (1 − p)`, symmetric in price, largest at 50c.

| venue | rate | charged to |
|---|---|---|
| Kalshi | 0.07 | taker; verified against the published fee table |
| Polymarket | see below | taker only — *"Makers are never charged fees"* |

**Polymarket's reported fee contradicts its published schedule.** All three
surfaces — Gamma `takerBaseFee`, the CLOB market record `taker_base_fee`, and
the authoritative `GET /fee-rate?token_id=…` — return **1000** for every
soccer market sampled (88/88 markets, 73/73 tokens, all five leagues,
2026-08-24). The published sports rate is **0.05**, with a stated ceiling of
$1.25 per 100 shares at 50c. Read as basis points, 1000 is 0.10 — exactly
twice that. Polymarket's own client carries an open, unanswered issue about
this contradiction ([py-clob-client#326](https://github.com/Polymarket/py-clob-client/issues/326)),
which also documents a second formula in `CalculatorHelper.sol` that disagrees
with the docs at any price other than 0.50.

We charge the **reported** field (0.10), not the published rate: over-charging
costs a missed trade, under-charging books a bad one. Both numbers are recorded
on the instrument's `fee_model` so a report can show either.

Two earlier readings of this field were wrong and both were caught in
production, not by tests:
* as a **fraction** — 10,000× too big, produced `breakeven_prob = 2.56` on a
  23c contract. Caught by the board's quant member on a live run.
* as a flat rate on **notional** — 5.6× too big at 23c ($10.00 vs $1.77 per
  100 shares). It also ignored that the fee is symmetric in price.

`p(1−p)` is formed from **integer cents**. In floating point
`0.7 × (1−0.7) ≠ 0.3 × (1−0.3)`, which charged different fees for the two sides
of the same contract.

## Completeness audit (2026-08-24)

Run because two earlier coverage claims were wrong. Method: exhaustive
enumeration, then classify EVERY distinct item and show the parser's verdict,
so a gap is visible rather than assumed.

### Polymarket — 55 distinct question shapes, all accounted for
`GET /series` for events has no residual pages (paging stops on a short page;
the cap was removed after it silently truncated MLS at 241 open events).

**Mapped (10 shapes, 4,399 markets):** exact score 1,952 · totals 734 ·
home team total 363 · away team total 363 · home spread 246 · away spread 246 ·
home win 124 · draw 124 · away win 124 · BTTS 123.

**Unmapped (45 shapes, 6,088 markets) — every one out of model scope:**
corners (889+504+504+378+378+126+126) · halves (363+363+242x4+122x3+121x5) ·
any-other-score 122 · first-to-score 122x3 · player anytime-goalscorer (many,
1-2 each). **No unclassified residue.**

### Kalshi — `/series?category=Sports` returns 3,493 with NO cursor
That is the complete set; `Soccer`/`Football` are not valid categories. Open
markets per league, retried on failure:

| League | supported open | unsupported open |
|---|---|---|
| la_liga | 327 (GAME 81, TOTAL 84, SPREAD 56, BTTS 14, TEAMTOTAL 18, SCORE 74) | 296 |
| mls | 255 (GAME 90, TOTAL 90, SPREAD 60, BTTS 15) | 90 |
| premier_league | 209 (GAME 63, TOTAL 60, SPREAD 40, BTTS 10, TEAMTOTAL 6, SCORE 30) | 384 |
| bundesliga | 153 (GAME 54, TOTAL 54, SPREAD 36, BTTS 9) | 132 |
| ligue_1 | 153 (GAME 54, TOTAL 54, SPREAD 36, BTTS 9) | 132 |
| **total** | **1,097** | **1,034** |

Bundesliga/Ligue 1/MLS currently list no TEAMTOTAL or SCORE markets; those open
nearer to fixtures. This is a genuine absence, confirmed by retry, not a
swallowed error.

### A silent-failure bug this audit exposed
The first run of this audit reported **la_liga GAME = 0**. The series actually
returns 81. The audit (and the provider) used a bare `except: continue`, so a
transient error was indistinguishable from "no markets". `discover()` is now
`strict=True` by default and RAISES `DiscoveryError`; tolerating partial data
requires opting out and inspecting `last_errors`. Pinned by tests.

### Residual uncertainty — stated, not hidden
* This is a **snapshot**. If a venue adds a family we *could* model (e.g. a
  double-chance or draw-no-bet market), it is recorded as unsupported and
  abstained from until a parser is written. That is fail-closed, and it means
  coverage can silently lag a venue's expansion. Re-run this audit periodically.
* Kalshi cup-context families (`ADVANCE`, and MLS `CUP`/`EAST`/`WEST`) are
  classified unsupported without a settlement study; they may be modellable
  later.
* Polymarket does not expose a settlement-basis flag, so
  `settles_on_regulation` is None there, which blocks cross-venue equivalence
  by design until the rules text is parsed.


---

## Why corners, halves and goalscorers are abstained from (decision, 2026-08-24)

Not a data gap. Feasibility was checked and the data exists:

| Family | Data | Source |
|---|---|---|
| Corners | available | `wonCorners` per team, boxscore |
| Halves | available | `keyEvents` goals carry `period` (1/2) + clock -> HT score |
| Goalscorers | available | per-player `totalGoals`, `totalShots`, `appearances`, `starter` |

The blocker is the MODEL. Dixon-Coles produces one joint distribution over
full-match regulation goals; every supported family is a different sum over
that single grid. Corners are not goals, halves need within-match timing the
model does not represent, and goalscorers need a player/minutes layer that does
not exist. Each would require its own model, its own walk-forward validation
and its own calibration before anything could be priced -- which is exactly
what the mission requires.

Cost of abstaining, measured: corners ~2,905 Polymarket markets (28%), halves
~2,700 (26%), goalscorers ~50 (<1%). So roughly 54% of the discovered universe
is deliberately out of scope.

**Decision: keep abstaining** and finish the decision/board/broker phases first.
If revisited, the order should be halves (cheapest, reuses the DC machinery),
then corners (separate count model), and goalscorers last or never on these
venues -- it depends on expected minutes, the exact blind spot the news gate
exists to cover, and is a rounding error in market count.

Backfill cost if built: corners and halves both need the per-match `summary`
endpoint, one request per match, 26,411 matches across the five leagues.
