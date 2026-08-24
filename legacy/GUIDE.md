# Operating Guide — World Cup Kalshi Model

Everything you need to run, maintain, and sharpen this system yourself. Read once
end-to-end, then use it as a reference.

> **Reality check.** This finds *model-vs-market value*; it does not guarantee profit.
> A liquid market is hard to beat. Your edge, if any, lives in (a) thin prop markets
> (correct score, team totals, lower-profile games), (b) correlated parlays, and
> (c) acting on lineup/injury news before prices move. Treat the backtest, not the
> live edges, as your source of truth about whether the model works.

---

## 0. TL;DR daily routine (game day)
```
1. Update Elo (if a match day passed):  edit data/incoming/elo.csv -> python build_database.py
2. Update players (optional, weekly):   FBref CSVs -> python build_database.py --expand-players
3. ~1-2 hrs before kickoff:             check lineups/injuries (Section 4)
4. Run it:                              python main.py     (set FIXTURES first)
5. Verify connection any time:          python test_kalshi.py
```

---

## 1. The data you import (what moves accuracy, in order)

| Priority | Data | File | Where to get it |
|---|---|---|---|
| 1 | Results history (for backtesting/tuning) | `data/incoming/results.csv` | Kaggle "International football results" dataset; or `python backtest.py --from-statsbomb` |
| 2 | Player xG (scorer markets) | `data/incoming/wc_players.csv` + `club_players.csv` | FBref → table → "Share & Export → Get table as CSV" |
| 3 | Team Elo | `data/incoming/elo.csv` | eloratings.net / Wikipedia "World Football Elo Ratings" / Kaggle Elo CSV |
| 4 | Historical odds (to tune MODEL_TRUST) | add `mkt_home,mkt_draw,mkt_away` to `results.csv` | football-data.co.uk (club leagues), oddsportal, any odds archive |
| 5 | Team xG (optional) | `data/incoming/wc_teams.csv` | FBref → WC Squad Standard Stats |

All importers match team names flexibly (United States→USA, Korea Republic→South Korea,
etc.). After dropping files, run `python build_database.py`. Anything you don't provide
keeps its current value. Add `--dry-run` to preview without writing.

### elo.csv format
```
Team,Elo
Argentina,2144
Qatar,1437
```
### FBref player CSV
Export the **Standard Stats** player table. The parser needs `Player`, `xG`, and
`Min` or `90s` (plus `Squad`, `Gls` if present). `wc_players.csv` = current 2026 WC;
`club_players.csv` = a recent club season (the large-sample prior). Blend with
`--recency-weight N` (higher = lean on current WC form).

### results.csv format (for backtesting)
```
date,home,away,home_score,away_score[,mkt_home,mkt_draw,mkt_away]
```
`mkt_*` may be implied probabilities OR decimal odds (auto-detected). More matches =
better; you want **thousands** (a full international history), not one tournament.

---

## 2. Sharpen the model with the backtest (do this first, then periodically)

```
python backtest.py --from-statsbomb      # quick real-data demo (WC 2018+2022, ~128 games)
python backtest.py                        # uses whatever data/incoming/results.csv you have
```
It prints four things:

1. **Brier + log-loss vs a base-rate baseline.** If the model doesn't beat the
   baseline, it has no skill yet — fix data/constants before betting.
2. **Calibration table.** Each row: predicted vs observed. Rows flagged `<-- off`
   mean that probability band is miscalibrated. Want pred ≈ obs across the board.
3. **Grid search.** The best `ELO_PER_GOAL` and `BASE_TOTAL_GOALS` — **paste these
   into `ratings.py`** (top of file). (`K` is backtest-only.)
4. **Model-trust sweep** (only if `results.csv` has odds). The `MODEL_TRUST` value
   with the lowest log-loss — **set this in `main.py`**.

> Use a results history of thousands of matches for trustworthy numbers. The
> `--from-statsbomb` set is too small (Elo barely converges) — it only proves the
> tool works.

---

## 3. The two key tuning knobs you set from the backtest

| Knob | File | Meaning |
|---|---|---|
| `ELO_PER_GOAL`, `BASE_TOTAL_GOALS` | `ratings.py` | how Elo gaps map to expected goals — from grid search |
| `MODEL_TRUST` | `main.py` | 1.0 = pure model, 0 = pure market. Lower = more conservative. From blend sweep |

If you have no historical odds yet, leave `MODEL_TRUST = 0.5` (a sane default that
hedges your model against the market). Raise it only after the backtest shows the
model genuinely beats the market.

---

## 4. Pre-bet checklist (the human edge I can't automate)

Do this ~1–2 hours before kickoff:
- [ ] **Confirmed lineup.** Is the key striker/playmaker actually starting? A benched
      star swings a match more than any model tweak. If a rated player is out, lower
      their `expected_minutes` (or remove them) in `data/players.json` and re-run.
- [ ] **Injuries/suspensions/rotation** (esp. dead-rubber group games — teams rest players).
- [ ] **Books are live.** `python test_kalshi.py` — WC markets only get quotes near
      kickoff. Empty books = no bet.
- [ ] **Weather/red-card context** for in-play, if relevant.

---

## 5. Run it live & read the output

Set the matchups in `main.py` (`FIXTURES`), then `python main.py`. For each game it
prints model headline numbers, the Kalshi mapping status, then ranked parlays:
```
3-leg parlay | hit 41.2% [40.8,41.6] | 4.10x | EV +18.9% | half-Kelly 9.1% of bankroll
    - [Brazil v X] Over 2.5 goals (YES @ 47c, odds 2.13, model 55% vs market 47%, edge +3.8%)
```
- **hit** = correlation-aware joint probability (blended). **multiplier** = combined payout.
- **EV** = expected value per unit. Only bet **EV > 0**.
- **half-Kelly** = suggested stake as % of bankroll (already halved for safety).
- If nothing clears your floors, it shows **closest +EV near-misses** and which floor failed.

Your floors live in `main.py`: `MIN_CONFIDENCE`, `MIN_MULTIPLIER`, `MIN_LEG_EDGE`,
`MAX_LEGS`, `RANK_BY`. **Keep `MIN_MULTIPLIER ≥ 1 / MIN_CONFIDENCE`** or the system
warns you the floors are mutually money-losing.

**Cross-game parlays** are automatic — list multiple games in `FIXTURES` and the
optimizer mixes legs across them (independent across games, correlation-aware within
each game). Two knobs control diversification:
- `MIN_DISTINCT_GAMES` (default 1) — set to `2`+ to *require* a parlay span multiple
  games. Forcing cross-game lowers correlation risk, since different games are
  statistically independent.
- `MAX_LEGS_PER_GAME` (default 3) — caps how many legs may come from one match, so a
  single game can't dominate a combo.

---

## 6. Staking discipline (this is where bankrolls die)
- The tool suggests **half-Kelly**; consider **quarter-Kelly** for parlays (variance
  compounds across legs).
- Set `BANKROLL` in `main.py` to your real, dedicated number — not your net worth.
- Never raise stakes to chase losses. Parlays are high-variance; expect long droughts.
- Place every bet **yourself** in Kalshi. The code never trades, by design.

---

## 7. What "good" looks like
- Backtest: model log-loss clearly below baseline, calibration tight, and (with odds)
  best `MODEL_TRUST` > 0.5 — i.e., the model adds info beyond the market.
- Live: small, frequent positive-EV edges (1–6%), mostly in props/parlays, not on
  efficient match-winner lines. Giant edges (>15%) usually mean stale prices or a model
  error — double-check before trusting them.

---

## 8. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `0 candidate legs` | Books empty (pre-kickoff) — run near game time; or edges below `MIN_LEG_EDGE`. |
| `[WARN] fill_synthetic=True` | You're in a test path using FAKE prices — never bet on these. `main.py` never does this. |
| `'unicodeescape'` error on a path | Use a raw string `r"C:\..."` or forward slashes. |
| Player shows 0 xG after build | No attacking sample in source; seed value is kept intentionally. |
| Estimated Elo reappears | You imported a partial `elo.csv`; teams not in it keep prior values. |

---

## 9. File map
`ratings.py` (Elo→goals, tuned constants) · `simulate.py` (Dixon–Coles 100k MC, all
markets) · `kalshi_client.py` (orderbook pricing, fixture→market auto-map) ·
`optimizer.py` (edge, correlation-aware joint, model↔market blend, Kelly, parlay search,
near-miss) · `main.py` (config + pipeline) · `build_database.py` (Elo/player/team
imports) · `backtest.py` (calibration + tuning) · `test_kalshi.py` (connection check) ·
`test_live_ready.py` (pre-liquidity smoke test).
