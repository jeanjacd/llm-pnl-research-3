# World Cup 2026 — Kalshi Betting Model

A Python pipeline that encapsulates teams as strength ratings, simulates matches
100,000 times with a Dixon–Coles model, prices the resulting markets against Kalshi,
and finds the best **correlation-aware** parlays subject to a confidence floor and a
minimum payout multiplier.

> **Not financial advice. No guarantees.** This produces *model* probabilities and
> *suggested* stakes. It never places a trade — you do that yourself in Kalshi.

## Setup
```bash
pip install -r requirements.txt
python main.py --demo        # runs end-to-end with synthetic prices, no keys needed
```

To go live:
1. In `kalshi_client.py`, set `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`
   (download the RSA private key `.pem` when you create the API key in Kalshi).
2. In `main.py`, fill `KALSHI_TICKER_MAP` (inside `map_kalshi_prices`) to map each
   simulated market to the real Kalshi ticker you want to bet.
3. Refresh player data daily (see **Daily data refresh** below).
4. `python main.py`

## Daily data refresh (strongest free xG)
Player rates blend **current 2026 WC form** with a **recent club-season prior**, using
Opta-grade FBref data. FBref blocks scrapers but allows in-browser CSV export, so the
flow is manual-but-high-quality:
1. FBref → 2026 World Cup *Player Standard Stats* → "Share & Export → Get table as CSV"
   → paste into `data/incoming/wc_players.csv`.
2. FBref → a recent club season *Player Standard Stats* (e.g. Big-5) → CSV →
   `data/incoming/club_players.csv`.
3. `python build_database.py --expand-players` (use `--recency-weight 3` to lean harder
   on current form). See `data/incoming/README.md` and the `.example.csv` files.

The team-strength backbone is **current Elo** in `teams.json` (it shifts after every
match — update those numbers when you refresh). No FBref CSVs handy? `python
build_database.py --statsbomb-fallback` uses 2022 WC as a clearly-labeled stale prior.

## Files
| File | Role |
|------|------|
| `data/teams.json`   | 48 World Cup teams. Elo **exact** for the top 20 (eloratings/Wikipedia, 2026-06-22), **estimated** for the rest. `attack_rating`/`defense_rating` filled with **real StatsBomb xG** for ~26 teams after a build. |
| `data/players.json` | Attackers per team. After a build, rates carry a `source` (`fbref blend (WC+club)`, etc.); unmatched players keep seed estimates. |
| `data/incoming/`    | Drop your daily FBref CSV exports here (`wc_players.csv`, `club_players.csv`, optional `wc_teams.csv`). |
| `ratings.py`        | Loads the DB; converts Elo → expected goals (λ), preferring real xG ratings when present; builds scorer distributions. |
| `simulate.py`       | Dixon–Coles bivariate Poisson Monte Carlo (100k). Returns per-iteration boolean market outcomes + Wilson CIs. |
| `kalshi_client.py`  | RSA-signed Kalshi REST client. Market data + price→probability/odds. **API key fillers here.** |
| `optimizer.py`      | Edge filter, correlation-aware joint probability, EV ranking, Half-Kelly staking, exhaustive parlay search, **and near-miss reporting** when nothing clears both floors. |
| `main.py`           | Config knobs + the full pipeline. `--demo` for synthetic prices. Warns if your floors are mutually -EV. |
| `build_database.py` | Blends current-WC + club-season FBref CSVs into player rates (minutes-weighted shrinkage); optional shrunk team ratings from `wc_teams.csv`; `--statsbomb-fallback` for a stale 2022 prior. |

## The model in one paragraph
Each team is reduced to a single strength number (Elo, preferring real xG ratings if
present). For a matchup, Elo difference → expected goal supremacy → per-side expected
goals (λ). A Dixon–Coles adjusted bivariate Poisson is sampled 100,000 times; every
market (result, totals, BTTS, team totals, anytime scorer) is just a count over those
samples, reported with a Wilson confidence interval. Kalshi prices become implied
probabilities; legs with model edge ≥ threshold form the candidate pool. Parlays are
scored by **true joint probability** — legs in the same match are AND-ed at the
iteration level (correlation captured exactly), legs across matches multiplied — then
filtered by your confidence/multiplier floors and ranked by expected value, with a
Half-Kelly stake suggestion.

## Tuning
- `ratings.py`: `ELO_PER_GOAL`, `BASE_TOTAL_GOALS`, `HOME_FIELD_ADVANTAGE`.
- `simulate.py`: `DC_RHO` — Dixon–Coles low-score correlation.
- `build_database.py`: `DEFAULT_RECENCY_WEIGHT`, `PRIOR_PSEUDO_90S`, `LEAGUE_AVG_XG90`
  (player blend); `TEAM_PRIOR_MATCHES`, `TEAM_AVG_XG` (team-rating shrinkage).
- `main.py`: `MIN_CONFIDENCE`, `MIN_MULTIPLIER`, `MIN_LEG_EDGE`, `MAX_LEGS`, `RANK_BY`.

## Honest limitations
- Elo for ~28 lower-ranked teams is estimated; refresh before betting them.
- Until you run a build with FBref CSVs, player rates are approximate seed placeholders.
- Early in the tournament the WC sample is tiny, so the blend leans on the club-season
  prior by design — raise `--recency-weight` as more WC matches are played.
- Within-match scorer markets are sampled independently per player (correlated with the
  scoreline, mildly independent across two scorers on the same team).
