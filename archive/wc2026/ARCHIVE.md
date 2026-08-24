# Archive: the 2026 World Cup / international era

**Archived:** 2026-07-20
**Why:** the 2026 World Cup chapter of this project is over. The project is being
repurposed as an MLS expected-value betting engine on Kalshi. The statistical core
(`wc2026/model/`, `wc2026/sim/match.py`) is unchanged and frozen; only the data era
is retired. Nothing was deleted — this is an archive, not a purge.

## What is here

| File | What it is |
|---|---|
| `data/raw/results.csv` | The international results dataset (martj42/international_results, ~49.5k rows, 1872–2026-07). History used to fit the WC model **and** the live 2026 WC fixtures/results. |
| `data/format_2026.json` | The official 2026 World Cup knockout-bracket structure consumed by `wc2026/sim/tournament.py`. |
| `data/fitted_params.json` | CV-tuned parameters (half_life=365, rho=-0.10, blend_k=8.0; validation log-loss 0.8667 on 2023-06→2026-06). **Tuned on international data — must never be applied to MLS fits.** |
| `data/adjustments.json` | The WC-era human-curated soft-factor layer (empty `teams` at archive time). |

`bold_play_shortlist.py` (the WC-era Kalshi shortlist script, read-only market
scanner) was moved to `legacy/bold_play_shortlist.py` at the same time: it hardwired
WC series tickers and the archived dataset, and is superseded by `wc2026/betting/`.

## Code paths

`wc2026/sim/tournament.py` and the WC bracket logic remain intact in the codebase
(frozen-engine history; may serve future tournaments). `wc2026/config.py` points its
WC-specific paths (`WC_*`, `FORMAT_JSON`) into this archive, so the legacy
`tournament` workflow still works — but no *active* MLS code path reads anything here.

## How to restore

Copy the four files back to their original locations (`data/raw/results.csv`,
`data/format_2026.json`, `data/fitted_params.json`, `data/adjustments.json`) and
re-point the paths in `wc2026/config.py` back to `data/`. The frozen engine will
reproduce the WC-era behaviour exactly; `python -m wc2026 tournament` needs only
these files.
