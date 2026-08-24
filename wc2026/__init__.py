"""
wc2026 -- an industry-standard Monte Carlo model for the 2026 FIFA World Cup.

A clean, modular, data-driven rebuild:
  data/   ingest + load a single verified public results dataset
  model/  transparent Elo prior + time-weighted Dixon-Coles bivariate-Poisson fit
  sim/    exact match scoreline distribution + Monte-Carlo tournament simulation
  eval/   walk-forward backtesting (log-loss / Brier / RPS / calibration) + tuning

Run `python -m wc2026 --help`.
"""
from __future__ import annotations

__version__ = "1.0.0"
