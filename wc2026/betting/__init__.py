"""
wc2026.betting
==============
The Kalshi EV-betting layer for MLS: connects the frozen engine's exact
scoreline distributions to live Kalshi markets.

Modules:
  config.py      every betting parameter and safety rail, documented
  fees.py        exact, integer-safe Kalshi fee math (verified vs the official
                 fee schedule PDF)
  kalshi.py      REST client -- RSA-PSS auth from environment variables only
  markets.py     MLS market discovery + settlement mapping onto the score grid
  ev.py          fee-aware, execution-aware expected value
  confidence.py  the documented confidence function -> dynamic Kelly multiplier
  kelly.py       simultaneous (joint-outcome) Kelly sizing with integer contracts
  bankroll.py    bankroll/position state, loss-limit accounting, audit log
  gate.py        the news-check gate (embedded Claude review; fail-closed)
  tracking.py    recommendation log, CLV and realized/counterfactual P&L
  pipeline.py    recommend / execute orchestration and the safety gauntlet
"""
