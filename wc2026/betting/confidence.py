"""
confidence.py
=============
The documented confidence function behind the dynamic fractional Kelly.

Design (explicit, per the project brief): stakes scale CONTINUOUSLY between
quarter Kelly (floor) and half Kelly (ceiling) as a function of confidence in
the edge. Confidence is a product of four independent discounts in [0, 1]:

  w_cal   calibration trust: from the walk-forward pooled trust curve for THAT
          league and market family (data/leagues/<id>/calibration/<family>.json,
          fitted on the calibration window only). In the bin containing the
          model's p, the measured |mean_pred - observed| gap is scaled by
          CAL_GAP_SCALE (0.15): a bin that has historically been off by >= 15
          points contributes zero confidence. Bins with fewer than MIN_BIN_N
          samples get w_cal = 0.5 (unproven, not distrusted).
  w_data  data richness: min(n_eff of the two teams) / (n_eff + N_EFF_K).
          Expansion teams and thin histories shrink stakes exactly where the
          Elo prior is doing the heavy lifting.
  w_liq   liquidity: depth at the executable touch relative to the contracts
          we want; a book that can absorb the full order contributes 1.
  w_time  proximity: within NEWS_SAFE_HOURS of kickoff -> 1 (lineups and news
          are largely known); further out decays toward TIME_FLOOR because
          unpriced team news is the dominant unmodelled risk.

  confidence = w_cal * w_data * w_liq * w_time
  kelly_multiplier = 0.25 + (0.50 - 0.25) * confidence      in [0.25, 0.50]

The multiplier applies to FULL-Kelly joint stakes (kelly.py); it can never
exceed half Kelly by construction, and any candidate that passed the edge
gates never sizes below quarter Kelly (of its joint full-Kelly stake).

The same trust curve supplies `calibrated_prob`, a conservative empirical
correction of the model probability used as a second EV gate (ev.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

CAL_GAP_SCALE = 0.15
MIN_BIN_N = 50
N_EFF_K = 15.0
NEWS_SAFE_HOURS = 48.0
TIME_FLOOR = 0.5


@dataclass
class TrustCurve:
    bins: list[dict]                 # {bin_lo, bin_hi, mean_pred, observed, n}

    @staticmethod
    def path_for(spec, family: str = "1x2") -> str:
        """Calibration file for one league AND one market family.

        A 1X2 trust curve says nothing about totals or exact scores, so
        each family is stored separately and must be validated before it
        is used (mission Phase 2).
        """
        import os
        return os.path.join(spec.calibration_dir, "%s.json" % family)

    @classmethod
    def load(cls, path: str) -> "TrustCurve":
        if not os.path.exists(path):
            raise FileNotFoundError(
                "No calibration at %s. Run `python -m wc2026 evaluate "
                "--league <id>` first -- the betting layer refuses to size "
                "stakes without a validated trust curve for that league "
                "and market family." % path)
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls(bins=payload["bins"])

    def _bin_for(self, p: float) -> dict | None:
        for b in self.bins:
            if b["bin_lo"] <= p < b["bin_hi"] or (b["bin_hi"] >= 1.0 and p >= b["bin_lo"]):
                return b
        return None

    def w_cal(self, p: float) -> float:
        b = self._bin_for(p)
        if b is None or b["n"] < MIN_BIN_N:
            return 0.5
        gap = abs(b["mean_pred"] - b["observed"])
        return float(np.clip(1.0 - gap / CAL_GAP_SCALE, 0.0, 1.0))

    def calibrated_prob(self, p: float) -> float:
        """Empirically corrected probability: p shifted by its bin's measured
        (observed - predicted) gap, clamped to [0.01, 0.99]. Bins that are too
        small to trust apply no correction."""
        b = self._bin_for(p)
        if b is None or b["n"] < MIN_BIN_N:
            return p
        return float(np.clip(p + (b["observed"] - b["mean_pred"]), 0.01, 0.99))


def w_data(n_eff_home: float, n_eff_away: float) -> float:
    n = min(n_eff_home, n_eff_away)
    return n / (n + N_EFF_K)


def w_liq(depth_at_touch: int, wanted_contracts: int) -> float:
    if wanted_contracts <= 0:
        return 1.0
    return float(np.clip(depth_at_touch / wanted_contracts, 0.0, 1.0))


def w_time(hours_to_kickoff: float) -> float:
    if hours_to_kickoff <= NEWS_SAFE_HOURS:
        return 1.0
    return float(np.clip(NEWS_SAFE_HOURS / hours_to_kickoff, TIME_FLOOR, 1.0))


def kelly_multiplier(confidence: float, lo: float = 0.25, hi: float = 0.50) -> float:
    c = float(np.clip(confidence, 0.0, 1.0))
    return lo + (hi - lo) * c
