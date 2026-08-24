"""
adjustments.py
==============
Loads the transparent soft-factor layer (data/adjustments.json) and applies it to
fitted ratings. These capture things no free verifiable feed provides -- injuries,
key-player availability, lineup/tactical changes, motivation -- as explicit,
auditable, human-curated multipliers on expected goals.

attack_mult  scales the goals a team is expected to SCORE.
defense_mult scales the goals a team is expected to CONCEDE (>1 = leakier).

In the log-goal parameterisation these fold cleanly into the ratings:
    attack_i  += log(attack_mult)         # more/fewer goals scored
    defense_i -= log(defense_mult)        # leakier defense => lower defense rating
"""
from __future__ import annotations

import json
import math
import os


def load_adjustments(path: str | None) -> dict[str, dict]:
    """Return {team: {attack_mult, defense_mult, note}} for an EXPLICIT path.

    There is no default file: a soft-factor layer belongs to exactly one
    league, and a shared default is how one league's injury note silently
    shifts another league's forecast. None or a missing file -> {}.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("teams", {}) or {}


def apply_adjustments(attack: dict[str, float], defense: dict[str, float],
                      adjustments: dict[str, dict], verbose: bool = True
                      ) -> list[str]:
    """Mutate attack/defense in place by the configured multipliers.

    Returns a human-readable log of everything applied, so assumptions are auditable.
    """
    applied: list[str] = []
    for team, adj in adjustments.items():
        am = float(adj.get("attack_mult", 1.0))
        dm = float(adj.get("defense_mult", 1.0))
        if am == 1.0 and dm == 1.0:
            continue
        if team in attack:
            attack[team] += math.log(am)
        if team in defense:
            defense[team] -= math.log(dm)
        note = adj.get("note", "")
        applied.append(f"{team}: attack x{am:.2f}, defense x{dm:.2f}  -- {note}")
    if verbose and applied:
        print("Applied soft-factor adjustments:")
        for line in applied:
            print("  " + line)
    return applied
