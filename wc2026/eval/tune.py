"""
tune.py
=======
Per-league hyperparameter selection.

Selection runs inside the DEV region only (see eval/nested.py) and writes the
winner to that league's own fitted_params.json. The final holdout is never
read here -- that is the whole point of the separation, and it is why the
numbers written by this module are explicitly labelled as inner-fold scores
rather than performance estimates.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .nested import make_split, select_hyperparameters


def tune_league(df: pd.DataFrame, spec, grid: dict | None = None,
                refit_freq: str = "Q", min_train: int = 400,
                verbose: bool = True) -> dict:
    """Select hyperparameters for one league and persist them."""
    split = make_split(df, spec.league_id)
    selection = select_hyperparameters(df, spec, split, grid=grid,
                                       refit_freq=refit_freq,
                                       min_train=min_train, verbose=verbose)
    payload = {
        "league_id": spec.league_id,
        "best": selection["best"],
        "selected_on": "DEV region only",
        "dev_window": split.as_dict()["dev"],
        "inner_log_loss": selection["inner_log_loss"],
        "n_inner_scored": selection["n_inner_scored"],
        "stability": selection["stability"],
        "grid": selection["grid"],
        "caveat": ("inner_log_loss is a SELECTION score, not an unbiased "
                   "performance estimate; see evaluation.json for the "
                   "holdout result."),
    }
    os.makedirs(spec.data_dir, exist_ok=True)
    with open(spec.fitted_params_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    if verbose:
        print("  %s -> %s" % (spec.league_id, json.dumps(selection["best"])))
        print("  wrote %s" % spec.fitted_params_json)
    return payload


def load_fitted_params(spec) -> dict:
    """Tuned parameters for a league, or {} when it has never been tuned.

    An empty result is meaningful: an untuned league must not silently inherit
    another league's fit, so callers should treat {} as "not validated".
    """
    if not os.path.exists(spec.fitted_params_json):
        return {}
    with open(spec.fitted_params_json, encoding="utf-8") as fh:
        return json.load(fh).get("best", {})


def effective_model(spec):
    """The league's ModelConfig with its own tuned parameters applied."""
    import dataclasses
    best = load_fitted_params(spec)
    if not best:
        return spec.model, False
    allowed = {k: v for k, v in best.items()
               if k in {f.name for f in dataclasses.fields(spec.model)}}
    return dataclasses.replace(spec.model, **allowed), True
