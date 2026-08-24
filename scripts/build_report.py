"""Build the public performance report.

Every number is labelled with what it actually is -- backtest, forward paper,
or counterfactual -- with its sample size. Nothing here may imply live trading:
live is zero and unsupported in this phase.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from wc2026.leagues import all_leagues


def evaluation_section() -> list:
    lines = ["## Model backtest (NOT market alpha)", "",
             "Nested evaluation: hyperparameters chosen on DEV, calibrator on",
             "CAL, headline computed once on an untouched HOLDOUT. The",
             "comparison is model vs BASE RATE, not model vs market price.", "",
             "| league | n | log-loss | base rate | improvement | 95% CI | cal. slope |",
             "|---|---|---|---|---|---|---|"]
    any_report = False
    for spec in all_leagues():
        if not os.path.exists(spec.eval_report_json):
            continue
        any_report = True
        report = json.load(open(spec.eval_report_json, encoding="utf-8"))
        hold = report["holdout"]
        m, b = hold["metrics"], hold["baseline"]
        extras = hold.get("extras", {})
        ci = extras.get("log_loss_ci", [float("nan")] * 2)
        lines.append("| %s | %d | %.4f | %.4f | %.1f%% | [%.4f, %.4f] | %.3f |"
                     % (spec.league_id, hold["n"], m["log_loss"],
                        b["log_loss"],
                        100 * (1 - m["log_loss"] / b["log_loss"]),
                        ci[0], ci[1],
                        extras.get("calibration_slope", float("nan"))))
    if not any_report:
        lines.append("| (no evaluation yet) | | | | | | |")
    return lines


def paper_section(state_path: str) -> list:
    lines = ["", "## Forward paper trading", ""]
    if not os.path.exists(state_path):
        lines += ["No paper portfolio yet.", ""]
        return lines
    from wc2026.paper.broker import PaperPortfolio
    summary = PaperPortfolio.load(state_path).summary()
    lines += ["| metric | value |", "|---|---|"]
    for key in ("starting_cash_usd", "cash_usd", "n_orders", "n_open_orders",
                "n_positions_open", "n_settled", "realized_pnl_usd",
                "fees_paid_usd", "fill_rate"):
        if key in summary:
            lines.append("| %s | %s |" % (key, summary[key]))
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="public_report.md")
    parser.add_argument("--state", default=os.path.join("data", "paper",
                                                        "portfolio.json"))
    args = parser.parse_args()

    lines = ["# wc2026 performance report", "",
             "**Status: PAPER TRADING ONLY.** Live trading is zero and",
             "unsupported in this phase. Figures below are separated into",
             "model backtest, forward paper results, and counterfactuals --",
             "they are not interchangeable.", ""]
    lines += evaluation_section()
    lines += paper_section(args.state)
    lines += ["", "## Coverage and abstention", "",
              "Supported market families are the six derived from the exact",
              "regulation scoreline grid. Corners, halves, player props,",
              "method-of-victory and season futures are DISCOVERED, RECORDED",
              "and ABSTAINED FROM -- no validated model exists for them, so no",
              "price is produced. See docs/VENUES.md for the audited counts.", ""]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
