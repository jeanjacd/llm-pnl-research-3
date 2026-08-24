"""Fail loudly if any league's dataset or manifest is missing or inconsistent."""
from __future__ import annotations

import hashlib
import json
import os
import sys

from wc2026.leagues import all_leagues


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    problems = []
    for spec in all_leagues():
        if not os.path.exists(spec.matches_csv):
            problems.append("%s: no matches.csv" % spec.league_id)
            continue
        if not os.path.exists(spec.manifest_json):
            problems.append("%s: no manifest" % spec.league_id)
            continue
        manifest = json.load(open(spec.manifest_json, encoding="utf-8"))
        actual = sha256(spec.matches_csv)
        if manifest.get("matches_csv_sha256") != actual:
            problems.append("%s: checksum mismatch (data changed without a "
                            "manifest update)" % spec.league_id)
        rows = sum(1 for _ in open(spec.matches_csv, encoding="utf-8")) - 1
        if rows != manifest.get("n_rows"):
            problems.append("%s: %d rows but manifest says %s"
                            % (spec.league_id, rows, manifest.get("n_rows")))
        print("%-15s rows=%-6d teams=%-4s sha=%s"
              % (spec.league_id, rows, manifest.get("n_teams"), actual[:12]))
    if problems:
        print("\nMANIFEST VALIDATION FAILED")
        for p in problems:
            print("  -", p)
        return 1
    print("\nall league manifests validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
