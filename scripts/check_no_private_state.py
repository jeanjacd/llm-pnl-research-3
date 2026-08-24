"""Fail the build if anything private is about to be published.

This is the last line of defence for mission rule 8. It is deliberately blunt:
it inspects what is actually tracked/present rather than trusting .gitignore,
because a single committed bankroll file cannot be un-published.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

# Paths that must never be tracked by git.
FORBIDDEN_PATHS = (
    "data/betting/",          # real bankroll, positions, realised P&L
    "data/private/",
    "data/paper/portfolio.json",   # the live paper ledger belongs on its branch
    ".env",
    ".venv/",
)

FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".sqlite", ".sqlite3")

# Content that must never appear in a tracked file.
SECRET_PATTERNS = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key block"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{10,}"), "Anthropic API key"),
    (re.compile(r"KALSHI_PRIVATE_KEY_PATH\s*=\s*['\"][^'\"]+"), "hardcoded key path"),
)

SCAN_SUFFIXES = (".py", ".md", ".json", ".yml", ".yaml", ".toml", ".txt", ".cfg")
# Real STATE (as opposed to a field name in source) is only ever a data file.
STATE_PATTERNS = (
    (re.compile(r'"bankroll_cents"\s*:\s*\d+'), "bankroll balance"),
    (re.compile(r'"pnl_cents"\s*:\s*-?\d+'), "realised P&L"),
    (re.compile(r'"order_id"\s*:\s*"[0-9a-f-]{8,}'), "live order id"),
)
STATE_SUFFIXES = (".json", ".jsonl")
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "archive",
             "legacy", "docs"}


def tracked_files() -> list:
    try:
        out = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def main() -> int:
    problems = []

    for path in tracked_files():
        norm = path.replace("\\", "/")
        for bad in FORBIDDEN_PATHS:
            if norm.startswith(bad):
                problems.append("tracked private path: %s" % norm)
        if norm.endswith(FORBIDDEN_SUFFIXES):
            problems.append("tracked secret-bearing file: %s" % norm)

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith(SCAN_SUFFIXES):
                continue
            full = os.path.join(root, name)
            norm = full.replace("\\", "/").lstrip("./")
            if norm.startswith("scripts/check_no_private_state.py"):
                continue
            if norm.startswith("data/betting/") or norm.startswith("data/paper/"):
                continue          # ignored by .gitignore; never published
            try:
                with open(full, encoding="utf-8", errors="ignore") as fh:
                    body = fh.read()
            except OSError:
                continue
            for pattern, label in SECRET_PATTERNS:
                if pattern.search(body):
                    problems.append("%s found in %s" % (label, norm))
            # Live state values only matter in data files; the same words
            # appearing as field names in source are not private data.
            if norm.endswith(STATE_SUFFIXES):
                for pattern, label in STATE_PATTERNS:
                    if pattern.search(body):
                        problems.append("%s found in %s" % (label, norm))

    if problems:
        print("PRIVATE STATE CHECK FAILED")
        for item in sorted(set(problems)):
            print("  -", item)
        return 1
    print("private state check passed: nothing sensitive is tracked or exposed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
