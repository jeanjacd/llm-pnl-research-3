"""Durable paper state on a dedicated automation branch.

GitHub runners are ephemeral, so the paper ledger cannot live in the workspace
between runs. It also must NOT live in a dependency cache: a cache is evictable
and unversioned, which is unacceptable for the only record of what the system
"traded".

Design: a small, sanitised, append-friendly state file committed to a dedicated
branch, written atomically, one writer at a time (the workflow's concurrency
group). Large raw snapshots stay as workflow artifacts instead.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

STATE_FILES = ("data/paper/portfolio.json",)
DEFAULT_BRANCH = "automation-state"


def run(*args, check=True):
    return subprocess.run(list(args), check=check, capture_output=True,
                          text=True)


def restore(branch: str) -> int:
    os.makedirs("data/paper", exist_ok=True)
    fetched = run("git", "fetch", "origin", branch, check=False)
    if fetched.returncode != 0:
        print("no %s branch yet; starting from an empty portfolio" % branch)
        return 0
    for path in STATE_FILES:
        got = run("git", "checkout", "FETCH_HEAD", "--", path, check=False)
        if got.returncode == 0:
            print("restored %s" % path)
        else:
            print("no %s on %s yet" % (path, branch))
    return 0


def save(branch: str) -> int:
    """Commit the state files ON TOP of the existing state branch.

    The obvious implementation -- commit onto the checked-out branch and
    `git push HEAD:<branch>` -- works exactly once. A runner checks out the
    default branch fresh each time, so the second run builds `main + state2`,
    which does not contain `main + state1` as an ancestor and is rejected
    non-fast-forward. Reproduced against a scratch remote: save #1 succeeds,
    save #2 fails.

    Force-pushing would paper over it and cost the history this branch exists
    to provide, so the commit is instead built in a throwaway worktree parented
    on the branch's current tip. The main working tree is never touched, which
    matters because it still holds the data the run just produced.
    """
    present = [p for p in STATE_FILES if os.path.exists(p)]
    if not present:
        print("nothing to save")
        return 0

    payload = {}
    for path in present:
        with open(path, "rb") as fh:
            payload[path] = fh.read()

    fetched = run("git", "fetch", "origin", branch, check=False)
    if fetched.returncode != 0:
        # No branch yet: the first push creates it, and there is nothing to
        # fast-forward from.
        return _save_onto_head(branch, present)

    workdir = tempfile.mkdtemp(prefix="state-sync-")
    try:
        added = run("git", "worktree", "add", "--detach", workdir,
                    "FETCH_HEAD", check=False)
        if added.returncode != 0:
            print("worktree failed:", added.stderr[:300])
            return 1
        for path, blob in payload.items():
            target = os.path.join(workdir, path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as fh:
                fh.write(blob)
        run("git", "-C", workdir, "config", "user.name", "wc2026-bot")
        run("git", "-C", workdir, "config", "user.email",
            "wc2026-bot@users.noreply.github.com")
        run("git", "-C", workdir, "add", "-f", *present)
        committed = run("git", "-C", workdir, "commit", "-m",
                        "paper state %s" % os.environ.get("GITHUB_RUN_ID", "local"),
                        check=False)
        if committed.returncode != 0:
            print("no changes to persist")
            return 0
        pushed = run("git", "-C", workdir, "push", "origin",
                     "HEAD:%s" % branch, check=False)
        if pushed.returncode != 0:
            print("push failed:", pushed.stderr[:300])
            return 1
    finally:
        run("git", "worktree", "remove", "--force", workdir, check=False)
        shutil.rmtree(workdir, ignore_errors=True)
    print("persisted paper state to %s" % branch)
    return 0


def _save_onto_head(branch: str, present) -> int:
    """First ever save: no branch to parent on, so the push creates it."""
    run("git", "config", "user.name", "wc2026-bot")
    run("git", "config", "user.email", "wc2026-bot@users.noreply.github.com")
    run("git", "add", "-f", *present)
    committed = run("git", "commit", "-m",
                    "paper state %s" % os.environ.get("GITHUB_RUN_ID", "local"),
                    check=False)
    if committed.returncode != 0:
        print("no changes to persist")
        return 0
    pushed = run("git", "push", "origin", "HEAD:%s" % branch, check=False)
    if pushed.returncode != 0:
        print("push failed:", pushed.stderr[:300])
        return 1
    print("persisted paper state to %s (branch created)" % branch)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["restore", "save"])
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    args = parser.parse_args()
    return restore(args.branch) if args.action == "restore" else save(args.branch)


if __name__ == "__main__":
    sys.exit(main())
