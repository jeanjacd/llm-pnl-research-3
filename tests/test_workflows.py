"""The scheduled automation is configuration, so it is tested like code.

A cron expression that silently stops matching, or a workflow that gains a
credential capable of placing a real order, would both fail quietly in
production. These assertions are cheap and catch exactly that.
"""
import datetime as dt
import glob
import os

import yaml

WORKFLOWS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         ".github", "workflows")


def load(name):
    with open(os.path.join(WORKFLOWS, name + ".yml"), encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    # PyYAML resolves the bare key `on:` to the boolean True (YAML 1.1).
    doc["triggers"] = doc.get("on", doc.get(True))
    return doc


def crons(doc):
    return [s["cron"] for s in (doc["triggers"].get("schedule") or [])]


def fire_times(expression, start, days=7):
    """Every UTC minute in a window at which a 5-field cron fires."""
    minute, hour, dom, month, dow = expression.split()

    def field(spec, lo, hi):
        out = set()
        for part in spec.split(","):
            step = 1
            if "/" in part:
                part, step_text = part.split("/")
                step = int(step_text)
            if part == "*":
                lo_, hi_ = lo, hi
            elif "-" in part:
                a, b = part.split("-")
                lo_, hi_ = int(a), int(b)
            else:
                lo_ = hi_ = int(part)
            out.update(range(lo_, hi_ + 1, step))
        return out

    minutes, hours = field(minute, 0, 59), field(hour, 0, 23)
    dows = field(dow, 0, 6)
    assert dom == "*" and month == "*", "day/month not modelled here"
    out, t = [], start
    while t < start + dt.timedelta(days=days):
        if (t.minute in minutes and t.hour in hours
                and (t.weekday() + 1) % 7 in dows):
            out.append(t)
        t += dt.timedelta(minutes=1)
    return out


MONDAY = dt.datetime(2026, 8, 24, tzinfo=dt.timezone.utc)


def test_every_workflow_file_is_valid_yaml_with_a_name():
    files = glob.glob(os.path.join(WORKFLOWS, "*.yml"))
    assert files, "no workflows found"
    for path in files:
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        assert doc.get("name"), path
        assert doc.get("on", doc.get(True)), path


def test_the_board_is_fired_externally_not_by_github():
    """`schedule:` is removed on purpose, so it must not creep back.

    GitHub's scheduler fired this workflow once in a 14h stretch while the
    external cron fired 7 for 7. Two triggers meant unpredictable extra runs,
    and a board run costs model time -- so the unreliable one was dropped. A
    re-added `schedule:` would quietly resume double-firing.
    """
    assert "schedule" not in load("matchday-board")["triggers"]


def test_the_boards_only_trigger_is_load_bearing():
    """With no schedule, `workflow_dispatch` is the ONLY way this runs.
    Removing it does not degrade the cadence -- it stops the board entirely."""
    assert "workflow_dispatch" in load("matchday-board")["triggers"]


def test_the_board_interval_is_shorter_than_the_board_window():
    """Otherwise a fixture slips between two runs and is never boarded."""
    from wc2026.paper.selection import BOARD_RUN_INTERVAL_HOURS, BOARD_WINDOW_HOURS
    assert BOARD_RUN_INTERVAL_HOURS <= 2 * BOARD_WINDOW_HOURS


def test_the_board_fetches_match_data_before_running():
    """`data/leagues/*/raw/` is gitignored, so a runner starts with none.
    Without this the cycle skips every league and reports a silent no-op."""
    text = open(os.path.join(WORKFLOWS, "matchday-board.yml"),
                encoding="utf-8").read()
    assert "wc2026 update" in text
    assert text.index("wc2026 update") < text.index("paper-cycle")


# --- the cheap half, split out so it can run daily ----------------------------
def test_maintenance_runs_every_day_including_midweek():
    """Settlement and fill replay cost no model time; a midweek match must
    settle on the day it finishes, not at the next matchday."""
    fires = fire_times(crons(load("paper-maintenance"))[0], MONDAY)
    assert len({f.strftime("%a") for f in fires}) == 7
    assert len(fires) == 14


def test_maintenance_makes_no_model_calls():
    """The whole point of the split is that this half is nearly free."""
    text = open(os.path.join(WORKFLOWS, "paper-maintenance.yml"),
                encoding="utf-8").read()
    assert "paper-maintain" in text
    for forbidden in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY",
                      "paper-cycle"):
        assert forbidden not in text, forbidden


def test_maintenance_refreshes_results_before_settling():
    text = open(os.path.join(WORKFLOWS, "paper-maintenance.yml"),
                encoding="utf-8").read()
    assert text.index("wc2026 update") < text.index("paper-maintain")


def test_maintenance_does_not_collide_with_the_daily_data_refresh():
    """Two workflows pulling every league from ESPN at the same instant."""
    a = set(fire_times(crons(load("paper-maintenance"))[0], MONDAY))
    b = set(fire_times(crons(load("daily-data"))[0], MONDAY))
    assert not (a & b)


def test_both_writers_share_one_portfolio_lock():
    """A settlement landing mid-submission would corrupt the ledger."""
    board = load("matchday-board")["concurrency"]
    maintain = load("paper-maintenance")["concurrency"]
    assert board["group"] == maintain["group"] == "paper-portfolio"
    assert maintain["cancel-in-progress"] is False


def test_a_scheduled_run_cannot_place_a_real_order():
    for name in ("matchday-board", "paper-maintenance"):
        text = open(os.path.join(WORKFLOWS, name + ".yml"),
                    encoding="utf-8").read()
        assert 'WC2026_PAPER_ONLY: "1"' in text, name
        for forbidden in ("KALSHI_API_KEY", "KALSHI_PRIVATE_KEY",
                          "POLYMARKET_KEY", "PRIVATE_KEY", "execute"):
            assert forbidden not in text, (name, forbidden)


def test_one_writer_at_a_time_on_the_portfolio():
    doc = load("matchday-board")
    assert doc["concurrency"]["group"] == "paper-portfolio"
    assert doc["concurrency"]["cancel-in-progress"] is False


def test_the_other_schedules_are_unchanged():
    assert crons(load("daily-data")) == ["40 5 * * *"]
    assert crons(load("weekly-evaluation")) == ["20 6 * * 1"]


def test_the_board_timeout_covers_a_busy_matchday():
    """A run killed mid-flight loses all of its work, including spent board
    calls. Simulated against the real fixture calendar at the 2h cadence, the
    busiest run boards 11 fixtures -- 60 minutes at the measured 330s each --
    plus discovery, which the window filter keeps small."""
    doc = load("matchday-board")
    timeout = doc["jobs"]["cycle"]["timeout-minutes"]
    worst_case_minutes = 11 * 330 / 60 + 21
    assert timeout >= worst_case_minutes, (
        "timeout %s min is below the measured worst case %.0f min"
        % (timeout, worst_case_minutes))


def test_the_board_finishes_before_the_next_run_is_due():
    """Otherwise the concurrency queue backs up run on run.

    Measured against the constant rather than a cron: the schedule now lives
    in the external trigger, but the bound it has to satisfy does not move."""
    from wc2026.paper.selection import BOARD_RUN_INTERVAL_HOURS
    gap_minutes = BOARD_RUN_INTERVAL_HOURS * 60
    assert load("matchday-board")["jobs"]["cycle"]["timeout-minutes"] < gap_minutes


def test_workflows_that_persist_state_can_actually_write():
    """`contents: read` made every scheduled run fail at its last step, after
    the work was already done -- including a spent board call. The failure was
    a 403 from github-actions[bot], not anything in the code."""
    for name in ("matchday-board", "paper-maintenance"):
        doc = load(name)
        text = open(os.path.join(WORKFLOWS, name + ".yml"),
                    encoding="utf-8").read()
        if "state_sync.py save" not in text:
            continue
        assert doc["permissions"]["contents"] == "write", (
            "%s pushes the paper ledger but only requests contents: %s"
            % (name, doc["permissions"]["contents"]))


def test_the_code_knows_the_real_board_cadence():
    """The retry deadline is derived from the run interval, so if the firing
    schedule changes and `BOARD_RUN_INTERVAL_HOURS` does not, retries stop
    being guaranteed -- invisibly, until a fixture kicks off unretried.

    This USED to compare the constant against the workflow's own cron. That
    cron is gone: firing moved to an external trigger to stop GitHub's
    unreliable scheduler spending board calls on unpredictable extra runs. No
    test can reach cron-job.org, so the check is necessarily weaker now, and
    saying so is the point -- the guarantee rests on a human keeping two
    places in step. What is still enforceable is that the workflow names the
    constant and states the interval it was configured with, so whoever
    changes one is told where the other lives.
    """
    from wc2026.paper.selection import BOARD_RUN_INTERVAL_HOURS
    text = open(os.path.join(WORKFLOWS, "matchday-board.yml"),
                encoding="utf-8").read()
    assert "BOARD_RUN_INTERVAL_HOURS" in text, (
        "the workflow must name the constant its cadence has to match")
    assert "every %dh" % BOARD_RUN_INTERVAL_HOURS in text, (
        "workflow does not state the %dh cadence the constant assumes"
        % BOARD_RUN_INTERVAL_HOURS)


def test_the_board_workflow_installs_the_binary_it_shells_out_to():
    """`run_board` calls `claude`, an npm package. It was never installed, so
    every board call fell closed to DEFER and seven fixtures were reported as
    considered deferrals when the board had never run."""
    text = open(os.path.join(WORKFLOWS, "matchday-board.yml"),
                encoding="utf-8").read()
    assert "@anthropic-ai/claude-code" in text
    assert text.index("npm install") < text.index("paper-cycle")


def test_a_missing_board_fails_the_job_rather_than_every_fixture():
    """A DEFER must never be able to mean 'the board was not installed'."""
    text = open(os.path.join(WORKFLOWS, "matchday-board.yml"),
                encoding="utf-8").read()
    assert "claude --version" in text
    assert text.index("claude --version") < text.index("paper-cycle")


# ── the published site ────────────────────────────────────────────────────────
def test_the_site_is_rebuilt_whenever_the_ledger_changes():
    """The page is generated from the ledger, so it must fire after every
    workflow that writes one -- otherwise the site silently shows a stale
    record while claiming to be current."""
    triggers = load("publish-site")["triggers"]
    watched = set(triggers["workflow_run"]["workflows"])
    writers = {name for name in ("matchday-board", "paper-maintenance")
               if "contents" in (load(name)["permissions"] or {})}
    assert writers <= watched, "a ledger writer is not being published from"


def test_the_watched_workflow_names_actually_exist():
    """`workflow_run` matches on the workflow's `name:`, not its filename. A
    rename would stop the trigger matching and nothing would fail -- the site
    would just quietly stop updating."""
    names = {load(f)["name"] for f in ("matchday-board", "paper-maintenance")}
    for watched in load("publish-site")["triggers"]["workflow_run"]["workflows"]:
        assert watched in names, (
            "publish-site watches %r, which is not any workflow's name" % watched)


def test_publishing_cannot_write_to_the_repository():
    """It reads the ledger and deploys. It has no business committing, and a
    deployment token plus write access is a wider blast radius than either
    needs on its own."""
    doc = load("publish-site")
    assert doc["permissions"]["contents"] == "read"
    assert doc["permissions"]["pages"] == "write"


def test_publishing_does_not_queue_behind_the_board():
    """Sharing `paper-portfolio` would make every deploy wait on a board run,
    and a long board would hold the site stale for an hour."""
    assert load("publish-site")["concurrency"]["group"] == "pages"
    assert load("publish-site")["concurrency"]["group"] != \
        load("matchday-board")["concurrency"]["group"]


def test_a_failed_board_does_not_publish():
    doc = load("publish-site")
    guard = str(doc["jobs"]["publish"].get("if") or "")
    assert "conclusion == 'success'" in guard


def test_the_site_restores_state_before_building_it():
    """`data/paper/` is gitignored; without the restore the build would render
    a page saying the book is empty, which is a claim about the trading."""
    text = open(os.path.join(WORKFLOWS, "publish-site.yml"),
                encoding="utf-8").read()
    assert "state_sync.py restore" in text
    assert text.index("state_sync.py restore") < text.index("wc2026.site.build")
