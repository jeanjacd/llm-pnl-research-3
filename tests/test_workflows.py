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


def test_the_board_runs_every_six_hours_every_day():
    """Every day, not weekends only: a Wednesday match has its T-24h on
    Tuesday, and the 162 midweek fixtures of 1,547 are 10.5% of the sample
    that any significance claim rests on."""
    fires = fire_times(crons(load("matchday-board"))[0], MONDAY)
    assert len(fires) == 28, "4 runs a day, 7 days"
    assert len({f.strftime("%a") for f in fires}) == 7
    for day in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"):
        hours = sorted(f.hour for f in fires if f.strftime("%a") == day)
        assert hours == [0, 6, 12, 18], day


def test_the_board_interval_is_shorter_than_the_board_window():
    """Otherwise a fixture slips between two runs and is never boarded."""
    from wc2026.paper.selection import BOARD_WINDOW_HOURS
    fires = fire_times(crons(load("matchday-board"))[0], MONDAY)
    gaps = {(fires[i + 1] - fires[i]).total_seconds() / 3600.0
            for i in range(len(fires) - 1)}
    assert max(gaps) <= 2 * BOARD_WINDOW_HOURS


def test_the_board_can_still_be_dispatched_by_hand():
    assert "workflow_dispatch" in load("matchday-board")["triggers"]


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
