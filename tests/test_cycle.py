"""End-to-end paper cycle: claim probabilities, orchestration, idempotency and
sanitised reporting. Fully offline -- fake providers, fake board."""
import datetime as dt
import os

import pytest

from wc2026.paper import cycle as cycle_mod
from wc2026.paper.broker import PaperPortfolio
from wc2026.paper.cycle import probability_for, render_summary, run_cycle
from wc2026.sim.match import predict_match
from wc2026.venues.base import KIND_BINARY, Book, Leg, MarketInstrument


def soon(hours=8):
    return (dt.datetime.now(dt.timezone.utc)
            + dt.timedelta(hours=hours)).isoformat()


# --------------------------------------------------------------------------- #
# claim -> probability, straight off the grid
# --------------------------------------------------------------------------- #
class _Ratings:
    """Minimal stand-in exposing what predict_match needs."""
    def __init__(self):
        import pandas as pd

        from wc2026.model.dixon_coles import DCRatings
        self.r = DCRatings(attack={"A": 0.2, "B": -0.1},
                           defense={"A": 0.1, "B": 0.0}, intercept=0.3,
                           home_adv=0.25, rho=-0.05,
                           n_eff={"A": 50.0, "B": 50.0}, teams=["A", "B"],
                           as_of=pd.Timestamp("2026-08-01"))


@pytest.fixture
def prediction():
    return predict_match(_Ratings().r, "A", "B", neutral=False)


def test_1x2_probabilities_sum_to_one(prediction):
    total = sum(probability_for(c, prediction)
                for c in ("home_win", "draw", "away_win"))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_negation_is_the_complement(prediction):
    p = probability_for("btts", prediction)
    assert probability_for("not_btts", prediction) == pytest.approx(1 - p)


@pytest.mark.parametrize("claim", [
    "home_win", "away_win", "draw", "btts", "total_over_2.5",
    "total_under_2.5", "home_over_1.5", "away_over_0.5",
    "home_wins_by_over_1.5", "away_wins_by_over_1.5", "score_2-1",
])
def test_every_supported_claim_resolves(prediction, claim):
    p = probability_for(claim, prediction)
    assert p is not None and 0.0 <= p <= 1.0


def test_unsupported_claims_return_none(prediction):
    for claim in ("player_anytime_goal", "total_corners_over_9.5",
                  "first_half_over_1.5", "score_99-99"):
        assert probability_for(claim, prediction) is None


def test_totals_are_monotone(prediction):
    assert (probability_for("total_over_0.5", prediction)
            > probability_for("total_over_2.5", prediction)
            > probability_for("total_over_4.5", prediction))


# --------------------------------------------------------------------------- #
# a full cycle, twice
# --------------------------------------------------------------------------- #
class FakeProvider:
    """Returns one attractively-priced supported market."""
    venue = "kalshi"

    def __init__(self, ask=30, size=500.0, hours=8):
        self.ask, self.size, self.hours = ask, size, hours
        self.calls = 0

    def discover(self, spec, **kw):
        self.calls += 1
        teams = kw.get("_teams") or ("A", "B")
        return [MarketInstrument(
            venue=self.venue, instrument_id="INST-1", kind=KIND_BINARY,
            title="A vs B", legs=(Leg.build("home_win", "ref", home=teams[0],
                                            away=teams[1],
                                            kickoff_utc=soon(self.hours)),),
            settles_on_regulation=True, league_id=spec.league_id,
            kickoff_utc=soon(self.hours), fee_model={"venue": "kalshi"},
            book=Book(yes_asks=((self.ask, self.size),),
                      yes_bids=((self.ask - 1, self.size),),
                      observed_at=dt.datetime.now(dt.timezone.utc).isoformat()))]


def _patch_league(monkeypatch, tmp_path):
    """Point the cycle at a tiny synthetic league with tuned params."""
    import pandas as pd

    from wc2026.paper import cycle as cycle_mod

    rows = []
    day = pd.Timestamp("2024-01-01")
    for k in range(400):
        home, away = ("A", "B") if k % 2 else ("B", "A")
        rows.append({"date": day + pd.Timedelta(days=k),
                     "kickoff_utc": day + pd.Timedelta(days=k, hours=19),
                     "home_team": home,
                     "away_team": away, "home_score": 2, "away_score": 1,
                     "tournament": "L", "tier": "league", "neutral": False,
                     "played": True})
    played = pd.DataFrame(rows)
    future = pd.DataFrame([{"date": pd.Timestamp("2026-12-01"),
                            "kickoff_utc": pd.Timestamp("2026-12-01 19:00"),
                            "home_team": "A", "away_team": "B",
                            "home_score": None, "away_score": None,
                            "tournament": "L", "tier": "league",
                            "neutral": False, "played": False}])
    frame = pd.concat([played, future], ignore_index=True)

    monkeypatch.setattr(cycle_mod.loader, "load_league", lambda s, **k: frame)
    monkeypatch.setattr(cycle_mod, "effective_model",
                        lambda spec: (spec.model, True))
    from wc2026.leagues import get_league
    monkeypatch.setattr(cycle_mod, "all_leagues",
                        lambda: [get_league("premier_league")])
    monkeypatch.setattr(cycle_mod, "SNAPSHOT_DIR", str(tmp_path / "snaps"))


def _approve_board(monkeypatch):
    from wc2026.paper import cycle as cycle_mod

    def fake_board(cases, fixture, **kw):
        return {"failed_closed": False, "failure": None,
                "coach": {"verdict": "ACCEPT", "rationale": "fine"},
                "decisions": {c.case_id: {
                    "case_id": c.case_id, "action": "PAPER_BUY_NOW",
                    "limit_price_cents": c.max_limit_price_cents,
                    "contracts": 5.0,
                    "decisive_reason": "approved"} for c in cases}}
    monkeypatch.setattr(cycle_mod, "run_board_slate", fake_board)


def test_cycle_builds_cases_and_submits(tmp_path, monkeypatch):
    _patch_league(monkeypatch, tmp_path)
    _approve_board(monkeypatch)
    state = str(tmp_path / "portfolio.json")
    stats = run_cycle(state_path=state, providers=[FakeProvider()],
                      verbose=False)
    assert stats["cases_built"] > 0
    assert stats["orders_submitted"] >= 1
    assert os.path.exists(state)


def test_running_the_cycle_twice_is_idempotent(tmp_path, monkeypatch):
    """THE requirement: a repeated scheduled run must not double-submit."""
    _patch_league(monkeypatch, tmp_path)
    _approve_board(monkeypatch)
    state = str(tmp_path / "portfolio.json")

    first = run_cycle(state_path=state, providers=[FakeProvider()],
                      verbose=False)
    portfolio_after_first = PaperPortfolio.load(state)
    n_orders_first = len(portfolio_after_first.orders)
    cash_first = portfolio_after_first.cash_cents

    second = run_cycle(state_path=state, providers=[FakeProvider()],
                       verbose=False)
    portfolio_after_second = PaperPortfolio.load(state)

    assert len(portfolio_after_second.orders) == n_orders_first
    assert portfolio_after_second.cash_cents == cash_first
    # unchanged markets are recognised and skipped rather than re-reviewed
    assert second["cases_skipped_unchanged"] >= first["cases_skipped_unchanged"]


def test_changed_price_reopens_a_case(tmp_path, monkeypatch):
    """A moved price must produce a fresh case, not be suppressed forever."""
    _patch_league(monkeypatch, tmp_path)
    _approve_board(monkeypatch)
    state = str(tmp_path / "portfolio.json")
    run_cycle(state_path=state, providers=[FakeProvider(ask=30)], verbose=False)
    moved = run_cycle(state_path=state, providers=[FakeProvider(ask=25)],
                      verbose=False)
    assert moved["cases_built"] > 0


def test_untuned_league_is_skipped(tmp_path, monkeypatch):
    """A league with no validated parameters must never be traded."""
    from wc2026.paper import cycle as cycle_mod
    _patch_league(monkeypatch, tmp_path)
    monkeypatch.setattr(cycle_mod, "effective_model",
                        lambda spec: (spec.model, False))
    stats = run_cycle(state_path=str(tmp_path / "p.json"),
                      providers=[FakeProvider()], verbose=False)
    assert stats["orders_submitted"] == 0
    assert any("untuned" in str(v.get("skipped", ""))
               for v in stats["leagues"].values())


def test_board_rejection_blocks_submission(tmp_path, monkeypatch):
    from wc2026.paper import cycle as cycle_mod
    _patch_league(monkeypatch, tmp_path)
    monkeypatch.setattr(cycle_mod, "run_board_slate",
                        lambda cases, fixture, **kw: {
                            "decisions": {}, "failed_closed": False,
                            "failure": "coach veto (REJECT)",
                            "coach": {"verdict": "REJECT",
                                      "rationale": "declined"}})
    stats = run_cycle(state_path=str(tmp_path / "p.json"),
                      providers=[FakeProvider()], verbose=False)
    assert stats["orders_submitted"] == 0


def test_provider_failure_does_not_abort_the_cycle(tmp_path, monkeypatch):
    class Broken:
        venue = "kalshi"

        def discover(self, spec, **kw):
            raise RuntimeError("venue down")

    _patch_league(monkeypatch, tmp_path)
    stats = run_cycle(state_path=str(tmp_path / "p.json"),
                      providers=[Broken()], verbose=False)
    assert stats["orders_submitted"] == 0
    errors = [e for v in stats["leagues"].values()
              for e in (v.get("errors") or [])]
    assert errors and "venue down" in errors[0]


# --------------------------------------------------------------------------- #
# the public summary must stay sanitised
# --------------------------------------------------------------------------- #
def test_summary_contains_no_secrets_or_transcripts():
    stats = {"started_at": "2026-08-24T00:00:00Z", "cases_built": 3,
             "orders_submitted": 1, "fills": 1,
             "portfolio": {"cash_usd": 987.65, "realized_pnl_usd": 1.23,
                           "fill_rate": 0.5},
             "leagues": {"premier_league": {"instruments": 10, "supported": 4,
                                            "changed": 4, "cases": 8,
                                            "submitted": 1}}}
    text = render_summary(stats)
    assert "PAPER TRADING ONLY" in text
    for banned in ("prompt", "rationale", "api_key", "sk-ant", "order_id",
                   "PRIVATE KEY", "instrument_id"):
        assert banned.lower() not in text.lower()


def test_summary_states_it_is_paper_only():
    assert "PAPER TRADING ONLY" in render_summary({"leagues": {}})


# --------------------------------------------------------------------------- #
# what a run actually costs
# --------------------------------------------------------------------------- #
def test_a_run_with_no_fixture_in_the_window_makes_no_model_calls(tmp_path,
                                                                  monkeypatch):
    """This is the cost model, so it is guarded rather than assumed.

    Each fixture is boarded once at ~T-24h, and 52 of 84 simulated runs board
    nothing at all. Those runs must cost ZERO model calls -- not a cheap call,
    not a cached one. The board is the only place this system spends anything
    per-decision, so a regression here would be silent and expensive.
    """
    _patch_league(monkeypatch, tmp_path)

    def explode(*args, **kwargs):
        raise AssertionError("a model was invoked on a run with no fixtures")

    monkeypatch.setattr("wc2026.board.orchestrator.invoke_member", explode)
    monkeypatch.setattr("wc2026.paper.cycle.run_board_slate", explode)

    # 40h out: comfortably inside the deterministic layer's 96h horizon, so
    # the case is built and PLACEABLE, but outside the board window's 30h
    # ceiling. That isolates the window as the thing doing the work -- a
    # fixture rejected on freshness instead would prove nothing about cost.
    stats = run_cycle(state_path=str(tmp_path / "portfolio.json"),
                      providers=[FakeProvider(hours=40)], verbose=False,
                      fill_probes={})
    assert stats["placeable_cases"] > 0, "must be placeable, or this proves nothing"
    assert stats["board_run"] == 0
    assert stats["fixtures_selected"] == 0
    assert stats["orders_submitted"] == 0
    assert "too_early" in " ".join(stats["drop_reasons"])
    # Everything up to the board still ran -- a cheap run, not a broken one.
    assert stats["cases_built"] > 0


def test_a_fixture_inside_the_window_does_reach_the_board(tmp_path, monkeypatch):
    """The mirror of the test above: proves it is the window doing the work."""
    _patch_league(monkeypatch, tmp_path)
    _approve_board(monkeypatch)
    stats = run_cycle(state_path=str(tmp_path / "portfolio.json"),
                      providers=[FakeProvider()], verbose=False,
                      fill_probes={})
    assert stats["fixtures_selected"] == 1
    assert stats["board_run"] == 1


# --------------------------------------------------------------------------- #
# why a fixture went the way it did
# --------------------------------------------------------------------------- #
def a_verdict(**kw):
    base = {"action": "DEFER", "failure": "", "quant": None, "coach": None,
            "judge": None}
    base.update(kw)
    return base


def test_a_quant_veto_is_attributed_to_the_quant_with_its_own_words():
    who, why = cycle_mod.board_reason(a_verdict(
        failure="quantitative veto (DEFER)",
        quant={"rationale": "Ladder reports a fee larger than the notional."}))
    assert who == "quant"
    assert "notional" in why


def test_a_coach_veto_is_attributed_to_the_coach():
    who, why = cycle_mod.board_reason(a_verdict(
        failure="coach veto (REJECT)",
        coach={"rationale": "Formation mismatch does not support a draw."}))
    assert who == "coach" and "Formation" in why


def test_a_coach_that_only_asked_for_reruns_still_gives_a_reason():
    """DEFER with no prose must not surface as an empty cell."""
    who, why = cycle_mod.board_reason(a_verdict(
        failure="coach requires a rerun or deferred",
        coach={"rationale": "", "required_reruns": ["confirm the keeper",
                                                    "confirm the back four"]}))
    assert who == "coach"
    assert "keeper" in why


def test_a_judge_decision_carries_its_decisive_reason():
    who, why = cycle_mod.board_reason(a_verdict(
        action="PAPER_PLACE_LIMIT",
        judge={"decisive_reason": "Edge survives the adverse-selection reserve."}))
    assert who == "judge" and "reserve" in why


def test_a_verdict_with_nothing_recorded_says_so_rather_than_blank():
    who, why = cycle_mod.board_reason(a_verdict())
    assert who == "board" and why.strip()


def test_the_reason_reaches_the_boarded_ledger_and_the_summary(tmp_path,
                                                               monkeypatch):
    """A DEFER must be answerable later without the transcript, which does
    not survive the runner."""
    _patch_league(monkeypatch, tmp_path)

    def veto(cases, fixture, **kw):
        return {"failed_closed": False, "decisions": {},
                "failure": "coach veto (REJECT)",
                "quant": {}, "judge": None,
                "coach": {"verdict": "REJECT",
                          "rationale": "Keeper ruled out an hour before kick-off."}}

    monkeypatch.setattr(cycle_mod, "run_board_slate", veto)
    stats = run_cycle(state_path=str(tmp_path / "portfolio.json"),
                      providers=[FakeProvider()], verbose=False, fill_probes={})

    assert stats["board_run"] == 1
    assert stats["orders_submitted"] == 0
    decision = stats["board_decisions"][0]
    assert decision["decided_by"] == "coach"
    assert "Keeper" in decision["reason"]
    assert decision["home"] and decision["away"]

    portfolio = PaperPortfolio.load(str(tmp_path / "portfolio.json"))
    stored = next(iter(portfolio.boarded.values()))
    assert "Keeper" in stored["reason"], "must survive a save/reload"

    rendered = render_summary(stats)
    assert "Board decisions this cycle" in rendered
    assert "Keeper" in rendered


def test_the_summary_shows_the_reason_without_dumping_the_transcript(tmp_path,
                                                                     monkeypatch):
    """The summary goes to a PUBLIC Action log. One line, never the findings."""
    _patch_league(monkeypatch, tmp_path)
    secret = "SOURCE-URL-THAT-MUST-NOT-APPEAR"

    def veto(cases, fixture, **kw):
        return {"failed_closed": False, "decisions": {},
                "failure": "coach requires a rerun or deferred",
                "quant": {}, "judge": None,
                "coach": {"verdict": "DEFER", "rationale": "x" * 900,
                          "findings": [{"text": secret}],
                          "sources": [secret]}}

    monkeypatch.setattr(cycle_mod, "run_board_slate", veto)
    stats = run_cycle(state_path=str(tmp_path / "portfolio.json"),
                      providers=[FakeProvider()], verbose=False, fill_probes={})
    rendered = render_summary(stats)
    assert secret not in rendered
    assert len(max(rendered.splitlines(), key=len)) < 400
    # A pipe in a rationale must not break the markdown table.
    rows = [line for line in rendered.splitlines()
            if line.startswith("| ") and " v " in line]
    assert rows and all(line.count("|") == 8 for line in rows)


def test_paper_board_transcripts_do_not_land_in_the_real_money_directory():
    """`data/betting/` is live-trading state; paper reasoning is not that.

    It also has to be somewhere the workflow uploads, or the explanation dies
    with the runner -- which is what was happening.
    """
    assert cycle_mod.PAPER_BOARD_LOG == os.path.join(
        "data", "paper", "board_audit.jsonl")
    assert "betting" not in cycle_mod.PAPER_BOARD_LOG


def test_a_board_that_could_not_run_is_not_recorded_as_a_decision(tmp_path,
                                                                  monkeypatch):
    """A missing binary, an expired token, a timeout -- all fail closed to
    DEFER, correctly. But they are not decisions, and must not consume the
    fixture's one board or its one retry, or an outage spends the whole slate.
    """
    _patch_league(monkeypatch, tmp_path)

    def broke(cases, fixture, **kw):
        return {"decisions": {}, "failed_closed": True,
                "failure": "quant: member invocation failed: "
                           "[Errno 2] No such file or directory: 'claude'",
                "quant": None, "coach": None, "judge": None}

    monkeypatch.setattr(cycle_mod, "run_board_slate", broke)
    state = str(tmp_path / "portfolio.json")
    stats = run_cycle(state_path=state, providers=[FakeProvider()],
                      verbose=False, fill_probes={})

    assert stats["board_failures"] == 1
    assert stats["orders_submitted"] == 0
    assert stats["board_decisions"] == [], "a breakage is not a decision"

    # The fixture stays unboarded, so the next run picks it up again.
    portfolio = PaperPortfolio.load(state)
    assert portfolio.boarded == {}

    rendered = render_summary(stats)
    assert "did not run" in rendered
    assert "No such file or directory" in rendered


def test_a_genuine_defer_still_consumes_the_fixture(tmp_path, monkeypatch):
    """The mirror: a real deferral IS recorded, or the retry cap means nothing."""
    _patch_league(monkeypatch, tmp_path)

    def declined(cases, fixture, **kw):
        return {"decisions": {}, "failed_closed": False,
                "failure": "coach requires a rerun or deferred",
                "quant": {}, "judge": None,
                "coach": {"verdict": "DEFER", "rationale": "Keeper unconfirmed."}}

    monkeypatch.setattr(cycle_mod, "run_board_slate", declined)
    state = str(tmp_path / "portfolio.json")
    stats = run_cycle(state_path=state, providers=[FakeProvider()],
                      verbose=False, fill_probes={})
    assert stats["board_failures"] == 0
    assert len(stats["board_decisions"]) == 1
    portfolio = PaperPortfolio.load(state)
    assert len(portfolio.boarded) == 1
    assert next(iter(portfolio.boarded.values()))["attempts"] == 1


def test_a_decision_no_board_ever_made_is_reopened(tmp_path):
    """`claude` was never installed, so every board call fell closed to DEFER
    and each was written into the ledger as a real deferral. Eight fixtures
    were held back for a single late retry on the strength of a decision
    nobody made -- five of them kicking off within a day."""
    p = PaperPortfolio(path=str(tmp_path / "p.json"))
    p.boarded = {
        "real": {"action": "DEFER", "attempts": 1, "decided_by": "coach",
                 "reason": "Keeper unconfirmed.", "markets_considered": 112},
        "phantom": {"action": "DEFER", "attempts": 1, "decided_by": "quant",
                    "reason": "quant: member invocation failed: [Errno 2] "
                              "No such file or directory: 'claude'"},
        "prehistoric": {"action": "DEFER"},
    }
    removed = cycle_mod.purge_phantom_decisions(p)
    assert set(removed) == {"phantom", "prehistoric"}
    assert list(p.boarded) == ["real"], "a genuine deferral must survive"


def test_purging_is_idempotent(tmp_path):
    p = PaperPortfolio(path=str(tmp_path / "p.json"))
    p.boarded = {"phantom": {"action": "DEFER",
                             "reason": "member invocation failed"}}
    assert len(cycle_mod.purge_phantom_decisions(p)) == 1
    assert cycle_mod.purge_phantom_decisions(p) == []


def test_a_real_decision_of_any_kind_is_never_purged(tmp_path):
    """Purging a genuine PASS would let the board be re-asked until it agrees."""
    p = PaperPortfolio(path=str(tmp_path / "p.json"))
    p.boarded = {
        "passed": {"action": "PASS", "decided_by": "coach",
                   "reason": "Formation mismatch.", "markets_considered": 98},
        "bought": {"action": "PAPER_PLACE_LIMIT", "decided_by": "judge",
                   "reason": "Edge survives the reserve.",
                   "markets_considered": 98, "markets_approved": 12},
    }
    assert cycle_mod.purge_phantom_decisions(p) == []
    assert len(p.boarded) == 2


def test_the_purge_runs_before_selection_so_the_fixture_is_not_lost(tmp_path,
                                                                   monkeypatch):
    """Repairing after selection would cost the fixture its board on the very
    run that fixed it."""
    _patch_league(monkeypatch, tmp_path)
    _approve_board(monkeypatch)
    state = str(tmp_path / "portfolio.json")

    seed = PaperPortfolio(path=state)
    key = "premier_league|A|B|%s" % soon()[:10]
    seed.boarded = {key: {"action": "DEFER", "attempts": 1,
                          "decided_by": "quant",
                          "reason": "quant: member invocation failed"}}
    seed.save(state)

    stats = run_cycle(state_path=state, providers=[FakeProvider()],
                      verbose=False, fill_probes={})
    assert stats["phantom_decisions_purged"] == 1
    assert stats["fixtures_selected"] == 1, "reopened, and boarded this run"
    assert "Reopened" in render_summary(stats)


def test_one_fixture_cannot_consume_the_bankroll(tmp_path, monkeypatch):
    """Trading the whole ladder concentrates risk on one match -- those markets
    share a scoreline grid, so a wrong read loses on many at once.
    Diversification across markets on ONE fixture is mostly illusory."""
    _patch_league(monkeypatch, tmp_path)

    def approve_big(cases, fixture, **kw):
        return {"failed_closed": False, "failure": None,
                "coach": {"verdict": "ACCEPT", "rationale": "ok"},
                "decisions": {c.case_id: {
                    "case_id": c.case_id, "action": "PAPER_PLACE_LIMIT",
                    "limit_price_cents": c.max_limit_price_cents,
                    "contracts": 500.0,          # deliberately enormous
                    "decisive_reason": "ok"} for c in cases}}

    monkeypatch.setattr(cycle_mod, "run_board_slate", approve_big)
    state = str(tmp_path / "portfolio.json")
    stats = run_cycle(state_path=state, providers=[FakeProvider()],
                      verbose=False, fill_probes={})

    portfolio = PaperPortfolio.load(state)
    risk = sum(o.limit_price_cents * o.requested_size
               for o in portfolio.orders.values())
    ceiling = portfolio.starting_cash_cents * cycle_mod.MAX_FIXTURE_EXPOSURE_FRACTION
    assert risk <= ceiling, "one fixture must not exceed its exposure cap"
    assert stats.get("exposure_capped", 0) >= 0


def test_a_verdict_made_on_one_market_of_many_is_reopened(tmp_path):
    """Until the slate board a fixture was judged on a single market and the
    verdict closed the whole match. Those verdicts answer a question about one
    price, not about the fixture, so they cannot stand once the board sees the
    whole ladder."""
    p = PaperPortfolio(path=str(tmp_path / "p.json"))
    p.boarded = {
        "old_pass": {"action": "PASS", "attempts": 1, "decided_by": "quant",
                     "reason": "Negative EV at touch."},
        "old_defer": {"action": "DEFER", "attempts": 1, "decided_by": "coach",
                      "reason": "Keeper unconfirmed."},
        "slate_pass": {"action": "PASS", "attempts": 1, "decided_by": "quant",
                       "reason": "Negative EV across the ladder.",
                       "markets_considered": 112, "markets_approved": 0},
    }
    removed = cycle_mod.purge_phantom_decisions(p)
    assert set(removed) == {"old_pass", "old_defer"}
    assert list(p.boarded) == ["slate_pass"], \
        "a verdict that DID see the whole fixture must stand"


def test_a_fixture_larger_than_one_prompt_is_boarded_in_full(tmp_path,
                                                             monkeypatch):
    """Every market reaches the board, in batches, with ONE coach call."""
    _patch_league(monkeypatch, tmp_path)
    seen, coach_calls = [], []

    def batched(cases, fixture, coach_cache=None, **kw):
        seen.append(len(cases))
        key = (fixture["home"], fixture["away"])
        if coach_cache is not None and key not in coach_cache:
            coach_cache[key] = True
            coach_calls.append(key)
        return {"failed_closed": False, "failure": None,
                "coach": {"verdict": "ACCEPT", "rationale": "ok"},
                "decisions": {c.case_id: {
                    "case_id": c.case_id, "action": "PAPER_PLACE_LIMIT",
                    "limit_price_cents": c.max_limit_price_cents,
                    "contracts": 1.0, "decisive_reason": "ok"} for c in cases}}

    monkeypatch.setattr(cycle_mod, "run_board_slate", batched)
    monkeypatch.setattr(cycle_mod, "chunk_slate",
                        lambda slate, size=1: [[c] for c in slate])
    stats = run_cycle(state_path=str(tmp_path / "p.json"),
                      providers=[FakeProvider()], verbose=False,
                      fill_probes={})
    assert stats["markets_boarded"] == sum(seen), "no market skipped"
    assert len(coach_calls) == 1, "one coach call per fixture, not per batch"


def test_one_broken_batch_does_not_cost_the_others(tmp_path, monkeypatch):
    """A truncated reply fails its batch closed; the rest of the fixture must
    still be tradeable."""
    _patch_league(monkeypatch, tmp_path)
    calls = {"n": 0}

    def flaky(cases, fixture, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"decisions": {}, "failed_closed": True,
                    "failure": "judge: response truncated"}
        return {"failed_closed": False, "failure": None,
                "coach": {"verdict": "ACCEPT", "rationale": "ok"},
                "decisions": {c.case_id: {
                    "case_id": c.case_id, "action": "PAPER_PLACE_LIMIT",
                    "limit_price_cents": c.max_limit_price_cents,
                    "contracts": 1.0, "decisive_reason": "ok"} for c in cases}}

    monkeypatch.setattr(cycle_mod, "run_board_slate", flaky)
    monkeypatch.setattr(cycle_mod, "chunk_slate",
                        lambda slate, size=1: [[c] for c in slate])
    stats = run_cycle(state_path=str(tmp_path / "p.json"),
                      providers=[FakeProvider()], verbose=False,
                      fill_probes={})
    assert stats["board_failures"] == 1
    if stats["markets_boarded"] > 1:
        assert stats["orders_submitted"] >= 1, "surviving batches still trade"
