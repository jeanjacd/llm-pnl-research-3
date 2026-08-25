"""End-to-end paper cycle: claim probabilities, orchestration, idempotency and
sanitised reporting. Fully offline -- fake providers, fake board."""
import datetime as dt
import os

import pytest

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

    def __init__(self, ask=30, size=500.0):
        self.ask, self.size = ask, size
        self.calls = 0

    def discover(self, spec, **kw):
        self.calls += 1
        teams = kw.get("_teams") or ("A", "B")
        return [MarketInstrument(
            venue=self.venue, instrument_id="INST-1", kind=KIND_BINARY,
            title="A vs B", legs=(Leg.build("home_win", "ref", home=teams[0],
                                            away=teams[1],
                                            kickoff_utc=soon()),),
            settles_on_regulation=True, league_id=spec.league_id,
            kickoff_utc=soon(), fee_model={"venue": "kalshi"},
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

    def fake_board(case, fixture, **kw):
        return {"case_id": case.case_id, "action": "PAPER_BUY_NOW",
                "judge": {"limit_price_cents": case.max_limit_price_cents,
                          "contracts": 5.0}}
    monkeypatch.setattr(cycle_mod, "run_board", fake_board)


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
    monkeypatch.setattr(cycle_mod, "run_board",
                        lambda case, fixture, **kw: {"case_id": case.case_id,
                                                     "action": "DEFER"})
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
