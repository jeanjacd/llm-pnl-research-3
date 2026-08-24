"""Loader: competition-tier classification and 2026 state extraction."""
import pandas as pd

from wc2026.data.loader import classify_tier, tournament_state


def test_tier_classification():
    assert classify_tier("FIFA World Cup") == "world_cup"
    assert classify_tier("FIFA World Cup qualification") == "qualifier"
    assert classify_tier("UEFA Euro qualification") == "qualifier"      # qualifier wins
    assert classify_tier("UEFA Euro") == "continental"
    assert classify_tier("Copa América") == "continental"
    assert classify_tier("UEFA Nations League") == "nations_league"
    assert classify_tier("Friendly") == "friendly"
    assert classify_tier("Some Random Cup") == "friendly"


def test_tournament_state_split():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-15", "2026-06-16", "2026-07-01"]),
        "home_team": ["A", "B", "C"], "away_team": ["X", "Y", "Z"],
        "home_score": [1.0, 2.0, None], "away_score": [0.0, 2.0, None],
        "tournament": ["FIFA World Cup"] * 3, "neutral": [True, True, True],
    })
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    st = tournament_state(df)
    assert st.n_played == 2 and st.n_upcoming == 1
    assert set(st.participants) == {"A", "B", "C", "X", "Y", "Z"}
    assert st.as_of == pd.Timestamp("2026-06-16")
