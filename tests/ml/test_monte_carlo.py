import pandas as pd
import numpy as np
from algomlb.ml.monte_carlo import (
    GameState,
    BullpenManager,
    SimulationEngine,
    PropAggregator,
)


def test_base_advancement():
    """Verify core baseball logic for forced and unforced advancements."""
    game = GameState()

    # Hit a double, batter on 2nd
    game.process_event("double")
    assert game.bases == [False, True, False]

    # Walk, runners on 1st and 2nd
    game.process_event("walk")
    assert game.bases == [True, True, False]

    # Single. Runner on 3rd scores (none here). Runner on 2nd to 3rd. Runner on 1st to 2nd. Batter to 1st.
    runs = game.process_event("single")
    assert runs == 0
    assert game.away_score == 0  # Top of the 1st
    assert game.bases == [True, True, True]

    # Home Run. 4 runs score (3 on base + batter). Bases clear.
    runs = game.process_event("home_run")
    assert runs == 4
    assert game.away_score == 4
    assert game.bases == [False, False, False]


def test_bullpen_leverage():
    """Verify bullpen manager correctly identifies high leverage scenarios."""
    df = pd.DataFrame(
        {
            "pitcher_id": [101, 102, 103],
            "team_id": [1, 1, 1],
            "role": ["long_rel", "setup", "closer"],
            "availability_score": [0.9, 0.8, 0.95],
        }
    )
    manager = BullpenManager(bullpen_df=df, hook_profiles=pd.DataFrame())
    game = GameState(inning=8, home_score=3, away_score=2)  # Home up by 1 in the 8th

    selected_arm = manager.select_arm(team_id=1, game=game)
    # Should select closer (103) because it's high leverage and availability is highest
    assert selected_arm == 103


def test_engine_reproducibility():
    """Verify the Monte Carlo engine is completely deterministic given a seed."""
    engine_1 = SimulationEngine(pa_model=None, bullpen_manager=None, seed=42)
    engine_2 = SimulationEngine(pa_model=None, bullpen_manager=None, seed=42)

    results_1 = engine_1.run_trials(trials=100)
    results_2 = engine_2.run_trials(trials=100)

    assert results_1 == results_2


def test_prop_aggregator():
    """Verify the aggregator calculates betting probabilities correctly."""
    # Dummy array representing Strikeouts for a pitcher across 10 trials
    trials = np.array([4, 5, 6, 7, 8, 5, 6, 7, 8, 9])

    # K's Over/Under 6.5
    probs = PropAggregator.calculate_over_under(trials, 6.5)
    assert probs["p_over"] == 0.50  # 7,8,7,8,9 are > 6.5
    assert probs["p_under"] == 0.50  # 4,5,6,5,6 are < 6.5

    # Summarize stats
    stats = PropAggregator.summarize_stat(trials)
    assert stats["median"] == 6.5
