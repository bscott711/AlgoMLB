import pytest
import pandas as pd
import numpy as np
from algomlb.ml.component_models import (
    PAOutcomeModel,
    BayesianShrinkage,
    ComponentEvaluator,
    TemporalLeakageError,
    BatterPreGameState,
    PitcherPreGameState,
)


def test_bayesian_shrinkage():
    # Weight of 100 PA's
    shrinkage = BayesianShrinkage(prior_weight=100)
    df = pd.DataFrame({"strikeouts": [10, 50, 0], "pa": [20, 100, 5]})

    # Base k_pct is 0.225
    # Row 0: (10 + 100*0.225) / (20 + 100) = 32.5 / 120 = 0.270833
    result = shrinkage.apply_shrinkage(df, "strikeouts", "pa", "k_pct")

    assert np.isclose(result.iloc[0], 0.270833, atol=1e-3)
    # Row 2 (0 K's in 5 PA's) should be pulled heavily up towards 0.225
    # (0 + 100*0.225) / (5 + 100) = 22.5 / 105 = 0.214285
    assert np.isclose(result.iloc[2], 0.2142, atol=1e-3)


def test_pa_outcome_model_probs_sum_to_one():
    # Setup mock data covering all 8 classes
    X = pd.DataFrame({"bat_speed": [70.1, 72.2, 68.3, 75.4, 71.0, 69.9, 76.5, 73.2]})
    y = pd.Series(
        [
            "strikeout",
            "walk",
            "hbp",
            "single",
            "double",
            "triple",
            "home_run",
            "out_in_play",
        ]
    )

    model = PAOutcomeModel(n_estimators=2, max_depth=2)
    model.train(X, y)

    batter = BatterPreGameState(batter_id=1, bat_speed=72.0)
    pitcher = PitcherPreGameState(pitcher_id=2)

    probs = model.predict_matchup(batter, pitcher, context={})

    # Verify probability distribution constraints
    assert probs.shape == (1, 8)
    assert np.isclose(np.sum(probs), 1.0)

    # Verify class mapping is exposed to the MC engine
    assert len(model.class_mapping_) == 8
    assert "home_run" in model.class_mapping_.values()


def test_strict_temporal_leakage():
    evaluator = ComponentEvaluator()

    train_df = pd.DataFrame(
        {
            "game_date": ["2023-05-01", "2023-05-15"],
            "f1": [1.0, 2.0],
            "y": ["strikeout", "walk"],
        }
    )

    # Test data overlaps/precedes train data (Min date is 05-10, Train max is 05-15)
    test_df = pd.DataFrame(
        {
            "game_date": ["2023-05-10", "2023-06-01"],
            "f1": [3.0, 4.0],
            "y": ["single", "hbp"],
        }
    )

    with pytest.raises(TemporalLeakageError, match="Temporal Leakage Detected"):
        evaluator.evaluate_walk_forward(
            None, train_df, test_df, ["f1"], "y", "game_date"
        )
