import pytest
import pandas as pd
import numpy as np
from algomlb.ml.prop_heads import PropCalibrator, PitcherPropHead, MarketAlignment


def test_target_labeling():
    head = PitcherPropHead("pitcher_strikeouts")
    actuals = pd.Series([5, 6, 7])
    lines = pd.Series([5.5, 6.5, 6.5])

    labels = head.generate_labels(actuals, lines)
    # 5 vs 5.5 -> 0 (Under)
    # 6 vs 6.5 -> 0 (Under)
    # 7 vs 6.5 -> 1 (Over)
    assert labels.tolist() == [0, 0, 1]


def test_prop_calibrator_logistic():
    # Synthetic data
    X_cal = pd.DataFrame(
        {"mc_p_over": [0.4, 0.6, 0.8, 0.2], "park_factor": [1.0, 1.1, 0.9, 1.0]}
    )
    y_cal = pd.Series([0, 1, 1, 0])

    calibrator = PropCalibrator(method="logistic")
    calibrator.train(X_cal, y_cal)

    probs = calibrator.predict_p_over(X_cal)
    assert len(probs) == 4
    assert all((p >= 0.0 and p <= 1.0) for p in probs)

    # Check that it somewhat learned the trend
    assert probs[2] > probs[3]

    metrics = calibrator.evaluate(X_cal, y_cal)
    assert "brier_score" in metrics
    assert "ece" in metrics


def test_prop_calibrator_isotonic():
    # Isotonic needs 1D input (first column)
    X_cal = pd.DataFrame({"mc_p_over": [0.1, 0.4, 0.7, 0.9]})
    y_cal = pd.Series([0, 0, 1, 1])

    calibrator = PropCalibrator(method="isotonic")
    calibrator.train(X_cal, y_cal)

    probs = calibrator.predict_p_over(X_cal)
    assert len(probs) == 4
    assert probs[0] <= probs[1] <= probs[2] <= probs[3]


def test_market_alignment_odds():
    # -110 odds implied prob is ~52.38%
    implied_neg = MarketAlignment.american_to_implied(-110)
    assert np.isclose(implied_neg, 0.5238, atol=1e-4)

    # +150 odds implied prob is 40.0%
    implied_pos = MarketAlignment.american_to_implied(150)
    assert np.isclose(implied_pos, 0.4000, atol=1e-4)


def test_kelly_criterion():
    # Scenario: +100 odds (2.0 decimal, b=1), True Prob = 55%
    # Full Kelly = (1 * 0.55 - 0.45) / 1 = 0.10 (10% of bankroll)
    # Quarter Kelly = 0.10 * 0.25 = 0.025 (2.5% of bankroll)
    stake = MarketAlignment.calculate_kelly_stake(
        calibrated_prob=0.55, american_odds=100, kelly_multiplier=0.25
    )
    assert np.isclose(stake, 0.025, atol=1e-4)

    # Scenario: Negative edge -> stake should be 0.0
    bad_stake = MarketAlignment.calculate_kelly_stake(
        calibrated_prob=0.45, american_odds=100
    )
    assert bad_stake == 0.0


def test_name_resolution():
    pid = MarketAlignment.resolve_player_id("Shohei Ohtani  ")
    assert pid == 660271

    with pytest.raises(ValueError):
        MarketAlignment.resolve_player_id("Unknown Player")
