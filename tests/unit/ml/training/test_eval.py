import numpy as np
import pytest
from algomlb.ml.training.backtester import calculate_20bin_ece


def test_calculate_20bin_ece_perfect():
    """Perfectly calibrated: y_prob == y_true (in aggregate)."""
    # Create 1000 samples evenly distributed
    y_prob = np.linspace(0.01, 0.99, 1000)
    # y_true generated from the probabilities to be 'perfectly calibrated' on average
    y_true = (np.random.random(1000) < y_prob).astype(int)

    ece = calculate_20bin_ece(y_true, y_prob)
    # Should be very low, but not zero due to random sampling variance
    assert ece < 0.1


def test_calculate_20bin_ece_poor():
    """Poorly calibrated: model predicts 0.9, actual win rate is 0.1."""
    y_prob = np.array([0.9] * 100)
    y_true = np.array([0] * 90 + [1] * 10)  # 10% win rate

    ece = calculate_20bin_ece(y_true, y_prob)
    # Expected: |0.9 - 0.1| = 0.8
    assert pytest.approx(ece, abs=0.01) == 0.8


def test_calculate_20bin_ece_empty():
    assert calculate_20bin_ece(np.array([]), np.array([])) == 0.0


def test_calculate_20bin_ece_edge_cases():
    """Check probabilities at 0.0 and 1.0."""
    y_prob = np.array([0.0, 1.0])
    y_true = np.array([0, 1])
    ece = calculate_20bin_ece(y_true, y_prob)
    assert ece == 0.0
