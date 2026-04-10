import pandas as pd
from datetime import datetime, timedelta
from algomlb.ml.training.backtester import TimeSeriesSplitter, TimeSplitConfig


def test_timeseries_splitter_no_overlap():
    """Assert that TimeSeriesSplitter never produces overlapping train/test sets."""
    # Create fake data over 3 months
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(120)]
    df = pd.DataFrame({"game_date": dates, "target": [0] * 120, "feat": [1] * 120})

    # 60 day train, 15 day test
    config = TimeSplitConfig(train_window_days=60, test_window_days=15)
    splitter = TimeSeriesSplitter(config)

    folds = splitter.split(df)

    assert len(folds) >= 1

    for train_df, test_df in folds:
        train_max = train_df["game_date"].max()
        test_min = test_df["game_date"].min()

        # Fundamental check
        assert train_max < test_min, (
            f"Overlap detected: Train max {train_max} vs Test min {test_min}"
        )

        # Assert test window size
        test_duration = (
            test_df["game_date"].max() - test_df["game_date"].min()
        ).days + 1
        assert test_duration <= 15


def test_timeseries_splitter_empty():
    splitter = TimeSeriesSplitter()
    assert splitter.split(pd.DataFrame()) == []


def test_timeseries_splitter_insufficient_data():
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({"game_date": dates})

    config = TimeSplitConfig(train_window_days=20, test_window_days=5)
    splitter = TimeSeriesSplitter(config)

    # Train window (20) + Test window (5) = 25 days needed. Only 10 provided.
    assert len(splitter.split(df)) == 0
