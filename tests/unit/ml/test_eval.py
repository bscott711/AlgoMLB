import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from algomlb.ml.eval import (
    compute_fold_metrics,
    compute_calibration_bins,
    compute_per_game_eval,
    persist_eval_results,
)


def test_compute_fold_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_fold_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    assert "log_loss" in metrics
    assert "brier" in metrics


def test_compute_calibration_bins():
    y_true = np.array([0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.15, 0.8, 0.85, 0.5])
    # 2 bins: [0, 0.5), [0.5, 1.0]
    df = compute_calibration_bins(y_true, y_prob, n_bins=2)
    assert len(df) == 2
    assert df.loc[0, "n_samples"] == 2  # 0.1, 0.15
    assert df.loc[1, "n_samples"] == 3  # 0.5, 0.8, 0.85


def test_compute_calibration_bins_empty():
    y_true = np.array([1])
    y_prob = np.array([0.9])
    df = compute_calibration_bins(y_true, y_prob, n_bins=2)
    # Bin 0 ([0, 0.5)) is empty
    assert df.loc[0, "n_samples"] == 0
    assert df.loc[0, "obs_rate"] == 0.0


def test_compute_per_game_eval():
    games_df = pd.DataFrame(
        {
            "game_pk": [1, 2],
            "game_date": ["2023-01-01", "2023-01-02"],
            "home_team": ["NYY", "BOS"],
            "away_team": ["BOS", "NYY"],
        },
        index=[0, 1],
    )
    y_true = pd.Series([1, 0], index=[0, 1])
    y_prob = np.array([0.9, 0.4])

    out = compute_per_game_eval(games_df, y_true, y_prob)
    assert out.loc[0, "game_pk"] == 1
    assert out.loc[0, "confidence_tier"] == "high"
    assert out.loc[1, "confidence_tier"] == "medium"
    assert out.loc[0, "correct"] == 1
    assert out.loc[1, "correct"] == 1


def test_persist_eval_results():
    mock_engine = MagicMock()
    metrics = {"accuracy": 0.8, "auc": 0.85, "log_loss": 0.4, "brier": 0.15}
    cal_bins = pd.DataFrame(
        [
            {
                "bin_index": 0,
                "bin_lower": 0.0,
                "bin_upper": 0.1,
                "pred_mean": 0.05,
                "obs_rate": 0.04,
                "n_samples": 10,
            }
        ]
    )

    with (
        patch("algomlb.db.models.UraniumEvalHistoryORM"),
        patch("algomlb.db.models.UraniumCalibrationBinORM"),
        patch("algomlb.ml.eval.pg_insert") as mock_insert,
    ):
        # Setup mock statement chain
        mock_stmt = MagicMock()
        mock_insert.return_value = mock_stmt
        mock_stmt.values.return_value = mock_stmt
        mock_stmt.on_conflict_do_update.return_value = mock_stmt

        persist_eval_results(
            mock_engine, "v1", 2023, 2019, 2022, 100, metrics, cal_bins
        )
        assert mock_engine.begin.called
        assert mock_insert.called
        assert mock_stmt.on_conflict_do_update.called
