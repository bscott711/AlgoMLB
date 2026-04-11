import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from algomlb.cli.ml import app

runner = CliRunner()


@pytest.fixture
def mock_session_factory():
    with patch("algomlb.cli.ml.get_session_factory") as mock:
        factory = mock.return_value
        factory.kw = {"bind": MagicMock()}
        yield mock


def test_cli_ml_fetch_history(mock_session_factory):
    with patch("algomlb.cli.ml.fetch_eval_history") as mock_fetch:
        mock_fetch.return_value = pd.DataFrame([{"id": 1}])

        result = runner.invoke(
            app, ["fetch-history", "--target", "home_win"], obj={"agent_mode": False}
        )
        assert result.exit_code == 0
        mock_fetch.assert_called()


def test_cli_ml_tune(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.run_optuna_study") as mock_opt,
    ):
        mock_load.return_value = {
            "games": pd.DataFrame([{"game_pk": 1, "game_date": "2023-01-01"}]),
            "pitcher_gold": pd.DataFrame(),
            "lineups": pd.DataFrame(),
            "batter_gold": pd.DataFrame(),
            "elo": pd.DataFrame(),
            "pythag": pd.DataFrame(),
            "re24": pd.DataFrame(),
        }
        mock_pipe.return_value.build_uranium_matrix.return_value = (
            pd.DataFrame({"feat": [1, 2]}, index=[0, 1]),
            pd.Series([1, 0], name="home_win"),
        )
        mock_opt.return_value.best_params = {"max_depth": 5}

        result = runner.invoke(
            app,
            ["tune", "--target", "home_win", "--trials", "1"],
            obj={"agent_mode": False},
        )
        assert result.exit_code == 0
        mock_opt.assert_called()


def test_cli_ml_backtest(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.OOFAccumulator") as mock_acc,
        patch("algomlb.cli.ml.load_optimized_params"),
        patch("algomlb.cli.ml.persist_eval_results"),
    ):
        mock_load.return_value = {
            "games": pd.DataFrame([{"game_pk": 1, "game_date": "2023-01-01"}]),
            "pitcher_gold": pd.DataFrame(),
            "lineups": pd.DataFrame(),
            "batter_gold": pd.DataFrame(),
            "elo": pd.DataFrame(),
            "pythag": pd.DataFrame(),
            "re24": pd.DataFrame(),
        }
        mock_pipe.return_value.build_uranium_matrix.return_value = (
            pd.DataFrame({"feat": [1, 2]}, index=[0, 1]),
            pd.Series([1, 0], name="home_win"),
        )
        # Harmony: oof_df must contain TWO CLASSES (0 and 1) for metrics calculation
        mock_acc.return_value.run_backtest.return_value = pd.DataFrame(
            {
                "home_win": [1, 0],
                "p_model": [0.8, 0.2],
                "game_date": ["2023-01-01", "2023-01-02"],
            }
        )

        result = runner.invoke(
            app, ["backtest", "--target", "home_win"], obj={"agent_mode": False}
        )
        assert result.exit_code == 0
        mock_acc.return_value.run_backtest.assert_called()


def test_cli_ml_explain(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.MLBModel"),
        patch("algomlb.cli.ml.compute_global_shap") as mock_shap,
        patch("algomlb.cli.ml.persist_global_shap"),
    ):
        mock_load.return_value = {
            "games": pd.DataFrame([{"game_pk": 1, "game_date": "2023-01-01"}]),
            "pitcher_gold": pd.DataFrame(),
            "lineups": pd.DataFrame(),
            "batter_gold": pd.DataFrame(),
            "elo": pd.DataFrame(),
            "pythag": pd.DataFrame(),
            "re24": pd.DataFrame(),
        }
        mock_pipe.return_value.build_uranium_matrix.return_value = (
            pd.DataFrame({"feat": [1]}, index=[0]),
            pd.Series([1], name="home_win"),
        )
        mock_shap.return_value = (
            None,
            pd.DataFrame([{"feature_name": "feat", "mean_abs_shap": 0.5}]),
        )

        result = runner.invoke(
            app, ["explain", "--target", "home_win"], obj={"agent_mode": False}
        )
        assert result.exit_code == 0
        mock_shap.assert_called()


def test_cli_ml_elo_backfill(mock_session_factory):
    with patch("algomlb.ml.elo.backfill_team_elo_history") as mock_backfill:
        result = runner.invoke(app, ["elo-backfill"], obj={"agent_mode": False})
        assert result.exit_code == 0
        assert mock_backfill.called


def test_cli_ml_decouple(mock_session_factory):
    with patch("algomlb.cli.ml.run_decoupler_pipeline") as mock_run:
        result = runner.invoke(app, ["decouple"], obj={"agent_mode": False})
        assert result.exit_code == 0
        assert mock_run.called


def test_cli_ml_tune_agent_mode(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.run_optuna_study") as mock_opt,
        patch("algomlb.cli.ml.emit_agent_result") as mock_emit,
    ):
        mock_load.return_value = {
            "games": pd.DataFrame([{"game_pk": 1, "game_date": "2023-01-01"}]),
            "pitcher_gold": pd.DataFrame(),
            "lineups": pd.DataFrame(),
            "batter_gold": pd.DataFrame(),
            "elo": pd.DataFrame(),
            "pythag": pd.DataFrame(),
            "re24": pd.DataFrame(),
        }
        mock_pipe.return_value.build_uranium_matrix.return_value = (
            pd.DataFrame({"feat": [1, 2]}, index=[0, 1]),
            pd.Series([1, 0], name="home_win"),
        )
        mock_opt.return_value.best_params = {"max_depth": 5}

        result = runner.invoke(
            app,
            ["tune", "--target", "home_win", "--trials", "1"],
            obj={"agent_mode": True},
        )
        assert result.exit_code == 0
        assert mock_emit.called


def test_evaluate_and_report_logic(mock_session_factory):
    from algomlb.cli.ml import _evaluate_and_report

    with (
        patch("algomlb.cli.ml.compute_fold_metrics") as mock_metrics,
        patch("algomlb.cli.ml.compute_calibration_bins"),
        patch("algomlb.cli.ml.persist_eval_results"),
    ):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        X_test = pd.DataFrame({"feat": [1, 2]})
        y_test = pd.Series([1, 0])

        mock_metrics.return_value = {
            "accuracy": 0.8,
            "auc": 0.9,
            "log_loss": 0.3,
            "brier": 0.1,
        }

        import datetime

        metrics = _evaluate_and_report(
            mock_model,
            X_test,
            y_test,
            "pa_outcome",
            datetime.date(2023, 1, 1),
            MagicMock(),
            "v1.0",
            2021,
            2023,
        )
        assert metrics["accuracy"] == 0.8
