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
    with patch("algomlb.cli.ml.HistoricalDataLoader") as mock_loader:
        mock_loader.return_value.fetch_pitching_stats.return_value = pd.DataFrame(
            [{"id": 1}]
        )
        mock_loader.return_value.fetch_team_batting.return_value = pd.DataFrame(
            [{"id": 2}]
        )

        result = runner.invoke(
            app, ["fetch-history", "--start-year", "2023"], obj={"agent_mode": False}
        )
        assert result.exit_code == 0
        mock_loader.return_value.fetch_pitching_stats.assert_called()


def test_cli_ml_train(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.MLBModel") as mock_model_cls,
        patch("algomlb.cli.ml._evaluate_and_report"),
    ):
        mock_load.return_value = {
            "games": pd.DataFrame(
                [
                    {"game_pk": 1, "game_date": "2023-01-01"},
                    {"game_pk": 2, "game_date": "2023-02-01"},
                ]
            ),
            "pitcher_gold": pd.DataFrame(columns=["player_id", "season"]),
            "lineups": pd.DataFrame(columns=["game_pk", "player_id"]),
            "batter_gold": pd.DataFrame(columns=["player_id", "season"]),
            "elo": pd.DataFrame(columns=["game_pk"]),
            "pythag": pd.DataFrame(columns=["game_pk"]),
            "re24": pd.DataFrame(columns=["player_id", "date"]),
        }
        mock_pipe.return_value.build_uranium_matrix.return_value = (
            pd.DataFrame({"feat": [1, 2]}, index=[0, 1]),
            pd.Series([1, 0]),
        )

        result = runner.invoke(
            app, ["train", "--test-year", "2023"], obj={"agent_mode": False}
        )
        assert result.exit_code == 0
        assert mock_model_cls.return_value.train.called


def test_cli_ml_elo_backfill(mock_session_factory):
    with patch("algomlb.ml.elo.backfill_team_elo_history") as mock_backfill:
        result = runner.invoke(app, ["elo-backfill"], obj={"agent_mode": False})
        assert result.exit_code == 0
        assert mock_backfill.called


def test_cli_ml_decouple(mock_session_factory):
    with patch("algomlb.cli.ml.run_decoupler_pipeline") as mock_run:
        result = runner.invoke(app, ["decouple", "full"], obj={"agent_mode": False})
        assert result.exit_code == 0
        mock_run.assert_called_with("full", "v1")


def test_cli_ml_optimize(mock_session_factory):
    # Patch the source since it's locally imported
    # We also MUST patch 'open' and 'Path.mkdir' to prevent disk writes to calibration data
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.ml.hyperopt.build_fold_data", return_value=[(1, 2, 3, 4)]),
        patch("algomlb.ml.hyperopt.optimize_model") as mock_opt,
        patch("pathlib.Path.mkdir"),  # Prevent directory creation
        patch("builtins.open", MagicMock()),  # Prevent file writing
    ):
        mock_load.return_value = {
            "games": pd.DataFrame(columns=["game_pk", "year"]),
            "pitcher_gold": pd.DataFrame(columns=["season"]),
            "batter_gold": pd.DataFrame(columns=["season"]),
            "lineups": pd.DataFrame(columns=["game_pk"]),
            "elo": pd.DataFrame(columns=["game_pk"]),
            "pythag": pd.DataFrame(columns=["game_pk"]),
            "re24": pd.DataFrame(),
        }
        mock_opt.return_value = ({}, MagicMock())

        result = runner.invoke(
            app, ["optimize", "--n-trials", "2"], obj={"agent_mode": False}
        )
        assert result.exit_code == 0
        assert mock_opt.called


def test_cli_ml_walk_forward(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.MLBModel"),
        patch(
            "algomlb.cli.ml._evaluate_and_report",
            return_value={"accuracy": 0.6, "auc": 0.7, "log_loss": 0.5},
        ) as mock_eval,
    ):
        mock_load.return_value = {
            "games": pd.DataFrame(
                [
                    {"game_pk": 1, "game_date": "2023-01-01", "year": 2023},
                    {"game_pk": 2, "game_date": "2024-01-01", "year": 2024},
                ]
            ),
            "pitcher_gold": pd.DataFrame({"season": [2023, 2024]}),
            "lineups": pd.DataFrame(columns=["game_pk"]),
            "batter_gold": pd.DataFrame(columns=["season"]),
            "elo": pd.DataFrame(columns=["game_pk"]),
            "pythag": pd.DataFrame(columns=["game_pk"]),
            "re24": pd.DataFrame(),
        }
        mock_pipe.return_value.build_uranium_matrix.return_value = (
            pd.DataFrame({"feat": [1, 2]}, index=[0, 1]),
            pd.Series([1, 0]),
        )

        result = runner.invoke(
            app,
            ["walk-forward", "--start-year", "2023", "--end-year", "2024"],
            obj={"agent_mode": True},
        )
        assert result.exit_code == 0
        assert mock_eval.called


def test_cli_ml_train_agent_mode(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.MLBModel"),
        patch("algomlb.cli.ml._evaluate_and_report", return_value={"accuracy": 0.8}),
        patch("algomlb.cli.ml.emit_agent_result") as mock_emit,
        patch("pathlib.Path.mkdir"),
        patch("builtins.open"),
    ):
        mock_load.return_value = {
            "games": pd.DataFrame(
                [
                    {
                        "game_pk": 1,
                        "game_date": "2023-01-01",
                        "home_team": "NYY",
                        "away_team": "BOS",
                        "home_pitcher_id": 1,
                        "away_pitcher_id": 2,
                        "home_score": 5,
                        "away_score": 3,
                    }
                ]
            ),
            "pitcher_gold": pd.DataFrame({"player_id": [1], "season": [2022]}),
            "lineups": pd.DataFrame(),
            "batter_gold": pd.DataFrame(),
            "elo": pd.DataFrame(),
            "pythag": pd.DataFrame(),
            "re24": pd.DataFrame(),
        }
        mock_pipe.return_value.build_uranium_matrix.return_value = (
            pd.DataFrame({"f": [1]}, index=[0]),
            pd.Series([1]),
        )

        result = runner.invoke(app, ["train"], obj={"agent_mode": True})
        assert result.exit_code == 0
        mock_emit.assert_called()


def test_cli_ml_optimize_errors(mock_session_factory):
    # Test year count error
    result = runner.invoke(
        app, ["optimize", "--start-year", "2023", "--end-year", "2023"]
    )
    assert result.exit_code == 1

    # Test no folds built
    with (
        patch("algomlb.cli.ml._load_ml_data"),
        patch("algomlb.ml.hyperopt.build_fold_data", return_value=[]),
    ):
        result = runner.invoke(
            app, ["optimize", "--start-year", "2022", "--end-year", "2023"]
        )
        assert result.exit_code == 1


def test_cli_ml_walk_forward_errors(mock_session_factory):
    # Year count error
    result = runner.invoke(
        app, ["walk-forward", "--start-year", "2023", "--end-year", "2023"]
    )
    assert result.exit_code == 1


def test_load_ml_data_re24_path(mock_session_factory):
    from algomlb.cli.ml import _load_ml_data

    with patch("pandas.read_sql") as mock_read:
        # games, pitcher, lineups, batter, elo, retrosheet
        mock_read.side_effect = [
            pd.DataFrame(
                {
                    "game_pk": [1],
                    "game_date": ["2023-04-01"],
                    "home_team": ["NYY"],
                    "away_team": ["BOS"],
                    "home_pitcher_id": [1],
                    "away_pitcher_id": [2],
                    "home_score": 5,
                    "away_score": 3,
                }
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "game_id": [1],
                    "date": ["2023-04-01"],
                    "pa_flag": [1],
                    "runs": [1],
                    "batter_id": [1],
                    "pitcher_id": [2],
                    "bat_team": ["NYY"],
                    "pit_team": ["BOS"],
                    "inning": [1],
                    "top_bot": [0],
                    "outs_pre": [0],
                    "outs_post": [1],
                    "br1_pre": [0],
                    "br2_pre": [0],
                    "br3_pre": [0],
                    "br1_post": [1],
                    "br2_post": [0],
                    "br3_post": [0],
                }
            ),
        ]
        with (
            patch("algomlb.ml.sabermetrics.compute_re24_per_pa"),
            patch("algomlb.ml.sabermetrics.compute_rolling_re24") as mock_re2,
        ):
            _load_ml_data(MagicMock(), "2023")
            assert mock_re2.called


def test_evaluate_and_report_success(mock_session_factory):
    from algomlb.cli.ml import _evaluate_and_report

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
    mock_model.get_feature_importance.return_value = pd.DataFrame(
        {"feature": ["f1"], "importance": [0.5]}
    )

    with (
        patch(
            "algomlb.ml.eval.compute_fold_metrics",
            return_value={"accuracy": 0.8, "auc": 0.8, "log_loss": 0.5},
        ),
        patch("algomlb.ml.eval.compute_calibration_bins"),
        patch("algomlb.ml.eval.persist_eval_results") as mock_persist,
    ):
        _evaluate_and_report(
            mock_model,
            pd.DataFrame({"f1": [1]}),
            pd.Series([1]),
            2023,
            MagicMock(),
            "v1",
            2019,
            2023,
        )
        assert mock_persist.called


def test_cli_ml_fetch_history_agent_mode(mock_session_factory):
    with (
        patch("algomlb.cli.ml.HistoricalDataLoader") as mock_loader,
        patch("algomlb.cli.ml.emit_agent_result") as mock_emit,
    ):
        mock_loader.return_value.fetch_pitching_stats.return_value = pd.DataFrame(
            {"id": [1]}
        )
        mock_loader.return_value.fetch_team_batting.return_value = pd.DataFrame(
            {"id": [2]}
        )

        result = runner.invoke(
            app, ["fetch-history", "--start-year", "2023"], obj={"agent_mode": True}
        )
        assert result.exit_code == 0
        mock_emit.assert_called_once()


def test_cli_ml_build_registry(mock_session_factory):
    with patch("algomlb.ml.registry.build_manager_registry") as mock_build:
        result = runner.invoke(app, ["build-registry"])
        assert result.exit_code == 0
        assert mock_build.called


def test_cli_ml_hook_backfill(mock_session_factory):
    with patch("algomlb.ml.hooks.backfill_hook_events") as mock_backfill:
        result = runner.invoke(app, ["hook-backfill"])
        assert result.exit_code == 0
        assert mock_backfill.called


def test_cli_ml_train_empty_matrix(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
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
            pd.DataFrame(),
            pd.Series(),
        )

        result = runner.invoke(app, ["train"], obj={"agent_mode": False})
        assert result.exit_code == 1


def test_load_ml_data_logic(mock_session_factory):
    from algomlb.cli.ml import _load_ml_data

    mock_engine = MagicMock()
    # Mocking read_sql to return simple dataframes
    with patch("pandas.read_sql") as mock_read:
        # 1. games, 2. pitcher_gold, 3. lineups, 4. batter_gold, 5. elo (fail), 6. retrosheet (fail)
        mock_read.side_effect = [
            pd.DataFrame(
                {
                    "game_pk": [1],
                    "game_date": ["2023-04-01"],
                    "home_team": ["NYY"],
                    "away_team": ["BOS"],
                    "home_pitcher_id": [1],
                    "away_pitcher_id": [2],
                    "home_score": [5],
                    "away_score": [3],
                }
            ),
            pd.DataFrame({"season": [2023]}),
            pd.DataFrame({"game_pk": [1]}),
            pd.DataFrame({"season": [2023]}),
            Exception("No table"),  # Elo
            Exception("No table"),  # Retrosheet
        ]

        data = _load_ml_data(mock_engine, "2023")
        assert "games" in data
        assert data["elo"].empty
        assert data["re24"].empty


def test_evaluate_and_report_logic(mock_session_factory):
    from algomlb.cli.ml import _evaluate_and_report

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
    mock_model.get_feature_importance.return_value = pd.DataFrame(
        {"feature": ["f1"], "importance": [0.5]}
    )

    with (
        patch(
            "algomlb.ml.eval.compute_fold_metrics",
            return_value={"accuracy": 0.8, "auc": 0.8, "log_loss": 0.5},
        ),
        patch(
            "algomlb.ml.eval.persist_eval_results",
            side_effect=Exception("persist fail"),
        ),
    ):
        metrics = _evaluate_and_report(
            mock_model,
            pd.DataFrame({"f1": [1, 2]}),
            pd.Series([1, 0]),
            2023,
            MagicMock(),
            "v1",
            2019,
            2023,
        )
        assert metrics["accuracy"] == 0.8


def test_evaluate_and_report_shap_fail(mock_session_factory):
    from algomlb.cli.ml import _evaluate_and_report

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

    with (
        patch(
            "algomlb.ml.eval.compute_fold_metrics",
            return_value={"accuracy": 0.8, "auc": 0.8, "log_loss": 0.5},
        ),
        patch("algomlb.ml.eval.compute_calibration_bins"),
        patch("algomlb.ml.eval.persist_eval_results"),
        patch("algomlb.ml.eval.persist_eval_results"),
    ):
        _evaluate_and_report(
            mock_model,
            pd.DataFrame({"f1": [1]}),
            pd.Series([1]),
            2023,
            MagicMock(),
            "v1",
            2019,
            2023,
        )


def test_cli_ml_train_no_games(mock_session_factory):
    with patch("algomlb.cli.ml._load_ml_data") as mock_load:
        mock_load.return_value = {
            "games": pd.DataFrame(),
            "pitcher_gold": pd.DataFrame(),
            "lineups": pd.DataFrame(),
            "batter_gold": pd.DataFrame(),
            "elo": pd.DataFrame(),
            "pythag": pd.DataFrame(),
            "re24": pd.DataFrame(),
        }
        result = runner.invoke(app, ["train"], obj={"agent_mode": False})
        assert result.exit_code == 1


def test_cli_ml_walk_forward_empty_split(mock_session_factory):
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.cli.ml.FeaturePipeline") as mock_pipe,
        patch("algomlb.cli.ml.load_optimized_params", create=True) as mock_opt,
    ):
        mock_opt.return_value = {"n_estimators": 100}
        mock_load.return_value = {
            "games": pd.DataFrame(
                [{"game_pk": 1, "game_date": "2023-01-01", "year": 2023}]
            ),
            "pitcher_gold": pd.DataFrame(columns=["season"]),
            "lineups": pd.DataFrame(columns=["game_pk"]),
            "batter_gold": pd.DataFrame(columns=["season"]),
            "elo": pd.DataFrame(columns=["game_pk"]),
            "pythag": pd.DataFrame(columns=["game_pk"]),
            "re24": pd.DataFrame(columns=["player_id"]),
        }
        mock_pipe.return_value.build_uranium_matrix.return_value = (
            pd.DataFrame(),
            pd.Series(),
        )

        result = runner.invoke(
            app,
            ["walk-forward", "--start-year", "2023", "--end-year", "2024"],
            obj={"agent_mode": False},
        )
        assert result.exit_code == 0


def test_cli_ml_decouple_error(mock_session_factory):
    with patch(
        "algomlb.cli.ml.run_decoupler_pipeline", side_effect=Exception("pipeline fail")
    ):
        result = runner.invoke(app, ["decouple", "full"])
        assert result.exit_code == 1
