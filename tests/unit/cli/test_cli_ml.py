import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch
import pandas as pd
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


def test_cli_ml_optimize(mock_session_factory):
    # Patch the source since it's locally imported
    with (
        patch("algomlb.cli.ml._load_ml_data") as mock_load,
        patch("algomlb.ml.hyperopt.build_fold_data", return_value=[(1, 2, 3, 4)]),
        patch("algomlb.ml.hyperopt.optimize_model") as mock_opt,
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


def test_cli_ml_decouple(mock_session_factory):
    with patch("algomlb.cli.ml.run_decoupler_pipeline") as mock_run:
        result = runner.invoke(app, ["decouple", "full"], obj={"agent_mode": False})
        assert result.exit_code == 0
        mock_run.assert_called_with("full", "v1")


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
            obj={"agent_mode": False},
        )
        assert result.exit_code == 0
        # Verification through mock calls since Loguru might write to stderr
        assert mock_eval.called
