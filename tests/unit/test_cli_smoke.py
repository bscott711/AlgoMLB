from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from algomlb.cli.main import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AlgoMLB" in result.stdout


def test_cli_db_stubs():
    with patch("algomlb.cli.db.command") as mock_command:
        assert runner.invoke(app, ["db", "init"]).exit_code == 0
        mock_command.upgrade.assert_called_once()
        assert runner.invoke(app, ["db", "status"]).exit_code == 0
        mock_command.current.assert_called_once()


def test_cli_db_init_failure():
    """Verify error handling in db init."""
    with patch("algomlb.cli.db.command") as mock_command:
        mock_command.upgrade.side_effect = Exception("DB Error")
        result = runner.invoke(app, ["db", "init"])
        assert result.exit_code == 1


def test_cli_ingest_stubs():
    # Use mocks to avoid connecting to DB and API key requirements
    with (
        patch("algomlb.cli.ingest.IngestionOrchestrator"),
        patch("algomlb.cli.ingest.get_session_factory"),
        patch("algomlb.cli.ingest.HistoricalDataLoader"),
    ):
        assert runner.invoke(app, ["ingest", "odds"]).exit_code == 0
        assert runner.invoke(app, ["ingest", "schedule"]).exit_code == 0
        assert runner.invoke(app, ["ingest", "historical"]).exit_code == 0


def test_cli_ml_stubs():
    # Mocking real work to keep smoke tests fast
    with (
        patch("algomlb.ml.hyperopt.build_fold_data") as mock_build,
        patch("algomlb.ml.hyperopt.optimize_model") as mock_opt,
        patch("algomlb.cli.ml.pd.read_sql") as mock_read,
        patch("algomlb.cli.ml.get_session_factory"),
    ):
        mock_build.return_value = {"fold1": {}}
        mock_opt.return_value = ({}, MagicMock())

        # Mocking sequential read_sql calls for games, pitchers, batters, lineups, elo
        mock_read.side_effect = [
            pd.DataFrame(
                [
                    {
                        "game_pk": 1,
                        "game_date": "2023-04-01",
                        "home_team": "A",
                        "away_team": "B",
                        "home_pitcher_id": 1,
                        "away_pitcher_id": 2,
                        "home_score": 5,
                        "away_score": 3,
                    }
                ]
            ),  # games_df
            pd.DataFrame([{"player_id": 1, "season": 2023}]),  # pitcher_gold_df
            pd.DataFrame([{"player_id": 2, "season": 2023}]),  # batter_gold_df
            pd.DataFrame([{"game_pk": 1, "game_date": "2023-04-01"}]),  # lineups_df
            pd.DataFrame(),  # elo_df
            pd.DataFrame(),  # retro_df
        ]

        # Basic smoke check for optimize
        result = runner.invoke(app, ["ml", "optimize", "--n-trials", "1"])
        assert result.exit_code == 0
        assert mock_build.called
        assert mock_opt.called


def test_cli_run_stubs():
    assert runner.invoke(app, ["run", "live"]).exit_code == 0


def test_cli_ui_stubs(monkeypatch: pytest.MonkeyPatch):
    import subprocess

    # Use main app to avoid sub-app invocation issues
    from algomlb.cli.main import app as main_app

    # Mock subprocess.run to avoid starting a real process
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: None)

    result = runner.invoke(main_app, ["ui", "launch"])
    assert result.exit_code == 0


def test_cli_db_status_failure():
    """Verify error handling in db status."""
    with patch("algomlb.cli.db.command") as mock_command:
        mock_command.current.side_effect = Exception("Status Error")
        result = runner.invoke(app, ["db", "status"])
        assert result.exit_code == 1


def test_cli_ingest_historical_statcast():
    """Verify statcast option in historical ingestion."""
    with (
        patch("algomlb.cli.ingest.IngestionOrchestrator"),
        patch("algomlb.cli.ingest.HistoricalDataLoader") as mock_loader,
        patch("algomlb.cli.ingest.get_session_factory"),
    ):
        # Mock returning a list so len() works for records_processed calculation
        mock_loader.return_value.fetch_statcast.return_value = [1, 2, 3]
        result = runner.invoke(
            app, ["ingest", "historical", "--statcast", "--start-year", "2024"]
        )
        assert result.exit_code == 0
        mock_loader.return_value.fetch_statcast.assert_called_once()


def test_cli_ingest_agent_mode():
    """Verify agent mode output for different commands."""
    with (
        patch("algomlb.cli.ingest.IngestionOrchestrator"),
        patch("algomlb.cli.ingest.get_session_factory"),
        patch("algomlb.cli.ingest.HistoricalDataLoader"),
        patch("algomlb.cli.ingest.emit_agent_result") as mock_emit,
    ):
        # Cover ingest odds agent mode
        result = runner.invoke(app, ["--agent-mode", "ingest", "odds"])
        assert result.exit_code == 0
        assert mock_emit.call_count == 1

        # Cover ingest historical agent mode (Line 119)
        result = runner.invoke(app, ["--agent-mode", "ingest", "historical"])
        assert result.exit_code == 0
        assert mock_emit.called


def test_cli_ingest_historical_range():
    """Verify start/end range in historical ingestion."""
    with (
        patch("algomlb.cli.ingest.HistoricalDataLoader") as mock_loader,
        patch("algomlb.cli.ingest.get_session_factory"),
    ):
        mock_loader.return_value.fetch_statcast.return_value = [1]
        result = runner.invoke(
            app,
            ["ingest", "historical", "--start", "2024-04-01", "--end", "2024-04-02"],
        )
        assert result.exit_code == 0
        mock_loader.return_value.fetch_statcast.assert_called_with(
            "2024-04-01", "2024-04-02"
        )


def test_cli_ingest_historical_yearly():
    """Verify start-year/end-year range in historical ingestion."""
    with (
        patch("algomlb.cli.ingest.IngestionOrchestrator") as mock_orch,
        patch("algomlb.cli.ingest.get_session_factory"),
    ):
        mock_orch.return_value.run_historical_ingestion.return_value = 5
        result = runner.invoke(
            app, ["ingest", "historical", "--start-year", "2020", "--end-year", "2021"]
        )
        assert result.exit_code == 0
        mock_orch.return_value.run_historical_ingestion.assert_called_with(2020, 2021)
