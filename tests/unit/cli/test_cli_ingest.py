import pytest
from typer.testing import CliRunner
from unittest.mock import patch
from algomlb.cli.ingest import app

runner = CliRunner()


@pytest.fixture
def mock_orchestrator():
    # Patch both the top-level name in the CLI module AND the source module
    with (
        patch("algomlb.cli.ingest.IngestionOrchestrator") as mock_top,
        patch("algomlb.ingestion.orchestrator.IngestionOrchestrator") as mock_src,
    ):
        # Link them to the same mock object
        obj = mock_top.return_value
        mock_src.return_value = obj
        yield obj


@pytest.fixture
def mock_session_factory():
    with patch("algomlb.cli.ingest.get_session_factory") as mock:
        yield mock


def test_cli_ingest_odds(mock_orchestrator, mock_session_factory):
    mock_orchestrator.run_odds_ingestion.return_value = 10
    result = runner.invoke(app, ["odds"], obj={"agent_mode": False})
    assert result.exit_code == 0
    # CliRunner might not capture loguru output if it's going to stderr
    # Let's check for orchestrator call instead or use result.stdout if captured
    mock_orchestrator.run_odds_ingestion.assert_called_once()


def test_cli_ingest_schedule(mock_orchestrator, mock_session_factory):
    mock_orchestrator.run_schedule_ingestion.return_value = 5
    result = runner.invoke(
        app,
        ["schedule", "--start", "2023-04-01", "--end", "2023-04-02"],
        obj={"agent_mode": False},
    )
    assert result.exit_code == 0
    mock_orchestrator.run_schedule_ingestion.assert_called()


def test_cli_ingest_historical(mock_orchestrator, mock_session_factory):
    mock_orchestrator.run_historical_ingestion.return_value = 100
    # 1. Yearly
    result = runner.invoke(
        app,
        ["historical", "--start-year", "2023", "--end-year", "2023"],
        obj={"agent_mode": False},
    )
    assert result.exit_code == 0
    mock_orchestrator.run_historical_ingestion.assert_called()

    # 2. Statcast range
    with patch("algomlb.cli.ingest.HistoricalDataLoader") as mock_loader:
        mock_loader.return_value.fetch_statcast.return_value = [1, 2, 3]  # length 3
        result = runner.invoke(
            app,
            ["historical", "--start", "2023-04-01", "--end", "2023-04-05"],
            obj={"agent_mode": False},
        )
        assert result.exit_code == 0
        mock_loader.return_value.fetch_statcast.assert_called()


def test_cli_ingest_ballparks(mock_session_factory):
    with patch("algomlb.cli.ingest.BallparkIngester") as mock_ingester:
        result = runner.invoke(app, ["ballparks", "--csv", "test.csv"], obj={})
        assert result.exit_code == 0
        mock_ingester.return_value.ingest_from_csv.assert_called_with("test.csv")


def test_cli_ingest_lineups(mock_session_factory):
    with patch("algomlb.ingestion.lineup_ingester.LineupIngester") as mock_ingester:
        mock_ingester.return_value.backfill_range.return_value = 50
        result = runner.invoke(
            app, ["lineups", "--start", "2023-04-01", "--end", "2023-04-02"], obj={}
        )
        assert result.exit_code == 0
        mock_ingester.return_value.backfill_range.assert_called()


def test_cli_ingest_gumbo(mock_orchestrator, mock_session_factory):
    mock_orchestrator.run_gumbo_ingestion.return_value = 25
    result = runner.invoke(app, ["gumbo", "--start", "2023-04-01"], obj={})
    assert result.exit_code == 0
    mock_orchestrator.run_gumbo_ingestion.assert_called()


def test_cli_ingest_managers(mock_session_factory):
    with patch(
        "algomlb.ingestion.managers_ingester.backfill_team_managers"
    ) as mock_backfill:
        result = runner.invoke(app, ["managers", "--start-year", "2023"])
        assert result.exit_code == 0
        mock_backfill.assert_called()
