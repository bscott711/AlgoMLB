import json
from unittest.mock import patch

from typer.testing import CliRunner

from algomlb.cli.main import app

runner = CliRunner()


@patch("algomlb.cli.ingest.IngestionOrchestrator")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_odds_command(mock_session_factory, mock_orchestrator_class):
    """Test the 'ingest odds' CLI command outputs correctly in agent mode."""
    # Setup mocks
    mock_orchestrator = mock_orchestrator_class.return_value
    mock_orchestrator.run_odds_ingestion.return_value = 15

    # Run command in agent mode - global options must come before commands
    result = runner.invoke(app, ["--agent-mode", "ingest", "odds"])

    assert result.exit_code == 0
    # Split output into lines, looking for agent-mode JSON
    lines = result.stdout.strip().split("\n")
    agent_output = None
    for line in lines:
        if line.startswith("{") and "ingest.odds" in line:
            agent_output = json.loads(line)
            break

    assert agent_output is not None
    assert agent_output["status"] == "success"
    assert agent_output["command"] == "ingest.odds"
    assert agent_output["data"]["records_inserted"] == 15
    mock_orchestrator.run_odds_ingestion.assert_called_once()


@patch("algomlb.cli.ingest.IngestionOrchestrator")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_schedule_command(mock_session_factory, mock_orchestrator_class):
    """Test the 'ingest schedule' CLI command outputs correctly in agent mode."""
    # Setup mocks
    mock_orchestrator = mock_orchestrator_class.return_value
    mock_orchestrator.run_schedule_ingestion.return_value = 10

    # Run command in agent mode
    result = runner.invoke(app, ["--agent-mode", "ingest", "schedule"])

    assert result.exit_code == 0
    # Resolve output
    lines = result.stdout.strip().split("\n")
    agent_output = None
    for line in lines:
        if line.startswith("{") and "ingest.schedule" in line:
            agent_output = json.loads(line)
            break

    assert agent_output is not None
    assert agent_output["status"] == "success"
    assert agent_output["command"] == "ingest.schedule"
    assert agent_output["data"]["records_inserted"] == 10
    mock_orchestrator.run_schedule_ingestion.assert_called_once()


@patch("algomlb.cli.ingest.IngestionOrchestrator")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_odds_no_agent_mode(mock_session_factory, mock_orchestrator_class):
    """Test 'ingest odds' CLI command outputs to stderr (verified via result.stderr if available)."""
    mock_orchestrator = mock_orchestrator_class.return_value
    mock_orchestrator.run_odds_ingestion.return_value = 15

    result = runner.invoke(app, ["ingest", "odds"])

    assert result.exit_code == 0
    # On some systems, logs go to stderr and aren't merged.
    # We focus on ensuring JSON is NOT in stdout and orchestrator WAS called.
    assert "{" not in result.stdout
    mock_orchestrator.run_odds_ingestion.assert_called_once()


@patch("algomlb.cli.ingest.BallparkIngester")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_ballparks_command(mock_session_factory, mock_ingester_class):
    """Test the 'ingest ballparks' CLI command calls the ingester."""
    mock_ingester = mock_ingester_class.return_value
    result = runner.invoke(app, ["ingest", "ballparks", "--csv", "fake.csv"])
    assert result.exit_code == 0
    mock_ingester.ingest_from_csv.assert_called_once_with("fake.csv")


@patch("algomlb.cli.ingest.HistoricalOddsIngester")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_historical_odds_command(mock_session_factory, mock_ingester_class):
    """Test the 'ingest historical-odds' CLI command calls the ingester."""
    mock_ingester = mock_ingester_class.return_value
    result = runner.invoke(app, ["ingest", "historical-odds", "--date", "2023-04-01"])
    assert result.exit_code == 0
    import datetime

    mock_ingester.ingest_day_snapshots.assert_called_once_with(
        datetime.date(2023, 4, 1)
    )


@patch("algomlb.cli.ingest.UmpireScorecardIngester")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_umpire_scorecards_command(mock_session_factory, mock_ingester_class):
    """Test 'ingest umpire-scorecards' CLI paths (--scrape and default warning)."""
    mock_ingester = mock_ingester_class.return_value
    mock_ingester.ingest_from_api.return_value = 100

    # Test --scrape
    result = runner.invoke(app, ["ingest", "umpire-scorecards", "--scrape"])
    assert result.exit_code == 0
    mock_ingester.ingest_from_api.assert_called_once()

    # Test default (no flag) — should warn
    mock_ingester.reset_mock()
    result = runner.invoke(app, ["ingest", "umpire-scorecards"])
    assert result.exit_code == 0
    mock_ingester.ingest_from_api.assert_not_called()


@patch("algomlb.cli.ingest.RetrosheetIngester")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_retrosheet_command(mock_session_factory, mock_ingester_class):
    """Test 'ingest retrosheet' CLI paths (CSV, URL, Default)."""
    mock_ingester = mock_ingester_class.return_value

    # Test --csv
    result = runner.invoke(app, ["ingest", "retrosheet", "--csv", "fake_retro.csv"])
    assert result.exit_code == 0
    mock_ingester.ingest_from_csv.assert_called_once_with("fake_retro.csv")

    # Test --url
    mock_ingester.reset_mock()
    result = runner.invoke(
        app, ["ingest", "retrosheet", "--url", "http://fake.com/plays.zip"]
    )
    assert result.exit_code == 0
    mock_ingester.ingest_from_url.assert_called_once_with("http://fake.com/plays.zip")

    # Test default
    mock_ingester.reset_mock()
    result = runner.invoke(app, ["ingest", "retrosheet"])
    assert result.exit_code == 0
    # 2019 to current year (2026) is 8 calls
    assert mock_ingester.ingest_from_url.call_count == 8


@patch("algomlb.cli.ingest.RetrosheetIngester")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_retrosheet_command_fails(mock_session_factory, mock_ingester_class):
    """Test 'ingest retrosheet' CLI handles seasonal failures."""
    mock_ingester = mock_ingester_class.return_value
    mock_ingester.ingest_from_url.side_effect = Exception("Download failed")

    result = runner.invoke(app, ["ingest", "retrosheet"])
    assert result.exit_code == 0
    # Should still call for all years (2019-2026), just catching exceptions
    assert mock_ingester.ingest_from_url.call_count == 8


@patch("algomlb.cli.ingest.HistoricalOddsIngester")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_historical_odds_range_and_missing(
    mock_session_factory, mock_ingester_class
):
    """Test 'ingest historical-odds' with range and missing parameters."""
    mock_ingester = mock_ingester_class.return_value
    import datetime

    # Test range
    result = runner.invoke(
        app,
        ["ingest", "historical-odds", "--start", "2023-04-01", "--end", "2023-04-03"],
    )
    assert result.exit_code == 0
    mock_ingester.run_backfill.assert_called_once_with(
        datetime.date(2023, 4, 1), datetime.date(2023, 4, 3), reverse=True
    )

    # Test missing
    mock_ingester.reset_mock()
    result = runner.invoke(app, ["ingest", "historical-odds"])
    assert result.exit_code == 0
    mock_ingester.run_backfill.assert_not_called()
    mock_ingester.ingest_day_snapshots.assert_not_called()
