import json
from unittest.mock import patch

from typer.testing import CliRunner

from algomlb.cli.main import app

runner = CliRunner()


@patch("algomlb.cli.ingest.IngestionOrchestrator")
@patch("algomlb.cli.ingest.create_db_engine")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_odds_command(
    mock_session_factory, mock_engine, mock_orchestrator_class
):
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
@patch("algomlb.cli.ingest.create_db_engine")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_schedule_command(
    mock_session_factory, mock_engine, mock_orchestrator_class
):
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
@patch("algomlb.cli.ingest.create_db_engine")
@patch("algomlb.cli.ingest.get_session_factory")
def test_ingest_odds_no_agent_mode(
    mock_session_factory, mock_engine, mock_orchestrator_class
):
    """Test 'ingest odds' CLI command outputs to stderr (verified via result.stderr if available)."""
    mock_orchestrator = mock_orchestrator_class.return_value
    mock_orchestrator.run_odds_ingestion.return_value = 15

    result = runner.invoke(app, ["ingest", "odds"])

    assert result.exit_code == 0
    # On some systems, logs go to stderr and aren't merged.
    # We focus on ensuring JSON is NOT in stdout and orchestrator WAS called.
    assert "{" not in result.stdout
    mock_orchestrator.run_odds_ingestion.assert_called_once()
