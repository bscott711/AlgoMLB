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


def test_cli_agent_mode(mock_orchestrator, mock_session_factory):
    with patch("algomlb.cli.ingest.emit_agent_result") as mock_emit:
        mock_orchestrator.run_odds_ingestion.return_value = 10
        result = runner.invoke(app, ["odds"], obj={"agent_mode": True})
        assert result.exit_code == 0
        mock_emit.assert_called_once()


def test_cli_ingest_historical_statcast_fallback(
    mock_orchestrator, mock_session_factory
):
    with patch("algomlb.cli.ingest.HistoricalDataLoader") as mock_loader:
        mock_loader.return_value.fetch_statcast.return_value = [1, 2]
        result = runner.invoke(
            app,
            ["historical", "--start-year", "2023", "--statcast"],
            obj={"agent_mode": True},
        )
        assert result.exit_code == 0
        mock_loader.return_value.fetch_statcast.assert_called_with(
            "2023-04-01", "2023-04-30"
        )


def test_cli_ingest_historical_odds(mock_session_factory):
    with patch("algomlb.cli.ingest.HistoricalOddsIngester") as mock_ingester:
        # 1. Single date
        result = runner.invoke(app, ["historical-odds", "--date", "2023-04-01"])
        assert result.exit_code == 0
        mock_ingester.return_value.ingest_day_snapshots.assert_called()

        # 2. Range
        result = runner.invoke(
            app, ["historical-odds", "--start", "2023-04-01", "--end", "2023-04-02"]
        )
        assert result.exit_code == 0
        mock_ingester.return_value.run_backfill.assert_called()

        # 3. Error path
        result = runner.invoke(app, ["historical-odds"])
        assert result.exit_code == 0


def test_cli_ingest_umpire_scorecards(mock_session_factory):
    with patch("algomlb.cli.ingest.UmpireScorecardIngester") as mock_ingester:
        # Success path
        result = runner.invoke(app, ["umpire-scorecards", "--scrape"])
        assert result.exit_code == 0
        mock_ingester.return_value.ingest_from_api.assert_called()

        # No source path
        result = runner.invoke(app, ["umpire-scorecards"])
        assert result.exit_code == 0


def test_cli_ingest_retrosheet(mock_session_factory):
    with patch("algomlb.cli.ingest.RetrosheetIngester") as mock_ingester:
        # CSV path
        result = runner.invoke(app, ["retrosheet", "--csv", "test.csv"])
        assert result.exit_code == 0
        mock_ingester.return_value.ingest_from_csv.assert_called_with("test.csv")

        # URL
        result = runner.invoke(app, ["retrosheet", "--url", "http://test.zip"])
        assert result.exit_code == 0
        mock_ingester.return_value.ingest_from_url.assert_called_with("http://test.zip")

        # Range (default) - mocks datetime to avoid long loop
        with patch("algomlb.cli.ingest.datetime") as mock_dt:
            mock_dt.datetime.now.return_value.year = 2020
            # Test exception in loop
            mock_ingester.return_value.ingest_from_url.side_effect = [
                None,
                Exception("fail"),
            ]
            result = runner.invoke(app, ["retrosheet", "--since", "2019"])
            assert result.exit_code == 0


def test_cli_ingest_lineups_single_game(mock_session_factory):
    with patch("algomlb.ingestion.lineup_ingester.LineupIngester") as mock_ingester:
        result = runner.invoke(app, ["lineups", "--game-pk", "12345"])
        assert result.exit_code == 0
        mock_ingester.return_value.ingest_game.assert_called()

        # Error path
        result = runner.invoke(app, ["lineups"])
        assert result.exit_code == 0


def test_cli_ingest_transactions(mock_orchestrator, mock_session_factory):
    with patch("algomlb.cli.ingest.emit_agent_result") as mock_emit:
        result = runner.invoke(
            app,
            ["transactions", "--start", "2023-04-01", "--end", "2023-04-02"],
            obj={"agent_mode": True},
        )
        assert result.exit_code == 0
        mock_orchestrator.run_transaction_ingestion.assert_called()
        mock_emit.assert_called()


def test_cli_ingest_weather(mock_orchestrator, mock_session_factory):
    result = runner.invoke(
        app, ["weather", "--start", "2023-04-01", "--end", "2023-04-02"], obj={}
    )
    assert result.exit_code == 0
    mock_orchestrator.run_weather_ingestion.assert_called()


def test_cli_ingest_statcast(mock_session_factory):
    with patch("algomlb.ingestion.statcast_ingester.StatcastIngester") as mock_ingester:
        result = runner.invoke(
            app, ["statcast", "--start", "2023-04-01", "--end", "2023-04-01"], obj={}
        )
        assert result.exit_code == 0
        mock_ingester.return_value.ingest_range.assert_called()


def test_cli_ingest_gumbo_single_game(mock_session_factory):
    with patch("algomlb.ingestion.gumbo_ingester.GumboIngester") as mock_ingester:
        result = runner.invoke(app, ["gumbo", "--game-pk", "12345"])
        assert result.exit_code == 0
        mock_ingester.return_value.ingest_game.assert_called_with(12345)


def test_cli_ingest_schedule_agent_mode(mock_orchestrator, mock_session_factory):
    with patch("algomlb.cli.ingest.emit_agent_result") as mock_emit:
        result = runner.invoke(
            app,
            ["schedule", "--start", "2023-04-01", "--end", "2023-04-02"],
            obj={"agent_mode": True},
        )
        assert result.exit_code == 0
        mock_emit.assert_called()
