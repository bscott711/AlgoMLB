import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch
from algomlb.cli.sync import app

runner = CliRunner()


@pytest.fixture
def mock_session_factory():
    with patch("algomlb.cli.sync.get_session_factory") as mock:
        factory = mock.return_value
        session = MagicMock()
        factory.return_value.__enter__.return_value = session
        factory.kw = {"bind": MagicMock()}
        yield mock


def test_cli_sync_daily_success(mock_session_factory):
    """Verify that the daily sync command orchestrates ingestion and processing correctly."""
    with (
        patch("algomlb.cli.sync.IngestionOrchestrator") as mock_orch_cls,
        patch("algomlb.cli.sync.OddsAPIClient"),
        patch("algomlb.cli.sync.MLBStatsAPIClient"),
        patch("algomlb.cli.sync.HistoricalDataLoader"),
        patch("algomlb.cli.sync.PlayerTransactionsIngester"),
        patch("algomlb.cli.sync.OpenMeteoIngester"),
        patch("algomlb.cli.sync.StatcastIngester"),
        patch("algomlb.cli.sync.UmpireScorecardIngester"),
        patch("algomlb.cli.sync.GumboIngester"),
        patch("algomlb.ml.silver_processor.process_silver_incremental"),
        patch("algomlb.ml.rolling_service.RollingService"),
        patch("algomlb.ml.rolling_processor.RollingProcessor"),
        patch("algomlb.config.settings.get_settings"),
    ):
        # Typer collapses the 'daily' command name because it is the only one in the app
        result = runner.invoke(app, ["--date", "2023-04-02"])
        assert result.exit_code == 0
        assert mock_orch_cls.called


def test_cli_sync_daily_default_date(mock_session_factory):
    """Verify that daily sync runs correctly with default date parameters."""
    with (
        patch("algomlb.cli.sync.IngestionOrchestrator"),
        patch("algomlb.cli.sync.OddsAPIClient"),
        patch("algomlb.cli.sync.MLBStatsAPIClient"),
        patch("algomlb.cli.sync.HistoricalDataLoader"),
        patch("algomlb.cli.sync.PlayerTransactionsIngester"),
        patch("algomlb.cli.sync.OpenMeteoIngester"),
        patch("algomlb.cli.sync.StatcastIngester"),
        patch("algomlb.cli.sync.UmpireScorecardIngester"),
        patch("algomlb.cli.sync.GumboIngester"),
        patch("algomlb.ml.silver_processor.process_silver_incremental"),
        patch("algomlb.ml.rolling_service.RollingService"),
        patch("algomlb.ml.rolling_processor.RollingProcessor"),
        patch("algomlb.config.settings.get_settings"),
    ):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
