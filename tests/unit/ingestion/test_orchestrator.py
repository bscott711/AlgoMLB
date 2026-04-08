import pytest
from datetime import date
from unittest.mock import MagicMock
from algomlb.ingestion.orchestrator import IngestionOrchestrator


@pytest.fixture
def mock_deps():
    return {
        "repo": MagicMock(),
        "odds_client": MagicMock(),
        "stats_client": MagicMock(),
        "historical_loader": MagicMock(),
        "transactions_ingester": MagicMock(),
        "openmeteo_ingester": MagicMock(),
        "statcast_ingester": MagicMock(),
        "umpire_ingester": MagicMock(),
        "gumbo_ingester": MagicMock(),
    }


@pytest.fixture
def orchestrator(mock_deps):
    return IngestionOrchestrator(**mock_deps)


def test_run_historical_ingestion(orchestrator, mock_deps):
    mock_deps["historical_loader"].fetch_pitching_stats.return_value = [1, 2]
    mock_deps["historical_loader"].fetch_team_batting.return_value = [3]
    assert orchestrator.run_historical_ingestion(2023, 2023) == 3


def test_run_odds_ingestion(orchestrator, mock_deps):
    mock_deps["odds_client"].fetch_live_odds.return_value = [{"id": 1}]
    assert orchestrator.run_odds_ingestion() == 1
    assert mock_deps["repo"].save_live_odds.called


def test_run_schedule_ingestion_chunking(orchestrator, mock_deps):
    # Dec 31 to Jan 1 to force chunking across years
    start = date(2023, 12, 31)
    end = date(2024, 1, 1)
    mock_deps["stats_client"].fetch_daily_schedule.return_value = []

    orchestrator.run_schedule_ingestion(start, end)
    # Should be called twice (once for Dec 31, once for Jan 1)
    assert mock_deps["stats_client"].fetch_daily_schedule.call_count == 2


def test_run_schedule_ingestion_defaults(orchestrator, mock_deps):
    mock_deps["stats_client"].fetch_daily_schedule.return_value = []
    orchestrator.run_schedule_ingestion()  # No dates passed
    assert mock_deps["stats_client"].fetch_daily_schedule.called


def test_run_transaction_ingestion(orchestrator, mock_deps):
    orchestrator.run_transaction_ingestion()
    assert mock_deps["transactions_ingester"].ingest_range.called


def test_run_weather_ingestion(orchestrator, mock_deps):
    orchestrator.run_weather_ingestion()
    assert mock_deps["openmeteo_ingester"].ingest_range.called


def test_run_statcast_ingestion(orchestrator, mock_deps):
    orchestrator.run_statcast_ingestion()
    assert mock_deps["statcast_ingester"].ingest_range.called


def test_run_umpire_ingestion(orchestrator, mock_deps):
    orchestrator.run_umpire_ingestion()
    assert mock_deps["umpire_ingester"].ingest_from_api.called


def test_run_gumbo_ingestion_logic(orchestrator, mock_deps):
    # Mocking the select statement and scalars session flow
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = ["101", "bad_id"]
    mock_deps["repo"].session.execute.return_value.scalars.return_value = mock_scalars

    # 1. Successful Gumbo run
    orchestrator.run_gumbo_ingestion(date(2023, 1, 1), date(2023, 1, 1))
    # Should call ingest_games with [101] (bad_id filtered)
    mock_deps["gumbo_ingester"].ingest_games.assert_called_once_with([101])

    # 2. Gumbo ingester is None
    orchestrator.gumbo_ingester = None
    assert orchestrator.run_gumbo_ingestion() == 0
