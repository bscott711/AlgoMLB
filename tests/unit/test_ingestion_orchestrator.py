import datetime
from unittest.mock import MagicMock
import pytest

from algomlb.domain import Game, GameStatus, Odds
from algomlb.ingestion import IngestionOrchestrator


@pytest.fixture
def mock_dependencies():
    repo = MagicMock()
    odds_client = MagicMock()
    stats_client = MagicMock()
    historical_loader = MagicMock()
    transactions_ingester = MagicMock()
    return repo, odds_client, stats_client, historical_loader, transactions_ingester


def test_run_odds_ingestion(mock_dependencies):
    repo, odds_client, stats_client, historical_loader, transactions_ingester = (
        mock_dependencies
    )
    orchestrator = IngestionOrchestrator(
        repo, odds_client, stats_client, historical_loader, transactions_ingester
    )

    # Mock return value
    d = datetime.date(2026, 3, 30)
    mock_odds = [
        Odds(
            odds_game_id="g1",
            home_team="Home",
            away_team="Away",
            game_date=d,
            sportsbook="sb1",
            market_type="h2h",
            outcome="A",
            price=1.9,
        ),
        Odds(
            odds_game_id="g1",
            home_team="Home",
            away_team="Away",
            game_date=d,
            sportsbook="sb1",
            market_type="h2h",
            outcome="H",
            price=2.0,
        ),
    ]
    odds_client.fetch_live_odds.return_value = mock_odds

    count = orchestrator.run_odds_ingestion()

    assert count == 2
    odds_client.fetch_live_odds.assert_called_once()
    assert repo.save_live_odds.call_count == 2


def test_run_schedule_ingestion(mock_dependencies):
    repo, odds_client, stats_client, historical_loader, transactions_ingester = (
        mock_dependencies
    )
    orchestrator = IngestionOrchestrator(
        repo, odds_client, stats_client, historical_loader, transactions_ingester
    )

    # Mock return value
    mock_games = [
        Game(
            game_id="g1",
            date=datetime.date(2026, 3, 30),
            home_team="Team H",
            away_team="Team A",
            status=GameStatus.SCHEDULED,
        )
    ]
    stats_client.fetch_daily_schedule.return_value = mock_games

    count = orchestrator.run_schedule_ingestion()

    assert count == 1
    stats_client.fetch_daily_schedule.assert_called_once()
    repo.save_game.assert_called_once_with(mock_games[0])


def test_run_odds_ingestion_empty(mock_dependencies):
    repo, odds_client, stats_client, historical_loader, transactions_ingester = (
        mock_dependencies
    )
    orchestrator = IngestionOrchestrator(
        repo, odds_client, stats_client, historical_loader, transactions_ingester
    )

    odds_client.fetch_live_odds.return_value = []

    count = orchestrator.run_odds_ingestion()

    assert count == 0
    repo.save_live_odds.assert_not_called()


def test_run_historical_ingestion(mock_dependencies):
    repo, odds_client, stats_client, historical_loader, transactions_ingester = (
        mock_dependencies
    )
    orchestrator = IngestionOrchestrator(
        repo, odds_client, stats_client, historical_loader, transactions_ingester
    )

    import pandas as pd

    historical_loader.fetch_pitching_stats.return_value = pd.DataFrame({"id": [1]})
    historical_loader.fetch_team_batting.return_value = pd.DataFrame({"playerid": [2]})

    count = orchestrator.run_historical_ingestion(2023, 2023)

    assert count == 2
    historical_loader.fetch_pitching_stats.assert_called_once_with(2023, 2023)
    historical_loader.fetch_team_batting.assert_called_once_with(2023, 2023)


def test_run_transaction_ingestion(mock_dependencies):
    repo, odds_client, stats_client, historical_loader, transactions_ingester = (
        mock_dependencies
    )
    orchestrator = IngestionOrchestrator(
        repo, odds_client, stats_client, historical_loader, transactions_ingester
    )

    transactions_ingester.ingest_range.return_value = 5
    d1 = datetime.date(2024, 1, 1)
    d2 = datetime.date(2024, 1, 31)

    count = orchestrator.run_transaction_ingestion(d1, d2)

    assert count == 5
    transactions_ingester.ingest_range.assert_called_once_with(d1, d2)


def test_run_transaction_ingestion_defaults(mock_dependencies):
    repo, odds_client, stats_client, historical_loader, transactions_ingester = (
        mock_dependencies
    )
    orchestrator = IngestionOrchestrator(
        repo, odds_client, stats_client, historical_loader, transactions_ingester
    )

    transactions_ingester.ingest_range.return_value = 1
    # Test only start_date provided
    d1 = datetime.date(2024, 1, 1)
    count = orchestrator.run_transaction_ingestion(start_date=d1)
    assert count == 1
    # Final date should be today
    args = transactions_ingester.ingest_range.call_args[0]
    assert args[0] == d1
    assert args[1] == datetime.date.today()

    # Test no dates provided
    count = orchestrator.run_transaction_ingestion()
    assert count == 1
    args = transactions_ingester.ingest_range.call_args[0]
    assert args[1] == datetime.date.today()
    assert args[0] == datetime.date.today() - datetime.timedelta(days=7)
