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
    return repo, odds_client, stats_client


def test_run_odds_ingestion(mock_dependencies):
    repo, odds_client, stats_client = mock_dependencies
    orchestrator = IngestionOrchestrator(repo, odds_client, stats_client)

    # Mock return value
    mock_odds = [
        Odds(game_id="g1", sportsbook="sb1", market="h2h:A", price=1.9),
        Odds(game_id="g1", sportsbook="sb1", market="h2h:H", price=2.0),
    ]
    odds_client.fetch_live_odds.return_value = mock_odds

    count = orchestrator.run_odds_ingestion()

    assert count == 2
    odds_client.fetch_live_odds.assert_called_once()
    assert repo.save_live_odds.call_count == 2


def test_run_schedule_ingestion(mock_dependencies):
    repo, odds_client, stats_client = mock_dependencies
    orchestrator = IngestionOrchestrator(repo, odds_client, stats_client)

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
    repo, odds_client, stats_client = mock_dependencies
    orchestrator = IngestionOrchestrator(repo, odds_client, stats_client)

    odds_client.fetch_live_odds.return_value = []

    count = orchestrator.run_odds_ingestion()

    assert count == 0
    repo.save_live_odds.assert_not_called()
