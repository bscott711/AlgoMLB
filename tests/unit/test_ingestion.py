import httpx
import pytest
import respx
from unittest.mock import patch, MagicMock

from algomlb.domain import GameStatus
from algomlb.ingestion import MLBStatsAPIClient, OddsAPIClient
from algomlb.ingestion.http_client import BaseAPIClient


@pytest.fixture
def mock_settings():
    """Mock the settings object to bypass real validation and env dependency."""
    with (
        patch("algomlb.ingestion.mlb_stats.get_settings") as mlb_mock,
        patch("algomlb.ingestion.odds_api.get_settings") as odds_mock,
    ):
        settings = MagicMock()
        settings.api.mlb_stats_url = "https://stats.test"
        settings.api.odds_api_key.get_secret_value.return_value = "test-key"

        mlb_mock.return_value = settings
        odds_mock.return_value = settings
        yield settings


@respx.mock
def test_base_client_retry_logic():
    """Verify that BaseAPIClient retries as expected on 500 errors and eventually fails."""
    client = BaseAPIClient(base_url="https://api.test")

    # Mock a persistent 500 error
    route = respx.get("https://api.test/fail").mock(return_value=httpx.Response(500))

    with pytest.raises(httpx.HTTPStatusError):
        client._request("GET", "/fail")

    # Tenacity with stop_after_attempt(3) should have tried 3 times
    assert route.called
    assert route.call_count == 3


@respx.mock
def test_odds_api_client_parsing(mock_settings):
    """Verify that OddsAPIClient correctly parses sample JSON into domain models."""
    client = OddsAPIClient(base_url="https://api.odds.test")

    sample_json = [
        {
            "id": "game_1",
            "sport_key": "baseball_mlb",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Away Team", "price": 1.91},
                                {"name": "Home Team", "price": 1.95},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    respx.get("https://api.odds.test/v4/sports/baseball_mlb/odds/").mock(
        return_value=httpx.Response(200, json=sample_json)
    )

    odds_list = client.fetch_live_odds()
    assert len(odds_list) == 2
    assert odds_list[0].game_id == "game_1"
    assert odds_list[0].sportsbook == "DraftKings"
    assert "h2h:Away Team" in odds_list[0].market
    assert odds_list[0].price == 1.91


def test_odds_api_client_missing_key():
    """Verify that OddsAPIClient raises RuntimeError if API key is missing."""
    with patch("algomlb.ingestion.odds_api.get_settings") as mock_get_settings:
        settings = MagicMock()
        settings.api.odds_api_key = None
        mock_get_settings.return_value = settings

        with pytest.raises(RuntimeError, match="The Odds API key is not configured"):
            OddsAPIClient()


@respx.mock
def test_mlb_stats_api_client_parsing(mock_settings):
    """Verify that MLBStatsAPIClient correctly parses sample schedule JSON with various statuses."""
    client = MLBStatsAPIClient(base_url="https://stats.test")

    sample_json = {
        "dates": [
            {
                "date": "2026-03-30",
                "games": [
                    {
                        "gamePk": 123456,
                        "gameDate": "2026-03-30T19:07:00Z",
                        "status": {"detailedState": "Final"},
                        "teams": {
                            "away": {
                                "team": {"name": "New York Yankees"},
                                "score": 4,
                                "probablePitcher": {"fullName": "Gerrit Cole"},
                            },
                            "home": {
                                "team": {"name": "Toronto Blue Jays"},
                                "score": 2,
                                "probablePitcher": {"fullName": "Kevin Gausman"},
                            },
                        },
                    },
                    {
                        "gamePk": 111,
                        "status": {"detailedState": "In Progress"},
                        "teams": {
                            "away": {"team": {"name": "Team A"}},
                            "home": {"team": {"name": "Team B"}},
                        },
                    },
                    {
                        "gamePk": 222,
                        "status": {"detailedState": "Cancelled"},
                        "teams": {
                            "away": {"team": {"name": "Team A"}},
                            "home": {"team": {"name": "Team B"}},
                        },
                    },
                    {
                        "gamePk": 333,
                        "status": {"detailedState": "Postponed"},
                        "teams": {
                            "away": {"team": {"name": "Team A"}},
                            "home": {"team": {"name": "Team B"}},
                        },
                    },
                ],
            }
        ]
    }

    respx.get("https://stats.test/schedule").mock(
        return_value=httpx.Response(200, json=sample_json)
    )

    games = client.fetch_daily_schedule()
    assert len(games) == 4

    # Check statuses
    assert games[0].status == GameStatus.COMPLETED
    assert games[1].status == GameStatus.IN_PROGRESS
    assert games[2].status == GameStatus.CANCELLED
    assert games[3].status == GameStatus.POSTPONED

    # Check first game details
    assert games[0].game_id == "123456"
    assert games[0].away_pitcher == "Gerrit Cole"
    assert games[0].home_score == 2


@respx.mock
def test_client_cleanup():
    """Test client close method."""
    client = BaseAPIClient(base_url="https://api.test")
    client.close()
    assert client.client.is_closed
