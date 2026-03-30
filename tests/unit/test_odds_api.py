import pytest
from unittest.mock import MagicMock
from algomlb.ingestion.odds_api import OddsAPIClient
from algomlb.config import get_settings


@pytest.fixture
def mock_settings(monkeypatch):
    """Ensure the Odds API key is available for testing."""
    settings = get_settings()
    # Mock the secret value to avoid real API key requirement
    mock_key = MagicMock()
    mock_key.get_secret_value.return_value = "test_key"
    monkeypatch.setattr(settings.api, "odds_api_key", mock_key)
    return settings


def test_fetch_live_odds_basic(mock_settings, respx_mock):
    """Test fetching and parsing live odds from the API."""
    client = OddsAPIClient()
    mock_data = [
        {
            "id": "game1",
            "home_team": "NYY",
            "away_team": "TOR",
            "commence_time": "2023-04-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {"key": "h2h", "outcomes": [{"name": "NYY", "price": 1.91}]}
                    ],
                }
            ],
        }
    ]
    respx_mock.get("https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/").respond(
        json=mock_data
    )

    odds = client.fetch_live_odds()
    assert len(odds) == 1
    assert odds[0].odds_game_id == "game1"
    assert odds[0].home_team == "NYY"
    assert odds[0].price == 1.91


def test_fetch_historical_odds_basic(mock_settings, respx_mock):
    """Test fetching and parsing historical odds snapshots."""
    client = OddsAPIClient()
    mock_data = {
        "timestamp": "2023-04-01T10:00:00Z",
        "data": [
            {
                "id": "game1",
                "home_team": "NYY",
                "away_team": "TOR",
                "commence_time": "2023-04-01T20:00:00Z",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {"key": "h2h", "outcomes": [{"name": "NYY", "price": -110}]}
                        ],
                    }
                ],
            }
        ],
    }
    respx_mock.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds-history/"
    ).respond(json=mock_data)

    odds = client.fetch_historical_odds("2023-04-01T10:00:00Z")
    assert len(odds) == 1
    assert odds[0].price == -110
    assert odds[0].timestamp.hour == 10


def test_fetch_live_odds_empty(mock_settings, respx_mock):
    """Test handling of empty API response."""
    client = OddsAPIClient()
    respx_mock.get("https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/").respond(
        json=[]
    )
    odds = client.fetch_live_odds()
    assert odds == []


def test_fetch_live_odds_bad_date(mock_settings, respx_mock):
    """Test resilience against malformed date strings in API."""
    client = OddsAPIClient()
    mock_data = [
        {
            "id": "game1",
            "home_team": "NYY",
            "away_team": "TOR",
            "commence_time": "INVALID_DATE",
            "bookmakers": [],
        }
    ]
    respx_mock.get("https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/").respond(
        json=mock_data
    )
    odds = client.fetch_live_odds()
    # Should still succeed but use current date
    assert len(odds) == 0  # no bookmakers, so no odds items


def test_fetch_historical_odds_bad_date(mock_settings, respx_mock):
    """Test resilience against malformed commence_time in historical API."""
    client = OddsAPIClient()
    mock_data = {
        "timestamp": "2023-04-01T10:00:00Z",
        "data": [
            {
                "id": "game1",
                "home_team": "NYY",
                "away_team": "TOR",
                "commence_time": "BAD_DATE",
                "bookmakers": [],
            }
        ],
    }
    respx_mock.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds-history/"
    ).respond(json=mock_data)
    odds = client.fetch_historical_odds("2023-04-01T10:00:00Z")
    assert odds == []
