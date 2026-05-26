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
        "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"
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
        "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"
    ).respond(json=mock_data)
    odds = client.fetch_historical_odds("2023-04-01T10:00:00Z")
    assert odds == []


def test_fetch_live_odds_with_rotation(monkeypatch, respx_mock):
    """Test that key rotation occurs automatically when the primary key is exhausted."""
    settings = get_settings()

    mock_key_1 = MagicMock()
    mock_key_1.get_secret_value.return_value = "test_key_1"
    mock_key_2 = MagicMock()
    mock_key_2.get_secret_value.return_value = "test_key_2"

    monkeypatch.setattr(settings.api, "odds_api_key", mock_key_1)
    monkeypatch.setattr(settings.api, "odds_api_key_secondary", mock_key_2)

    client = OddsAPIClient()

    mock_data = [
        {
            "id": "game2",
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

    # First mock request with key 1: return 401 OUT_OF_USAGE_CREDITS
    respx_mock.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
        params={"apiKey": "test_key_1", "regions": "us", "markets": "h2h", "oddsFormat": "decimal"}
    ).respond(
        status_code=401,
        json={"message": "Usage quota has been reached.", "error_code": "OUT_OF_USAGE_CREDITS"}
    )

    # Second mock request with key 2: return success 200
    respx_mock.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
        params={"apiKey": "test_key_2", "regions": "us", "markets": "h2h", "oddsFormat": "decimal"}
    ).respond(
        status_code=200,
        json=mock_data
    )

    odds = client.fetch_live_odds()
    assert len(odds) == 1
    assert odds[0].odds_game_id == "game2"
    assert client._active_key_index == 1


def test_key_status_manager(monkeypatch, tmp_path):
    """Test that key exhaustion is persistently cached and correctly expires."""
    import datetime
    import json
    from algomlb.ingestion.odds_api import is_key_exhausted, mark_key_exhausted, load_status
    
    # Mock STATUS_FILE to a temporary path so we don't pollute the actual cache during testing
    test_status_file = str(tmp_path / ".odds_api_status.json")
    monkeypatch.setattr("algomlb.ingestion.odds_api.STATUS_FILE", test_status_file)
    
    key = "test_exhaustion_key"
    assert not is_key_exhausted(key)
    
    # Mark as exhausted
    mark_key_exhausted(key)
    assert is_key_exhausted(key)
    
    # Load status dict and check values
    status_dict = load_status()
    assert key in status_dict
    assert status_dict[key]["status"] == "exhausted"
    
    # Mock future time to check reset (mock reset_at to a past time)
    status_dict[key]["reset_at"] = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)).isoformat()
    # Save the mocked status
    with open(test_status_file, "w") as f:
        json.dump(status_dict, f)
        
    # Check that it resets automatically when reset_at is in the past
    assert not is_key_exhausted(key)
    # Check it updated the file to active
    assert load_status()[key]["status"] == "active"
