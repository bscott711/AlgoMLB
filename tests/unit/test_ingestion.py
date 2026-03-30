from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx
from unittest.mock import MagicMock, patch

from algomlb.domain import GameStatus
from algomlb.ingestion import MLBStatsAPIClient, OddsAPIClient
from algomlb.ingestion.historical import HistoricalDataLoader
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
            "commence_time": "2026-03-30T19:07:00Z",
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
        },
        {
            "id": "game_2",
            "sport_key": "baseball_mlb",
            "commence_time": "bad_date",
            "bookmakers": [],
        },
    ]

    respx.get("https://api.odds.test/v4/sports/baseball_mlb/odds/").mock(
        return_value=httpx.Response(200, json=sample_json)
    )

    odds_list = client.fetch_live_odds()
    assert len(odds_list) == 2
    assert odds_list[0].odds_game_id == "game_1"
    assert odds_list[0].sportsbook == "DraftKings"
    assert odds_list[0].market_type == "h2h"
    assert odds_list[0].outcome == "Away Team"
    assert odds_list[0].price == 1.91


def test_circuit_breaker():
    """Verify circuit breaker opens after failures."""
    from algomlb.ingestion.http_client import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=2)
    assert cb.is_available() is True
    cb.record_failure()
    assert cb.is_available() is True
    cb.record_failure()
    assert cb.is_available() is False
    # Manually transition to HALF_OPEN to test recovery
    cb.state = "HALF_OPEN"
    cb.record_success()
    assert cb.is_available() is True
    assert cb.state == "CLOSED"
    assert cb.failure_count == 0


def test_historical_loader_persist_empty():
    """Verify empty DataFrame handling."""
    from algomlb.ingestion.historical import HistoricalDataLoader
    from unittest.mock import MagicMock

    mock_repo = MagicMock()
    loader = HistoricalDataLoader(repo=mock_repo)
    loader._persist_pitch_events(pd.DataFrame())
    loader._validate_completeness(pd.DataFrame(), ["era"])
    assert not mock_repo.save_pitch_events.called


def test_historical_loader_validate_completeness():
    """Verify completeness validation logic."""
    from algomlb.ingestion.historical import HistoricalDataLoader
    from unittest.mock import MagicMock

    loader = HistoricalDataLoader(repo=MagicMock())
    # > 20% NaNs
    df = pd.DataFrame({"era": [1.0, 2.0, 3.0, None, None]})
    with pytest.raises(ValueError, match="Excessive NaNs detected in era"):
        loader._validate_completeness(df, ["era"])


def test_circuit_breaker_open_runtime_error():
    """Verify _request throws RuntimeError when circuit breaker is OPEN."""
    from algomlb.ingestion.http_client import BaseAPIClient
    import time

    client = BaseAPIClient(base_url="https://api.test")
    client._circuit_breaker.state = "OPEN"
    client._circuit_breaker.last_failure_time = time.time() + 9999.0
    with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
        client._request("GET", "/")


def test_api_client_failure_record(respx_mock):
    """Verify HTTP failure records in circuit breaker."""
    from algomlb.ingestion.http_client import BaseAPIClient
    import httpx

    url = "https://api.test/error"
    respx_mock.get(url).mock(return_value=httpx.Response(500))

    client = BaseAPIClient(base_url="https://api.test")
    with pytest.raises(httpx.HTTPStatusError):
        client._request("GET", "/error")

    # 3 retries = 3 failure records
    assert client._circuit_breaker.failure_count == 3


def test_historical_loader_parse_date_string():
    """Verify date parsing formats."""
    from algomlb.ingestion.historical import HistoricalDataLoader
    from unittest.mock import MagicMock
    import datetime

    loader = HistoricalDataLoader(repo=MagicMock())
    row = pd.Series({"game_date": "2024-04-01", "pitcher": 123, "batter": 456})
    event = loader._row_to_pitch_event(row, pd.Timestamp("2024-04-01").date())
    assert event.game_date.year == 2024

    # Test datetime.datetime and datetime.date persistence parsing
    df2 = pd.DataFrame(
        [{"game_date": datetime.datetime(2024, 4, 1), "pitcher": 123, "batter": 456}]
    )
    loader._persist_pitch_events(df2)
    df3 = pd.DataFrame(
        [{"game_date": datetime.date(2024, 4, 1), "pitcher": 123, "batter": 456}]
    )
    loader._persist_pitch_events(df3)


def test_circuit_breaker_half_open_logic():
    """Verify HALF_OPEN status transition."""
    from algomlb.ingestion.http_client import CircuitBreaker
    import time

    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.is_available() is False

    time.sleep(0.02)
    assert cb.is_available() is True
    assert cb.state == "HALF_OPEN"


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


def test_historical_loader_pitching(tmp_path: Path):
    """Verify that HistoricalDataLoader fetches pitching stats and persists them."""
    repo = MagicMock()
    loader = HistoricalDataLoader(repo, cache_dir=tmp_path)

    with patch("pybaseball.pitching_stats") as mock_stats:
        # Mock dataframe with standard pybaseball-style columns (PascalCase)
        df = pd.DataFrame(
            {
                "ID": [101, 102],
                "ERA": [3.50, 4.20],
                "FIP": [3.40, 4.10],
                "xFIP": [3.30, 4.00],
                "SIERA": [3.20, 3.90],
            }
        )
        mock_stats.return_value = df

        result_df = loader.fetch_pitching_stats(2023, 2023)

        assert len(result_df) == 2
        assert repo.save_historical_data.called
        # Verify columns are cleaned (ID -> id)
        assert "id" in result_df.columns
        assert "era" in result_df.columns

        # Verify persistence called with ORMs
        orms = repo.save_historical_data.call_args[0][0]
        assert len(orms) == 8  # 2 players * 4 metrics (era, fip, xfip, siera)
        assert orms[0].player_id == 101

        # Test cache hit (mock_stats should not be called again)
        mock_stats.reset_mock()
        result_df_cached = loader.fetch_pitching_stats(2023, 2023)
        assert len(result_df_cached) == 2
        mock_stats.assert_not_called()


def test_historical_loader_statcast(tmp_path: Path):
    """Verify that HistoricalDataLoader fetches Statcast data and persists it."""
    repo = MagicMock()
    loader = HistoricalDataLoader(repo, cache_dir=tmp_path)

    with patch("pybaseball.statcast") as mock_statcast:
        df = pd.DataFrame(
            {
                "game_date": ["bad_date", "2023-04-01", "2023-04-01"],
                "game_pk": [999, 601.0, 601.0],
                "pitcher": [1, 1, 1],
                "batter": [4, 2, 3],
                "release_speed": [90.0, 95.0, 94.0],
            }
        )
        mock_statcast.return_value = df

        result_df = loader.fetch_statcast("2023-04-01", "2023-04-01")

        assert len(result_df) == 3
        assert repo.save_pitch_events.called
        events = repo.save_pitch_events.call_args[0][0]
        # 'bad_date' should trigger the except block (Line 145-146)
        assert len(events) == 2

        # Test cache hit (Line 98)
        mock_statcast.reset_mock()
        result_df_cached = loader.fetch_statcast("2023-04-01", "2023-04-01")
        assert len(result_df_cached) == 3
        mock_statcast.assert_not_called()
