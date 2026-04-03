import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from algomlb.ingestion.openmeteo_ingester import OpenMeteoIngester


@pytest.fixture
def mock_session():
    mock = MagicMock()
    mock.__enter__.return_value = mock
    return mock


@pytest.fixture
def mock_om_client():
    with patch("openmeteo_requests.Client") as mock:
        yield mock.return_value


@pytest.fixture
def ingester(mock_session, mock_om_client):
    return OpenMeteoIngester(session_factory=lambda: mock_session)


def test_fetch_season_weather_success(mock_om_client, ingester):
    # Mocking Hourly data response from OpenMeteo SDK
    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = 1711843200  # 2024-03-31 00:00:00
    mock_hourly.TimeEnd.return_value = 1711843200 + 3600
    mock_hourly.Interval.return_value = 3600

    # 7 variables
    vars_mocks = [MagicMock() for _ in range(7)]
    for v in vars_mocks:
        v.ValuesAsNumpy.return_value = np.array([75.0])

    mock_hourly.Variables.side_effect = lambda idx: vars_mocks[idx]

    mock_res = MagicMock()
    mock_res.Hourly.return_value = mock_hourly
    mock_om_client.weather_api.return_value = [mock_res]

    df = ingester.fetch_season_weather(
        2024, start_date="2024-03-31", end_date="2024-03-31"
    )

    assert not df.empty
    assert "temp" in df.columns
    assert df.iloc[0]["temp"] == 75.0
    assert df.iloc[0]["stadium_code"] == "ARI"  # First in dict


def test_ingest_range_integration(mock_session, ingester):
    # Mock database games
    game = MagicMock()
    game.game_id = "123"
    game.game_date = datetime.date(2024, 4, 1)
    game.game_datetime = datetime.datetime(2024, 4, 1, 19, 0, tzinfo=datetime.UTC)
    ballpark = MagicMock()
    ballpark.id = 1
    ballpark.team_name = "ARI"
    ballpark.hp_bearing_deg = 0.0

    mock_session.execute.return_value.all.return_value = [(game, ballpark)]

    # Mock fetch_season_weather to return exactly what's needed for "AZ" at 19:00:00Z
    weather_df = pd.DataFrame(
        {
            "ballpark_id": [1] * 5,
            "stadium_code": ["AZ"] * 5,
            "datetime_utc": [
                pd.Timestamp("2024-04-01 19:00:00", tz="UTC") + pd.Timedelta(hours=i)
                for i in range(5)
            ],
            "temp": [80.0] * 5,
            "wind_speed": [10.0] * 5,
            "wind_dir": [180.0] * 5,
            "precip": [0.0] * 5,
            "humidity": [30.0] * 5,
            "pressure": [1013.0] * 5,
            "cloud_cover": [0.0] * 5,
        }
    )

    with patch.object(ingester, "fetch_weather_batch", return_value=weather_df):
        with patch(
            "algomlb.ingestion.openmeteo_ingester.TEAM_NAME_MAP", {"AZ": ["ARI"]}
        ):
            ingester.ingest_range(datetime.date(2024, 4, 1), datetime.date(2024, 4, 1))

    assert mock_session.merge.called, "Should have merged 1 record"
    assert mock_session.commit.called


def test_get_hourly_vars_safety(ingester):
    h = MagicMock()
    # Missing variable 6
    h.Variables.side_effect = lambda idx: MagicMock() if idx < 6 else None
    res = ingester._get_hourly_vars(h)
    assert res is None


def test_create_weather_orm_logic(ingester):
    rows = [
        {
            "temp": 70.0,
            "wind_speed": 10.0,
            "wind_dir": 0.0,
            "precip": 0.0,
            "humidity": 50.0,
            "pressure": 1010.0,
            "cloud_cover": 0.0,
        }
        for _ in range(5)
    ]
    orm = ingester._create_weather_orm(2024, "game1", 90.0, rows)
    assert orm.game_id == "game1"
    assert orm.temp_t0_f == 70.0
    # Bearing 90 (East), Wind 0 (North -> South) => Pure crosswind (10mph), 0 headwind
    assert abs(orm.headwind_t0_mph) < 0.1
    assert abs(orm.crosswind_t0_mph - 10.0) < 0.1


def test_ingest_range_empty_db(mock_session, ingester):
    mock_session.execute.return_value.all.return_value = []
    ingester.ingest_range(datetime.date(2024, 4, 1), datetime.date(2024, 4, 1))
    assert not mock_session.commit.called


def test_ingest_range_empty_api(ingester):
    with patch.object(ingester, "fetch_season_weather", return_value=pd.DataFrame()):
        ingester.ingest_range(datetime.date(2024, 1, 1), datetime.date(2024, 1, 1))


def test_map_game_to_weather_missing_datetime(ingester):
    game = MagicMock()
    game.game_datetime = None
    res = ingester._map_game_to_weather(2024, game, MagicMock(), {})
    assert res is None


def test_map_game_to_weather_missing_ballpark_code(ingester):
    game = MagicMock()
    game.game_datetime = datetime.datetime.now()
    ballpark = MagicMock()
    ballpark.team_name = "UNKNOWN_TEAM"
    with patch("algomlb.ingestion.openmeteo_ingester.TEAM_NAME_MAP", {}):
        res = ingester._map_game_to_weather(2024, game, ballpark, {})
    assert res is None


def test_map_game_to_weather_insufficient_rows(ingester):
    game = MagicMock()
    game.game_datetime = datetime.datetime.now()
    ballpark = MagicMock()
    ballpark.team_name = "ARI"
    with patch("algomlb.ingestion.openmeteo_ingester.TEAM_NAME_MAP", {"AZ": ["ARI"]}):
        res = ingester._map_game_to_weather(2024, game, ballpark, {})  # Empty lookup
    assert res is None


def test_fetch_season_weather_missing_hourly(mock_om_client, ingester):
    mock_res = MagicMock()
    mock_res.Hourly.return_value = None
    mock_om_client.weather_api.return_value = [mock_res]
    df = ingester.fetch_season_weather(2024)
    assert df.empty


def test_fetch_season_weather_url_branches(mock_om_client, ingester):
    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = 0
    mock_hourly.TimeEnd.return_value = 3600
    mock_hourly.Interval.return_value = 3600
    vars_mocks = [MagicMock() for _ in range(7)]
    for v in vars_mocks:
        v.ValuesAsNumpy.return_value = np.array([0.0])
    mock_hourly.Variables.side_effect = lambda idx: vars_mocks[idx]
    mock_res = MagicMock()
    mock_res.Hourly.return_value = mock_hourly
    mock_om_client.weather_api.return_value = [mock_res]

    with patch(
        "algomlb.ingestion.openmeteo_ingester.MLB_STADIUM_COORDS", {"TEST": (0, 0)}
    ):
        df1 = ingester.fetch_season_weather(2026)
        assert df1.iloc[0]["stadium_code"] == "TEST"
        df2 = ingester.fetch_season_weather(2023)
        assert df2.iloc[0]["stadium_code"] == "TEST"
        df3 = ingester.fetch_season_weather(2020)
        assert df3.iloc[0]["stadium_code"] == "TEST"


def test_fetch_season_weather_incomplete_data(mock_om_client, ingester):
    # Case: h.Variables returns None halfway through
    mock_hourly = MagicMock()
    mock_hourly.Variables.side_effect = lambda idx: MagicMock() if idx < 3 else None
    mock_res = MagicMock()
    mock_res.Hourly.return_value = mock_hourly
    mock_om_client.weather_api.return_value = [mock_res]
    with patch(
        "algomlb.ingestion.openmeteo_ingester.MLB_STADIUM_COORDS", {"TEST": (0, 0)}
    ):
        df = ingester.fetch_season_weather(2024)
    assert df.empty


def test_ingest_range_batch_logic(ingester, mock_session):
    # Test batch commit and progress logging
    games = [(MagicMock(), MagicMock()) for _ in range(201)]
    for i, (g, bp) in enumerate(games):
        g.game_id = f"g{i}"
        g.game_date = datetime.date(2024, 4, 1)
        g.game_datetime = datetime.datetime(2024, 4, 1, 10, 0, tzinfo=datetime.UTC)
        bp.team_name = "ARI"
        bp.hp_bearing_deg = 0.0

    weather_df = pd.DataFrame(
        {
            "ballpark_id": [1] * 5,
            "stadium_code": ["AZ"] * 5,
            "datetime_utc": [
                pd.Timestamp("2024-04-01 10:00:00", tz="UTC") + pd.Timedelta(hours=i)
                for i in range(5)
            ],
            "temp": [70.0] * 5,
            "wind_speed": [5.0] * 5,
            "wind_dir": [90.0] * 5,
            "precip": [0.0] * 5,
            "humidity": [50.0] * 5,
            "pressure": [1010.0] * 5,
            "cloud_cover": [0.0] * 5,
        }
    )

    with patch.object(ingester, "fetch_weather_batch", return_value=weather_df):
        with patch(
            "algomlb.ingestion.openmeteo_ingester.TEAM_NAME_MAP", {"AZ": ["ARI"]}
        ):
            # Mock the execute call in ingest_range
            mock_session.execute.return_value.all.return_value = games
            for _, bp in games:
                bp.id = 1  # Standardize ID to match weather_df
            ingester.ingest_range(datetime.date(2024, 4, 1), datetime.date(2024, 4, 1))
            # 201 games => commits at 200 and end. total >= 2
            assert mock_session.commit.call_count >= 2


def test_process_year_weather_empty_df(ingester):
    """Test early return when fetch_weather_batch returns empty DF."""
    with patch.object(ingester, "fetch_weather_batch", return_value=pd.DataFrame()):
        # Should return early and NOT call _persist_year_weather
        with patch.object(ingester, "_persist_year_weather") as mock_persist:
            ingester._process_year_weather(
                2024, datetime.date(2024, 4, 1), datetime.date(2024, 4, 1), []
            )
            mock_persist.assert_not_called()
