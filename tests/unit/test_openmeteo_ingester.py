import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.orm import Session

from algomlb.ingestion.openmeteo_ingester import OpenMeteoIngester


@pytest.fixture
def mock_session():
    return MagicMock(spec=Session)


@pytest.fixture
def mock_om_client():
    with patch("openmeteo_requests.Client") as mock:
        yield mock.return_value


@pytest.fixture
def ingester(mock_session, mock_om_client):
    return OpenMeteoIngester(mock_session, proxies=["http://proxy1"])


def test_fetch_game_weather_progression(mock_om_client, ingester):
    # Mock ERA5 response
    mock_hourly = MagicMock()
    # 7 variables: temp, wind_speed, wind_dir, precip, humidity, pressure, cloud_cover
    data_points = 48

    era5_vars = [
        np.full(data_points, 75.0),  # 0: temp
        np.full(data_points, 10.0),  # 1: wind_speed
        np.full(data_points, 180.0),  # 2: wind_dir
        np.full(data_points, 0.0),  # 3: precip
        np.full(data_points, 50.0),  # 4: humidity
        np.full(data_points, 1013.0),  # 5: pressure
        np.full(data_points, 20.0),  # 6: cloud_cover
    ]
    mock_hourly.Variables.side_effect = lambda idx: MagicMock(
        ValuesAsNumpy=lambda: era5_vars[idx]
    )

    # Mock Forecast response
    mock_f_hourly = MagicMock()
    f_vars = [
        np.full(data_points, 70.0),  # 0: temp
        np.full(data_points, 15.0),  # 1: wind_speed
        np.full(data_points, 200.0),  # 2: wind_dir
        np.full(data_points, 10.0),  # 3: precip_prob
        np.full(data_points, 30.0),  # 4: cloud_cover
    ]
    mock_f_hourly.Variables.side_effect = lambda idx: MagicMock(
        ValuesAsNumpy=lambda: f_vars[idx]
    )

    mock_resp_era5 = MagicMock()
    mock_resp_era5.Hourly.return_value = mock_hourly

    mock_resp_forecast = MagicMock()
    mock_resp_forecast.Hourly.return_value = mock_f_hourly

    mock_om_client.weather_api.side_effect = [
        [mock_resp_era5],  # Archive call
        [mock_resp_forecast],  # Forecast call
    ]

    res = ingester.fetch_game_weather_progression(
        stadium_key="angel_stadium",
        lat=33.8,
        lon=-117.8,
        game_datetime_utc=datetime.datetime(2024, 5, 20, 19, 0, tzinfo=datetime.UTC),
        timezone="America/Los_Angeles",
    )

    assert res["temp_t0_f"] == 75.0
    assert res["forecast_temp_f"] == 70.0
    assert res["delta_temp_f"] == 5.0
    assert res["wind_speed_t0"] == 10.0
    assert "headwind_t0_mph" in res
    assert res["forecast_source"] == "openmeteo_forecast"


@patch("algomlb.ingestion.openmeteo_ingester.logger")
def test_ingest_game_not_found(mock_logger, ingester, mock_session):
    mock_session.execute.return_value.first.return_value = None

    ingester.ingest_game("744955")
    mock_logger.warning.assert_called_with("Could not find game/ballpark for 744955")


def test_forecast_fallback_to_era5_proxy(mock_om_client, ingester):
    # ERA5 works
    mock_hourly = MagicMock()
    era5_vars = [
        np.full(48, 80.0),  # 0: temp
        np.full(48, 5.0),  # 1: wind_speed
        np.full(48, 90.0),  # 2: wind_dir
        np.full(48, 0.0),  # 3: precip
        np.full(48, 40.0),  # 4: humidity
        np.full(48, 1010.0),  # 5: pressure
        np.full(48, 10.0),  # 6: cloud_cover
    ]
    mock_hourly.Variables.side_effect = lambda idx: MagicMock(
        ValuesAsNumpy=lambda: era5_vars[idx]
    )
    mock_resp_era5 = MagicMock()
    mock_resp_era5.Hourly.return_value = mock_hourly

    # Forecast fails
    mock_om_client.weather_api.side_effect = [
        [mock_resp_era5],
        RuntimeError("API Down"),
    ]

    res = ingester.fetch_game_weather_progression(
        stadium_key="angel_stadium",
        lat=33.8,
        lon=-117.8,
        game_datetime_utc=datetime.datetime(2024, 5, 20, 19, 0, tzinfo=datetime.UTC),
        timezone="America/Los_Angeles",
    )

    assert res["forecast_source"] == "era5_proxy"
    assert res["forecast_temp_f"] == 80.0
    assert res["delta_temp_f"] == 0.0


@patch("algomlb.ingestion.openmeteo_ingester.logger")
def test_ingest_game_missing_datetime(mock_logger, ingester, mock_session):
    game_orm = MagicMock()
    game_orm.game_datetime = None
    ballpark_orm = MagicMock()
    mock_session.execute.return_value.first.return_value = (game_orm, ballpark_orm)

    ingester.ingest_game("744955")
    mock_logger.warning.assert_called_with("Game 744955 missing game_datetime")


@patch("algomlb.ingestion.openmeteo_ingester.logger")
def test_ingest_game_success(mock_logger, ingester, mock_session):
    game_orm = MagicMock()
    game_orm.game_datetime = datetime.datetime(2024, 5, 1, 19, 0, tzinfo=datetime.UTC)
    ballpark_orm = MagicMock()
    ballpark_orm.ballpark = "Angel Stadium"
    ballpark_orm.team_name = "LAA"
    ballpark_orm.latitude = 33.8
    ballpark_orm.longitude = -117.8
    mock_session.execute.return_value.first.return_value = (game_orm, ballpark_orm)

    # Mock fetch_game_weather_progression to return dummy data instead of calling API
    with patch.object(ingester, "fetch_game_weather_progression") as mock_fetch:
        mock_fetch.return_value = {"temp_t0_f": 75.0, "forecast_source": "test"}
        ingester.ingest_game("744955")

    mock_session.merge.assert_called()
    mock_session.commit.assert_called()


@patch("algomlb.ingestion.openmeteo_ingester.logger")
def test_ingest_range(mock_logger, ingester, mock_session):
    # Mock games in range
    game_ids = ["gid1", "gid2"]
    mock_session.execute.return_value.scalars.return_value.all.return_value = game_ids

    with patch.object(ingester, "ingest_game") as mock_ingest:
        ingester.ingest_range(datetime.date(2024, 1, 1), datetime.date(2024, 1, 2))
        assert mock_ingest.call_count == 2
        # Comparing SQLAlchemy statement contents is tricky, but let's just assert execution
        mock_ingest.assert_any_call("gid1")
        mock_ingest.assert_any_call("gid2")


@patch("algomlb.ingestion.openmeteo_ingester.logger")
def test_ingest_game_exception_handling(mock_logger, ingester, mock_session):
    game_orm = MagicMock()
    game_orm.game_datetime = datetime.datetime(2024, 5, 1, 19, 0, tzinfo=datetime.UTC)
    ballpark_orm = MagicMock()
    ballpark_orm.ballpark = "Angel Stadium"
    ballpark_orm.team_name = "LAA"
    mock_session.execute.return_value.first.return_value = (game_orm, ballpark_orm)

    with patch.object(
        ingester, "fetch_game_weather_progression", side_effect=Exception("API Error")
    ):
        ingester.ingest_game("744955")

    mock_logger.error.assert_called()
    mock_session.rollback.assert_called()


def test_stadium_key_resolution(ingester):
    # ... abbreviation match as before ...
    bp = MagicMock()
    bp.ballpark = ""
    bp.team_name = "SF"
    with patch.object(ingester, "fetch_game_weather_progression") as mock_fetch:
        mock_fetch.return_value = {}
        game = MagicMock()
        game.game_datetime = datetime.datetime.now(datetime.UTC)
        ingester.session.execute.return_value.first.return_value = (game, bp)
        ingester.ingest_game("g1")
        assert mock_fetch.call_args[0][0] == "at_t_park"


def test_midnight_rollover(ingester, mock_om_client):
    # Mock ERA5 with rollover index (idx >= 24)
    # This requires 48h of data or just forcing first_pitch_hour=23
    mock_hourly = MagicMock()
    mock_hourly.Variables.side_effect = lambda idx: MagicMock(
        ValuesAsNumpy=lambda: np.full(48, 70.0)
    )
    mock_resp_era5 = MagicMock()
    mock_resp_era5.Hourly.return_value = mock_hourly
    # Mock Forecast
    mock_f_hourly = MagicMock()
    mock_f_hourly.Variables.side_effect = lambda idx: MagicMock(
        ValuesAsNumpy=lambda: np.full(48, 70.0)
    )
    mock_resp_forecast = MagicMock()
    mock_resp_forecast.Hourly.return_value = mock_f_hourly

    mock_om_client.weather_api.side_effect = [[mock_resp_era5], [mock_resp_forecast]]

    # 11 PM local time
    dt = datetime.datetime(2024, 5, 1, 23, 0).replace(tzinfo=datetime.UTC)
    res = ingester.fetch_game_weather_progression(
        "generic", 40, -70, dt, "America/New_York"
    )
    assert res["temp_t4_f"] == 70.0


def test_ingest_range_progress_logging(ingester, mock_session):
    # Test that it logs every 50 games
    count = 100
    mock_session.execute.return_value.scalars.return_value.all.return_value = [
        f"g{i}" for i in range(count)
    ]

    with patch.object(ingester, "ingest_game") as mock_ingest:
        with patch("algomlb.ingestion.openmeteo_ingester.logger") as mock_logger:
            ingester.ingest_range(datetime.date(2024, 5, 1), datetime.date(2024, 5, 1))
            assert mock_ingest.call_count == 100
            # Should log at 50th and 100th
            assert mock_logger.info.call_count >= 2


def test_rotate_proxy(ingester):
    ingester.proxies = ["p1", "p2"]
    ingester._proxy_idx = 0
    ingester._rotate_proxy()
    assert ingester._proxy_idx == 1
    assert ingester.cache_session.proxies["http"] == "p2"
    ingester._rotate_proxy()
    assert ingester._proxy_idx == 0
    assert ingester.cache_session.proxies["http"] == "p1"


def test_ingest_game_proxy_retry(ingester, mock_session):
    # Mock fallback to logic that fails once then succeeds
    # This is tricky because we use _ingest_game_logic
    with patch.object(ingester, "_ingest_game_logic") as mock_logic:
        mock_logic.side_effect = [Exception("Connection refused"), None]
        ingester.proxies = ["p1", "p2"]
        ingester.ingest_game("123")
        assert mock_logic.call_count == 2
        assert ingester._proxy_idx == 1


def test_ingest_game_permanent_failure(ingester, mock_session):
    with patch.object(ingester, "_ingest_game_logic") as mock_logic:
        mock_logic.side_effect = Exception("Auth failure")  # Not a proxy error usually
        ingester.ingest_game("123")
        assert mock_logic.call_count == 1
        mock_session.rollback.assert_called()
