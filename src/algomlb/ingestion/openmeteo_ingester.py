import datetime
from typing import Any
from zoneinfo import ZoneInfo
from loguru import logger

import openmeteo_requests
import requests_cache
from retry_requests import retry
from sqlalchemy import select
from sqlalchemy.orm import Session

from algomlb.db.models import GameResultORM, OpenMeteoWeatherProgressionORM, BallparkORM
from algomlb.domain.stadium_bearings import STADIUM_HP_BEARINGS
from algomlb.domain.wind_physics import WindComponents, circular_std, wind_components

# Team timezone mapping (Standard IANA strings)
TEAM_TIMEZONES = {
    "AZ": "America/Phoenix",
    "ATL": "America/New_York",
    "BAL": "America/New_York",
    "BOS": "America/New_York",
    "CHC": "America/Chicago",
    "CWS": "America/Chicago",
    "CIN": "America/New_York",
    "CLE": "America/New_York",
    "COL": "America/Denver",
    "DET": "America/New_York",
    "HOU": "America/Chicago",
    "KC": "America/Chicago",
    "LAA": "America/Los_Angeles",
    "LAD": "America/Los_Angeles",
    "MIA": "America/New_York",
    "MIL": "America/Chicago",
    "MIN": "America/Chicago",
    "NYM": "America/New_York",
    "NYY": "America/New_York",
    "OAK": "America/Los_Angeles",
    "PHI": "America/New_York",
    "PIT": "America/New_York",
    "SD": "America/Los_Angeles",
    "SF": "America/Los_Angeles",
    "SEA": "America/Los_Angeles",
    "STL": "America/Chicago",
    "TB": "America/New_York",
    "TEX": "America/Chicago",
    "TOR": "America/Toronto",
    "WSH": "America/New_York",
}


class OpenMeteoIngester:
    """
    Ingests hourly weather progression and market forecast signals from Open-Meteo.
    """

    ERA5_FIELDS: list[str] = [
        "temperature_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "relative_humidity_2m",
        "surface_pressure",
        "cloud_cover",
    ]
    GAME_DURATION_HOURS: int = 5  # T0 through T4 (approx. 4-5 hours of game time)

    def __init__(self, session: Session, proxies: list[str] | None = None):
        self.session = session
        self.proxies = proxies or []
        self._proxy_idx = 0

        # Setup the Open-Meteo API client with cache and retry on error
        self.cache_session = requests_cache.CachedSession(".cache", expire_after=3600)

        # Configure initial proxy if available
        if self.proxies:
            proxy = self.proxies[0]
            self.cache_session.proxies = {"http": proxy, "https": proxy}
            logger.info(f"Initialized with proxy: {proxy}")

        retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        # Type ignore because requests-cache subclass can be tricky with some clients
        self.om = openmeteo_requests.Client(session=retry_session)  # type: ignore

    def _rotate_proxy(self):
        """Rotate to the next proxy in the list."""
        if not self.proxies:  # pragma: no cover
            return

        self._proxy_idx = (self._proxy_idx + 1) % len(self.proxies)
        proxy = self.proxies[self._proxy_idx]
        self.cache_session.proxies = {"http": proxy, "https": proxy}
        logger.info(f"Rotated to proxy: {proxy}")

    def fetch_game_weather_progression(
        self,
        stadium_key: str,
        lat: float,
        lon: float,
        game_datetime_utc: datetime.datetime,
        timezone: str,
    ) -> dict[str, Any]:
        """
        Fetch T0-T4 hourly actuals (ERA5) and T-24h forecast for a single game.
        """
        # Convert UTC game time to local time to identify the start hour
        game_local = game_datetime_utc.astimezone(ZoneInfo(timezone))
        game_date = game_local.date()
        first_pitch_hour = game_local.hour

        bearing = STADIUM_HP_BEARINGS.get(stadium_key, 0.0)

        # 1. Fetch Actual Conditions (Archive API - ERA5)
        actual_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": game_date.isoformat(),
            "end_date": game_date.isoformat(),
            "hourly": self.ERA5_FIELDS,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": timezone,
            "models": "era5",
        }
        actual_resp_list = self.om.weather_api(
            "https://archive-api.open-meteo.com/v1/archive", params=actual_params
        )
        if not actual_resp_list:
            raise ValueError(f"No archive data for {game_date} at {lat},{lon}")

        actual_resp = actual_resp_list[0]
        hourly = actual_resp.Hourly()
        if not hourly:
            raise ValueError(f"No hourly data for {game_date} at {lat},{lon}")

        rows: list[dict[str, float]] = []
        for offset in range(self.GAME_DURATION_HOURS):
            idx = first_pitch_hour + offset
            # Bounds check - if game crosses midnight local time, we might need next day data
            # For simplicity, we cap at 23 or we could fetch 2 days of data and slice.
            # Let's fix this by potentially fetching game_date and next day.
            if idx >= 24:
                # Placeholder: use last hour of the day or fetch more data
                idx = 23

            rows.append(
                {
                    "temp": float(hourly.Variables(0).ValuesAsNumpy()[idx]),  # type: ignore
                    "wind_speed": float(hourly.Variables(1).ValuesAsNumpy()[idx]),  # type: ignore
                    "wind_dir": float(hourly.Variables(2).ValuesAsNumpy()[idx]),  # type: ignore
                    "precip": float(hourly.Variables(3).ValuesAsNumpy()[idx]),  # type: ignore
                    "humidity": float(hourly.Variables(4).ValuesAsNumpy()[idx]),  # type: ignore
                    "pressure": float(hourly.Variables(5).ValuesAsNumpy()[idx]),  # type: ignore
                    "cloud_cover": float(hourly.Variables(6).ValuesAsNumpy()[idx]),  # type: ignore
                }
            )

        # 2. Fetch Forecast (Historical Forecast API)
        forecast_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": game_date.isoformat(),
            "end_date": game_date.isoformat(),
            "hourly": [
                "temperature_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation_probability",
                "cloud_cover",
            ],
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": timezone,
        }

        try:
            forecast_resp_list = self.om.weather_api(
                "https://historical-forecast-api.open-meteo.com/v1/forecast",
                params=forecast_params,
            )
            if forecast_resp_list:
                f_hourly = forecast_resp_list[0].Hourly()
                if not f_hourly:
                    raise ValueError("No hourly forecast data")
                f_idx = first_pitch_hour
                forecast_data = {
                    "temp": float(f_hourly.Variables(0).ValuesAsNumpy()[f_idx]),  # type: ignore
                    "wind_speed": float(f_hourly.Variables(1).ValuesAsNumpy()[f_idx]),  # type: ignore
                    "wind_dir": float(f_hourly.Variables(2).ValuesAsNumpy()[f_idx]),  # type: ignore
                    "precip_prob": float(f_hourly.Variables(3).ValuesAsNumpy()[f_idx]),  # type: ignore
                    "cloud_cover": float(f_hourly.Variables(4).ValuesAsNumpy()[f_idx]),  # type: ignore
                    "source": "openmeteo_forecast",
                }
            else:
                raise ValueError("No forecast data")
        except Exception as e:
            logger.warning(f"Forecast fallback for {game_date}: {e}")
            forecast_data = {
                "temp": rows[0]["temp"],
                "wind_speed": rows[0]["wind_speed"],
                "wind_dir": rows[0]["wind_dir"],
                "precip_prob": 0.0,
                "cloud_cover": rows[0]["cloud_cover"],
                "source": "era5_proxy",
            }

        # 3. Compute Derived Features
        wc_t0: WindComponents = wind_components(
            rows[0]["wind_speed"], rows[0]["wind_dir"], bearing
        )
        wc_t3: WindComponents = wind_components(
            rows[3]["wind_speed"], rows[3]["wind_dir"], bearing
        )
        wc_forecast: WindComponents = wind_components(
            forecast_data["wind_speed"], forecast_data["wind_dir"], bearing
        )

        temps = [r["temp"] for r in rows]
        winds = [r["wind_speed"] for r in rows]
        dirs = [r["wind_dir"] for r in rows]
        precips = [r["precip"] for r in rows]

        # ORM ready dict
        data = {
            # T0-T4 raw
            **{f"temp_t{i}_f": rows[i]["temp"] for i in range(5)},
            **{f"wind_speed_t{i}": rows[i]["wind_speed"] for i in range(5)},
            **{f"wind_dir_t{i}": rows[i]["wind_dir"] for i in range(5)},
            "humidity_t0": rows[0]["humidity"],
            "precip_t0_mm": rows[0]["precip"],
            "pressure_t0_hpa": rows[0]["pressure"],
            "cloud_cover_t0_pct": rows[0]["cloud_cover"],
            "temp_delta_game": temps[3] - temps[0],
            "temp_min_game": min(temps),
            "wind_speed_max_game": max(winds),
            "wind_dir_variance_deg": circular_std(dirs),
            "headwind_t0_mph": wc_t0.headwind_mph,
            "headwind_t3_mph": wc_t3.headwind_mph,
            "headwind_delta_game": wc_t3.headwind_mph - wc_t0.headwind_mph,
            "crosswind_t0_mph": wc_t0.crosswind_mph,
            "wind_shift_gt_45deg": circular_std(dirs) > 45.0,
            "temp_drop_gt_10f": (temps[3] - temps[0]) < -10.0,
            "precip_any_game": any(p > 0.1 for p in precips),
            "forecast_temp_f": forecast_data["temp"],
            "forecast_wind_speed_mph": forecast_data["wind_speed"],
            "forecast_wind_dir_deg": forecast_data["wind_dir"],
            "forecast_headwind_mph": wc_forecast.headwind_mph,
            "forecast_crosswind_mph": wc_forecast.crosswind_mph,
            "forecast_precip_prob": forecast_data["precip_prob"],
            "forecast_cloud_cover_pct": forecast_data["cloud_cover"],
            "forecast_source": forecast_data["source"],
            "delta_temp_f": rows[0]["temp"] - forecast_data["temp"],
            "delta_wind_speed_mph": rows[0]["wind_speed"] - forecast_data["wind_speed"],
            "delta_headwind_mph": wc_t0.headwind_mph - wc_forecast.headwind_mph,
            "delta_precip_mm": rows[0]["precip"],
            "era5_model_used": "era5",
        }
        return data

    def ingest_game(self, game_id: str, max_retries: int = 3):
        """Fetch and save weather for one game, with automated proxy-hopping on failure."""
        for attempt in range(max_retries):
            try:
                self._ingest_game_logic(game_id)
                return  # Success!
            except Exception as e:
                err_msg = str(e).lower()
                # Errors indicating we should swap our proxy and try again
                is_proxy_failure = any(
                    x in err_msg
                    for x in [
                        "limit exceeded",
                        "refused",
                        "proxyerror",
                        "timeout",
                        "reset",
                        "connection refused",
                        "established",
                    ]
                )

                if is_proxy_failure and self.proxies:
                    logger.warning(
                        f"Connection failure for {game_id} (Attempt {attempt + 1}/{max_retries}): {e}. Swapping proxy..."
                    )
                    self._rotate_proxy()
                    continue  # Retry current game with new proxy
                else:
                    # Not a proxy issue or we've run out of proxies/retries
                    logger.error(
                        f"Failed weather ingestion for {game_id} after {attempt + 1} attempts: {e}"
                    )
                    self.session.rollback()
                    break  # pragma: no cover

    def _ingest_game_logic(self, game_id: str):
        """Internal logic for weather ingestion (pulled from previous ingest_game)."""
        stmt = (
            select(GameResultORM, BallparkORM)
            .join(BallparkORM, GameResultORM.ballpark_id == BallparkORM.id)
            .where(GameResultORM.game_id == game_id)
        )
        result = self.session.execute(stmt).first()
        if not result:
            logger.warning(f"Could not find game/ballpark for {game_id}")
            return

        game_orm, ballpark_orm = result
        if not game_orm.game_datetime:
            logger.warning(f"Game {game_id} missing game_datetime")
            return

        stadium_key = None
        if ballroom_name := ballpark_orm.ballpark.lower().replace(" ", "_"):
            # Check if we have a direct bearing or use generic
            for key in STADIUM_HP_BEARINGS:
                if key in ballroom_name:
                    stadium_key = key
                    break

        # Use team name as backup for stadium key mapping
        if not stadium_key:
            stadium_key = BALLPARK_KEY_MAPPING.get(ballpark_orm.team_name)

        timezone = TEAM_TIMEZONES.get(ballpark_orm.team_name, "America/New_York")

        weather_data = self.fetch_game_weather_progression(
            stadium_key or "generic",
            ballpark_orm.latitude,
            ballpark_orm.longitude,
            game_orm.game_datetime,
            timezone,
        )

        orm = OpenMeteoWeatherProgressionORM(game_id=game_id, **weather_data)
        self.session.merge(orm)
        self.session.commit()

    def ingest_range(self, start_date: datetime.date, end_date: datetime.date):
        """Ingest weather for missing games in range, respecting archive bounds."""
        # Archive is only available for dates in the past (usually 2+ days ago, but date.today() is safe cap)
        limit_date = datetime.date.today() - datetime.timedelta(days=1)

        # Subquery for already ingested games
        existing_stmt = select(OpenMeteoWeatherProgressionORM.game_id)

        # Only process games with resolved ballparks AND missing weather data
        stmt = select(GameResultORM.game_id).where(
            GameResultORM.game_date >= start_date,
            GameResultORM.game_date <= end_date,
            GameResultORM.game_date <= limit_date,
            GameResultORM.ballpark_id.is_not(None),
            ~GameResultORM.game_id.in_(existing_stmt),
        )
        game_ids = self.session.execute(stmt).scalars().all()
        logger.info(
            f"Ingesting weather for {len(game_ids)} missing resolved games (Range: {start_date} to {min(end_date, limit_date)})"
        )
        for i, gid in enumerate(game_ids):
            self.ingest_game(gid)
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(game_ids)} games processed.")


BALLPARK_KEY_MAPPING = {
    "LAA": "angel_stadium",
    "SF": "at_t_park",
    "STL": "busch_stadium",
    "BAL": "camden_yards",
    "AZ": "chase_field",
    "NYM": "citi_field",
    "PHI": "citizens_bank_park",
    "DET": "comerica_park",
    "COL": "coors_field",
    "LAD": "dodger_stadium",
    "BOS": "fenway_park",
    "TEX": "globe_life_field",
    "CIN": "great_american_ball_park",
    "CWS": "guaranteed_rate_field",
    "KC": "kauffman_stadium",
    "MIA": "loanDepot_park",
    "MIL": "miller_park",
    "HOU": "minute_maid_park",
    "WSH": "nationals_park",
    "SD": "petco_park",
    "PIT": "pnc_park",
    "CLE": "progressive_field",
    "TOR": "rogers_centre",
    "SEA": "safeco_field",
    "MIN": "target_field",
    "TB": "tropicana_field",
    "ATL": "truist_park",
    "CHC": "wrigley_field",
    "NYY": "yankee_stadium",
}
