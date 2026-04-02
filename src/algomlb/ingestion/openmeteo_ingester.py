import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import openmeteo_requests
import pandas as pd
import requests_cache
from loguru import logger
from retry_requests import retry
from sqlalchemy import select

from algomlb.db.models import BallparkORM, GameResultORM, OpenMeteoWeatherProgressionORM
from algomlb.domain.wind_physics import circular_std, wind_components

# =============================================================================
# CONFIGURATION (PROVEN COORDINATES)
# =============================================================================

MLB_STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4455, -112.0667),
    "ATL": (33.8907, -84.4677),
    "BAL": (39.2840, -76.6200),
    "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),
    "CWS": (41.8299, -87.6338),
    "CIN": (39.0979, -84.5072),
    "CLE": (41.4962, -81.6852),
    "COL": (39.7559, -104.9942),
    "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),
    "KC": (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827),
    "LAD": (34.0739, -118.2400),
    "MIA": (25.7783, -80.2195),
    "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2776),
    "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),
    "OAK": (37.7516, -122.2005),
    "PHI": (39.9061, -75.1665),
    "PIT": (40.4469, -80.0057),
    "SD": (32.7076, -117.1570),
    "SF": (37.7786, -122.3893),
    "SEA": (47.5914, -122.3325),
    "STL": (38.6226, -90.1928),
    "TB": (27.7682, -82.6534),
    "TEX": (32.7473, -97.0818),
    "TOR": (43.6414, -79.3894),
    "WSH": (38.8730, -77.0074),
}

# Mapping to DB Team Names (for ballpark lookups)
TEAM_NAME_MAP = {
    "ARI": ["Arizona Diamondbacks", "AZ", "ARI"],
    "ATL": ["Atlanta Braves", "ATL"],
    "BAL": ["Baltimore Orioles", "BAL"],
    "BOS": ["Boston Red Sox", "BOS"],
    "CHC": ["Chicago Cubs", "CHC"],
    "CWS": ["Chicago White Sox", "CWS"],
    "CIN": ["Cincinnati Reds", "CIN"],
    "CLE": ["Cleveland Guardians", "CLE"],
    "COL": ["Colorado Rockies", "COL"],
    "DET": ["Detroit Tigers", "DET"],
    "HOU": ["Houston Astros", "HOU"],
    "KC": ["Kansas City Royals", "KC"],
    "LAA": ["Los Angeles Angels", "LAA"],
    "LAD": ["Los Angeles Dodgers", "LAD"],
    "MIA": ["Miami Marlins", "MIA"],
    "MIL": ["Milwaukee Brewers", "MIL"],
    "MIN": ["Minnesota Twins", "MIN"],
    "NYM": ["New York Mets", "NYM"],
    "NYY": ["New York Yankees", "NYY"],
    "OAK": ["Oakland Athletics", "OAK"],
    "PHI": ["Philadelphia Phillies", "PHI"],
    "PIT": ["Pittsburgh Pirates", "PIT"],
    "SD": ["San Diego Padres", "SD"],
    "SF": ["San Francisco Giants", "SF"],
    "SEA": ["Seattle Mariners", "SEA"],
    "STL": ["St. Louis Cardinals", "STL"],
    "TB": ["Tampa Bay Rays", "TB"],
    "TEX": ["Texas Rangers", "TEX"],
    "TOR": ["Toronto Blue Jays", "TOR"],
    "WSH": ["Washington Nationals", "WSH"],
}


class OpenMeteoIngester:
    """
    INGESTION PRODUCTION ENGINE
    Based on the proven backfill.py coordinate-batching strategy.

    Processes season-long hourly vectors to bypass IP rate-limiting
    and perform direct-to-db game progressions.
    """

    def __init__(self, session_factory: Any):
        self.session_factory = session_factory
        self.cache_session = requests_cache.CachedSession(
            ".openmeteo_cache", expire_after=-1, allowable_codes=(200,)
        )
        retry_session = retry(self.cache_session, retries=10, backoff_factor=1.0)
        self.om = openmeteo_requests.Client(session=retry_session)  # type: ignore

    def fetch_season_weather(
        self,
        year: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Downloads weather for stadiums. Defaults to full season batch if range not provided."""
        if year >= 2026:  # For current season, use the high-reliability primary API
            url = "https://api.open-meteo.com/v1/forecast"
            models = "best_match"
        elif year >= 2022:
            url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
            models = "best_match"
        else:
            url = "https://archive-api.open-meteo.com/v1/archive"
            models = "era5"

        teams = list(MLB_STADIUM_COORDS.keys())
        lats, lons = zip(*MLB_STADIUM_COORDS.values())

        # Heuristic: Start 6h before potential first game on start_date, end 12h after potential last game on end_date
        s_date = start_date or f"{year}-03-20"
        e_date = end_date or (
            f"{year}-11-05"
            if year < datetime.date.today().year
            else datetime.date.today().strftime("%Y-%m-%d")
        )

        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": s_date,
            "end_date": e_date,
            "hourly": [
                "temperature_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
                "relative_humidity_2m",
                "surface_pressure",
                "cloud_cover",
            ],
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "GMT",
            "models": models,
        }

        all_data = []
        batch_size = 5  # Small batches to avoid 'Too many concurrent requests' (often triggered by large payload parsing)

        for i in range(0, len(teams), batch_size):
            batch_teams = teams[i : i + batch_size]
            batch_lats = lats[i : i + batch_size]
            batch_lons = lons[i : i + batch_size]

            params["latitude"] = batch_lats
            params["longitude"] = batch_lons

            logger.info(f"📡 Downloading {year} weather for stadiums {batch_teams}...")
            responses = self.om.weather_api(url, params=params)

            for j, res in enumerate(responses):
                h = res.Hourly()
                if not h:
                    continue
                # Extraction logic with type-safety
                vars = self._get_hourly_vars(h)
                if not vars:
                    logger.warning(f"Incomplete weather variables for {batch_teams[j]}")
                    continue

                df = pd.DataFrame(
                    {
                        "stadium_code": batch_teams[j],
                        "datetime_utc": pd.date_range(
                            start=pd.to_datetime(h.Time(), unit="s", utc=True),
                            end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
                            freq=pd.Timedelta(seconds=h.Interval()),
                            inclusive="left",
                            tz="UTC",
                        ),
                        "temp": vars[0].ValuesAsNumpy(),
                        "wind_speed": vars[1].ValuesAsNumpy(),
                        "wind_dir": vars[2].ValuesAsNumpy(),
                        "precip": vars[3].ValuesAsNumpy(),
                        "humidity": vars[4].ValuesAsNumpy(),
                        "pressure": vars[5].ValuesAsNumpy(),
                        "cloud_cover": vars[6].ValuesAsNumpy(),
                    }
                )
                df.ffill(inplace=True)
                df.fillna(0.0, inplace=True)
                all_data.append(df)

            # Tiny sleep to breathe
            import time

            time.sleep(1.0)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _get_hourly_vars(self, h):
        """Helper to safely extract all required hourly variables."""
        v0 = h.Variables(0)
        v1 = h.Variables(1)
        v2 = h.Variables(2)
        v3 = h.Variables(3)
        v4 = h.Variables(4)
        v5 = h.Variables(5)
        v6 = h.Variables(6)
        if any(v is None for v in [v0, v1, v2, v3, v4, v5, v6]):
            return None
        return [v0, v1, v2, v3, v4, v5, v6]

    def ingest_range(self, start_date: datetime.date, end_date: datetime.date) -> None:
        """Primary Workflow: Directly streams API vectors into the database."""
        with self.session_factory() as session:
            # 1. Fetch missing games in range
            existing = select(OpenMeteoWeatherProgressionORM.game_id)
            stmt = (
                select(GameResultORM, BallparkORM)
                .join(BallparkORM, GameResultORM.ballpark_id == BallparkORM.id)
                .where(
                    GameResultORM.game_date >= start_date,
                    GameResultORM.game_date <= end_date,
                    ~GameResultORM.game_id.in_(existing),
                )
            )
            games = session.execute(stmt).all()
            if not games:
                logger.info("✅ Database is already up to date in this range.")
                return

            games_by_year = {}
            for g, bp in games:
                y = g.game_date.year
                if y not in games_by_year:
                    games_by_year[y] = []
                games_by_year[y].append((g, bp))

        for year in sorted(games_by_year.keys(), reverse=True):
            self._process_year_weather(year, start_date, end_date, games_by_year[year])

    def _process_year_weather(self, year, start_date, end_date, year_games):
        """Processes weather for a specific year's batch of games."""
        df_weather = self.fetch_season_weather(
            year,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        if df_weather.empty:
            return

        # Use ISO-timestamp keys for absolute join stability
        df_weather["ts_iso"] = df_weather["datetime_utc"].dt.strftime(
            "%Y-%m-%dT%H:00:00Z"
        )
        weather_lookup = df_weather.set_index(["stadium_code", "ts_iso"]).to_dict(
            "index"
        )

        with self.session_factory() as bulk_session:
            inserted = 0
            for game, ballpark in year_games:
                orm = self._map_game_to_weather(year, game, ballpark, weather_lookup)
                if orm:
                    bulk_session.merge(orm)
                    inserted += 1
                    if inserted % 50 == 0:
                        logger.info(
                            f"⏳ Processed {inserted} / {len(year_games)} games for {year}..."
                        )
                    if inserted % 200 == 0:
                        bulk_session.commit()

            bulk_session.commit()
        logger.info(f"✅ Ingested {inserted} weather progressions for {year}")

    def _map_game_to_weather(self, year, game, ballpark, lookup):
        """Maps a single game to its weather progression ORM."""
        if not game.game_datetime:
            return None

        code = next(
            (
                k
                for k, synonyms in TEAM_NAME_MAP.items()
                if ballpark.team_name in synonyms
            ),
            None,
        )
        if not code:
            return None

        t0 = game.game_datetime.astimezone(ZoneInfo("UTC")).replace(
            minute=0, second=0, microsecond=0
        )
        rows = []
        for h in range(5):
            ts_iso = (t0 + datetime.timedelta(hours=h)).strftime("%Y-%m-%dT%H:00:00Z")
            row = lookup.get((code, ts_iso))
            if row:
                rows.append(row)

        if len(rows) < 4:
            return None

        return self._create_weather_orm(
            year, game.game_id, ballpark.hp_bearing_deg or 0.0, rows
        )

    def _create_weather_orm(self, year, game_id, bearing, rows):
        """Builds the OpenMeteoWeatherProgressionORM with physics calculations."""
        wc_t0 = wind_components(rows[0]["wind_speed"], rows[0]["wind_dir"], bearing)
        wc_t3 = wind_components(
            rows[min(3, len(rows) - 1)]["wind_speed"],
            rows[min(3, len(rows) - 1)]["wind_dir"],
            bearing,
        )

        # Weather vectors
        temps = [r["temp"] for r in rows]
        winds = [r["wind_speed"] for r in rows]
        dirs = [r["wind_dir"] for r in rows]

        return OpenMeteoWeatherProgressionORM(
            game_id=game_id,
            **{f"temp_t{i}_f": temps[i] for i in range(len(rows))},
            **{f"wind_speed_t{i}": winds[i] for i in range(len(rows))},
            **{f"wind_dir_t{i}": dirs[i] for i in range(len(rows))},
            humidity_t0=rows[0]["humidity"],
            precip_t0_mm=rows[0]["precip"],
            pressure_t0_hpa=rows[0]["pressure"],
            cloud_cover_t0_pct=rows[0]["cloud_cover"],
            temp_delta_game=temps[min(3, len(rows) - 1)] - temps[0],
            temp_min_game=min(temps),
            wind_speed_max_game=max(winds),
            wind_dir_variance_deg=circular_std(dirs),
            headwind_t0_mph=wc_t0.headwind_mph,
            headwind_t3_mph=wc_t3.headwind_mph,
            headwind_delta_game=wc_t3.headwind_mph - wc_t0.headwind_mph,
            crosswind_t0_mph=wc_t0.crosswind_mph,
            wind_shift_gt_45deg=circular_std(dirs) > 45.0,
            temp_drop_gt_10f=(temps[min(3, len(rows) - 1)] - temps[0]) < -10.0,
            precip_any_game=any(r["precip"] > 0.1 for r in rows),
            forecast_temp_f=rows[0]["temp"],
            forecast_wind_speed_mph=rows[0]["wind_speed"],
            forecast_wind_dir_deg=rows[0]["wind_dir"],
            forecast_headwind_mph=wc_t0.headwind_mph,
            forecast_crosswind_mph=wc_t0.crosswind_mph,
            forecast_precip_prob=float(any(r["precip"] > 0.1 for r in rows)),
            forecast_cloud_cover_pct=rows[0]["cloud_cover"],
            forecast_source="historical_forecast_seamless"
            if year >= 2022
            else "era5_archive",
            era5_model_used="best_match" if year >= 2022 else "era5",
        )
