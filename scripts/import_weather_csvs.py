import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger
from sqlalchemy import select

from algomlb.db.models import (
    BallparkORM,
    GameResultORM,
    OpenMeteoDailyForecastORM,
    OpenMeteoWeatherProgressionORM,
)
from algomlb.db.session import get_session_factory
from algomlb.domain.wind_physics import circular_std, wind_components

# =============================================================================
# CONFIGURATION
# =============================================================================

# Mapping from DB Team Short Codes to CSV Stadium Codes
TEAM_NAME_MAP = {
    "AZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC",
    "CWS": "CWS",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KC",
    "LAA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYM",
    "NYY": "NYY",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SD",
    "SF": "SF",
    "SEA": "SEA",
    "STL": "STL",
    "TB": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WSH",
}


class WeatherCSVImporter:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def run(self):
        session_factory = get_session_factory()
        with session_factory() as session:
            # 1. Map Stadium Code to Ballpark ID / Bearing
            ballparks = session.execute(select(BallparkORM)).scalars().all()
            # Map ID -> Stadium Name/Code for fast lookup
            id_to_code = {
                bp.id: TEAM_NAME_MAP.get(bp.team_name, bp.team_name) for bp in ballparks
            }
            id_to_bp = {bp.id: bp for bp in ballparks}

            # 2. Identify Years represented in data dir
            hourly_files = list(self.data_dir.glob("mlb_weather_hourly_*.csv"))
            years = sorted(
                [int(f.name.split("_")[-1].split(".")[0]) for f in hourly_files]
            )

            for year in years:
                logger.info(f"🏆 Processing {year} data ingestion...")

                # Fetch games for this year using a simple filter
                stmt = select(GameResultORM).filter(
                    GameResultORM.game_date >= datetime.date(year, 1, 1),
                    GameResultORM.game_date <= datetime.date(year, 12, 31),
                )
                games = session.execute(stmt).scalars().all()
                if not games:
                    logger.warning(f"⚠️ No games found in DB for {year}.")
                    continue

                logger.info(
                    f"📍 Found {len(games)} games for {year}. Matching weather data..."
                )

                # Load CSVs
                h_path = self.data_dir / f"mlb_weather_hourly_{year}.csv"
                f_path = self.data_dir / f"mlb_forecast_daily_{year}.csv"

                df_hourly = pd.read_csv(h_path)
                df_hourly["ts_iso"] = pd.to_datetime(
                    df_hourly["datetime_utc"], utc=True
                ).dt.strftime("%Y-%m-%dT%H:00:00Z")
                hourly_lookup = df_hourly.set_index(["stadium", "ts_iso"]).to_dict(
                    "index"
                )

                forecast_lookup = {}
                if f_path.exists():
                    df_forecast = pd.read_csv(f_path)
                    df_forecast["date_str"] = pd.to_datetime(
                        df_forecast["date_utc"]
                    ).dt.strftime("%Y-%m-%d")
                    forecast_lookup = df_forecast.set_index(
                        ["stadium", "date_str"]
                    ).to_dict("index")

                with session_factory() as bulk_session:
                    inserted_p = 0
                    inserted_f = 0

                    for game in games:
                        code = id_to_code.get(game.ballpark_id)
                        if not code:
                            logger.warning(
                                f"Skipping game {game.game_id}: No code found for ballpark {game.ballpark_id}"
                            )
                            continue

                        ballpark = id_to_bp.get(game.ballpark_id)
                        if not ballpark:
                            logger.warning(
                                f"Skipping game {game.game_id}: Ballpark {game.ballpark_id} not found in DB"
                            )
                            continue
                        bearing = ballpark.hp_bearing_deg or 0.0

                        # A. WEATHER PROGRESSION (HOURLY)
                        if game.game_datetime:
                            # Use TZ-aware compare for safety
                            t0 = game.game_datetime.astimezone(ZoneInfo("UTC")).replace(
                                minute=0, second=0, microsecond=0
                            )
                            rows = []
                            for h_offset in range(5):
                                ts_iso = (
                                    t0 + datetime.timedelta(hours=h_offset)
                                ).strftime("%Y-%m-%dT%H:00:00Z")
                                row = hourly_lookup.get((code, ts_iso))
                                if row:
                                    rows.append(row)

                            if len(rows) >= 4:
                                try:
                                    wc_t0 = wind_components(
                                        rows[0]["wind_speed_10m_mph"],
                                        rows[0]["wind_direction_10m_deg"],
                                        bearing,
                                    )
                                    wc_t3 = wind_components(
                                        rows[min(3, len(rows) - 1)][
                                            "wind_speed_10m_mph"
                                        ],
                                        rows[min(3, len(rows) - 1)][
                                            "wind_direction_10m_deg"
                                        ],
                                        bearing,
                                    )

                                    temps = [r["temperature_2m_f"] for r in rows]
                                    winds = [r["wind_speed_10m_mph"] for r in rows]
                                    dirs = [r["wind_direction_10m_deg"] for r in rows]

                                    bulk_session.merge(
                                        OpenMeteoWeatherProgressionORM(
                                            game_id=game.game_id,
                                            **{
                                                f"temp_t{i}_f": rows[i][
                                                    "temperature_2m_f"
                                                ]
                                                for i in range(len(rows))
                                            },
                                            **{
                                                f"wind_speed_t{i}": rows[i][
                                                    "wind_speed_10m_mph"
                                                ]
                                                for i in range(len(rows))
                                            },
                                            **{
                                                f"wind_dir_t{i}": rows[i][
                                                    "wind_direction_10m_deg"
                                                ]
                                                for i in range(len(rows))
                                            },
                                            humidity_t0=rows[0][
                                                "relative_humidity_2m_pct"
                                            ],
                                            precip_t0_mm=rows[0]["precipitation_mm"],
                                            pressure_t0_hpa=rows[0][
                                                "surface_pressure_hpa"
                                            ],
                                            cloud_cover_t0_pct=rows[0][
                                                "cloud_cover_pct"
                                            ],
                                            temp_delta_game=rows[min(3, len(rows) - 1)][
                                                "temperature_2m_f"
                                            ]
                                            - rows[0]["temperature_2m_f"],
                                            temp_min_game=min(temps),
                                            wind_speed_max_game=max(winds),
                                            wind_dir_variance_deg=circular_std(dirs),
                                            headwind_t0_mph=wc_t0.headwind_mph,
                                            headwind_t3_mph=wc_t3.headwind_mph,
                                            headwind_delta_game=wc_t3.headwind_mph
                                            - wc_t0.headwind_mph,
                                            crosswind_t0_mph=wc_t0.crosswind_mph,
                                            wind_shift_gt_45deg=circular_std(dirs)
                                            > 45.0,
                                            temp_drop_gt_10f=(
                                                rows[min(3, len(rows) - 1)][
                                                    "temperature_2m_f"
                                                ]
                                                - rows[0]["temperature_2m_f"]
                                            )
                                            < -10.0,
                                            precip_any_game=any(
                                                r["precipitation_mm"] > 0.1
                                                for r in rows
                                            ),
                                            forecast_temp_f=rows[0]["temperature_2m_f"],
                                            forecast_wind_speed_mph=rows[0][
                                                "wind_speed_10m_mph"
                                            ],
                                            forecast_wind_dir_deg=rows[0][
                                                "wind_direction_10m_deg"
                                            ],
                                            forecast_headwind_mph=wc_t0.headwind_mph,
                                            forecast_crosswind_mph=wc_t0.crosswind_mph,
                                            forecast_precip_prob=float(
                                                any(
                                                    r["precipitation_mm"] > 0.1
                                                    for r in rows
                                                )
                                            ),
                                            forecast_cloud_cover_pct=rows[0][
                                                "cloud_cover_pct"
                                            ],
                                            forecast_source="historical_forecast_seamless"
                                            if year >= 2022
                                            else "era5_archive",
                                            era5_model_used="best_match"
                                            if year >= 2021
                                            else "era5",
                                        )
                                    )
                                    inserted_p += 1
                                except Exception as e:
                                    logger.error(
                                        f"Error processing weather for game {game.game_id}: {e}"
                                    )

                        # B. DAILY FORECASTS
                        f_row = forecast_lookup.get(
                            (code, game.game_date.strftime("%Y-%m-%d"))
                        )
                        if f_row:
                            bulk_session.merge(
                                OpenMeteoDailyForecastORM(
                                    game_id=game.game_id,
                                    temp_max_f=f_row["temperature_2m_max_f"],
                                    temp_min_f=f_row["temperature_2m_min_f"],
                                    precip_sum_mm=f_row["precipitation_sum_mm"],
                                    wind_speed_max_mph=f_row["wind_speed_10m_max_mph"],
                                    weather_code=int(f_row["weather_code"]),
                                    precip_prob_max_pct=f_row[
                                        "precipitation_probability_max_pct"
                                    ],
                                    uv_index_max=f_row["uv_index_max"],
                                    sunshine_duration_sec=f_row[
                                        "sunshine_duration_sec"
                                    ],
                                )
                            )
                            inserted_f += 1

                        if (inserted_p + inserted_f) % 500 == 0:
                            bulk_session.commit()

                    bulk_session.commit()
                logger.info(
                    f"✅ Year {year}: Processed {inserted_p} progression rows, {inserted_f} forecast rows."
                )


if __name__ == "__main__":
    importer = WeatherCSVImporter()
    importer.run()
