"""
src/algomlb/scripts/decoupler_cli.py

CLI for the Batted Ball Flight Decoupler.
Handles data loading, training, calibration, and backfilling.
"""

from __future__ import annotations

import argparse

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from algomlb.db.session import get_session_factory
from algomlb.db.models import (
    StatcastBattedBallORM,
)
from algomlb.ml.batted_ball_decoupler import BattedBallFlightDecoupler


def load_dataset(engines, start_year: int, end_year: int) -> pd.DataFrame:
    """Load Statcast + Weather + Ballpark data for the given range."""
    # We join Statcast with Weather on game_pk = game_id (cast to string)
    # and Ballpark on game.ballpark_id = ballpark.id
    query = f"""
    SELECT 
        s.game_pk, s.game_date, s.batter, s.pitcher, p.stand,
        s.launch_speed, s.launch_angle, s.hc_x, s.hc_y, s.hit_distance_sc,
        s.bb_type, s.events,
        g.ballpark_id as venue_id,
        w.temp_t0_f as temperature_f,
        w.pressure_t0_hpa as pressure_hpa,
        w.humidity_t0 as relative_humidity,
        w.wind_speed_t0 as wind_speed_mph,
        w.wind_dir_t0 as wind_direction_deg,
        w.precip_t0_mm as precipitation_mm_hr,
        b.hp_lat, b.hp_lon, b.pm_lat, b.pm_lon, b.hp_bearing_deg
    FROM statcast_raw s
    JOIN game_results g ON CAST(s.game_pk AS TEXT) = g.game_id
    JOIN pitch_events p ON CAST(s.game_pk AS TEXT) = p.game_id
        AND s.at_bat_number = p.at_bat_number 
        AND s.pitch_number = p.pitch_number
    LEFT JOIN openmeteo_weather_progression w ON g.game_id = w.game_id
    LEFT JOIN ballparks b ON g.ballpark_id = b.id
    WHERE s.game_date >= '{start_year}-01-01' AND s.game_date <= '{end_year}-12-31'
      AND s.launch_speed IS NOT NULL 
      AND s.launch_angle IS NOT NULL
      AND s.hc_x IS NOT NULL 
      AND s.hc_y IS NOT NULL
      AND s.hit_distance_sc IS NOT NULL
      AND s.events NOT IN ('strikeout', 'walk', 'hit_by_pitch', 'intent_walk', 'sac_bunt')
    """
    with engines() as session:
        df = pd.read_sql(text(query), session.bind)

    # Rename columns to match Decoupler expectations
    df = df.rename(columns={"batter": "batter_id", "pitcher": "pitcher_id"})
    return df


def run_pipeline():
    parser = argparse.ArgumentParser(description="AlgoMLB Batted Ball Decoupler CLI")
    parser.add_argument("action", choices=["train", "calibrate", "backfill", "full"])
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    session_factory = get_session_factory()
    decoupler = BattedBallFlightDecoupler(version=args.version)

    if args.action in ["train", "full"]:
        print("🚀 Loading Training Data (2019-2023)...")
        train_df = load_dataset(session_factory, 2019, 2023)
        print(f"✅ Loaded {len(train_df)} records. Training baseline...")
        decoupler.train_baseline(train_df)
        print("⭐ Baseline model trained.")

    if args.action in ["calibrate", "full"]:
        if args.action == "calibrate":
            decoupler.load()
        print("🧪 Loading Calibration Data (2024)...")
        val_df = load_dataset(session_factory, 2024, 2024)

        # Get ballpark coords for physics
        with session_factory() as session:
            coords = pd.read_sql(
                text(
                    "SELECT id, hp_lat, hp_lon, pm_lat, pm_lon, hp_bearing_deg FROM ballparks"
                ),
                session.bind,
            )

        print("⚙️ Calibrating coefficients (β, γ, δ)...")
        decoupler.calibrate(val_df, coords)
        print(f"📊 Coefficients: {decoupler.coeffs}")
        decoupler.save()

    if args.action in ["backfill", "full"]:
        decoupler.load()
        print("🌊 Loading Full Dataset for Decoupling (2019-2026)...")
        df = load_dataset(session_factory, 2019, 2026)

        with session_factory() as session:
            coords = pd.read_sql(
                text(
                    "SELECT id, hp_lat, hp_lon, pm_lat, pm_lon, hp_bearing_deg FROM ballparks"
                ),
                session.bind,
            )

        print("🧬 Decoupling environmental effects...")
        df = decoupler.preprocess(df, coords)
        results = decoupler.decouple(df)

        print("💾 Saving results to statcast_batted_balls...")
        # Chunked upsert to avoid memory issues
        chunk_size = 5000
        with session_factory() as session:
            # Clear existing for the range if needed, or just insert
            # For backfill simplicity, we'll clear first
            session.execute(text("TRUNCATE TABLE statcast_batted_balls"))
            session.commit()

            for i in tqdm(range(0, len(results), chunk_size)):
                chunk = results.iloc[i : i + chunk_size]
                # Map to ORM
                records = []
                for _, row in chunk.iterrows():
                    records.append(
                        StatcastBattedBallORM(
                            game_pk=int(row["game_pk"]),
                            game_date=row["game_date"],
                            batter_id=int(row["batter_id"]),
                            pitcher_id=int(row["pitcher_id"]),
                            venue_id=int(row["venue_id"]),
                            stand=row["stand"],
                            launch_speed=row["launch_speed"],
                            launch_angle=row["launch_angle"],
                            hc_x=row["hc_x"],
                            hc_y=row["hc_y"],
                            spray_angle=row["spray_angle"],
                            hit_distance_sc=row["hit_distance_sc"],
                            bb_type=row["bb_type"],
                            events=row["events"],
                            is_rhb=int(row["is_rhb"]),
                            temperature_f=row["temperature_f"],
                            pressure_hpa=row["pressure_hpa"],
                            relative_humidity=row["relative_humidity"],
                            wind_speed_mph=row["wind_speed_mph"],
                            wind_direction_deg=row["wind_direction_deg"],
                            precipitation_mm_hr=row["precipitation_mm_hr"],
                            cf_bearing_deg=row["cf_bearing_deg"],
                            air_density_ratio=row["air_density_ratio"],
                            tailwind_component=row["tailwind_component"],
                            baseline_distance=row["baseline_distance"],
                            total_delta=row["total_delta"],
                            delta_density=row["delta_density"],
                            delta_wind=row["delta_wind"],
                            delta_precip=row["delta_precip"],
                            environmental_factor=row["environmental_factor"],
                            spin_contact_factor=row["spin_contact_factor"],
                        )
                    )
                session.add_all(records)
                session.commit()

    print("🏁 Decoupler pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
