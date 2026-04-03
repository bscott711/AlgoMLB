from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import (
    select,
)
from sqlalchemy.dialects.postgresql import insert

from algomlb.config.settings import get_settings
from algomlb.db.models import (
    StatcastPlayerGameLog,
    StatcastProcessRegistry,
    StatcastRawORM,
)
from algomlb.db.session import get_engine

logger = logging.getLogger(__name__)
SETTINGS = get_settings()


def apply_bayesian_shrinkage(
    current_val: float,
    current_n: int,
    prior_val: Optional[float],
    prior_k: int,
) -> float:
    """
    Apply Bayesian shrinkage toward a player's own prior-year metric.
    Formula: (n * current + k * prior) / (n + k)
    If prior is None (rookie), returns current_val (no shrinkage).
    """
    if prior_val is None or pd.isna(prior_val):
        return current_val
    return (current_n * current_val + prior_k * prior_val) / (current_n + prior_k)


def summarize_to_silver(
    df: pd.DataFrame,
    prior_year_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Summarize pitch-level Statcast data into a game-level silver layer.
    Regresses metrics toward prior_year_stats if provided.
    """
    if df.empty:
        return pd.DataFrame()

    # Define Whiff & Barrel flags
    df["is_whiff"] = df["description"].isin(
        ["swinging_strike", "swinging_strike_blocked"]
    )
    df["is_barrel"] = df["launch_speed_angle"] == 6

    # 1. Pitcher Aggregation
    p_agg = (
        df.groupby(["game_pk", "pitcher", "game_date"])
        .agg(
            pitches=("pitch_number", "count"),
            strikes=(
                "description",
                lambda x: x.isin(
                    [
                        "strike",
                        "foul",
                        "swinging_strike",
                        "called_strike",
                        "swinging_strike_blocked",
                    ]
                ).sum(),
            ),
            whiffs=("is_whiff", "sum"),
            k=("events", lambda x: (x == "strikeout").sum()),
            bb=("events", lambda x: (x == "walk").sum()),
            avg_release_speed=("release_speed", "mean"),
            avg_pfx_x=("pfx_x", "mean"),
            avg_pfx_z=("pfx_z", "mean"),
            avg_pitcher_xwoba=("estimated_woba_using_speedangle", "mean"),
        )
        .reset_index()
        .rename(columns={"pitcher": "player_id"})
    )
    p_agg["role"] = "PITCHER"

    # 2. Batter Aggregation
    # In Statcast, multiple rows per PA exist. PA summary should only look at pitches with 'events'
    b_agg = (
        df.groupby(["game_pk", "batter", "game_date"])
        .agg(
            pas=("at_bat_number", "nunique"),
            hits=(
                "events",
                lambda x: x.isin(["single", "double", "triple", "home_run"]).sum(),
            ),
            batter_k=("events", lambda x: (x == "strikeout").sum()),
            batter_bb=("events", lambda x: (x == "walk").sum()),
            barrels=("is_barrel", "sum"),
            avg_launch_speed=("launch_speed", "mean"),
            avg_launch_angle=("launch_angle", "mean"),
            avg_batter_xwoba=("estimated_woba_using_speedangle", "mean"),
        )
        .reset_index()
        .rename(columns={"batter": "player_id"})
    )
    # Correct ABs = PAs - Walks - Sacs - HBP
    # For now, simplistic approximation: AB = PA - BB
    b_agg["abs"] = b_agg["pas"] - b_agg["batter_bb"]
    b_agg["role"] = "BATTER"

    logs = pd.concat([p_agg, b_agg], ignore_index=True)

    # 3. Bayesian Shrinkage (Optional)
    if prior_year_stats is not None and not prior_year_stats.empty:
        # Merge prior stats
        # prior_year_stats grain: [player_id, role, metric_avg]
        logs = logs.merge(
            prior_year_stats,
            on=["player_id", "role"],
            how="left",
            suffixes=("", "_prior"),
        )

        # Apply to Pitchers (K%, Whiff%)
        # Note: In a real implementation, we'd apply to more metrics.
        # This implementation shows the pattern.
        pk = SETTINGS.ml.quant_pitcher_shrinkage_k
        bk = SETTINGS.ml.quant_batter_shrinkage_k

        # Example: Pitcher xwOBA shrinkage
        if "avg_pitcher_xwoba_prior" in logs.columns:
            mask = logs["role"] == "PITCHER"
            logs.loc[mask, "avg_pitcher_xwoba"] = logs[mask].apply(
                lambda r: apply_bayesian_shrinkage(
                    r["avg_pitcher_xwoba"],
                    r["pitches"],
                    r["avg_pitcher_xwoba_prior"],
                    pk,
                ),
                axis=1,
            )

        # Example: Batter xwOBA shrinkage
        if "avg_batter_xwoba_prior" in logs.columns:
            mask = logs["role"] == "BATTER"
            logs.loc[mask, "avg_batter_xwoba"] = logs[mask].apply(
                lambda r: apply_bayesian_shrinkage(
                    r["avg_batter_xwoba"], r["pas"], r["avg_batter_xwoba_prior"], bk
                ),
                axis=1,
            )

    return logs


def fetch_prior_year_stats(year: int) -> pd.DataFrame:
    """
    Load a summary of the entire prior year into memory for shrinkage.
    Normally this would read from the Silver layer for the previous year.
    """
    engine = get_engine()
    # Simple seasonal rollup
    query = f"""
        SELECT player_id, role, 
               AVG(avg_pitcher_xwoba) as avg_pitcher_xwoba, 
               AVG(avg_batter_xwoba) as avg_batter_xwoba
        FROM statcast_player_game_logs
        WHERE EXTRACT(year FROM game_date) = {year}
        GROUP BY player_id, role
    """
    try:
        return pd.read_sql(query, engine)
    except Exception:
        return pd.DataFrame()


def process_silver_incremental(batch_size: int = 50000):
    """
    Idempotent incremental processor for Silver Layer.
    """
    engine = get_engine()
    target = "statcast_player_game_logs"

    with engine.connect() as conn:
        # 1. Get Checkpoint
        res = conn.execute(
            select(StatcastProcessRegistry.last_processed_ingested_at).where(
                StatcastProcessRegistry.target_table == target
            )
        ).fetchone()

        last_ingested = res[0] if res else datetime(2019, 1, 1)

        # 2. Find new game_pks
        raw_rows = conn.execute(
            select(
                StatcastRawORM.game_pk,
                StatcastRawORM.game_date,
                StatcastRawORM.ingested_at,
            )
            .where(StatcastRawORM.ingested_at > last_ingested)
            .order_by(StatcastRawORM.ingested_at.asc())
            .limit(batch_size)
        ).fetchall()

        if not raw_rows:
            logger.info("No new data to summarize for Silver.")
            return

        target_game_pks = list(set([r[0] for r in raw_rows]))
        max_ingested_in_batch = max([r[2] for r in raw_rows])

        # 3. Load full pitch detail for these games
        query = select(StatcastRawORM).where(
            StatcastRawORM.game_pk.in_(target_game_pks)
        )
        df = pd.read_sql(query, engine)

        # Load prior year context (simplistic: current year - 1)
        current_year = df["game_date"].iloc[0].year
        prior_stats = fetch_prior_year_stats(current_year - 1)

        # 4. Summarize
        silver_df = summarize_to_silver(df, prior_stats)

        # 5. Upsert
        if not silver_df.empty:
            _upsert_silver(silver_df)

        # 6. Update Checkpoint
        conn.execute(
            insert(StatcastProcessRegistry)
            .values(
                target_table=target, last_processed_ingested_at=max_ingested_in_batch
            )
            .on_conflict_do_update(
                index_elements=["target_table"],
                set_={"last_processed_ingested_at": max_ingested_in_batch},
            )
        )
        conn.commit()
        logger.info(
            f"Summarized {len(target_game_pks)} games into Silver. Resetting checkpoint to {max_ingested_in_batch}."
        )


def _upsert_silver(df: pd.DataFrame):
    """
    Idempotent write into statcast_player_game_logs.
    """
    engine = get_engine()
    records = df.to_dict(orient="records")

    for record in records:
        # Convert nan to None for DB
        record = {k: (v if not pd.isna(v) else None) for k, v in record.items()}
        # Optimization: remove prior columns used in calculation
        record = {k: v for k, v in record.items() if not str(k).endswith("_prior")}

        stmt = insert(StatcastPlayerGameLog).values(**record)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_statcast_player_game_log",
            set_={
                k: v
                for k, v in record.items()
                if k not in ["game_pk", "player_id", "role"]
            },
        )
        with engine.begin() as conn:
            conn.execute(stmt)
