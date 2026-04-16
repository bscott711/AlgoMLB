import pandas as pd
import numpy as np
from sqlalchemy import text, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from algomlb.db.session import get_session_factory
from algomlb.db.models import PitcherDailyUsageORM
from algomlb.core.logger import logger

def backfill_usage(years):
    session_factory = get_session_factory()
    engine = session_factory.kw['bind']
    
    for year in years:
        logger.info(f"Processing pitcher usage backfill for {year}...")
        
        # Query retrosheet events for the year
        # We use a subquery/CTE to identify the last pitcher in each game-team stint
        query = f"""
            WITH stint_stats AS (
                SELECT game_id, pit_team, pitcher_id, date,
                       sum(nump) as pitches,
                       count(*) as pas,
                       sum(CASE WHEN outs_post >= outs_pre THEN outs_post - outs_pre ELSE 3 - outs_pre END) as outs,
                       max(inning) as max_inn,
                       max(play_number) as last_play
                FROM retrosheet_events
                WHERE EXTRACT(YEAR FROM date) = {year} AND pa_flag = 1
                GROUP BY game_id, pit_team, pitcher_id, date
            ),
            game_last_play AS (
                SELECT game_id, pit_team, max(play_number) as game_max_play
                FROM retrosheet_events
                WHERE EXTRACT(YEAR FROM date) = {year} AND pa_flag = 1
                GROUP BY game_id, pit_team
            )
            SELECT s.pitcher_id, s.date, 
                   sum(s.pitches) as total_pitches,
                   sum(s.pas) as total_pas,
                   sum(s.outs) as total_outs,
                   max(s.max_inn) as max_inning,
                   max(CASE WHEN s.last_play = g.game_max_play THEN 1 ELSE 0 END) as finished
            FROM stint_stats s
            JOIN game_last_play g ON s.game_id = g.game_id AND s.pit_team = g.pit_team
            GROUP BY s.pitcher_id, s.date
        """
        df = pd.read_sql(query, engine)
        if df.empty:
            logger.warning(f"No data found for {year}")
            continue
            
        logger.info(f"  Extracted {len(df)} pitcher-days for {year}. Persisting...")
        
        records = []
        for _, row in df.iterrows():
            records.append({
                "pitcher_id": str(row["pitcher_id"]),
                "game_date": row["date"],
                "season": year,
                "pitches_thrown": int(row["total_pitches"]),
                "pa_count": int(row["total_pas"]),
                "outs_recorded": int(row["total_outs"]),
                "max_inning": int(row["max_inning"]),
                "is_game_finished": bool(row["finished"])
            })
            
        chunk_size = 5000
        total_persisted = 0
        with engine.begin() as conn:
            for i in range(0, len(records), chunk_size):
                chunk = records[i : i + chunk_size]
                stmt = pg_insert(PitcherDailyUsageORM).values(chunk)
                upsert = stmt.on_conflict_do_update(
                    index_elements=["pitcher_id", "game_date"],
                    set_={
                        "pitches_thrown": stmt.excluded.pitches_thrown,
                        "pa_count": stmt.excluded.pa_count,
                        "outs_recorded": stmt.excluded.outs_recorded,
                    }
                )
                conn.execute(upsert)
                total_persisted += len(chunk)
                
        logger.success(f"  Successfully persisted {total_persisted} records for {year}.")

if __name__ == "__main__":
    # Work back from 2026 to 2019 as requested
    years_to_process = list(range(2026, 2018, -1))
    backfill_usage(years_to_process)
