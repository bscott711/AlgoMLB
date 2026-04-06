from __future__ import annotations

import datetime
import logging
from typing import Final, Optional

import pandas as pd
from pybaseball import statcast, cache

from algomlb.db.repository import DatabaseRepository

logger = logging.getLogger(__name__)

# Constants for column mapping and validation
RAW_COLUMNS: Final = [
    "pitch_type",
    "game_date",
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "player_name",
    "batter",
    "pitcher",
    "events",
    "description",
    "description",
    "zone",
    "des",
    "game_type",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "type",
    "hit_location",
    "bb_type",
    "balls",
    "strikes",
    "game_year",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "on_3b",
    "on_2b",
    "on_1b",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "hc_x",
    "hc_y",
    "hc_y",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "sz_top",
    "sz_bot",
    "hit_distance_sc",
    "launch_speed",
    "launch_angle",
    "effective_speed",
    "release_spin_rate",
    "release_extension",
    "game_pk",
    "fielder_2",
    "fielder_3",
    "fielder_4",
    "fielder_5",
    "fielder_6",
    "fielder_7",
    "fielder_8",
    "fielder_9",
    "release_pos_y",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "babip_value",
    "iso_value",
    "launch_speed_angle",
    "at_bat_number",
    "pitch_number",
    "pitch_name",
    "home_score",
    "away_score",
    "bat_score",
    "fld_score",
    "post_away_score",
    "post_home_score",
    "post_bat_score",
    "post_fld_score",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "spin_axis",
    "delta_home_win_exp",
    "delta_run_exp",
    "bat_speed",
    "swing_length",
    "estimated_slg_using_speedangle",
    "delta_pitcher_run_exp",
    "hyper_speed",
    "home_score_diff",
    "bat_score_diff",
    "home_win_exp",
    "bat_win_exp",
    "age_pit_legacy",
    "age_bat_legacy",
    "age_pit",
    "age_bat",
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game",
    "batter_days_since_prev_game",
    "pitcher_days_until_next_game",
    "batter_days_until_next_game",
    "api_break_z_with_gravity",
    "api_break_x_arm",
    "api_break_x_batter_in",
    "arm_angle",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
]


class StatcastIngester:
    """
    Dedicated adapter for fetching and persisting raw Statcast event data.
    Uses 7-day chunking to avoid server timeouts and ensure progressive commits.
    """

    def __init__(self, repo: Optional[DatabaseRepository] = None):
        if repo:
            self.repo = repo
        else:
            from algomlb.db.session import get_session_factory

            session_factory = get_session_factory()
            # Note: Caller is responsible for closing the session if they don't provide a repo.
            # However, for CLI use, this is usually transactional within the command.
            self.repo = DatabaseRepository(session_factory())

        # Enable pybaseball caching to handle large queries and recover from failures automatically
        cache.enable()

    def fetch_statcast_chunk(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        team: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Internal helper to fetch and normalize a single chunk of Statcast data.
        """
        s_str = start_date.strftime("%Y-%m-%d")
        e_str = end_date.strftime("%Y-%m-%d")

        try:
            chunk = statcast(start_dt=s_str, end_dt=e_str, team=team, verbose=False)  # type: ignore[arg-type]
            if chunk is None or chunk.empty:
                return pd.DataFrame()

            logger.info(f"    Fetched {len(chunk)} rows from pybaseball for {s_str}")

            # Filter to only Regular Season and Postseason games
            if "game_type" in chunk.columns:
                # R: Regular Season, F: Wild Card, D: Division Series, L: League Championship, W: World Series
                chunk = chunk[chunk["game_type"].isin(["R", "F", "D", "L", "W"])]
                logger.info(f"    {len(chunk)} rows remain after R/F/D/L/W filtering")
            else:
                logger.warning(f"    'game_type' column missing in chunk for {s_str}")

            if chunk.empty:
                return pd.DataFrame()

            # Normalize columns to lowercase and clean up names
            chunk.columns = [
                str(c).lower().replace(" ", "_").replace(".", "_")
                for c in chunk.columns
            ]

            # Filter to canonical columns
            available = [c for c in RAW_COLUMNS if c in chunk.columns]
            return chunk[available].copy()

        except Exception as e:
            logger.error(f"    Failed to fetch Statcast chunk {s_str}-{e_str}: {e}")
            return pd.DataFrame()

    def _process_rows(self, df: pd.DataFrame) -> list[dict]:
        """Convert DataFrame to list of dicts with JSON-safe NULLs."""
        rows = df.to_dict(orient="records")
        for row in rows:
            for k, v in row.items():
                if isinstance(v, float) and pd.isna(v):
                    row[k] = None
        return rows

    def ingest_range(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        team: Optional[str] = None,
        dry_run: bool = False,
    ) -> int:
        """
        Fetch and upsert Statcast events for a date range in 7-day chunks.
        Commits each chunk individually for robustness.
        """
        logger.info(
            f"🚀 Starting Statcast ingestion: {start_date} → {end_date} (team={team or 'all'})"
        )

        total_ingested = 0
        curr_start = start_date

        while curr_start <= end_date:
            curr_end = min(curr_start + datetime.timedelta(days=6), end_date)
            df = self.fetch_statcast_chunk(curr_start, curr_end, team=team)

            if not df.empty:
                if dry_run:
                    logger.info(f"    [dry-run] Would ingest {len(df)} rows.")
                else:
                    rows = self._process_rows(df)
                    count = self.repo.save_statcast_raw(rows)
                    total_ingested += count
                    logger.info(f"    ✅ Committed {count} rows to database.")
            else:
                logger.debug(f"    No data to ingest for {curr_start} to {curr_end}")

            curr_start = curr_end + datetime.timedelta(days=1)

        logger.info(f"🏁 Completed! Total rows ingested: {total_ingested}")
        return total_ingested


if __name__ == "__main__":
    import argparse
    from algomlb.core.logger import logger

    parser = argparse.ArgumentParser(description="Ingest raw Statcast data")
    parser.add_argument(
        "--start", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--team", type=str, help="Optional team abbreviation")
    parser.add_argument(
        "--dry-run", action="store_true", help="Fetch only, do not save"
    )

    args = parser.parse_args()

    start_dt = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()

    ingester = StatcastIngester()
    ingester.ingest_range(start_dt, end_dt, team=args.team, dry_run=args.dry_run)
