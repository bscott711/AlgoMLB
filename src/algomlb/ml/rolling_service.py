from datetime import date, timedelta
import pandas as pd
from sqlalchemy import select
from algomlb.db.models import StatcastPlayerGameLog, PlayerRollingFeaturesORM
from algomlb.db.repository import DatabaseRepository
from algomlb.ml.rolling_processor import RollingProcessor, PlayerRollingRecord
from algomlb.domain import PlayerRole
from algomlb.core.logger import logger


class RollingService:
    def __init__(self, db: DatabaseRepository, processor: RollingProcessor):
        self.db = db
        self.processor = processor

    def process_date_range(
        self, start_date: date, end_date: date, dry_run: bool = False
    ) -> int:
        """Process rolling features for a date range daily."""
        total_written = 0
        current = start_date
        while current <= end_date:
            written = self.process_single_date(current, dry_run=dry_run)
            total_written += written
            if written > 0:
                logger.success(f"[{current}] Completed rolling features. {written} records saved.")
            elif current.day == 1: # Log start of month as a heartbeat even if empty
                logger.info(f"[{current}] Year/Month tick...")
                
            current += timedelta(days=1)
        return total_written

    def process_single_date(self, target_date: date, dry_run: bool = False) -> int:
        """Process rolling features for a single date."""
        # 1. Identify all active (player, role) on target_date from player_game_logs
        # These are the players we need features FOR on this day.
        # Actually, in some contexts, we might want features for all rostered players,
        # but for this Epic, we consume the Silver log as defined.

        stmt = (
            select(StatcastPlayerGameLog.player_id, StatcastPlayerGameLog.role)
            .where(StatcastPlayerGameLog.game_date == target_date)
            .distinct()
        )

        active_pairs = self.db.session.execute(stmt).fetchall()
        if not active_pairs:
            logger.info(f"No active players in Silver log for {target_date}. Skipping.")
            return 0

        player_ids = list(set([p[0] for p in active_pairs]))
        season_start = self.db.get_season_start_date(target_date.year)

        # 2. Bulk fetch history for all involved players (for the current season)
        # Filters: game_date < target_date (look-back only) AND >= season_start
        history_stmt = (
            select(StatcastPlayerGameLog)
            .where(StatcastPlayerGameLog.player_id.in_(player_ids))
            .where(StatcastPlayerGameLog.game_date < target_date)
            .where(StatcastPlayerGameLog.game_date >= season_start)
            .order_by(StatcastPlayerGameLog.game_date.asc())
        )

        history_df = pd.read_sql(history_stmt, self.db.session.connection())
        if history_df.empty:
            logger.info(
                f"No history found for players active on {target_date} (Season: {target_date.year}). Cold starts only."
            )

        # 3. Compute for each pair
        records_to_save = []
        for player_id, role_str in active_pairs:
            role = PlayerRole(role_str)
            player_history = pd.DataFrame()
            if not history_df.empty and "player_id" in history_df.columns:
                player_history = history_df[
                    (history_df["player_id"] == player_id)
                    & (history_df["role"] == role_str)
                ]

            record_data = self.processor.compute_for_player(
                player_id=player_id,
                target_date=target_date,
                role=role,
                history=player_history,
                season_start=season_start,
            )

            # Map dataclass to ORM
            orm_record = PlayerRollingFeaturesORM(
                player_id=record_data.player_id,
                game_date=record_data.game_date,
                season=record_data.season,
                role=record_data.role,
                window_games=record_data.window_games,
                n_games_used=record_data.n_games_used,
                days_since_last_game=record_data.days_since_last_game,
                baseline_quality=record_data.baseline_quality,
                shrinkage_applied=record_data.shrinkage_applied,
            )

            # Feature mapping (brute force for accuracy)
            self._map_features_to_orm(orm_record, record_data)
            records_to_save.append(orm_record)

        # 4. Save
        if not dry_run:
            rows = self.db.save_player_rolling_features_records(records_to_save)
            logger.info(f"Upserted {rows} rolling features for {target_date}.")
            return rows
        else:
            logger.info(
                f"[DRY-RUN] Would have updated {len(records_to_save)} records for {target_date}."
            )
            return len(records_to_save)

    def _map_features_to_orm(
        self, orm: PlayerRollingFeaturesORM, data: PlayerRollingRecord
    ):
        """Map feature fields from dataclass to ORM (handling None/Nullable)."""
        attrs = [
            # Base Features
            "roll_pitches",
            "roll_strikes_pct",
            "roll_whiff_pct",
            "roll_k_pct",
            "roll_bb_pct",
            "roll_avg_release_speed",
            "roll_avg_pfx_x",
            "roll_avg_pfx_z",
            "roll_avg_pitcher_xwoba",
            "roll_pitcher_xwoba_shrunk",
            "roll_pas",
            "roll_hits_per_pa",
            "roll_k_pct_batter",
            "roll_bb_pct_batter",
            "roll_barrel_pct",
            "roll_avg_launch_speed",
            "roll_avg_launch_angle",
            "roll_avg_batter_xwoba",
            "roll_batter_xwoba_shrunk",
            # Momentum & Trends (EMA)
            "ema_pitcher_xwoba_3g",
            "ema_pitcher_xwoba_7g",
            "ema_edge_pct_3g",
            "ema_velo_degradation_3g",
            "ema_batter_xwoba_3g",
            "ema_batter_xwoba_7g",
            "ema_bat_speed_3g",
            "ema_attack_angle_3g",
            "ema_chase_pct_3g",
            "ema_iz_whiff_pct_3g",
            # Volatility
            "std_pitcher_xwoba_15g",
            "std_edge_pct_15g",
            "std_release_pos_z_15g",
            "std_batter_xwoba_15g",
            "std_launch_angle_15g",
            # Fatigue & Stuff stability
            "fatigue_index_7d",
            "fatigue_index_14d",
            "delta_spin_rate_3g",
            "delta_extension_3g",
            "delta_fb_velo_3g",
            # Seasonal Splits
            "seasonal_xwoba_vs_rh",
            "seasonal_xwoba_vs_lh",
        ]
        for attr in attrs:
            try:
                val = getattr(data, attr)
                if val is not None and pd.isna(val):
                    setattr(orm, attr, None)
                else:
                    setattr(orm, attr, val)
            except AttributeError:
                continue
