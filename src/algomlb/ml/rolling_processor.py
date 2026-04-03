from datetime import date
from typing import Optional
import pandas as pd
from dataclasses import dataclass
from algomlb.domain import PlayerRole, BaselineQuality


@dataclass
class PlayerRollingRecord:
    player_id: int
    game_date: date
    season: int
    role: PlayerRole
    window_games: int
    n_games_used: int
    days_since_last_game: Optional[int]
    baseline_quality: BaselineQuality
    shrinkage_applied: bool

    # Pitcher Feature Columns
    roll_pitches: Optional[float] = None
    roll_strikes_pct: Optional[float] = None
    roll_whiff_pct: Optional[float] = None
    roll_k_pct: Optional[float] = None
    roll_bb_pct: Optional[float] = None
    roll_avg_release_speed: Optional[float] = None
    roll_avg_pfx_x: Optional[float] = None
    roll_avg_pfx_z: Optional[float] = None
    roll_avg_pitcher_xwoba: Optional[float] = None
    roll_pitcher_xwoba_shrunk: Optional[float] = None

    # Batter Feature Columns
    roll_pas: Optional[float] = None
    roll_hits_per_pa: Optional[float] = None
    roll_k_pct_batter: Optional[float] = None
    roll_bb_pct_batter: Optional[float] = None
    roll_barrel_pct: Optional[float] = None
    roll_avg_launch_speed: Optional[float] = None
    roll_avg_launch_angle: Optional[float] = None
    roll_avg_batter_xwoba: Optional[float] = None
    roll_batter_xwoba_shrunk: Optional[float] = None


class RollingProcessor:
    def __init__(self, config):
        self.config = config

    def apply_shrinkage(
        self, observed: float, n: int, league_mean: float, k: float
    ) -> float:
        """Apply Bayesian shrinkage regression (n / (n + k))."""
        if n == 0:
            return league_mean
        weight = n / (n + k)
        return weight * observed + (1 - weight) * league_mean

    def compute_for_player(
        self,
        player_id: int,
        target_date: date,
        role: PlayerRole,
        history: pd.DataFrame,
        season_start: date,
    ) -> PlayerRollingRecord:
        """
        Compute rolling features for a single player/role.
        history: sorted player_game_logs.
        """
        # 1. Filter within season and before target date
        eligible = self._get_eligible_history(history, season_start, target_date)

        window = (
            self.config.pitcher_rolling_games
            if role == PlayerRole.PITCHER
            else self.config.batter_rolling_games
        )

        # Take last N
        windowed = eligible.tail(window)
        n_used = len(windowed)

        # Baseline Quality
        quality = self._determine_quality(role, n_used)

        # Staleness
        days_since = self._calculate_staleness(eligible, target_date)

        record = PlayerRollingRecord(
            player_id=player_id,
            game_date=target_date,
            season=target_date.year,
            role=role,
            window_games=window,
            n_games_used=n_used,
            days_since_last_game=days_since,
            baseline_quality=quality,
            shrinkage_applied=(n_used < window and n_used > 0),
        )

        if n_used == 0:
            return record

        if role == PlayerRole.PITCHER:
            self._compute_pitcher(record, windowed)
        else:
            self._compute_batter(record, windowed)

        return record

    def _compute_pitcher(self, record: PlayerRollingRecord, df: pd.DataFrame):
        total_pitches = df["pitches"].sum()
        if total_pitches > 0:
            record.roll_pitches = float(total_pitches)
            record.roll_strikes_pct = float(df["strikes"].sum() / total_pitches)
            record.roll_whiff_pct = float(df["whiffs"].sum() / total_pitches)
            record.roll_k_pct = float(df["k"].sum() / total_pitches)
            record.roll_bb_pct = float(df["bb"].sum() / total_pitches)

        record.roll_avg_release_speed = (
            float(df["avg_release_speed"].mean())
            if not df["avg_release_speed"].isnull().all()
            else None
        )
        record.roll_avg_pfx_x = (
            float(df["avg_pfx_x"].mean()) if not df["avg_pfx_x"].isnull().all() else None
        )
        record.roll_avg_pfx_z = (
            float(df["avg_pfx_z"].mean()) if not df["avg_pfx_z"].isnull().all() else None
        )

        observed_xwoba = df["avg_pitcher_xwoba"].mean()
        record.roll_avg_pitcher_xwoba = (
            float(observed_xwoba) if not pd.isna(observed_xwoba) else None
        )

        # Shrinkage
        record.roll_pitcher_xwoba_shrunk = float(
            self.apply_shrinkage(
                float(observed_xwoba) if not pd.isna(observed_xwoba) else 0.0,
                len(df),
                self.config.league_mean_pitcher_xwoba,
                self.config.rolling_shrinkage_k,
            )
        )

    def _compute_batter(self, record: PlayerRollingRecord, df: pd.DataFrame):
        total_pas = df["pas"].sum()
        if total_pas > 0:
            record.roll_pas = float(total_pas)
            record.roll_hits_per_pa = float(df["hits"].sum() / total_pas)
            record.roll_k_pct_batter = float(df["batter_k"].sum() / total_pas)
            record.roll_bb_pct_batter = float(df["batter_bb"].sum() / total_pas)
            record.roll_barrel_pct = float(df["barrels"].sum() / total_pas)

        record.roll_avg_launch_speed = (
            float(df["avg_launch_speed"].mean())
            if not df["avg_launch_speed"].isnull().all()
            else None
        )
        record.roll_avg_launch_angle = (
            float(df["avg_launch_angle"].mean())
            if not df["avg_launch_angle"].isnull().all()
            else None
        )

        observed_xwoba = df["avg_batter_xwoba"].mean()
        record.roll_avg_batter_xwoba = (
            float(observed_xwoba) if not pd.isna(observed_xwoba) else None
        )

        # Shrinkage
        record.roll_batter_xwoba_shrunk = float(
            self.apply_shrinkage(
                float(observed_xwoba) if not pd.isna(observed_xwoba) else 0.0,
                len(df),
                self.config.league_mean_batter_xwoba,
                self.config.rolling_shrinkage_k,
            )
        )

    def _get_eligible_history(
        self, history: pd.DataFrame, season_start: date, target_date: date
    ) -> pd.DataFrame:
        """Filter history for points-in-time calculation."""
        if history.empty or "game_date" not in history.columns:
            return pd.DataFrame(columns=["game_date"])
        return history[
            (history["game_date"] >= season_start)
            & (history["game_date"] < target_date)
        ].copy()

    def _determine_quality(self, role: PlayerRole, n_used: int) -> BaselineQuality:
        """Determine baseline quality based on role-specific window sizes."""
        if n_used == 0:
            return BaselineQuality.COLD_START

        if role == PlayerRole.PITCHER:
            return BaselineQuality.FULL if n_used >= 5 else BaselineQuality.PARTIAL
        return BaselineQuality.FULL if n_used >= 20 else BaselineQuality.PARTIAL

    def _calculate_staleness(
        self, eligible: pd.DataFrame, target_date: date
    ) -> Optional[int]:
        """Calculate days since last appearance."""
        if eligible.empty:
            return None
        last_date = eligible["game_date"].max()
        return (target_date - last_date).days
