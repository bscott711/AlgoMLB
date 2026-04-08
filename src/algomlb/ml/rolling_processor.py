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

    # Momentum & Trends (EMA)
    ema_pitcher_xwoba_3g: Optional[float] = None
    ema_pitcher_xwoba_7g: Optional[float] = None
    ema_edge_pct_3g: Optional[float] = None
    ema_velo_degradation_3g: Optional[float] = None

    # Volatility (Consistency)
    std_pitcher_xwoba_15g: Optional[float] = None
    std_edge_pct_15g: Optional[float] = None
    std_release_pos_z_15g: Optional[float] = None

    # Fatigue & Stuff Stability
    fatigue_index_7d: Optional[float] = None
    fatigue_index_14d: Optional[float] = None
    delta_spin_rate_3g: Optional[float] = None
    delta_extension_3g: Optional[float] = None
    delta_fb_velo_3g: Optional[float] = None

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

    # Momentum & Trends (EMA)
    ema_batter_xwoba_3g: Optional[float] = None
    ema_batter_xwoba_7g: Optional[float] = None
    ema_bat_speed_3g: Optional[float] = None
    ema_attack_angle_3g: Optional[float] = None
    ema_chase_pct_3g: Optional[float] = None
    ema_iz_whiff_pct_3g: Optional[float] = None

    # Volatility (Consistency)
    std_batter_xwoba_15g: Optional[float] = None
    std_launch_angle_15g: Optional[float] = None

    # Shared Splits
    seasonal_xwoba_vs_rh: Optional[float] = None
    seasonal_xwoba_vs_lh: Optional[float] = None


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
            self._compute_pitcher(record, windowed, eligible)
        else:
            self._compute_batter(record, windowed, eligible)

        return record

    def _compute_pitcher(
        self, record: PlayerRollingRecord, df: pd.DataFrame, eligible: pd.DataFrame
    ):
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
            float(df["avg_pfx_x"].mean())
            if not df["avg_pfx_x"].isnull().all()
            else None
        )
        record.roll_avg_pfx_z = (
            float(df["avg_pfx_z"].mean())
            if not df["avg_pfx_z"].isnull().all()
            else None
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

        # --- SHARP FEATURES (Phase 2) ---
        last_3 = df.tail(3)
        last_15 = df.tail(15)

        # EMA Momentum
        record.ema_pitcher_xwoba_3g = self._get_ema(df["avg_pitcher_xwoba"], 3)
        record.ema_pitcher_xwoba_7g = self._get_ema(df["avg_pitcher_xwoba"], 7)
        record.ema_edge_pct_3g = self._get_ema(df["edge_pct"], 3)
        record.ema_velo_degradation_3g = self._get_ema(
            df["fastball_velo_degradation"], 3
        )

        # Volatility
        record.std_pitcher_xwoba_15g = self._get_std(last_15["avg_pitcher_xwoba"])
        record.std_edge_pct_15g = self._get_std(last_15["edge_pct"])
        record.std_release_pos_z_15g = self._get_std(last_15["std_release_pos_z"])

        # Fatigue Index (Whiteside-style)
        record.fatigue_index_7d = self._get_fatigue_index(eligible, record.game_date, 7)
        record.fatigue_index_14d = self._get_fatigue_index(
            eligible, record.game_date, 14
        )

        # Stuff Stability (Deltas)
        season_avg_spin = eligible["avg_spin_rate"].mean()
        record.delta_spin_rate_3g = (
            float(last_3["avg_spin_rate"].mean() - season_avg_spin)
            if not pd.isna(season_avg_spin)
            else 0.0
        )

        season_avg_ext = eligible["avg_release_extension"].mean()
        record.delta_extension_3g = (
            float(last_3["avg_release_extension"].mean() - season_avg_ext)
            if not pd.isna(season_avg_ext)
            else 0.0
        )

        season_avg_velo = (
            eligible["fb_speed"].mean() if "fb_speed" in eligible.columns else None
        )
        if season_avg_velo:
            record.delta_fb_velo_3g = float(last_3["fb_speed"].mean() - season_avg_velo)

        # Platoon Stability (Seasonal)
        record.seasonal_xwoba_vs_rh = float(eligible["xwoba_vs_rh"].mean())
        record.seasonal_xwoba_vs_lh = float(eligible["xwoba_vs_lh"].mean())

    def _compute_batter(
        self, record: PlayerRollingRecord, df: pd.DataFrame, eligible: pd.DataFrame
    ):
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
        # --- SHARP FEATURES (Phase 2) ---
        last_15 = df.tail(15)

        # EMA Momentum
        record.ema_batter_xwoba_3g = self._get_ema(df["avg_batter_xwoba"], 3)
        record.ema_batter_xwoba_7g = self._get_ema(df["avg_batter_xwoba"], 7)
        record.ema_bat_speed_3g = self._get_ema(df["avg_bat_speed"], 3)
        record.ema_attack_angle_3g = self._get_ema(df["avg_attack_angle"], 3)

        # Plate Discipline Trends (Calculated from counts in game logs)
        chase_pct = df["chase_count"] / df["pas"]
        record.ema_chase_pct_3g = self._get_ema(chase_pct, 3)

        iz_whiff_pct = df["in_zone_whiff_count"] / df["pas"]
        record.ema_iz_whiff_pct_3g = self._get_ema(iz_whiff_pct, 3)

        # Volatility
        record.std_batter_xwoba_15g = self._get_std(last_15["avg_batter_xwoba"])
        record.std_launch_angle_15g = self._get_std(last_15["std_launch_angle"])

        # Platoon Stability (Seasonal)
        record.seasonal_xwoba_vs_rh = float(eligible["xwoba_vs_rh"].mean())
        record.seasonal_xwoba_vs_lh = float(eligible["xwoba_vs_lh"].mean())

    def _get_ema(self, series: pd.Series, span: int) -> Optional[float]:
        if series.dropna().empty:
            return None
        return float(series.ewm(span=span, adjust=False).mean().iloc[-1])

    def _get_std(self, series: pd.Series) -> Optional[float]:
        if len(series.dropna()) < 2:
            return 0.0
        return float(series.std())

    def _get_fatigue_index(
        self, history: pd.DataFrame, target_date: date, days: int
    ) -> float:
        """
        Whiteside-style Fatigue Index: sum(pitches * 0.7^days_ago)
        """
        cutoff = target_date - pd.Timedelta(days=days)
        # Convert game_date to datetime for comparison if needed, but here they should be date objects
        recent = history[history["game_date"] >= cutoff].copy()
        if recent.empty:
            return 0.0

        # Calculate days_ago using date objects
        recent["days_ago"] = [(target_date - d).days for d in recent["game_date"]]
        recent["fatigue_contribution"] = recent["pitches"] * (
            0.7 ** recent["days_ago"].astype(float)
        )
        return float(recent["fatigue_contribution"].sum())

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
