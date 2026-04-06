import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Handles data transformation and Uranium training matrix construction."""

    # Columns to aggregate for team-level batting features
    BATTER_AGG_COLS = [
        "roll_pas", "roll_hits_per_pa", "roll_k_pct_batter", "roll_bb_pct_batter",
        "roll_barrel_pct", "roll_avg_launch_speed", "roll_avg_launch_angle",
        "roll_avg_batter_xwoba", "roll_batter_xwoba_shrunk",
        "ema_batter_xwoba_3g", "ema_batter_xwoba_7g",
        "ema_bat_speed_3g", "ema_attack_angle_3g",
        "ema_chase_pct_3g", "ema_iz_whiff_pct_3g",
        "std_batter_xwoba_15g", "std_launch_angle_15g",
        "seasonal_xwoba_vs_rh", "seasonal_xwoba_vs_lh",
    ]

    def _aggregate_team_batting(
        self,
        lineups_df: pd.DataFrame,
        batter_gold_df: pd.DataFrame,
        side: str,
    ) -> pd.DataFrame:
        """
        For each game, join the 9 starters (for 'home' or 'away') to their
        batter Gold features and return a single row of aggregated team batting.
        """
        side_lineups = lineups_df[lineups_df["team_side"] == side].copy()
        if side_lineups.empty:
            return pd.DataFrame()

        # Ensure date types match
        side_lineups["game_date"] = pd.to_datetime(side_lineups["game_date"])
        batter_gold_df["game_date"] = pd.to_datetime(batter_gold_df["game_date"])

        # Join each starter to their Gold BATTER record for that game_date
        merged = pd.merge(
            side_lineups,
            batter_gold_df,
            on=["player_id", "game_date"],
            how="left",
        )

        # Determine which aggregation columns actually exist
        available_cols = [c for c in self.BATTER_AGG_COLS if c in merged.columns]
        if not available_cols:
            logger.warning(f"No batter Gold columns found for {side} side aggregation.")
            return pd.DataFrame()

        # Aggregate: mean across the 9 starters per game
        prefix = "h_bat_" if side == "home" else "a_bat_"
        team_agg = (
            merged.groupby("game_pk")[available_cols]
            .mean()
            .reset_index()
        )
        # Rename with prefix
        team_agg = team_agg.rename(columns={c: f"{prefix}{c}" for c in available_cols})

        return team_agg

    def build_uranium_matrix(
        self,
        games_df: pd.DataFrame,
        pitcher_gold_df: pd.DataFrame,
        lineups_df: pd.DataFrame | None = None,
        batter_gold_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Merge Gold Layer features onto historical games.

        Phase 1: Pitcher matchup (h_sp_ / a_sp_) — exact temporal join.
        Phase 2: Team batting (h_bat_ / a_bat_) — lineup-aware aggregate.

        Args:
            games_df: DataFrame of GameResultORM records.
            pitcher_gold_df: Gold features for PITCHER role.
            lineups_df: (Optional) DataFrame from game_lineups table.
            batter_gold_df: (Optional) Gold features for BATTER role.

        Returns:
            X: Feature matrix
            y: Target series (home_win)
        """
        if games_df.empty or pitcher_gold_df.empty:
            logger.warning("Empty dataframes passed to Uranium pipeline.")
            return pd.DataFrame(), pd.Series()

        # Ensure datetime/date matching
        games_df = games_df.copy()
        pitcher_gold_df = pitcher_gold_df.copy()
        games_df["game_date"] = pd.to_datetime(games_df["game_date"])
        pitcher_gold_df["game_date"] = pd.to_datetime(pitcher_gold_df["game_date"])

        # ── 1. Pitcher Spine ──────────────────────────────────────────────
        df = pd.merge(
            games_df,
            pitcher_gold_df.add_prefix("h_sp_"),
            left_on=["home_pitcher_id", "game_date"],
            right_on=["h_sp_player_id", "h_sp_game_date"],
            how="left",
        )
        df = pd.merge(
            df,
            pitcher_gold_df.add_prefix("a_sp_"),
            left_on=["away_pitcher_id", "game_date"],
            right_on=["a_sp_player_id", "a_sp_game_date"],
            how="left",
        )

        # ── 2. Team Batting Spine ─────────────────────────────────────────
        if lineups_df is not None and batter_gold_df is not None:
            batter_gold_df = batter_gold_df.copy()

            # Ensure game_pk is integer in both
            if "game_id" in df.columns:
                df["game_pk_int"] = pd.to_numeric(df["game_id"], errors="coerce")
            elif "game_pk" in df.columns:
                df["game_pk_int"] = df["game_pk"]

            lineups_df = lineups_df.copy()

            home_bat = self._aggregate_team_batting(lineups_df, batter_gold_df, "home")
            away_bat = self._aggregate_team_batting(lineups_df, batter_gold_df, "away")

            if not home_bat.empty:
                df = pd.merge(df, home_bat, left_on="game_pk_int", right_on="game_pk", how="left", suffixes=("", "_hbat"))
                df = df.drop(columns=["game_pk_hbat"], errors="ignore")
            if not away_bat.empty:
                df = pd.merge(df, away_bat, left_on="game_pk_int", right_on="game_pk", how="left", suffixes=("", "_abat"))
                df = df.drop(columns=["game_pk_abat"], errors="ignore")

            logger.info(f"Team batting features attached. Shape after join: {df.shape}")

        # ── 3. Target Label ───────────────────────────────────────────────
        if "home_score" in df.columns and "away_score" in df.columns:
            df = df.dropna(subset=["home_score", "away_score"])
            df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

        if "home_win" not in df.columns:
            logger.error("No target label 'home_win' could be resolved.")
            return pd.DataFrame(), pd.Series()

        y = df["home_win"]

        # ── 4. Feature Selection ──────────────────────────────────────────
        keep_prefixes = ["h_sp_", "a_sp_", "h_bat_", "a_bat_"]
        drop_patterns = [
            "player_id", "_game_date", "season", "role",
            "computed_at", "id", "game_pk"
        ]

        feature_cols = [
            c for c in df.columns
            if any(c.startswith(p) for p in keep_prefixes)
            and not any(p in c for p in drop_patterns)
        ]

        X = df[feature_cols].copy()
        X = X.select_dtypes(include=["number"])

        # Log missing data
        missing_ratios = X.isna().mean()
        high_missing = missing_ratios[missing_ratios > 0.3]
        if not high_missing.empty:
            logger.info(f"High missing feature rates (>30%): {high_missing.to_dict()}")

        # Robust imputation
        X = X.fillna(X.median()).fillna(0)

        # Drop constant columns
        X = X.loc[:, (X != X.iloc[0]).any()]

        logger.info(f"Uranium Matrix built: {X.shape[0]} games, {X.shape[1]} features.")
        return X, y
