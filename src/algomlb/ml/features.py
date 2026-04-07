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
        elo_df: pd.DataFrame | None = None,
        pythag_df: pd.DataFrame | None = None,
        re24_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Merge Gold Layer features onto historical games.

        Phase 1: Pitcher matchup (h_sp_ / a_sp_) — exact temporal join.
        Phase 2: Team batting (h_bat_ / a_bat_) — lineup-aware aggregate.
        Phase 2b: Team Elo spine — slow-moving franchise strength prior.
        Phase 2c: Pythagorean expectation — fundamental run production law.
        Phase 2d: RE24 — context-neutral run production/prevention.

        Args:
            games_df: DataFrame of GameResultORM records.
            pitcher_gold_df: Gold features for PITCHER role.
            lineups_df: (Optional) DataFrame from game_lineups table.
            batter_gold_df: (Optional) Gold features for BATTER role.
            elo_df: (Optional) DataFrame from team_elo_history table.
            pythag_df: (Optional) DataFrame from sabermetrics.compute_pythagorean_features.
            re24_df: (Optional) DataFrame from sabermetrics.compute_rolling_re24.

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

        # ── 2b. Team Elo Spine ────────────────────────────────────────────
        if elo_df is not None and not elo_df.empty:
            elo = elo_df.copy()
            elo["game_date"] = pd.to_datetime(elo["game_date"])

            # One row per (game_pk, is_home)
            home_elo = elo[elo["is_home"]].rename(
                columns={
                    "elo_pre": "home_team_elo_pre",
                    "elo_post": "home_team_elo_post",
                }
            )[["game_pk", "home_team_elo_pre", "home_team_elo_post"]]

            away_elo = elo[~elo["is_home"]].rename(
                columns={
                    "elo_pre": "away_team_elo_pre",
                    "elo_post": "away_team_elo_post",
                }
            )[["game_pk", "away_team_elo_pre", "away_team_elo_post"]]

            # Ensure df has game_pk_int from earlier logic
            if "game_pk_int" not in df.columns:
                if "game_id" in df.columns:
                    df["game_pk_int"] = pd.to_numeric(df["game_id"], errors="coerce")
                elif "game_pk" in df.columns:
                    df["game_pk_int"] = df["game_pk"]

            df = df.merge(
                home_elo,
                left_on="game_pk_int",
                right_on="game_pk",
                how="left",
                suffixes=("", "_homeelo"),
            ).drop(columns=["game_pk_homeelo"], errors="ignore")

            df = df.merge(
                away_elo,
                left_on="game_pk_int",
                right_on="game_pk",
                how="left",
                suffixes=("", "_awayelo"),
            ).drop(columns=["game_pk_awayelo"], errors="ignore")

            df["elo_diff"] = df["home_team_elo_pre"] - df["away_team_elo_pre"]
            logger.info(f"Elo features attached. Shape after join: {df.shape}")

        # ── 2c. Pythagorean Expectation Spine ─────────────────────────────
        if pythag_df is not None and not pythag_df.empty:
            pyth = pythag_df.copy()

            home_pyth = pyth[pyth["is_home"]].rename(
                columns={
                    "pythag_win_pct": "h_pythag_win_pct",
                    "roll_run_diff": "h_roll_run_diff",
                    "roll_rs_per_game": "h_roll_rs_per_game",
                    "roll_ra_per_game": "h_roll_ra_per_game",
                }
            )[["game_pk", "h_pythag_win_pct", "h_roll_run_diff", "h_roll_rs_per_game", "h_roll_ra_per_game"]]

            away_pyth = pyth[~pyth["is_home"]].rename(
                columns={
                    "pythag_win_pct": "a_pythag_win_pct",
                    "roll_run_diff": "a_roll_run_diff",
                    "roll_rs_per_game": "a_roll_rs_per_game",
                    "roll_ra_per_game": "a_roll_ra_per_game",
                }
            )[["game_pk", "a_pythag_win_pct", "a_roll_run_diff", "a_roll_rs_per_game", "a_roll_ra_per_game"]]

            if "game_pk_int" not in df.columns:
                if "game_id" in df.columns:
                    df["game_pk_int"] = pd.to_numeric(df["game_id"], errors="coerce")
                elif "game_pk" in df.columns:
                    df["game_pk_int"] = df["game_pk"]

            df = df.merge(
                home_pyth, left_on="game_pk_int", right_on="game_pk",
                how="left", suffixes=("", "_hpyth"),
            ).drop(columns=["game_pk_hpyth"], errors="ignore")
            df = df.merge(
                away_pyth, left_on="game_pk_int", right_on="game_pk",
                how="left", suffixes=("", "_apyth"),
            ).drop(columns=["game_pk_apyth"], errors="ignore")

            df["pythag_diff"] = df["h_pythag_win_pct"] - df["a_pythag_win_pct"]
            logger.info(f"Pythagorean features attached. Shape after join: {df.shape}")

        # ── 2d. RE24 Spine ─────────────────────────────────────────────
        if re24_df is not None and not re24_df.empty:
            re24 = re24_df.copy()
            re24["game_date"] = pd.to_datetime(re24["game_date"])

            # Pitcher RE24: join directly by pitcher_id + game_date
            pitcher_re24 = re24[re24["role"] == "PITCHER"][["player_id", "game_date", "roll_re24"]].copy()

            # Home SP RE24
            h_pre24 = pitcher_re24.rename(columns={"roll_re24": "h_sp_roll_re24", "player_id": "h_sp_pid_re24"})
            df = df.merge(
                h_pre24, left_on=["home_pitcher_id", "game_date"],
                right_on=["h_sp_pid_re24", "game_date"], how="left",
            ).drop(columns=["h_sp_pid_re24"], errors="ignore")

            # Away SP RE24
            a_pre24 = pitcher_re24.rename(columns={"roll_re24": "a_sp_roll_re24", "player_id": "a_sp_pid_re24"})
            df = df.merge(
                a_pre24, left_on=["away_pitcher_id", "game_date"],
                right_on=["a_sp_pid_re24", "game_date"], how="left",
            ).drop(columns=["a_sp_pid_re24"], errors="ignore")

            # Batter RE24: aggregate across lineup (mean of 9 starters)
            if lineups_df is not None and not lineups_df.empty:
                batter_re24 = re24[re24["role"] == "BATTER"][["player_id", "game_date", "roll_re24"]].copy()
                lu = lineups_df.copy()
                lu["game_date"] = pd.to_datetime(lu["game_date"])
                lu_re24 = lu.merge(batter_re24, on=["player_id", "game_date"], how="left")

                for side, prefix in [("home", "h_bat_roll_re24"), ("away", "a_bat_roll_re24")]:
                    side_agg = (
                        lu_re24[lu_re24["team_side"] == side]
                        .groupby("game_pk")["roll_re24"]
                        .mean()
                        .reset_index()
                        .rename(columns={"roll_re24": prefix})
                    )
                    if "game_pk_int" in df.columns:
                        df = df.merge(side_agg, left_on="game_pk_int", right_on="game_pk", how="left", suffixes=("", f"_{side}re24"))
                        df = df.drop(columns=[f"game_pk_{side}re24"], errors="ignore")

            # Compute diff
            if "h_sp_roll_re24" in df.columns and "a_sp_roll_re24" in df.columns:
                df["re24_sp_diff"] = df["h_sp_roll_re24"] - df["a_sp_roll_re24"]

            logger.info(f"RE24 features attached. Shape after join: {df.shape}")

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
        # Non-prefixed features we want to keep explicitly:
        extra_features = [
            "home_team_elo_pre",
            "away_team_elo_pre",
            "elo_diff",
            "h_pythag_win_pct",
            "a_pythag_win_pct",
            "pythag_diff",
            "h_roll_run_diff",
            "a_roll_run_diff",
            "h_roll_rs_per_game",
            "a_roll_rs_per_game",
            "h_roll_ra_per_game",
            "a_roll_ra_per_game",
            "h_sp_roll_re24",
            "a_sp_roll_re24",
            "h_bat_roll_re24",
            "a_bat_roll_re24",
            "re24_sp_diff",
        ]

        drop_patterns = [
            "player_id", "_game_date", "season", "role",
            "computed_at", "id", "game_pk"
        ]

        feature_cols = [
            c for c in df.columns
            if (
                any(c.startswith(p) for p in keep_prefixes)
                or c in extra_features
            )
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

