import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Handles data transformation and Uranium training matrix construction."""

    # Columns to aggregate for team-level batting features
    BATTER_AGG_COLS = [
        "roll_pas",
        "roll_hits_per_pa",
        "roll_k_pct_batter",
        "roll_bb_pct_batter",
        "roll_barrel_pct",
        "roll_avg_launch_speed",
        "roll_avg_launch_angle",
        "roll_avg_batter_xwoba",
        "roll_batter_xwoba_shrunk",
        "ema_batter_xwoba_3g",
        "ema_batter_xwoba_7g",
        "ema_bat_speed_3g",
        "ema_attack_angle_3g",
        "ema_chase_pct_3g",
        "ema_iz_whiff_pct_3g",
        "std_batter_xwoba_15g",
        "std_launch_angle_15g",
        "seasonal_xwoba_vs_rh",
        "seasonal_xwoba_vs_lh",
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

        # Ensure date and ID types match
        side_lineups["game_date"] = pd.to_datetime(side_lineups["game_date"]).dt.date
        side_lineups["game_pk"] = pd.to_numeric(
            side_lineups["game_pk"], errors="coerce"
        ).astype(float)
        batter_gold_df["game_date"] = pd.to_datetime(
            batter_gold_df["game_date"]
        ).dt.date
        batter_gold_df["player_id"] = pd.to_numeric(
            batter_gold_df["player_id"], errors="coerce"
        ).astype(float)
        side_lineups["player_id"] = pd.to_numeric(
            side_lineups["player_id"], errors="coerce"
        ).astype(float)

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
        team_agg = merged.groupby("game_pk")[available_cols].mean().reset_index()
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
        Merge Gold Layer features onto historical games using a modular pipeline.
        """
        if games_df.empty or pitcher_gold_df.empty:
            logger.warning("Empty dataframes passed to Uranium pipeline.")
            return pd.DataFrame(), pd.Series()

        # 1. Normalization
        games_df, pitcher_gold_df, lineups_df, batter_gold_df = (
            self._prepare_data_for_merge(
                games_df, pitcher_gold_df, lineups_df, batter_gold_df
            )
        )

        # 2. Pitcher Matchup Spine
        df = self._attach_pitcher_spines(games_df, pitcher_gold_df)

        # 3. Team Features (Batting, Elo, Pythag, RE24)
        df = self._attach_team_spines(
            df, lineups_df, batter_gold_df, elo_df, pythag_df, re24_df
        )

        # 4. Finalize features and labels
        return self._finalize_features(df)

    def _prepare_data_for_merge(
        self,
        games_df: pd.DataFrame,
        pitcher_gold_df: pd.DataFrame,
        lineups_df: pd.DataFrame | None = None,
        batter_gold_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """Normalize IDs and dates across dataframes."""
        games_df = games_df.copy()
        pitcher_gold_df = pitcher_gold_df.copy()

        # Force IDs to float64 for merge compatibility
        for col in ["home_pitcher_id", "away_pitcher_id", "game_pk", "game_id"]:
            if col in games_df.columns:
                games_df[col] = pd.to_numeric(games_df[col], errors="coerce").astype(
                    float
                )

        pitcher_gold_df["player_id"] = pd.to_numeric(
            pitcher_gold_df["player_id"], errors="coerce"
        ).astype(float)

        games_df["game_date"] = pd.to_datetime(games_df["game_date"]).dt.date
        pitcher_gold_df["game_date"] = pd.to_datetime(
            pitcher_gold_df["game_date"]
        ).dt.date

        if lineups_df is not None:
            lineups_df = lineups_df.copy()
            lineups_df["game_pk"] = pd.to_numeric(
                lineups_df["game_pk"], errors="coerce"
            ).astype(float)
            lineups_df["game_date"] = pd.to_datetime(lineups_df["game_date"]).dt.date

        if batter_gold_df is not None:
            batter_gold_df = batter_gold_df.copy()

        return games_df, pitcher_gold_df, lineups_df, batter_gold_df

    def _attach_pitcher_spines(
        self, games_df: pd.DataFrame, pitcher_gold_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge home and away starting pitcher features."""
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
        return df

    def _attach_team_spines(
        self,
        df: pd.DataFrame,
        lineups_df: pd.DataFrame | None,
        batter_gold_df: pd.DataFrame | None,
        elo_df: pd.DataFrame | None,
        pythag_df: pd.DataFrame | None,
        re24_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Attach batting aggregates and team-level metrics."""
        # Ensure game_pk_int is present for team joins
        if "game_pk_int" not in df.columns:
            id_col = "game_id" if "game_id" in df.columns else "game_pk"
            df["game_pk_int"] = pd.to_numeric(df[id_col], errors="coerce").astype(float)

        # 1. Batting
        if lineups_df is not None and batter_gold_df is not None:
            h_bat = self._aggregate_team_batting(lineups_df, batter_gold_df, "home")
            a_bat = self._aggregate_team_batting(lineups_df, batter_gold_df, "away")
            if not h_bat.empty:
                df = df.merge(
                    h_bat,
                    left_on="game_pk_int",
                    right_on="game_pk",
                    how="left",
                    suffixes=("", "_hbat"),
                ).drop(columns=["game_pk_hbat"], errors="ignore")
            if not a_bat.empty:
                df = df.merge(
                    a_bat,
                    left_on="game_pk_int",
                    right_on="game_pk",
                    how="left",
                    suffixes=("", "_abat"),
                ).drop(columns=["game_pk_abat"], errors="ignore")

        # 2. Elo
        if elo_df is not None and not elo_df.empty:
            df = self._attach_elo_metrics(df, elo_df)

        # 3. Pythag
        if pythag_df is not None and not pythag_df.empty:
            df = self._attach_pythagorean_metrics(df, pythag_df)

        # 4. RE24
        if re24_df is not None and not re24_df.empty:
            df = self._attach_re24_metrics(df, re24_df, lineups_df)

        return df

    def _attach_elo_metrics(
        self, df: pd.DataFrame, elo_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process and merge Elo metrics."""
        elo = elo_df.copy()
        elo["game_pk"] = pd.to_numeric(elo["game_pk"], errors="coerce").astype("Int64")
        home_elo = elo[elo["is_home"]].rename(
            columns={"elo_pre": "home_team_elo_pre", "elo_post": "home_team_elo_post"}
        )[["game_pk", "home_team_elo_pre", "home_team_elo_post"]]
        away_elo = elo[~elo["is_home"]].rename(
            columns={"elo_pre": "away_team_elo_pre", "elo_post": "away_team_elo_post"}
        )[["game_pk", "away_team_elo_pre", "away_team_elo_post"]]

        df = df.merge(
            home_elo, left_on="game_pk_int", right_on="game_pk", how="left"
        ).drop(columns=["game_pk_y"], errors="ignore")
        df = df.merge(
            away_elo, left_on="game_pk_int", right_on="game_pk", how="left"
        ).drop(columns=["game_pk_y"], errors="ignore")
        df["elo_diff"] = df["home_team_elo_pre"] - df["away_team_elo_pre"]
        return df

    def _attach_pythagorean_metrics(
        self, df: pd.DataFrame, pythag_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process and merge Pythagorean Sabermetrics."""
        pyth = pythag_df.copy()
        pyth["game_pk"] = pd.to_numeric(pyth["game_pk"], errors="coerce").astype(
            "Int64"
        )
        for side, prefix in [("home", "h_"), ("away", "a_")]:
            side_pyth = pyth[pyth["is_home"] == (side == "home")].rename(
                columns={
                    "pythag_win_pct": f"{prefix}pythag_win_pct",
                    "roll_run_diff": f"{prefix}roll_run_diff",
                    "roll_rs_per_game": f"{prefix}roll_rs_per_game",
                    "roll_ra_per_game": f"{prefix}roll_ra_per_game",
                }
            )[
                [
                    "game_pk",
                    f"{prefix}pythag_win_pct",
                    f"{prefix}roll_run_diff",
                    f"{prefix}roll_rs_per_game",
                    f"{prefix}roll_ra_per_game",
                ]
            ]
            df = df.merge(
                side_pyth, left_on="game_pk_int", right_on="game_pk", how="left"
            ).drop(columns=["game_pk_y"], errors="ignore")

        df["pythag_diff"] = df["h_pythag_win_pct"] - df["a_pythag_win_pct"]
        return df

    def _attach_re24_metrics(
        self,
        df: pd.DataFrame,
        re24_df: pd.DataFrame,
        lineups_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Process and merge RE24 run expectancy metrics."""
        re24 = re24_df.copy()
        re24["game_date"] = pd.to_datetime(re24["game_date"]).dt.date
        p_re24 = re24[re24["role"] == "PITCHER"][
            ["player_id", "game_date", "roll_re24"]
        ]

        # Individual Pitcher RE24
        for side, prefix in [("home", "h_"), ("away", "a_")]:
            target_pid = f"{side}_pitcher_id"
            merged_re24 = p_re24.rename(
                columns={"roll_re24": f"{prefix}sp_roll_re24", "player_id": "pid"}
            )
            df = df.merge(
                merged_re24,
                left_on=[target_pid, "game_date"],
                right_on=["pid", "game_date"],
                how="left",
            ).drop(columns=["pid"], errors="ignore")

        # Aggregated Batter RE24
        if lineups_df is not None:
            b_re24 = re24[re24["role"] == "BATTER"][
                ["player_id", "game_date", "roll_re24"]
            ]
            lu_re24 = lineups_df.merge(
                b_re24, on=["player_id", "game_date"], how="left"
            )
            for side in ["home", "away"]:
                prefix = f"{side[0]}_bat_roll_re24"
                side_agg = (
                    lu_re24[lu_re24["team_side"] == side]
                    .groupby("game_pk")["roll_re24"]
                    .mean()
                    .reset_index()
                    .rename(columns={"roll_re24": prefix})
                )
                df = df.merge(
                    side_agg, left_on="game_pk_int", right_on="game_pk", how="left"
                ).drop(columns=["game_pk_y"], errors="ignore")

        if "h_sp_roll_re24" in df.columns and "a_sp_roll_re24" in df.columns:
            df["re24_sp_diff"] = df["h_sp_roll_re24"] - df["a_sp_roll_re24"]
        return df

    def _finalize_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Resolve labels, select features, and impute missing values."""
        if "home_score" in df.columns and "away_score" in df.columns:
            df = df.dropna(subset=["home_score", "away_score"])
            df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

        if "home_win" not in df.columns:
            logger.error("No target label 'home_win' could be resolved.")
            return pd.DataFrame(), pd.Series()

        y = df["home_win"]

        # Selection logic
        keep_prefixes = ["h_sp_", "a_sp_", "h_bat_", "a_bat_"]
        extra = [
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
        feature_cols = [
            c
            for c in df.columns
            if (any(c.startswith(p) for p in keep_prefixes) or c in extra)
            and not any(
                p in c
                for p in [
                    "player_id",
                    "_game_date",
                    "season",
                    "role",
                    "computed_at",
                    "id",
                    "game_pk",
                ]
            )
        ]

        X = df[feature_cols].copy().select_dtypes(include=["number"])
        X = X.fillna(X.median()).fillna(0)
        X = X.loc[:, (X != X.iloc[0]).any()]  # Drop constants

        logger.info(f"Uranium Matrix built: {X.shape[0]} games, {X.shape[1]} features.")
        return X, y
